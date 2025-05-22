#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import division
import numpy as np
import rospy
from nav_msgs.msg import Odometry
from mav_msgs.msg import Actuators
from std_msgs.msg import Float64MultiArray

import utils.dynamics_utils2 as dyn
from utils.dynamics_utils2 import pget
from utils.obstacle_parser import parse_obstacles

from components import (
    State, TrajectoryModule, StateEstimator, FlightStateManager,
    ControlAllocator, SafetyManager, ActuationHandler, TelemetryPublisher
)

LOG_T = 0.5
DBG = True

class Controller(object):
    def __init__(self):
        self.ns = pget("namespace", "hummingbird")
        self.use_gz = pget("use_model_states", False)

        # for online time-scaling of the helix (see Szadeczky-Kiss & Kiss 2006)
        self.t_traj    = 0.0                       # our internal trajectory time
        self.t_last    = rospy.Time.now().to_sec() # last wall-clock timestamp
        self.prev_phase = None                     # to detect entering TRAJ

        # for avoidance bookkeeping
        self.in_avoidance      = False
        self.avoid_start_wall  = None

        self.model = dyn.DroneModel()

        self.trajectory_module = TrajectoryModule()
        
        initial_takeoff_x = pget("takeoff_x", self.trajectory_module.r0)
        initial_takeoff_y = pget("takeoff_y", 0.0)
        initial_takeoff_z = pget("takeoff_height", 3.0)

        def get_gains_tuple(tag, k1, k2, a1, a2):
            return [pget("%s%s" % (tag, n), dflt)
                    for n, dflt in zip(("pos1", "pos2", "att1", "att2"),
                                       (k1, k2, a1, a2))]
        
        g_take = get_gains_tuple("k_take", 0.22, 0.8, 2.05, 4.1)
        g_traj = get_gains_tuple("k_traj", 0.75, 4.1, 16.00, 32.0)

        self.flight_state_manager = FlightStateManager(
            initial_takeoff_x, initial_takeoff_y, initial_takeoff_z,
            g_take, g_traj, self.trajectory_module
        )
        
        self.state_estimator = StateEstimator()
        self.control_allocator = ControlAllocator(self.model)

        obstacles = parse_obstacles(pget)
        # keep a local copy so we can check distances
        self.obstacles = obstacles
        cbf_params = dict(
            beta=pget("zcbf_beta", 1.5),
            a1=pget("zcbf_a1", 1.5),
            a2=pget("zcbf_a2", 1.6),
            gamma=pget("zcbf_gamma", 8.4),
            kappa=pget("zcbf_kappa", 0.8),
            order_a=pget("zcbf_order_a", 0)
        )
        self.cbf_slack_pub = rospy.Publisher("~cbf/slack", Float64MultiArray, queue_size=1)
        self.safety_manager = SafetyManager(self.model, obstacles, cbf_params, self.cbf_slack_pub)
        
        self.cmd_pub = rospy.Publisher(self.ns + "/command/motor_speed", Actuators, queue_size=1)
        self.actuation_handler = ActuationHandler(self.model, self.cmd_pub)
        
        self.telemetry_publisher = TelemetryPublisher()

        self.last_odom_msg = None
        
        if self.use_gz:
            from gazebo_msgs.msg import ModelStates
            self.sub = rospy.Subscriber("/gazebo/model_states",
                                        ModelStates, self.cb_model,
                                        queue_size=5, buff_size=2**24)
        else:
            self.sub = rospy.Subscriber(self.ns + "/ground_truth/odometry",
                                        Odometry, self.cb_odom, queue_size=10)

        control_rate = pget("control_rate", 500.0)
        self.timer = rospy.Timer(rospy.Duration(1.0 / control_rate),
                                 self.loop, reset=True)
        rospy.loginfo("Controller initialized, starting in phase %s", self.prev_phase)
        rospy.on_shutdown(self.shutdown)

    def cb_odom(self, msg):
        self.last_odom_msg = msg

    def cb_model(self, msg):
        try:
            idx = msg.name.index(self.ns)
        except ValueError:
            try:
                idx = msg.name.index(self.ns + "/")
            except ValueError:
                return
        
        o = Odometry()
        o.header.stamp = rospy.Time.now()
        o.header.frame_id = "world"
        o.child_frame_id = self.ns + "/base_link"
        o.pose.pose, o.twist.twist = msg.pose[idx], msg.twist[idx]
        self.last_odom_msg = o

    def loop(self, event):
        if self.last_odom_msg is None:
            return
        
        now = rospy.Time.now()
        now_sec = now.to_sec()

        current_kinematics = self.state_estimator.process_odometry(self.last_odom_msg, self.use_gz)
        # get the FSM reference & gains (we'll override only in TRAJ)
        refs_fsm, current_gains = self.flight_state_manager.update_state(
            now, current_kinematics
        )
        phase = refs_fsm["state_name"]

        # if we just entered TRAJ, reset our internal trajectory clock
        if phase == "TRAJ" and self.prev_phase != "TRAJ":
            self.t_traj = 0.0
            self.t_last = now_sec

        #  ---- ONLINE TIME-SCALING -----------------------------------------
        if phase == "TRAJ":
            # 1) wall-clock delta
            dt_real = now_sec - self.t_last

            # 2) tracking error at last desired point
            pos_d_prev, _, _, _, _ = self.trajectory_module.get_reference(self.t_traj)
            p_act = current_kinematics["p_vec"]
            e = np.linalg.norm(p_act - pos_d_prev)

            # 3) compute slowdown factor sigma in (0,1]
            # only slow down if within clearance of any obstacle
            clearance_thresh = pget("time_scale_dist", 0.8)
            sigma = 1.0
            if self.obstacles.size > 0:
                # obstacle array columns: [ox, oy, oz, r_o, ...]
                centers = self.obstacles[:, :3]
                radii   = self.obstacles[:, 3]
                # distance to each obstacle surface
                d_to_obs = np.linalg.norm(centers - p_act[None,:], axis=1) - (radii + self.model.r_drone)
                if np.any(d_to_obs < clearance_thresh):
                    k_e = pget("time_scale_k", 0.5)
                    sigma = 1.0 / (1.0 + k_e * e)

            # 4) advance our internal trajectory time
            self.t_traj += sigma * dt_real
            self.t_last = now_sec

            # 5) query the helix at the new time
            posd, vd, ad, yd, rd = \
                self.trajectory_module.get_reference(self.t_traj)


            reference_signals = {
                "tgt": posd,
                "vd" : vd,
                "ad" : ad,
                "yd" : yd,
                "rd" : rd,
                "t_traj_secs": self.t_traj,
                "state_name": phase
            }
        else:
            # not in trajectory phase: use FSM references directly
            reference_signals = refs_fsm

        # remember for next iteration
        self.prev_phase = phase

        clf_output_dict = self.control_allocator.compute_nominal_control(
            current_kinematics, reference_signals, current_gains
        )
        U_nominal_clf = clf_output_dict["U_nom"]

        U_filtered, slack = self.safety_manager.filter_control(
            phase,
            U_nominal_clf,
            current_kinematics, 
            current_gains,
            reference_signals 
        )

        # ─── detect *actual* avoidance ─────────────────────────────────
        # only true if safety QP actually altered the command
        obs_active = not np.allclose(U_filtered, U_nominal_clf, atol=1e-3)

        # ─── avoidance start/stop detection ─────────────────────────────
        if obs_active and not self.in_avoidance:
            # just entered avoidance
            self.in_avoidance     = True
            self.avoid_start_wall = event.current_real

        if not obs_active and self.in_avoidance:
            # just exited avoidance — realign trajectory time
            self._realign_traj_clock(event.current_real, current_kinematics["p_vec"])
            self.in_avoidance = False

        # ─── freeze helix clock while *truly* avoiding ───────────────────
        if obs_active and self.flight_state_manager.t0_traj is not None:
            dt = event.current_real - event.last_real
            self.flight_state_manager.t0_traj += dt
        
        _, w_sq_final = self.actuation_handler.generate_and_publish_motor_commands(
            U_filtered, now
        )
        
        self.telemetry_publisher.publish_telemetry(
            current_kinematics,
            self.flight_state_manager.get_current_state_name(),
            U_filtered,
            w_sq_final,
            clf_output_dict,
            reference_signals,
            U_nominal_clf
        )
        # ─── throttled kinematics logging (0.5 s) ───────────────────────────────
        if DBG:
            # warn once every LOG_T s if our expected kinematic keys are missing
            if "v_world" not in current_kinematics or "omega_b" not in current_kinematics:
                rospy.logwarn_throttle(LOG_T,
                    "Unexpected kinematics keys, available: %s",
                    list(current_kinematics.keys())
                )
            # extract actual kinematics (using the correct key names)
            p = current_kinematics.get("p_vec", np.zeros(3))
            # velocity in world frame
            v = current_kinematics.get("v_world", np.zeros(3))
            # acceleration (fallback to zero if not provided)
            a = current_kinematics.get("a_vec", np.zeros(3))
            # angular velocity in body frame
            w = current_kinematics.get("omega_b", np.zeros(3))
            # extract desired signals
            pd = reference_signals.get("tgt", np.zeros(3))
            vd = reference_signals.get("vd", np.zeros(3))
            ad = reference_signals.get("ad", np.zeros(3))
            rospy.loginfo_throttle(LOG_T,
                "[%s] ACTUAL  pos=[%.4f, %.4f, %.4f] vel=[%.4f, %.4f, %.4f] acc=[%.4f, %.4f, %.4f] ang_vel=[%.4f, %.4f, %.4f] | "
                "DESIRED pos=[%.4f, %.4f, %.4f] vel=[%.4f, %.4f, %.4f] acc=[%.4f, %.4f, %.4f]",
                self.flight_state_manager.get_current_state_name(),
                p[0], p[1], p[2],
                v[0], v[1], v[2],
                a[0], a[1], a[2],
                w[0], w[1], w[2],
                pd[0], pd[1], pd[2],
                vd[0], vd[1], vd[2],
                ad[0], ad[1], ad[2]
            )

        if DBG:
            rospy.loginfo_throttle(LOG_T,
                "[%s] U=[%.4f %.4f %.4f %.4f]  |  Nom=[%.4f %.4f %.4f %.4f]",
                self.flight_state_manager.get_current_state_name(),
                U_filtered[0], U_filtered[1], U_filtered[2], U_filtered[3],
                U_nominal_clf[0], U_nominal_clf[1], U_nominal_clf[2], U_nominal_clf[3])

    def shutdown(self):
        for _ in range(10):
            self.actuation_handler.send_single_stop_command()
            rospy.sleep(0.01)

    def _realign_traj_clock(self, wall_time, current_p):
        """
        Project current position onto the helix to find the best t*, and
        reset t0_traj so that (wall_time – t0_traj) == t*.
        """
        # 1) how far along we *think* we are right now:
        old_t = (wall_time - self.flight_state_manager.t0_traj).to_sec()

        # 2) choose a window to search — e.g. the duration we just spent avoiding
        dt_avoid = (wall_time - self.avoid_start_wall).to_sec()
        t_min, t_max = old_t, old_t + dt_avoid*1.5

        # 3) coarse 1D search (50 samples)
        traj = self.trajectory_module.trajectory
        ts   = np.linspace(t_min, t_max, 50)
        ps   = np.array([traj.ref(t)[0] for t in ts])    # positions
        d2   = np.sum((ps - current_p[None,:])**2, axis=1)
        t_star = ts[np.argmin(d2)]

        # 4) reset the "clock"
        new_t0 = wall_time - rospy.Duration(t_star)
        self.flight_state_manager.t0_traj = new_t0

if __name__ == "__main__":
    rospy.init_node("clf_hummingbird_trajectory_controller", anonymous=True)
    try:
        Controller()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass