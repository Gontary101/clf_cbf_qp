#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import division
import math, ast, numpy as np, rospy
from enum import Enum
from nav_msgs.msg import Odometry
from mav_msgs.msg import Actuators
from geometry_msgs.msg import Point, Vector3
from std_msgs.msg import Float64MultiArray, String
from tf.transformations import euler_from_quaternion
from trajectory.helix import HelixTrajectory
# --- local helpers -----------------------------------------------------------
import utils.dynamics_utils2 as dyn
from utils.dynamics_utils2 import pget, rotation_matrix 
from clf_backstepping import CLFBackstepping
from obstacle_avoidance.zcbf_filter     import ZCBFFilter as SAFETYFilter
#from obstacle_avoidance.ecbf_forms_filter import ECBFFormsFilter as SAFETYFilter
#from obstacle_avoidance.C3BF_filter import C3BFFilter as SAFETYFilter

LOG_T = 1.0     # seconds between throttled debug prints
DBG   = True    # set False to silence controller‑side logs


class State(Enum):
    TAKEOFF = 1
    HOVER   = 2
    TRAJ    = 3
    LAND    = 4
    IDLE    = 5


class Controller(object):
    """Top‑level node that glues together dynamics, CLF tracking and ZCBF CSFs."""

    # ------------------------------------------------------------------ init -
    def __init__(self):
        # ------------- generic / namespace ----------------------------------
        ns          = pget("namespace", "hummingbird")
        self.ns     = ns
        self.use_gz = pget("use_model_states", False)

        # ------------- quadrotor model --------------------------------------
        self.model  = dyn.DroneModel()

        # ------------- trajectory selection -----------------------------------
        # instantiate helix; later you can swap in other trajectory classes
        self.trajectory   = HelixTrajectory()
        self.omega_traj   = self.trajectory.omega
        self.yaw_fix      = self.trajectory.yaw_fix

        # ------------- take‑off / hover goal --------------------------------
        self.x_to = pget("takeoff_x",      self.trajectory.r0) # Will be overridden
        self.y_to = pget("takeoff_y",      0.0)
        self.z_to = pget("takeoff_height", 3.0)

        # ------------- controller gains -------------------------------------
        def gains(tag, k1, k2, a1, a2):
            return [pget("%s%s" % (tag, n), dflt)
                    for n, dflt in zip(("pos1", "pos2", "att1", "att2"),
                                       (k1, k2, a1, a2))]
        self.g_take = gains("k_take", 0.22, 0.8,  2.05,  4.1) # Defaults, will be overridden by launch
        self.g_traj = gains("k_traj", 0.75, 4.1, 16.00, 32.0) # Defaults, will be overridden by launch

        # ------------- CLF back‑stepping controller --------------------------
        self.clf = CLFBackstepping(self.model)

        # ------------- obstacle list & ZCBF filter ---------------------------
        self.obs = self._parse_obstacles()
        cbf_par  = dict(beta   = pget("zcbf_beta",   1.0),
                        a1     = pget("zcbf_a1",     0.2),
                        a2     = pget("zcbf_a2",     1.0),
                        gamma  = pget("zcbf_gamma",  2.4),
                        kappa  = pget("zcbf_kappa",  1.0),
                        order_a= pget("zcbf_order_a", 0)) 
        self.cbf_pub = rospy.Publisher("~cbf/slack",
                                       Float64MultiArray, queue_size=1)
        self.zcbf = SAFETYFilter(self.model, self.obs, cbf_par,
                               cbf_pub=self.cbf_pub)

        # ------------- publishers -------------------------------------------
        self.cmd_pub = rospy.Publisher(ns + "/command/motor_speed",
                                       Actuators, queue_size=1)

        misc_topics = [
            ("control/state",              String),
            ("control/U",                  Float64MultiArray),
            ("control/omega_sq",           Float64MultiArray),
            ("error/position",             Point),
            ("error/velocity",             Vector3),
            ("error/attitude_deg",         Point),
            ("error/rates_deg_s",          Vector3),
            ("control/desired_position",   Point),
            ("control/desired_velocity",   Vector3),
            ("control/desired_acceleration", Vector3),
            ("control/desired_attitude_deg", Point),
            ("control/virtual_inputs",     Point),
            ("state/position",             Point),          
            ("state/velocity_world",       Vector3),         
            ("state/orientation_eul",      Point),           
            ("state/rates_body",           Vector3),         
            ("control/U_nominal",          Float64MultiArray)
        ]
        self.pubs = {n: rospy.Publisher("~" + n, m, queue_size=1)
                     for n, m in misc_topics}

        # ------------- state, subs, timers -----------------------------------
        self.last        = None
        self.state       = State.TAKEOFF
        self.t0_traj     = None
        self.hover_ok_t  = None
        self.model_states = None
        self.trajectory.xy_offset = None
        self.trajectory.z_offset  = None
        # *********************************************************************

        self.psi_traj0   = self.trajectory.ref(0.0)[3]

        if self.use_gz:
            from gazebo_msgs.msg import ModelStates
            self.sub = rospy.Subscriber("/gazebo/model_states",
                                        ModelStates, self.cb_model,
                                        queue_size=5, buff_size=2**24)
        else:
            self.sub = rospy.Subscriber(ns + "/ground_truth/odometry",
                                        Odometry, self.cb_odom, queue_size=10)

        rate = pget("control_rate", 500.0)
        self.timer = rospy.Timer(rospy.Duration(1.0 / rate),
                                 self.loop, reset=True)
        rospy.on_shutdown(self.shutdown)

    # ------------------------------------------------ private helpers -------
    def _parse_obstacles(self):
        default_obs_str = "[]" 



        try:
            raw = pget("dynamic_obstacles", default_obs_str)
            lst = ast.literal_eval(raw)
            if not isinstance(lst, list):
                rospy.logwarn("Parsed dynamic_obstacles is not a list. Using empty list.")
                lst = []
            
            # Validate format of each obstacle if list is not empty
            if lst and not all(
                    isinstance(o, (list, tuple)) and len(o) == 10 and
                    all(isinstance(n, (int, float)) for n in o)
                    for o in lst):
                rospy.logwarn("Invalid format in dynamic_obstacles. Expected list of 10-element lists/tuples. Using empty list.")
                lst = []

            obs_array = np.array(lst, dtype=float)
            if obs_array.ndim == 1 and obs_array.size == 0: # Handles empty list input
                 obs_array = np.empty((0, 10), dtype=float)
            elif obs_array.ndim == 1 and obs_array.shape[0] == 10: # Single obstacle
                obs_array = obs_array.reshape(1, 10)
            elif obs_array.ndim != 2 or (obs_array.size > 0 and obs_array.shape[1] != 10):
                rospy.logwarn("Parsed dynamic_obstacles does not have shape (N, 10). Using empty list.")
                obs_array = np.empty((0,10), dtype=float)
            
            if obs_array.size > 0:
                rospy.loginfo("Loaded %d dynamic obstacles.", obs_array.shape[0])
            else:
                rospy.loginfo("No dynamic obstacles loaded or an error occurred in parsing.")
            return obs_array

        except (ValueError, SyntaxError) as e:
             rospy.logwarn("Error parsing dynamic_obstacles parameter '%s': %s. Using empty list.", raw, e)
             return np.empty((0,10), dtype=float)
        except Exception as e: # Catch any other unexpected errors
             rospy.logwarn("Unexpected error processing dynamic_obstacles '%s': %s. Using empty list.", raw, e)
             return np.empty((0,10), dtype=float)


    # ------------------------------------------------ subscriber callbacks --
    def cb_odom(self, msg):   self.last = msg
    def cb_model(self, msg):
        self.model_states = msg
        try: idx = msg.name.index(self.ns)
        except ValueError:
            try: idx = msg.name.index(self.ns + "/")
            except ValueError: return
        o = Odometry()
        o.header.stamp    = rospy.Time.now()
        o.header.frame_id = "world"
        o.child_frame_id  = self.ns + "/base_link"
        o.pose.pose, o.twist.twist = msg.pose[idx], msg.twist[idx]
        self.last = o

    # ------------------------------------------------ main control loop -----
    def loop(self, _evt):
        if self.use_gz and getattr(self, 'model_states', None):
            obs_list = []
            for name, pose, twist in zip(self.model_states.name,
                                         self.model_states.pose,
                                         self.model_states.twist):
                if name.startswith('sphere_obstacle_'):
                    x, y, z = (pose.position.x,
                               pose.position.y,
                               pose.position.z)
                    vx, vy, vz = (twist.linear.x,
                                  twist.linear.y,
                                  twist.linear.z)
                    # no accel in ModelStates → zero
                    ax = ay = az = 0.0
                    # sphere radius
                    r = 1.0
                    obs_list.append([x, y, z, vx, vy, vz, ax, ay, az, r])
            if obs_list:
                self.obs = np.array(obs_list, dtype=float)
                self.zcbf.obs = self.obs
                if DBG:
                    rospy.loginfo_throttle(LOG_T, "Updated dynamic obstacles: %d obstacles detected", len(obs_list))
                    for i, obs in enumerate(obs_list):
                        rospy.loginfo_throttle(LOG_T, "Obstacle %d: pos=[%.2f, %.2f, %.2f] vel=[%.2f, %.2f, %.2f] r=%.2f",
                            i, obs[0], obs[1], obs[2], obs[3], obs[4], obs[5], obs[9])
        if self.last is None:
            return
        now = rospy.Time.now()
        # ----------------------------------------------------------------
        if self.state == State.TRAJ and self.t0_traj is not None:
            t_traj = (now - self.t0_traj).to_sec()
        else:
            t_traj = 0.0
        # ----------------------------------------------------------------

        # ----- current state -------------------------------------------------
        p  = self.last.pose.pose.position
        q  = self.last.pose.pose.orientation
        v  = self.last.twist.twist.linear
        w  = self.last.twist.twist.angular

        phi, th, psi = euler_from_quaternion((q.x, q.y, q.z, q.w))
        R_mat = rotation_matrix(phi, th, psi)
        p_vec = np.array([p.x, p.y, p.z])

        if self.use_gz:
            v_world   = np.array([v.x, v.y, v.z])
            omega_b   = np.dot(R_mat.T, np.array([w.x, w.y, w.z]))
        else:
            v_body    = np.array([v.x, v.y, v.z])
            v_world   = np.dot(R_mat, v_body)
            omega_b   = np.array([w.x, w.y, w.z])

        # ----- state machine -------------------------------------------------
        if self.state in (State.TAKEOFF, State.HOVER):
            tgt  = np.array([self.x_to, self.y_to, self.z_to])
            vd   = ad_nom = np.zeros(3)
            # yd, rd selection logic is crucial here
            yd, rd = ((self.psi_traj0, 0.0) if self.state == State.HOVER
                      else (self.yaw_fix, 0.0))
            gains = self.g_take

            pos_thr = pget("hover_pos_threshold", 0.15)
            vel_thr = pget("hover_vel_threshold", 0.10)
            err_z   = abs(p_vec[2] - tgt[2])
            err_v   = np.linalg.norm(v_world - vd)

            if (self.state == State.TAKEOFF
                and err_z < pos_thr and err_v < vel_thr):
                rospy.loginfo("TRANSITION  →  HOVER")
                self.state, self.hover_ok_t = State.HOVER, None

            if self.state == State.HOVER:
                if err_z < pos_thr and err_v < vel_thr:
                    if self.hover_ok_t is None:
                        self.hover_ok_t = now
                    elif (now - self.hover_ok_t) >= rospy.Duration(
                            pget("hover_stabilization_secs", 2.0)):
                        rospy.loginfo("TRANSITION  →  TRAJ")
                        self.state, self.t0_traj = State.TRAJ, now
                        self.trajectory.xy_offset = np.array([self.x_to - self.trajectory.r0, self.y_to])
                        self.trajectory.z_offset  = self.z_to

                else:
                    self.hover_ok_t = None

        elif self.state == State.TRAJ:
            if self.t0_traj is None: # guard
                rospy.logwarn_throttle(5.0, "In TRAJ state but t0_traj is None. Reverting to HOVER.")
                self.state = State.HOVER
                return
            # use the same t_traj that we just computed above
            posd, vd, ad_nom, yd, rd = self.trajectory.ref(t_traj)
            tgt   = posd
            gains = self.g_traj

        else:
            tgt   = p_vec # Hold current position
            vd    = ad_nom = np.zeros(3)
            yd, rd = psi, 0.0 # Hold current yaw
            gains = self.g_take

        # ----- nominal CLF control ------------------------------------------
        # pack state & reference (including our trajectory timestamp!)
        st = dict(p_vec=p_vec, v_vec=v_world,
                  phi=phi, th=th, psi=psi,
                  omega_body=omega_b, R_mat=R_mat)
        ref = dict(tgt    = tgt,
                   vd     = vd,
                   ad     = ad_nom,
                   yd     = yd,
                   rd     = rd,
                   t_traj_secs = t_traj,
                   state_name = self.state.name)
        out = self.clf.compute(st, ref, gains)      # -> dict
        U_nom = out["U_nom"]

  
                # --- If we're using the ECBFFormsFilter it expects these extras ---
        st['gains']  = gains     # from either takeoff or traj
        st['ref']    = ref       # original ref dict {tgt, vd, ad, yd, rd}
        st['ad_nom'] = ad_nom    # the nominal acceleration before filtering
        U, _ = self.zcbf.filter(self.state.name, U_nom, st, out)

        # ----- motor allocation & publish ------------------------------------
        w_cmd, w_sq = self.model.thrust_torques_to_motor_speeds(U)
        
        
        m_msg = Actuators()
        m_msg.header.stamp       = now
        m_msg.angular_velocities = w_cmd.tolist()
        self.cmd_pub.publish(m_msg)

        self.pubs["state/position"].publish(Point(*p_vec))
        self.pubs["state/velocity_world"].publish(Vector3(*v_world))
        self.pubs["state/orientation_eul"].publish(Point(phi, th, psi)) # Publish radians
        self.pubs["state/rates_body"].publish(Vector3(*omega_b))      # Publish rad/s
        self.pubs["control/U_nominal"].publish(Float64MultiArray(data=U_nom))
        # -- debug / telemetry
        self.pubs["control/state"].publish(String(data=self.state.name))
        self.pubs["control/U"].publish(Float64MultiArray(data=U))
        self.pubs["control/omega_sq"].publish(Float64MultiArray(data=w_sq))
        self.pubs["error/position"].publish(Point(*out["ex1"]))
        self.pubs["error/velocity"].publish(Vector3(*out["ex2"]))
        self.pubs["error/attitude_deg"].publish(
            Point(*(math.degrees(i) for i in out["e_th"])))
        self.pubs["error/rates_deg_s"].publish(
            Vector3(*(math.degrees(i) for i in out["e_w"])))
        self.pubs["control/desired_position"].publish(Point(*tgt))
        self.pubs["control/desired_velocity"].publish(Vector3(*vd))
        self.pubs["control/desired_acceleration"].publish(Vector3(*ad_nom))
        self.pubs["control/desired_attitude_deg"].publish(
            Point(*(math.degrees(i) for i in
                    (out["phi_d"], out["theta_d"], yd))))
        self.pubs["control/virtual_inputs"].publish(
            Point(out["Uex"], out["Uey"], 0.0))

        if DBG:
            rospy.loginfo_throttle(LOG_T,
                "[%s] U=[%.2f %.2f %.2f %.2f]  |  Nom=[%.2f %.2f %.2f %.2f]",
                self.state.name,
                U[0], U[1], U[2], U[3],
                U_nom[0], U_nom[1], U_nom[2], U_nom[3])

    # ------------------------------------------------ shutdown -----
    def shutdown(self):
        stop = Actuators()
        stop.angular_velocities = [0.0] * 4
        for _ in range(10):
            self.cmd_pub.publish(stop)
            rospy.sleep(0.01)


# ------------------------------------------------------------------- main ---
if __name__ == "__main__":
    rospy.init_node("clf_hummingbird_trajectory_controller", anonymous=True)
    try:
        Controller()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass