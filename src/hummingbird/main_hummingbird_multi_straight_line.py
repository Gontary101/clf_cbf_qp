#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import division
import math, ast, numpy as np, rospy
from enum import Enum
from nav_msgs.msg import Odometry
from mav_msgs.msg import Actuators
from geometry_msgs.msg import Point, Vector3, Pose, Twist
from std_msgs.msg import Float64MultiArray, String
from tf.transformations import euler_from_quaternion
from gazebo_msgs.msg import ModelStates 

from trajectory.straight_line import StraightLineTrajectory
import utils.dynamics_utils2 as dyn
from utils.dynamics_utils2 import pget, rotation_matrix
from clf_backstepping import CLFBackstepping
from obstacle_avoidance.zcbf_filter     import ZCBFFilter as SAFETYFilter
#from obstacle_avoidance.C3BF_filter import C3BFFilter as SAFETYFilter

LOG_T = 1.0
DBG   = False

class State(Enum):
    TAKEOFF = 1
    HOVER   = 2
    TRAJ    = 3
    LAND    = 4
    IDLE    = 5

class Controller(object):

    def __init__(self):
        self.ns     = rospy.get_namespace().strip('/') # Get namespace automatically
        if not self.ns: # Handle case where namespace might be global
             self.ns = pget("namespace", "iris") # Fallback to param if needed
             rospy.logwarn("Could not determine namespace automatically, using param: %s", self.ns)
        else:
             rospy.loginfo("Controller running in namespace: %s", self.ns)

        self.use_gz = pget("use_model_states", False)
        if not self.use_gz:
             rospy.logerr("Multi-drone collision avoidance requires use_model_states=true to access /gazebo/model_states.")
             rospy.signal_shutdown("Configuration Error")
             return

        self.model  = dyn.DroneModel()
        self.trajectory   = StraightLineTrajectory()
        self.yaw_fix      = self.trajectory.yaw_fix

        self.x_hover_target = self.trajectory.p0[0]
        self.y_hover_target = self.trajectory.p0[1]
        self.z_hover_target = self.trajectory.p0[2]

        def gains(tag, k1, k2, a1, a2):
            return [pget("%s%s" % (tag, n), dflt)
                    for n, dflt in zip(("pos1", "pos2", "att1", "att2"),
                                       (k1, k2, a1, a2))]
        self.g_take = gains("k_take", 0.22, 0.8,  2.05,  4.1)
        self.g_traj = gains("k_traj", 0.75, 4.1, 16.00, 32.0)

        self.clf = CLFBackstepping(self.model)

        # --- Multi-Drone Setup ---
        try:
            all_ns_str = pget("all_drone_namespaces", "[]")
            self.all_drone_namespaces = ast.literal_eval(all_ns_str)
            if not isinstance(self.all_drone_namespaces, list):
                raise ValueError("all_drone_namespaces is not a list")
            rospy.loginfo("[%s] Aware of drones: %s", self.ns, self.all_drone_namespaces)
        except (ValueError, SyntaxError) as e:
            rospy.logerr("[%s] Invalid all_drone_namespaces parameter '%s': %s", self.ns, all_ns_str, e)
            rospy.signal_shutdown("Configuration Error")
            return

        # Store state of other drones {ns: {'pose': Pose, 'twist': Twist, 'time': Time}}
        self.other_drone_states = {
            other_ns: {'pose': None, 'twist': None, 'time': rospy.Time(0)}
            for other_ns in self.all_drone_namespaces if other_ns != self.ns
        }
        self.state_timeout = rospy.Duration(1.0) # Consider other drone state stale after 1s

        # Static obstacles parsed from param (can still be used if needed)
        self.static_obs = self._parse_obstacles()

        cbf_par  = dict(beta   = pget("zcbf_beta",   1.0),
                        a1     = pget("zcbf_a1",     0.2),
                        a2     = pget("zcbf_a2",     1.0),
                        gamma  = pget("zcbf_gamma",  2.4),
                        kappa  = pget("zcbf_kappa",  1.0),
                        order_a= pget("zcbf_order_a", 0))
        self.cbf_pub = rospy.Publisher("~cbf/slack",
                                       Float64MultiArray, queue_size=1)
        # Initialize CBF with empty dynamic obstacles; will be updated in loop
        self.zcbf = SAFETYFilter(self.model, np.empty((0, 10)), cbf_par,
                               cbf_pub=self.cbf_pub)

        # --- Publishers ---
        # Use relative topic names (will be automatically namespaced)
        self.cmd_pub = rospy.Publisher("command/motor_speed",
                                       Actuators, queue_size=1)
        misc_topics = [
            ("control/state",              String), ("control/U", Float64MultiArray),
            ("control/omega_sq",           Float64MultiArray), ("error/position", Point),
            ("error/velocity",             Vector3), ("error/attitude_deg", Point),
            ("error/rates_deg_s",          Vector3), ("control/desired_position", Point),
            ("control/desired_velocity",   Vector3), ("control/desired_acceleration", Vector3),
            ("control/desired_attitude_deg", Point), ("control/virtual_inputs", Point),
        ]
        self.pubs = {n: rospy.Publisher("~" + n, m, queue_size=1) for n, m in misc_topics}

        # --- State Variables ---
        self.last        = None # Stores Odometry derived from ModelStates for this drone
        self.state       = State.TAKEOFF
        self.t0_traj     = None
        self.hover_ok_t  = None
        self.initial_xy_pos = None
        self.yaw_ramp_start_t = None
        self.current_hover_yaw = self.yaw_fix
        self.yaw_ramp_duration = rospy.Duration(pget("hover_yaw_ramp_secs", 2.0))
        self.target_hover_yaw = self.trajectory.psi_d
        self.trajectory.xy_offset = None
        self.trajectory.z_offset  = None

        # --- Subscribers ---
        # Subscribe to global ModelStates for own state and others' states
        self.model_state_sub = rospy.Subscriber("/gazebo/model_states",
                                                ModelStates, self.cb_model,
                                                queue_size=1, buff_size=2**24) # Increased buffer

        rate = pget("control_rate", 200.0)
        self.timer = rospy.Timer(rospy.Duration(1.0 / rate),
                                 self.loop, reset=True)
        rospy.on_shutdown(self.shutdown)

    def _parse_obstacles(self):
        default_obs_str = "[]"
        try:
            raw = pget("dynamic_obstacles", default_obs_str) # Keep param name for potential static obs
            lst = ast.literal_eval(raw)
            if not isinstance(lst, list): lst = []
            if lst and not all(isinstance(o, (list, tuple)) and len(o) == 10 and all(isinstance(n, (int, float)) for n in o) for o in lst): lst = []
            obs_array = np.array(lst, dtype=float)
            if obs_array.ndim == 1 and obs_array.size == 0: obs_array = np.empty((0, 10), dtype=float)
            elif obs_array.ndim == 1 and obs_array.shape[0] == 10: obs_array = obs_array.reshape(1, 10)
            elif obs_array.ndim != 2 or (obs_array.size > 0 and obs_array.shape[1] != 10): obs_array = np.empty((0,10), dtype=float)
            if obs_array.size > 0: rospy.loginfo("[%s] Loaded %d static obstacles.", self.ns, obs_array.shape[0])
            return obs_array
        except (ValueError, SyntaxError) as e:
             rospy.logwarn("[%s] Error parsing static obstacles parameter '%s': %s. Using empty list.", self.ns, raw, e)
             return np.empty((0,10), dtype=float)
        except Exception as e:
             rospy.logwarn("[%s] Unexpected error processing static obstacles '%s': %s. Using empty list.", self.ns, raw, e)
             return np.empty((0,10), dtype=float)

    def cb_model(self, msg):
        now = rospy.Time.now() # Use consistent time
        found_self = False
        for i, name in enumerate(msg.name):
            # Update own state
            if name == self.ns:
                o = Odometry()
                o.header.stamp    = now
                o.header.frame_id = "world"
                o.child_frame_id  = self.ns + "/base_link" # Standard convention
                o.pose.pose  = msg.pose[i]
                o.twist.twist = msg.twist[i]
                self.last = o
                found_self = True
            # Update other drone states
            elif name in self.other_drone_states:
                self.other_drone_states[name]['pose'] = msg.pose[i]
                self.other_drone_states[name]['twist'] = msg.twist[i]
                self.other_drone_states[name]['time'] = now

        if not found_self and self.last is None: # Only warn if never found
             rospy.logwarn_throttle(5.0, "[%s] Own state not found in /gazebo/model_states", self.ns)


    def loop(self, _evt):
        if self.last is None:
            rospy.logwarn_throttle(2.0, "[%s] Waiting for initial state...", self.ns)
            return
        now = rospy.Time.now()
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
        v_world   = np.array([v.x, v.y, v.z]) # Gazebo twist is in world frame
        omega_b   = np.dot(R_mat.T, np.array([w.x, w.y, w.z])) # Gazebo angular twist is in body frame

        vd = ad_nom = np.zeros(3)
        gains = self.g_take

        if self.state == State.TAKEOFF:
            if self.initial_xy_pos is None:
                self.initial_xy_pos = p_vec[:2].copy()
                rospy.loginfo("[%s] Takeoff initiated. Holding XY at [%.2f, %.2f], climbing to Z=%.2f",
                              self.ns, self.initial_xy_pos[0], self.initial_xy_pos[1], self.z_hover_target)
            tgt = np.array([self.initial_xy_pos[0], self.initial_xy_pos[1], self.z_hover_target])
            yd, rd = self.yaw_fix, 0.0
            pos_thr_z = pget("hover_pos_threshold", 0.15)
            vel_thr   = pget("hover_vel_threshold", 0.10)
            err_z     = abs(p_vec[2] - self.z_hover_target)
            err_v_z   = abs(v_world[2] - vd[2])
            if err_z < pos_thr_z and err_v_z < vel_thr:
                rospy.loginfo("[%s] Takeoff Z reached. TRANSITION  →  HOVER", self.ns)
                self.state = State.HOVER
                self.hover_ok_t = None
                self.yaw_ramp_start_t = now
                self.current_hover_yaw = self.yaw_fix

        elif self.state == State.HOVER:
            tgt = np.array([self.x_hover_target, self.y_hover_target, self.z_hover_target])
            if self.yaw_ramp_start_t is not None and self.yaw_ramp_duration.to_sec() > 1e-6:
                elapsed_ramp_time = (now - self.yaw_ramp_start_t).to_sec()
                ramp_fraction = np.clip(elapsed_ramp_time / self.yaw_ramp_duration.to_sec(), 0.0, 1.0)
                delta_psi = self.target_hover_yaw - self.yaw_fix
                delta_psi = (delta_psi + math.pi) % (2 * math.pi) - math.pi
                self.current_hover_yaw = self.yaw_fix + delta_psi * ramp_fraction
                self.current_hover_yaw = (self.current_hover_yaw + math.pi) % (2 * math.pi) - math.pi
                yd, rd = self.current_hover_yaw, 0.0
                if ramp_fraction >= 1.0:
                    rospy.loginfo_once("[%s] Hover yaw ramp complete.", self.ns)
                    self.yaw_ramp_start_t = None
                    yd, rd = self.target_hover_yaw, 0.0
            else:
                 yd, rd = self.target_hover_yaw, 0.0

            pos_thr = pget("hover_pos_threshold", 0.15)
            vel_thr = pget("hover_vel_threshold", 0.10)
            err_pos = np.linalg.norm(p_vec - tgt)
            err_vel = np.linalg.norm(v_world - vd)
            yaw_ramp_complete = (self.yaw_ramp_start_t is None)

            if err_pos < pos_thr and err_vel < vel_thr and yaw_ramp_complete:
                if self.hover_ok_t is None:
                    self.hover_ok_t = now
                elif (now - self.hover_ok_t) >= rospy.Duration(pget("hover_stabilization_secs", 1.0)):
                    rospy.loginfo("[%s] Hover stable. TRANSITION  →  TRAJ", self.ns)
                    self.state = State.TRAJ
                    self.t0_traj = now
                    self.trajectory.xy_offset = p_vec[:2] - self.trajectory.p0[:2]
                    self.trajectory.z_offset  = p_vec[2]  - self.trajectory.p0[2]
                    rospy.loginfo("[%s] Calculated trajectory offsets: XY=[%.2f, %.2f], Z=%.2f",
                                  self.ns, self.trajectory.xy_offset[0], self.trajectory.xy_offset[1],
                                  self.trajectory.z_offset)
            else:
                self.hover_ok_t = None

        elif self.state == State.TRAJ:
            if self.t0_traj is None:
                rospy.logwarn_throttle(5.0, "[%s] In TRAJ state but t0_traj is None. Reverting to HOVER.", self.ns)
                self.state = State.HOVER
                self.hover_ok_t = None
                self.yaw_ramp_start_t = now
                self.current_hover_yaw = psi
                return
            posd, vd, ad_nom, yd, rd = self.trajectory.ref(t_traj)
            tgt   = posd
            gains = self.g_traj

        else: # LAND, IDLE
            tgt   = p_vec
            vd    = ad_nom = np.zeros(3)
            yd, rd = psi, 0.0
            gains = self.g_take

        # --- Build Dynamic Obstacle List for CBF ---
        dynamic_obs_list = []
        for other_ns, other_state in self.other_drone_states.items():
            if other_state['pose'] is not None and (now - other_state['time']) < self.state_timeout:
                op = other_state['pose'].position
                ov = other_state['twist'].linear
                # Use zero acceleration for other drones as it's not directly available
                # The CBF formulation expects [x,y,z, vx,vy,vz, ax,ay,az, radius]
                obs_entry = [op.x, op.y, op.z, ov.x, ov.y, ov.z, 0.0, 0.0, 0.0, self.model.r_drone]
                dynamic_obs_list.append(obs_entry)
            #else:
            #    rospy.logwarn_throttle(5.0, "[%s] State for drone '%s' is stale or missing.", self.ns, other_ns)

        # Combine static (if any) and dynamic obstacles
        dynamic_obs_array = np.array(dynamic_obs_list, dtype=float)
        if self.static_obs.size > 0:
             combined_obs = np.vstack((self.static_obs, dynamic_obs_array)) if dynamic_obs_array.size > 0 else self.static_obs
        else:
             combined_obs = dynamic_obs_array

        # Update the obstacles used by the ZCBF filter instance for this cycle
        self.zcbf.obs = combined_obs

        # ----- nominal CLF control ------------------------------------------
        st = dict(p_vec=p_vec, v_vec=v_world, phi=phi, th=th,
                  psi=psi,  omega_body=omega_b, R_mat=R_mat)
        ref = dict(tgt=tgt, vd=vd, ad=ad_nom, yd=yd, rd=rd, t_traj_secs=t_traj)
        out = self.clf.compute(st, ref, gains)
        U_nom = out["U_nom"]

        # ----- ZCBF safety filter -------------------------------------------
        # The filter will now use the updated self.zcbf.obs
        st['gains']  = gains
        st['ref']    = ref
        st['ad_nom'] = ad_nom
        U, _ = self.zcbf.filter(self.state.name, U_nom, st, out) # Pass state name

        # ----- motor allocation & publish ------------------------------------
        w_cmd, w_sq = self.model.thrust_torques_to_motor_speeds(U)

        m_msg = Actuators()
        m_msg.header.stamp       = now
        m_msg.angular_velocities = w_cmd.tolist()
        self.cmd_pub.publish(m_msg)

        # -- debug / telemetry
        self.pubs["control/state"].publish(String(data=self.state.name))
        self.pubs["control/U"].publish(Float64MultiArray(data=U))
        self.pubs["control/omega_sq"].publish(Float64MultiArray(data=w_sq))
        self.pubs["error/position"].publish(Point(*out["ex1"]))
        self.pubs["error/velocity"].publish(Vector3(*out["ex2"]))
        self.pubs["error/attitude_deg"].publish(Point(*(math.degrees(i) for i in out["e_th"])))
        self.pubs["error/rates_deg_s"].publish(Vector3(*(math.degrees(i) for i in out["e_w"])))
        self.pubs["control/desired_position"].publish(Point(*tgt))
        self.pubs["control/desired_velocity"].publish(Vector3(*vd))
        self.pubs["control/desired_acceleration"].publish(Vector3(*ad_nom))
        self.pubs["control/desired_attitude_deg"].publish(Point(*(math.degrees(i) for i in (out["phi_d"], out["theta_d"], yd))))
        self.pubs["control/virtual_inputs"].publish(Point(out["Uex"], out["Uey"], 0.0))

        if DBG and combined_obs.size > 0:
            rospy.loginfo_throttle(LOG_T,
                "[%s] State=%s | #Obs=%d | Tgt=[%.1f %.1f %.1f] | U=[%.1f %.1f %.1f %.1f]",
                self.ns, self.state.name, combined_obs.shape[0],
                tgt[0], tgt[1], tgt[2],
                U[0], U[1], U[2], U[3])
        elif DBG:
             rospy.loginfo_throttle(LOG_T,
                "[%s] State=%s | #Obs=0 | Tgt=[%.1f %.1f %.1f] | U=[%.1f %.1f %.1f %.1f]",
                self.ns, self.state.name,
                tgt[0], tgt[1], tgt[2],
                U[0], U[1], U[2], U[3])

        # Add position and attitude logging
        rospy.loginfo_throttle(1.0,
            "[%s] Pos=[%.2f %.2f %.2f] | PosErr=[%.2f %.2f %.2f] | Att=[%.1f %.1f %.1f]° | AttErr=[%.1f %.1f %.1f]°",
            self.ns,
            p_vec[0], p_vec[1], p_vec[2],
            out["ex1"][0], out["ex1"][1], out["ex1"][2],
            math.degrees(phi), math.degrees(th), math.degrees(psi),
            math.degrees(out["e_th"][0]), math.degrees(out["e_th"][1]), math.degrees(out["e_th"][2]))


    def shutdown(self):
        rospy.loginfo("[%s] Shutting down controller.", self.ns)
        stop = Actuators()
        stop.angular_velocities = [0.0] * 4
        rate = rospy.Rate(100)
        for _ in range(10):
            if rospy.is_shutdown(): break
            self.cmd_pub.publish(stop)
            rate.sleep()


if __name__ == "__main__":
    rospy.init_node("clf_hummingbird_trajectory_controller", anonymous=True) # Node name is generic, namespace distinguishes
    try:
        controller = Controller()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
    except Exception as e:
        rospy.logerr("Unhandled exception in controller: %s", e)