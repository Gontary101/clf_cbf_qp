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
from trajectory.straight_line import StraightLineTrajectory
from gazebo_msgs.msg import ModelStates
# --- local helpers -----------------------------------------------------------
import utils.dynamics_utils2 as dyn
from utils.dynamics_utils2 import pget, rotation_matrix 
from clf_backstepping import CLFBackstepping
from obstacle_avoidance.zcbf_filter     import ZCBFFilter as SAFETYFilter


LOG_T = 1.0
DBG   = True 


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
        # force use of ModelStates for multi-drone collision avoidance
        self.use_gz = pget("use_model_states", True)

        # ------------- quadrotor model --------------------------------------
        self.model  = dyn.DroneModel()

        # ------------- trajectory selection -----------------------------------
        # instantiate straight-line trajectory
        self.trajectory   = StraightLineTrajectory()
        self.yaw_fix      = self.trajectory.yaw_fix

        # ------------- take‑off / hover goal --------------------------------
        # take-off / hover goal — read the private '~trajectory_start_point' param,
        # parse it if it's a string, else fall back to trajectory.p0
        raw_pt = pget("trajectory_start_point", None)
        if isinstance(raw_pt, str):
            try:
                pt = ast.literal_eval(raw_pt)
            except Exception:
                rospy.logwarn("[%s] Could not parse trajectory_start_point '%s', using default p0", self.ns, raw_pt)
                pt = tuple(self.trajectory.p0)
        elif isinstance(raw_pt, (list, tuple)) and len(raw_pt) >= 3:
            pt = raw_pt
        else:
            pt = tuple(self.trajectory.p0)
        self.x_to, self.y_to, self.z_to = float(pt[0]), float(pt[1]), float(pt[2])

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
                        order_a= pget("zcbf_order_a", 0)) # Defaults, will be overridden
        self.cbf_pub = rospy.Publisher("~cbf/slack",
                                       Float64MultiArray, queue_size=1)
        self.zcbf = SAFETYFilter(self.model, self.obs, cbf_par,
                                 cbf_pub=self.cbf_pub)

        # --- Multi-Drone Setup ------------------------------------------------
        try:
            all_ns_str = pget("all_drone_namespaces", "[]")
            self.all_drone_namespaces = ast.literal_eval(all_ns_str)
            if not isinstance(self.all_drone_namespaces, list):
                raise ValueError("not a list")
            rospy.loginfo("[%s] Aware of drones: %s", self.ns, self.all_drone_namespaces)
        except Exception as e:
            rospy.logerr("[%s] Invalid all_drone_namespaces='%s': %s", self.ns, all_ns_str, e)
            rospy.signal_shutdown("Configuration Error")
            return

        # initialise container for others' states
        self.other_drone_states = {
            other: {'pose': None, 'twist': None, 'time': rospy.Time(0)}
            for other in self.all_drone_namespaces if other != self.ns
        }
        self.state_timeout = rospy.Duration(1.0)

        # ------------- publishers -------------------------------------------
        self.cmd_pub = rospy.Publisher("command/motor_speed",
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
        
        # ******** INITIALIZE OFFSETS *BEFORE* USING THEM IN trajectory ********
        self.trajectory.xy_offset = None
        self.trajectory.z_offset  = None
        # *********************************************************************

        self.psi_traj0   = self.trajectory.ref(0.0)[3]

        # subscribe to ModelStates for this + other drones
        if self.use_gz:
            self.model_state_sub = rospy.Subscriber("/gazebo/model_states",
                                                    ModelStates, self.cb_model,
                                                    queue_size=1, buff_size=2**24)
        else:
            rospy.logerr("[%s] Multi-drone requires use_model_states=True", self.ns)
            rospy.signal_shutdown("Configuration Error")
            return

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
        now = rospy.Time.now()
        found_self = False
        for i, name in enumerate(msg.name):
            if name == self.ns:
                o = Odometry()
                o.header.stamp    = now
                o.header.frame_id = "world"
                o.child_frame_id  = self.ns + "/base_link"
                o.pose.pose       = msg.pose[i]
                o.twist.twist     = msg.twist[i]
                self.last = o
                found_self = True
            elif name in self.other_drone_states:
                self.other_drone_states[name]['pose']  = msg.pose[i]
                self.other_drone_states[name]['twist'] = msg.twist[i]
                self.other_drone_states[name]['time']  = now
        if not found_self and self.last is None:
            rospy.logwarn_throttle(5.0, "[%s] Own state not found in /gazebo/model_states", self.ns)

    # ------------------------------------------------ main control loop -----
    def loop(self, _evt):
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

            if self.state == State.TAKEOFF and p_vec[2] >= self.z_to - pos_thr:
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
                        self.trajectory.xy_offset = np.array([self.x_to - self.trajectory.p0[0], self.y_to - self.trajectory.p0[1]])
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

        # ----- build dynamic obstacle list from other drones ---------------
        dynamic_obs = []
        for ns, s in self.other_drone_states.items():
            if s['pose'] is not None and (now - s['time']) < self.state_timeout:
                p_o = s['pose'].position
                v_o = s['twist'].linear
                dynamic_obs.append([p_o.x, p_o.y, p_o.z,
                                     v_o.x, v_o.y, v_o.z,
                                     0.0, 0.0, 0.0, self.model.r_drone])
        dyn_arr = np.array(dynamic_obs, dtype=float)
        if self.obs.size > 0:
            combined_obs = np.vstack((self.obs, dyn_arr)) if dyn_arr.size > 0 else self.obs
        else:
            combined_obs = dyn_arr
        self.zcbf.obs = combined_obs

        # ----- ZCBF safety filter -------------------------------------------
        st['gains']  = gains
        st['ref']    = ref
        st['ad_nom'] = ad_nom
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