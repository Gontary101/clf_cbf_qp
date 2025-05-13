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
#from trajectory.helix import HelixTrajectory
from trajectory.straight_line import StraightLineTrajectory
# --- local helpers -----------------------------------------------------------
import dynamics_utils as dyn
from dynamics_utils import pget, rotation_matrix
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
        ns          = pget("namespace", "iris")
        self.ns     = ns
        self.use_gz = pget("use_model_states", False)

        # ------------- quadrotor model --------------------------------------
        self.model  = dyn.DroneModel()

        # ------------- trajectory selection -----------------------------------
        self.trajectory   = StraightLineTrajectory()
        self.yaw_fix      = self.trajectory.yaw_fix # Yaw used during initial takeoff climb

        # ------------- take‑off / hover goal --------------------------------
        # These define the target point for the HOVER state (start of trajectory)
        self.x_hover_target = self.trajectory.p0[0]
        self.y_hover_target = self.trajectory.p0[1]
        self.z_hover_target = self.trajectory.p0[2]

        # ------------- controller gains -------------------------------------
        def gains(tag, k1, k2, a1, a2):
            return [pget("%s%s" % (tag, n), dflt)
                    for n, dflt in zip(("pos1", "pos2", "att1", "att2"),
                                       (k1, k2, a1, a2))]
        self.g_take = gains("k_take", 0.22, 0.8,  2.05,  4.1)
        self.g_traj = gains("k_traj", 0.75, 4.1, 16.00, 32.0)

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
        ]
        self.pubs = {n: rospy.Publisher("~" + n, m, queue_size=1)
                     for n, m in misc_topics}

        # ------------- state, subs, timers -----------------------------------
        self.last        = None
        self.state       = State.TAKEOFF
        self.t0_traj     = None
        self.hover_ok_t  = None # Time when hover position/velocity criteria met

        # --- New variables for takeoff and hover yaw control ---
        self.initial_xy_pos = None # Store initial XY position for takeoff
        self.yaw_ramp_start_t = None # Time when hover yaw ramp starts
        self.current_hover_yaw = self.yaw_fix # Start hover yaw at takeoff yaw
        self.yaw_ramp_duration = rospy.Duration(pget("hover_yaw_ramp_secs", 2.0)) # Duration for yaw ramp
        self.target_hover_yaw = self.trajectory.psi_d # Final yaw for hover/trajectory

        # ******** INITIALIZE OFFSETS *BEFORE* USING THEM IN trajectory ********
        self.trajectory.xy_offset = None
        self.trajectory.z_offset  = None
        # *********************************************************************

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
            if lst and not all(
                    isinstance(o, (list, tuple)) and len(o) == 10 and
                    all(isinstance(n, (int, float)) for n in o)
                    for o in lst):
                rospy.logwarn("Invalid format in dynamic_obstacles. Expected list of 10-element lists/tuples. Using empty list.")
                lst = []
            obs_array = np.array(lst, dtype=float)
            if obs_array.ndim == 1 and obs_array.size == 0:
                 obs_array = np.empty((0, 10), dtype=float)
            elif obs_array.ndim == 1 and obs_array.shape[0] == 10:
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
        except Exception as e:
             rospy.logwarn("Unexpected error processing dynamic_obstacles '%s': %s. Using empty list.", raw, e)
             return np.empty((0,10), dtype=float)

    # ------------------------------------------------ subscriber callbacks --
    def cb_odom(self, msg):   self.last = msg
    def cb_model(self, msg):
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
        if self.last is None:
            return
        now = rospy.Time.now()

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
        vd = ad_nom = np.zeros(3) # Common for takeoff/hover
        gains = self.g_take       # Common for takeoff/hover

        if self.state == State.TAKEOFF:
            # --- Takeoff Logic ---
            if self.initial_xy_pos is None: # First time in takeoff
                self.initial_xy_pos = p_vec[:2].copy()
                rospy.loginfo("Takeoff initiated. Holding XY at [%.2f, %.2f], climbing to Z=%.2f",
                              self.initial_xy_pos[0], self.initial_xy_pos[1], self.z_hover_target)

            # Target: Hold initial XY, reach target Z
            tgt = np.array([self.initial_xy_pos[0], self.initial_xy_pos[1], self.z_hover_target])
            yd, rd = self.yaw_fix, 0.0 # Use fixed takeoff yaw

            # Check transition condition (only Z matters for takeoff completion)
            pos_thr_z = pget("hover_pos_threshold", 0.15) # Use same threshold for Z
            vel_thr   = pget("hover_vel_threshold", 0.10)
            err_z     = abs(p_vec[2] - self.z_hover_target)
            err_v_z   = abs(v_world[2] - vd[2]) # Check only Z velocity

            # Use Z position and Z velocity for transition criteria
            if err_z < pos_thr_z and err_v_z < vel_thr:
                rospy.loginfo("Takeoff Z reached. TRANSITION  →  HOVER")
                self.state = State.HOVER
                self.hover_ok_t = None # Reset hover stabilization timer
                self.yaw_ramp_start_t = now # Start yaw ramp timer
                self.current_hover_yaw = self.yaw_fix # Ensure ramp starts from takeoff yaw

        elif self.state == State.HOVER:
            # --- Hover Logic ---
            # Target: Desired hover point (trajectory start)
            tgt = np.array([self.x_hover_target, self.y_hover_target, self.z_hover_target])

            # --- Yaw Ramping ---
            if self.yaw_ramp_start_t is not None and self.yaw_ramp_duration.to_sec() > 1e-6:
                elapsed_ramp_time = (now - self.yaw_ramp_start_t).to_sec()
                ramp_fraction = np.clip(elapsed_ramp_time / self.yaw_ramp_duration.to_sec(), 0.0, 1.0)

                # Interpolate yaw using shortest angle difference
                delta_psi = self.target_hover_yaw - self.yaw_fix
                # Wrap delta_psi to [-pi, pi]
                delta_psi = (delta_psi + math.pi) % (2 * math.pi) - math.pi
                self.current_hover_yaw = self.yaw_fix + delta_psi * ramp_fraction
                # Wrap current_hover_yaw to [-pi, pi]
                self.current_hover_yaw = (self.current_hover_yaw + math.pi) % (2 * math.pi) - math.pi

                yd, rd = self.current_hover_yaw, 0.0

                if ramp_fraction >= 1.0:
                    rospy.loginfo_once("Hover yaw ramp complete.")
                    self.yaw_ramp_start_t = None # Stop ramping
                    yd, rd = self.target_hover_yaw, 0.0 # Use final target yaw
            else:
                 # Ramp finished or duration is zero, use final target yaw
                 yd, rd = self.target_hover_yaw, 0.0

            # --- Check transition to TRAJ ---
            pos_thr = pget("hover_pos_threshold", 0.15)
            vel_thr = pget("hover_vel_threshold", 0.10)
            # Check error w.r.t HOVER target
            err_pos = np.linalg.norm(p_vec - tgt)
            err_vel = np.linalg.norm(v_world - vd)

            # Position, velocity, AND yaw ramp must be stable/complete
            yaw_ramp_complete = (self.yaw_ramp_start_t is None)

            if err_pos < pos_thr and err_vel < vel_thr and yaw_ramp_complete:
                if self.hover_ok_t is None:
                    self.hover_ok_t = now # Start stabilization timer
                elif (now - self.hover_ok_t) >= rospy.Duration(
                        pget("hover_stabilization_secs", 1.0)):
                    rospy.loginfo("Hover stable. TRANSITION  →  TRAJ")
                    self.state = State.TRAJ
                    self.t0_traj = now

                    # Calculate offsets based on CURRENT position vs trajectory start point (p0)
                    # This ensures smooth transition even if hover wasn't perfect
                    self.trajectory.xy_offset = p_vec[:2] - self.trajectory.p0[:2]
                    self.trajectory.z_offset  = p_vec[2]  - self.trajectory.p0[2]
                    rospy.loginfo("Calculated trajectory offsets: XY=[%.2f, %.2f], Z=%.2f",
                                  self.trajectory.xy_offset[0], self.trajectory.xy_offset[1],
                                  self.trajectory.z_offset)
            else:
                # Reset stabilization timer if criteria not met
                self.hover_ok_t = None

        elif self.state == State.TRAJ:
            # --- Trajectory Tracking Logic ---
            if self.t0_traj is None:
                rospy.logwarn_throttle(5.0, "In TRAJ state but t0_traj is None. Reverting to HOVER.")
                self.state = State.HOVER
                # Reset hover state variables if reverting
                self.hover_ok_t = None
                self.yaw_ramp_start_t = now # Restart yaw ramp if needed
                self.current_hover_yaw = psi # Start ramp from current actual yaw
                return

            # Get reference from trajectory object (already includes offsets)
            posd, vd, ad_nom, yd, rd = self.trajectory.ref((now - self.t0_traj).to_sec())
            tgt   = posd
            gains = self.g_traj # Use trajectory gains

        else: # LAND, IDLE
            # --- Default Hold Logic ---
            tgt   = p_vec # Hold current position
            vd    = ad_nom = np.zeros(3)
            yd, rd = psi, 0.0 # Hold current yaw
            gains = self.g_take # Use takeoff/hover gains

        # ----- nominal CLF control ------------------------------------------
        st = dict(p_vec=p_vec, v_vec=v_world, phi=phi, th=th,
                  psi=psi,  omega_body=omega_b, R_mat=R_mat)
        ref = dict(tgt=tgt, vd=vd, ad=ad_nom, yd=yd, rd=rd)
        out = self.clf.compute(st, ref, gains)
        U_nom = out["U_nom"]

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
                "[%s] Tgt=[%.2f %.2f %.2f] Yd=%.1f | ErrP=[%.2f %.2f %.2f] | U=[%.2f %.2f %.2f %.2f]",
                self.state.name,
                tgt[0], tgt[1], tgt[2], math.degrees(yd),
                out["ex1"][0], out["ex1"][1], out["ex1"][2],
                U[0], U[1], U[2], U[3])

    # ------------------------------------------------ shutdown -----
    def shutdown(self):
        stop = Actuators()
        stop.angular_velocities = [0.0] * 4
        for _ in range(10):
            self.cmd_pub.publish(stop)
            rospy.sleep(0.01)


# ------------------------------------------------------------------- main ---
if __name__ == "__main__":
    rospy.init_node("clf_iris_trajectory_controller", anonymous=True)
    try:
        Controller()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass