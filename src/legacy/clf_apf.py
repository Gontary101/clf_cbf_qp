#!/usr/bin/env python
# -*- coding: utf-8 -*-


from __future__ import division

import ast
import math

import numpy as np
import rospy
from enum import Enum
from geometry_msgs.msg import Point, Vector3
from mav_msgs.msg import Actuators
from nav_msgs.msg import Odometry
from std_msgs.msg import Float64MultiArray, String
from tf.transformations import euler_from_quaternion

# --------------------------------------------------------------------------- #
#  helpers                                                                    #
# --------------------------------------------------------------------------- #

clip = np.clip
rt = np.sqrt
dot = np.dot
norm = np.linalg.norm

pget = lambda n, d: rospy.get_param("~" + n, d)
EPS = 1.0e-4


class State(Enum):
    TAKEOFF = 1
    HOVER = 2
    TRAJ = 3
    LAND = 4
    IDLE = 5



class ClfIrisControllerAFP(object):
    def __init__(self):
        # -------------------- namespace -------------------------------- #
        ns = pget("namespace", "iris")
        self.ns = ns
        self.use_gz = pget("use_model_states", False)

        # -------------------- vehicle params --------------------------- #
        self.m = pget("mass", 1.5)
        self.g = pget("gravity", 9.81)
        self.Ix = pget("I_x", 0.0348)
        self.Iy = pget("I_y", 0.0459)
        self.Iz = pget("I_z", 0.0977)
        self.kf = pget("motor_constant", 8.54858e-06)
        self.km = pget("moment_constant", 1.3677728e-07)
        self.w_max = pget("max_rot_velocity", 838.0)
        self.min_f = pget("min_thrust_factor", 0.1)
        self.gc = pget("gravity_comp_factor", 1.022)
        self.max_tilt = math.radians(pget("max_tilt_angle_deg", 30.0))

        # -------------------- APF parameters --------------------------- #
        self.eta = pget("apf_eta", 1.2)        # strength of potential
        self.d0 = pget("apf_d0", 3.0)          # influence radius [m]
        self.k_rep = pget("apf_rep_gain", 1.0) # gain on translational field
        self.k_rot = pget("apf_rot_coef", 0.3) # gain on rotational field
        self.d_margin = pget("apf_margin", 0.2)         # extra clearance [m]
        self.a_apf_max = pget("apf_max_acc", 5.0)       # accel saturation [m/s²]
        self.apf_debug = pget("apf_debug", False)

        # -------------------- trajectory parameters -------------------- #
        self.d_start = pget("helix_start_diameter", 40.0)
        self.d_end = pget("helix_end_diameter", 15.0)
        self.height = pget("helix_height", 30.0)
        self.laps = pget("helix_laps", 4.0)
        self.omega_traj = pget("trajectory_omega", 0.1)
        self.yaw_fix = math.radians(pget("fixed_yaw_deg", 0.0))

        # derived
        self.r0 = 0.5 * self.d_start
        theta_tot = self.laps * 2.0 * math.pi
        self.k_r = (self.r0 - 0.5 * self.d_end) / theta_tot
        self.k_z = self.height / theta_tot

        # -------------------- take-off target -------------------------- #
        self.x_to = pget("takeoff_x", self.r0)
        self.y_to = pget("takeoff_y", 0.0)
        self.z_to = pget("takeoff_height", 3.0)

        # -------------------- controller gains ------------------------- #
        def gains(pref, k1, k2, a1, a2):
            return [pget(pref + k, d) for k, d in
                    zip(("pos1", "pos2", "att1", "att2"), (k1, k2, a1, a2))]

        self.g_take = gains("k_take", 0.22, 0.8, 2.05, 4.1)
        self.g_traj = gains("k_traj", 0.75, 4.1, 16.0, 32.0)

        # -------------------- control allocation ----------------------- #
        A = np.array([
            [self.kf, self.kf, self.kf, self.kf],
            [-0.22 * self.kf, 0.20 * self.kf, 0.22 * self.kf, -0.20 * self.kf],
            [-0.13 * self.kf, 0.13 * self.kf, -0.13 * self.kf, 0.13 * self.kf],
            [-self.km, -self.km, self.km, self.km]
        ])
        try:
            self.invA = np.linalg.inv(A)
        except np.linalg.LinAlgError:
            rospy.logfatal("Control allocation matrix A is singular")
            rospy.signal_shutdown("Singular allocation matrix")
            return

        # -------------------- obstacles -------------------------------- #
        self.r_drone = pget("drone_radius", 0.5)
        obs_raw = pget("static_obstacles", "[]")
        try:
            parsed_list = ast.literal_eval(obs_raw)
            if not isinstance(parsed_list, list):
                parsed_list = []
        except Exception:
            rospy.logwarn("Failed to parse static_obstacles")
            parsed_list = []

        parsed = []
        for o in parsed_list:
            if isinstance(o, (list, tuple)):
                if len(o) == 4:          # (x, y, z, radius)
                    parsed.append(list(o) + [0.0, 0.0, 0.0])
                elif len(o) == 7:        # (x, y, z, radius, vx, vy, vz)
                    parsed.append(list(o))
        self.obs = np.array(parsed, dtype=float)
        rospy.loginfo("Loaded %d static obstacles", self.obs.shape[0])

        # -------------------- pubs / subs ------------------------------ #
        self.cmd_pub = rospy.Publisher(
            ns + "/command/motor_speed", Actuators, queue_size=1)

        topic = lambda s: "~" + s
        pubs = [
            ("control/state", String),
            ("control/U", Float64MultiArray),
            ("control/omega_sq", Float64MultiArray),
            ("error/position", Point),
            ("error/velocity", Vector3),
            ("error/attitude_deg", Point),
            ("error/rates_deg_s", Vector3),
            ("control/desired_position", Point),
            ("control/desired_velocity", Vector3),
            ("control/desired_acceleration", Vector3),
            ("control/desired_attitude_deg", Point),
            ("control/virtual_inputs", Point),
            ("apf/acc_repulsive", Vector3),
        ]
        self.pubs = {n: rospy.Publisher(topic(n), t, queue_size=1)
                     for n, t in pubs}

        if self.use_gz:
            from gazebo_msgs.msg import ModelStates
            self.sub = rospy.Subscriber(
                "/gazebo/model_states", ModelStates,
                self.cb_model, queue_size=5, buff_size=2 ** 24)
        else:
            self.sub = rospy.Subscriber(
                ns + "/ground_truth/odometry", Odometry,
                self.cb_odom, queue_size=10)

        # state machine
        self.state = State.TAKEOFF
        self.last_odom = None
        self.t0_traj = None
        self.hover_ok_t = None
        self.xy_offset = None
        self.z_offset = None

        # control loop timer
        rate_hz = pget("control_rate", 100.0)
        self.timer = rospy.Timer(rospy.Duration(1.0 / rate_hz),
                                 self.loop, reset=True)

        rospy.on_shutdown(self.shutdown)
        rospy.loginfo("CLF-AFP controller initialised")

    # ------------------------------------------------------------------ #
    #  odometry callbacks                                                #
    # ------------------------------------------------------------------ #
    def cb_odom(self, msg):
        self.last_odom = msg

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
        o.pose.pose = msg.pose[idx]
        o.twist.twist = msg.twist[idx]
        self.last_odom = o

    # ------------------------------------------------------------------ #
    #  helpers                                                           #
    # ------------------------------------------------------------------ #
    @staticmethod
    def R(phi, th, psi):
        c, s = math.cos, math.sin
        return np.array([
            [c(th) * c(psi),
             s(phi) * s(th) * c(psi) - c(phi) * s(psi),
             c(phi) * s(th) * c(psi) + s(phi) * s(psi)],
            [c(th) * s(psi),
             s(phi) * s(th) * s(psi) + c(phi) * c(psi),
             c(phi) * s(th) * s(psi) - s(phi) * c(psi)],
            [-s(th),
             s(phi) * c(th),
             c(phi) * c(th)]
        ])

    def traj_ref(self, t):
        omt = self.omega_traj * t
        r = self.r0 - self.k_r * omt
        z = self.k_z * omt
        c, s = math.cos, math.sin
        pos = np.array([r * c(omt), r * s(omt), z])
        if self.xy_offset is not None:
            pos[0:2] += self.xy_offset
        if self.z_offset is not None:
            pos[2] += self.z_offset

        dr_dt = -self.k_r * self.omega_traj
        vel = np.array([
            dr_dt * c(omt) - r * s(omt) * self.omega_traj,
            dr_dt * s(omt) + r * c(omt) * self.omega_traj,
            self.k_z * self.omega_traj
        ])

        acc = np.zeros(3)  # negligible second derivative for this demo
        return pos, vel, acc, self.yaw_fix, 0.0

    # ------------------------------------------------------------------ #
    #  APF                                                               #
    # ------------------------------------------------------------------ #
    def compute_apf_repulsive(self, p_world):
        """Exact gradient-based APF with rotational component and saturation."""
        if not self.obs.size:
            return np.zeros(3)

        a_total = np.zeros(3)
        for ox, oy, oz, r_obs, vx_o, vy_o, vz_o in self.obs:
            r_vec = p_world - np.array([ox, oy, oz])
            dist = norm(r_vec)
            if dist < EPS:
                continue

            shell = r_obs + self.r_drone + self.d_margin
            d_eff = max(dist - shell, EPS)           # always positive
            if d_eff >= self.d0:
                continue                             # outside influence

            inv_d = 1.0 / d_eff
            coeff = self.eta * (inv_d - 1.0 / self.d0) * inv_d ** 2

            # translational component (Khatib 1986)
            a_rep = self.k_rep * coeff * (r_vec / dist)

            # rotational component (Batinovic 2023)
            a_rot = self.k_rot * np.cross([0.0, 0.0, 1.0], a_rep)

            a_obs = a_rep + a_rot

            # physical saturation
            a_norm = norm(a_obs)
            if a_norm > self.a_apf_max:
                a_obs = a_obs / a_norm * self.a_apf_max

            a_total += a_obs

        return a_total

    # ------------------------------------------------------------------ #
    #  main                                                              #
    # ------------------------------------------------------------------ #
    def loop(self, event):
        if self.last_odom is None:
            rospy.logwarn_throttle(5.0, "No odometry received yet")
            return

        now = rospy.Time.now()
        pos = self.last_odom.pose.pose.position
        ori = self.last_odom.pose.pose.orientation
        vel = self.last_odom.twist.twist.linear
        ang = self.last_odom.twist.twist.angular

        # ignore zero pose at sim start
        if (abs(pos.x) < EPS and abs(pos.y) < EPS and abs(pos.z) < EPS
                and self.state == State.TAKEOFF):
            return

        try:
            phi, th, psi = euler_from_quaternion([ori.x, ori.y, ori.z, ori.w])
        except ValueError:
            return

        p_world = np.array([pos.x, pos.y, pos.z])
        v_body = np.array([vel.x, vel.y, vel.z])
        v_world = dot(self.R(phi, th, psi), v_body)

        # ------------ state machine (takeoff / hover / traj) ------------ #
        if self.state in (State.TAKEOFF, State.HOVER):
            tgt = np.array([self.x_to, self.y_to, self.z_to])
            vd = np.zeros(3)
            ad_nom = np.zeros(3)
            yd, rd = self.yaw_fix, 0.0
            g1, g2, g3, g4 = self.g_take

            err_pos = norm(p_world - tgt)
            err_z = abs(pos.z - tgt[2])
            err_v = norm(v_world)
            hover_pos_thr = pget("hover_pos_threshold", 0.15)
            hover_vel_thr = pget("hover_vel_threshold", 0.1)

            if (self.state == State.TAKEOFF and
                    err_z < hover_pos_thr and err_v < hover_vel_thr):
                rospy.loginfo("TAKEOFF complete -> HOVER")
                self.state = State.HOVER
                self.hover_ok_t = None

            if self.state == State.HOVER:
                if err_pos < hover_pos_thr and err_v < hover_vel_thr:
                    if self.hover_ok_t is None:
                        self.hover_ok_t = now
                    elif now - self.hover_ok_t >= rospy.Duration(
                            pget("hover_stabilization_secs", 2.0)):
                        rospy.loginfo("HOVER stable -> TRAJ")
                        self.state = State.TRAJ
                        self.t0_traj = now
                        ref0, _, _, _, _ = self.traj_ref(0.0)
                        self.xy_offset = p_world[0:2] - ref0[0:2]
                        self.z_offset = p_world[2] - ref0[2]
                else:
                    self.hover_ok_t = None

        elif self.state == State.TRAJ:
            if self.t0_traj is None:
                self.state = State.HOVER
                return
            dt = (now - self.t0_traj).to_sec()
            tgt, vd, ad_nom, yd, rd = self.traj_ref(dt)
            g1, g2, g3, g4 = self.g_traj
        else:  # LAND / IDLE
            tgt = p_world.copy()
            vd = np.zeros(3)
            ad_nom = np.zeros(3)
            yd, rd = psi, 0.0
            g1, g2, g3, g4 = self.g_take

        # ------------------ APF avoidance (only on TRAJ) ---------------- #
        a_rep = np.zeros(3)
        if self.state == State.TRAJ:
            a_rep = self.compute_apf_repulsive(p_world)
            if self.apf_debug:
                self.pubs["apf/acc_repulsive"].publish(Vector3(*a_rep))

        acc_safe = ad_nom + a_rep

        # ----------------------- CLF controller ------------------------- #
        e_pos = p_world - tgt
        e_vel = v_world - vd

        tof = math.cos(phi) * math.cos(th)
        tof_safe = max(self.min_f, abs(tof)) * math.copysign(1.0, tof or 1.0)

        U1 = (self.m / tof_safe) * (-g1 * e_pos[2] + acc_safe[2] - g2 * e_vel[2]) \
             + self.m * self.g * self.gc / tof_safe
        U1 = max(0.0, U1)

        if U1 < 1.0e-6:
            Uex = Uey = 0.0
            phi_d, th_d = phi, th
        else:
            Uex = (self.m / U1) * (-g1 * e_pos[0] + acc_safe[0] - g2 * e_vel[0])
            Uey = (self.m / U1) * (-g1 * e_pos[1] + acc_safe[1] - g2 * e_vel[1])
            sp, cp = math.sin(yd), math.cos(yd)
            phi_d = math.asin(clip(Uex * sp - Uey * cp, -1.0, 1.0))
            cpd = math.cos(phi_d)
            if abs(cpd) < self.min_f:
                th_d = 0.0
            else:
                th_d = math.asin(clip((Uex * cp + Uey * sp) / cpd, -1.0, 1.0))

        phi_d = clip(phi_d, -self.max_tilt, self.max_tilt)
        th_d = clip(th_d, -self.max_tilt, self.max_tilt)

        e_att = np.array([
            phi - phi_d,
            th - th_d,
            (psi - yd + math.pi) % (2 * math.pi) - math.pi
        ])
        w_body = np.array([ang.x, ang.y, ang.z])
        w_des = np.array([0.0, 0.0, rd])
        e_rate = w_body - w_des

        U2 = self.Ix * (-g3 * e_att[0] - g4 * e_rate[0]) \
             + (self.Iy - self.Iz) * w_body[1] * w_body[2]
        U3 = self.Iy * (-g3 * e_att[1] - g4 * e_rate[1]) \
             + (self.Iz - self.Ix) * w_body[0] * w_body[2]
        U4 = self.Iz * (-g3 * e_att[2] - g4 * e_rate[2]) \
             + (self.Ix - self.Iy) * w_body[0] * w_body[1]

        U = np.array([U1, U2, U3, U4])
        omega_sq = dot(self.invA, U)
        omega_sq = clip(omega_sq, 0.0, None)
        w_cmd = clip(rt(omega_sq), 0.0, self.w_max)

        # ---------------------------- actuation ------------------------- #
        m = Actuators()
        m.header.stamp = now
        m.angular_velocities = w_cmd.tolist()
        self.cmd_pub.publish(m)

        # ---------------------------- debug ----------------------------- #
        self.pubs["control/state"].publish(String(data=self.state.name))
        self.pubs["control/U"].publish(Float64MultiArray(data=U))
        self.pubs["control/omega_sq"].publish(Float64MultiArray(data=omega_sq))
        self.pubs["error/position"].publish(Point(*e_pos))
        self.pubs["error/velocity"].publish(Vector3(*e_vel))
        self.pubs["error/attitude_deg"].publish(
            Point(*(math.degrees(v) for v in e_att)))
        self.pubs["error/rates_deg_s"].publish(
            Vector3(*(math.degrees(v) for v in e_rate)))
        self.pubs["control/desired_position"].publish(Point(*tgt))
        self.pubs["control/desired_velocity"].publish(Vector3(*vd))
        self.pubs["control/desired_acceleration"].publish(Vector3(*acc_safe))
        self.pubs["control/desired_attitude_deg"].publish(
            Point(*(math.degrees(v) for v in (phi_d, th_d, yd))))
        self.pubs["control/virtual_inputs"].publish(Point(Uex, Uey, 0.0))

    # ------------------------------------------------------------------ #
    #  shutdown                                                          #
    # ------------------------------------------------------------------ #
    def shutdown(self):
        rospy.loginfo("Shutting down CLF-AFP controller")
        if hasattr(self, "timer"):
            self.timer.shutdown()

        stop_msg = Actuators()
        stop_msg.angular_velocities = [0.0] * 4
        for _ in range(10):
            if rospy.is_shutdown():
                break
            if self.cmd_pub.get_num_connections() > 0:
                self.cmd_pub.publish(stop_msg)
            rospy.sleep(0.01)
        rospy.loginfo("Controller stopped")


# --------------------------------------------------------------------------- #
#  main                                                                       #
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    controller = None
    try:
        rospy.init_node("clf_afp_iris_controller", anonymous=True)
        controller = ClfIrisControllerAFP()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
    except Exception as e:
        rospy.logfatal("Unhandled exception: %s", e)
        import traceback
        traceback.print_exc()
    finally:
        if controller is not None:
            controller.shutdown()
