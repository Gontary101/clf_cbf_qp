#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import division
import rospy
import math
import numpy as np
import ast
from enum import Enum
from cvxopt import matrix, solvers
from nav_msgs.msg import Odometry
from mav_msgs.msg import Actuators
from geometry_msgs.msg import Point, Vector3
from std_msgs.msg import Float64MultiArray, String
from tf.transformations import euler_from_quaternion

clip = np.clip
rt = np.sqrt
dot = np.dot
norm = np.linalg.norm
solvers.options['show_progress'] = False
pget = lambda name, default: rospy.get_param("~" + name, default)
EPS = 1e-4

class State(Enum):
    TAKEOFF = 1
    HOVER = 2
    TRAJ = 3
    LAND = 4
    IDLE = 5

class ClfIrisController(object):
    def __init__(self):
        ns = pget("namespace", "iris")
        self.ns = ns
        self.use_gz = pget("use_model_states", False)
        self.traj_start_pos = None

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

        self.z_final = pget("z_final", 20.0)
        self.slope_deg = pget("slope_deg", 60.0)
        self.traj_speed = pget("traj_speed", 1.0)
        self.yaw_fix = math.radians(pget("fixed_yaw_deg", 0.0))

        tx = pget("takeoff_x", 0.0)
        ty = pget("takeoff_y", 0.0)
        th = pget("takeoff_height", 3.0)
        self.x_to, self.y_to, self.z_to = tx, ty, th

        self.delta_z = 0.0
        self.delta_xy = 0.0
        self.traj_direction_xy = np.array([1.0, 0.0])
        self.traj_vector = np.zeros(3)
        self.traj_length = 0.0
        self.total_traj_time = 0.0
        self.traj_velocity_vector = np.zeros(3)


        def gains(pref, k1, k2, a1, a2):
            return [pget(pref + k, i) for k, i in
                    zip(("pos1", "pos2", "att1", "att2"), (k1, k2, a1, a2))]
        self.g_take = gains("k_take", 0.22, 0.8, 2.05, 4.1)
        self.g_traj = gains("k_traj", 0.75, 4.1, 16.0, 32.0)

        A = np.array([
            [self.kf] * 4,
            [-0.22 * self.kf,  0.20 * self.kf,  0.22 * self.kf, -0.20 * self.kf],
            [-0.13 * self.kf,  0.13 * self.kf, -0.13 * self.kf,  0.13 * self.kf],
            [-self.km,        -self.km,         self.km,         self.km]
        ])
        try:
            self.invA = np.linalg.inv(A)
        except np.linalg.LinAlgError:
            rospy.logfatal("Control allocation matrix A is singular.")
            rospy.signal_shutdown("Singular Matrix A")
            return

        self.gamma = pget("cbf_gamma", 1.0)
        self.r_drone = pget("drone_radius", 0.5)

        obs_raw = pget("static_obstacles", "[]")
        obs_list = []
        try:
            parsed_list = ast.literal_eval(obs_raw)
            if isinstance(parsed_list, list):
                 obs_list = parsed_list
            else:
                 rospy.logwarn("Parsed static_obstacles is not a list, using empty.")
        except (ValueError, SyntaxError) as e:
            rospy.logwarn("Error parsing static_obstacles: %s. Using empty list.", e)
        except Exception as e:
             rospy.logwarn("Unexpected error parsing static_obstacles: %s", e)

        parsed = []
        for o in obs_list:
             if isinstance(o, (list, tuple)):
                if len(o) == 4:
                    parsed.append(list(o) + [0.0, 0.0, 0.0])
                elif len(o) == 7:
                    parsed.append(list(o))
                else:
                    rospy.logwarn("Skipping obstacle with incorrect length %d: %s", len(o), o)
             else:
                 rospy.logwarn("Skipping non-list/tuple obstacle: %s", o)

        self.obs = np.array(parsed, dtype=float)
        if self.obs.size > 0:
             rospy.loginfo("Loaded %d obstacles for C3BF.", self.obs.shape[0])
        else:
             rospy.loginfo("No valid static obstacles loaded for C3BF.")

        self.cbf_h_pub = rospy.Publisher("~cbf/h_values", Float64MultiArray, queue_size=1)
        self.cbf_slack_pub = rospy.Publisher("~cbf/slack", Float64MultiArray, queue_size=1)

        self.cmd_pub = rospy.Publisher(ns + '/command/motor_speed', Actuators, queue_size=1)

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
        ]
        self.pubs = {n: rospy.Publisher(topic(n), t, queue_size=1) for n, t in pubs}

        if self.use_gz:
            from gazebo_msgs.msg import ModelStates
            self.sub = rospy.Subscriber('/gazebo/model_states', ModelStates, self.cb_model, queue_size=5, buff_size=2**24)
        else:
            self.sub = rospy.Subscriber(ns + '/ground_truth/odometry', Odometry, self.cb_odom, queue_size=10)

        self.state = State.TAKEOFF
        self.last_odom = None
        self.t0_traj = None
        self.hover_ok_t = None

        rate = pget("control_rate", 100.0)
        if rate <= 0:
             rospy.logwarn("Control rate must be positive, defaulting to 100 Hz.")
             rate = 100.0
        self.timer = rospy.Timer(rospy.Duration(1.0 / rate), self.loop, reset=True)
        rospy.on_shutdown(self.shutdown)
        rospy.loginfo("C3BF Controller Initialized.")

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

    @staticmethod
    def R(phi, th, psi):
        c = math.cos
        s = math.sin
        return np.array([
            [c(th)*c(psi), s(phi)*s(th)*c(psi)-c(phi)*s(psi), c(phi)*s(th)*c(psi)+s(phi)*s(psi)],
            [c(th)*s(psi), s(phi)*s(th)*s(psi)+c(phi)*c(psi), c(phi)*s(th)*s(psi)-s(phi)*c(psi)],
            [-s(th), s(phi)*c(th), c(phi)*c(th)]
        ])

    def traj_ref(self, t):
        if self.traj_start_pos is None or self.total_traj_time <= EPS:
            rospy.logwarn_throttle(1.0, "Trajectory parameters not initialized.")
            return np.array([self.x_to, self.y_to, self.z_to]), np.zeros(3), np.zeros(3), self.yaw_fix, 0.0

        if t < 0:
            pos = self.traj_start_pos
            vel = np.zeros(3)
            acc = np.zeros(3)
        elif t >= self.total_traj_time:
            pos = self.traj_start_pos + self.traj_vector
            vel = np.zeros(3)
            acc = np.zeros(3)
        else:
            pos = self.traj_start_pos + t * self.traj_velocity_vector
            vel = self.traj_velocity_vector
            acc = np.zeros(3)

        return pos, vel, acc, self.yaw_fix, 0.0


    def build_c3bf_constraints(self, p_world, v_world):
        Gs = []
        hs = []
        h_values_debug = []

        if not self.obs.size:
             return None, None, []

        for i in range(self.obs.shape[0]):
            ox, oy, oz, r_obs, vx_o, vy_o, vz_o = self.obs[i, :]
            p_obs = np.array([ox, oy, oz])
            v_obs = np.array([vx_o, vy_o, vz_o])

            p_rel = p_world - p_obs
            v_rel = v_world - v_obs
            r_s = r_obs + self.r_drone

            d2 = dot(p_rel, p_rel)
            d = rt(d2)

            if d <= r_s - 0.4 + EPS:
                rospy.logwarn_throttle(2.0,"Drone too close (d=%.2f <= r_s=%.2f) to obstacle %d. Skipping constraint.", d, r_s, i)
                h_values_debug.append(-999.0)
                continue

            s2 = dot(v_rel, v_rel)
            s = rt(s2)

            term_inside_sqrt = d2 - r_s**2
            rho = rt(max(EPS**2, term_inside_sqrt))

            p_dot_v = dot(p_rel, v_rel)
            h_i = p_dot_v + s * rho
            h_values_debug.append(h_i)

            Lfh = s2 + (s / (rho + EPS)) * p_dot_v

            Lgh = p_rel + v_rel * (rho / (s + EPS))

            G_row = -Lgh
            b_scalar = Lfh + self.gamma * h_i

            Gs.append(G_row)
            hs.append(b_scalar)

        if not Gs:
            return None, None, h_values_debug

        try:
            G_matrix = matrix(np.vstack(Gs))
            h_vector = matrix(np.array(hs))
        except Exception as e:
             rospy.logerr("Error creating G/h matrices for CVXOPT: %s", e)
             return None, None, h_values_debug

        return G_matrix, h_vector, h_values_debug


    def loop(self, event):
        if self.last_odom is None:
            rospy.logwarn_throttle(5.0, "No odometry received.")
            return
        now = rospy.Time.now()

        p3 = self.last_odom.pose.pose.position
        q = self.last_odom.pose.pose.orientation
        v3 = self.last_odom.twist.twist.linear
        w3 = self.last_odom.twist.twist.angular

        if abs(p3.x) < EPS and abs(p3.y) < EPS and abs(p3.z) < EPS and norm([q.x,q.y,q.z,q.w]-np.array([0,0,0,1])) < EPS and self.state == State.TAKEOFF :
             rospy.loginfo_throttle(1.0, "Waiting for non-zero initial pose/orientation from odometry...")
             return

        x, y, z = p3.x, p3.y, p3.z
        try:
             phi, th, psi = euler_from_quaternion([q.x, q.y, q.z, q.w])
        except ValueError:
             rospy.logwarn_throttle(1.0, "Invalid quaternion received.")
             return

        p_world = np.array([x, y, z])
        R_b_w = self.R(phi, th, psi)
        v_world = dot(R_b_w, [v3.x, v3.y, v3.z])


        if self.state in (State.TAKEOFF, State.HOVER):
            tgt = np.array([self.x_to, self.y_to, self.z_to])
            vd = np.zeros(3)
            ad_nom = np.zeros(3)
            yd, rd = self.yaw_fix, 0.0
            g1, g2, g3, g4 = self.g_take

            err_pos = norm(p_world - tgt)
            err_z = abs(z - tgt[2])
            err_v = norm(v_world)
            hover_pos_thresh = pget("hover_pos_threshold", 0.15)
            hover_vel_thresh = pget("hover_vel_threshold", 0.1)

            if (self.state == State.TAKEOFF and err_z < hover_pos_thresh and err_v < hover_vel_thresh):
                rospy.loginfo("TAKEOFF complete, -> HOVER")
                self.state = State.HOVER
                self.hover_ok_t = None
            if self.state == State.HOVER:
                if (err_pos < hover_pos_thresh and err_v < hover_vel_thresh):
                    if self.hover_ok_t is None:
                        self.hover_ok_t = now
                    elif (now - self.hover_ok_t >= rospy.Duration(pget("hover_stabilization_secs", 2.0))):
                        rospy.loginfo("HOVER stable, -> TRAJ")
                        self.state = State.TRAJ
                        self.t0_traj = now
                        self.traj_start_pos = p_world.copy()
                        z0_actual = self.traj_start_pos[2]
                        self.delta_z = self.z_final - z0_actual

                        if self.delta_z <= EPS:
                            rospy.logwarn("Final Z (%.2f) is not above current Z (%.2f). Staying in HOVER.", self.z_final, z0_actual)
                            self.state = State.HOVER
                            self.t0_traj = None
                            self.traj_start_pos = None
                        else:
                            slope_rad = math.radians(self.slope_deg)
                            if abs(slope_rad - math.pi/2.0) < EPS:
                                self.delta_xy = 0.0
                            elif abs(math.tan(slope_rad)) < EPS:
                                rospy.logwarn("Slope angle is too close to zero. Setting delta_xy to a large number or handle differently.")
                                self.delta_xy = 1000.0 # Or revert to hover
                            else:
                                self.delta_xy = self.delta_z / math.tan(slope_rad)

                            # Using world X direction for horizontal movement
                            self.traj_direction_xy = np.array([1.0, 0.0])
                            self.traj_vector = np.array([self.delta_xy * self.traj_direction_xy[0],
                                                         self.delta_xy * self.traj_direction_xy[1],
                                                         self.delta_z])
                            self.traj_length = norm(self.traj_vector)

                            if self.traj_speed <= EPS:
                                rospy.logwarn("Trajectory speed is zero or negative (%.2f). Setting to 1.0 m/s.", self.traj_speed)
                                self.traj_speed = 1.0

                            if self.traj_length < EPS:
                                rospy.logwarn("Trajectory length is near zero (%.3f). Reverting to HOVER.", self.traj_length)
                                self.state = State.HOVER
                                self.t0_traj = None
                                self.traj_start_pos = None
                            else:
                                self.total_traj_time = self.traj_length / self.traj_speed
                                self.traj_velocity_vector = self.traj_speed * (self.traj_vector / self.traj_length)
                                rospy.loginfo("Starting straight line trajectory: Start=%.2f,%.2f,%.2f End=%.2f,%.2f,%.2f Speed=%.2f Time=%.2f",
                                              self.traj_start_pos[0], self.traj_start_pos[1], self.traj_start_pos[2],
                                              self.traj_start_pos[0]+self.traj_vector[0], self.traj_start_pos[1]+self.traj_vector[1], self.traj_start_pos[2]+self.traj_vector[2],
                                              self.traj_speed, self.total_traj_time)
                else:
                    self.hover_ok_t = None

        elif self.state == State.TRAJ:
            if self.t0_traj is None or self.traj_start_pos is None:
                rospy.logwarn("In TRAJ state but trajectory not initialized. Reverting to HOVER.")
                self.state = State.HOVER
                return
            dt = (now - self.t0_traj).to_sec()
            posd, vd, ad_nom, yd, rd = self.traj_ref(dt)
            tgt = posd
            g1, g2, g3, g4 = self.g_traj

            # Check if trajectory finished
            if dt >= self.total_traj_time:
                 rospy.loginfo("Trajectory finished. Reverting to HOVER at final position.")
                 self.state = State.HOVER
                 self.x_to, self.y_to, self.z_to = posd[0], posd[1], posd[2] # Update hover target
                 self.t0_traj = None
                 self.traj_start_pos = None
                 # Reset trajectory parameters
                 self.delta_z = 0.0
                 self.delta_xy = 0.0
                 self.traj_vector = np.zeros(3)
                 self.traj_length = 0.0
                 self.total_traj_time = 0.0
                 self.traj_velocity_vector = np.zeros(3)
                 # Use hover gains for the next loop iteration
                 g1, g2, g3, g4 = self.g_take
                 tgt = np.array([self.x_to, self.y_to, self.z_to])
                 vd = np.zeros(3)
                 ad_nom = np.zeros(3)
                 yd, rd = self.yaw_fix, 0.0


        else: # LAND, IDLE
            tgt = p_world.copy() # Hold current position if landing/idle logic isn't implemented
            vd = np.zeros(3)
            ad_nom = np.zeros(3)
            yd, rd = psi, 0.0 # Hold current yaw
            g1, g2, g3, g4 = self.g_take # Use takeoff/hover gains

        acc_safe = ad_nom.copy()
        h_debug_data = []

        if self.state == State.TRAJ and self.obs.size > 0:
            G, h, h_debug_data = self.build_c3bf_constraints(p_world, v_world)

            if G is not None and h is not None:
                P = matrix(np.eye(3))
                q = matrix(-ad_nom)
                try:
                    sol = solvers.qp(P, q, G, h)
                    if sol['status'] == 'optimal':
                        acc_safe = np.array(sol['x']).flatten()
                        slack = h - G * sol['x']
                        self.cbf_slack_pub.publish(Float64MultiArray(data=list(slack)))
                    else:
                         rospy.logwarn_throttle(1.0,"C3BF-QP solver status: %s. Using nominal accel.", sol['status'])
                         self.cbf_slack_pub.publish(Float64MultiArray(data=[]))
                except ValueError as e:
                    rospy.logwarn_throttle(1.0, "C3BF-QP ValueError (likely infeasible): %s. Using nominal accel.", e)
                    self.cbf_slack_pub.publish(Float64MultiArray(data=[]))
                except ArithmeticError as e:
                     rospy.logwarn_throttle(1.0, "C3BF-QP ArithmeticError: %s. Using nominal accel.", e)
                     self.cbf_slack_pub.publish(Float64MultiArray(data=[]))
                except Exception as e:
                     rospy.logerr_throttle(5.0, "Unhandled QP Solver Exception: %s", e)
                     self.cbf_slack_pub.publish(Float64MultiArray(data=[]))

        if h_debug_data:
            self.cbf_h_pub.publish(Float64MultiArray(data=h_debug_data))


        acc_used = acc_safe
        e_pos = p_world - tgt
        e_vel = v_world - vd

        tof = math.cos(phi) * math.cos(th)
        if abs(tof) < EPS:
             tof_safe = np.sign(tof + EPS) * max(abs(tof), EPS)
             rospy.logwarn_throttle(1.0,"tof near zero (%.3f), using safe value %.3f", tof, tof_safe)
        elif abs(tof) < self.min_f:
             tof_safe = np.sign(tof) * self.min_f
             rospy.logwarn_throttle(1.0,"tof below min_f (%.3f), clipping to %.3f", tof, tof_safe)
        else:
             tof_safe = tof

        U1 = (self.m / tof_safe) * (-g1 * e_pos[2] + acc_used[2] - g2 * e_vel[2]) \
             + self.m * self.g * self.gc / tof_safe
        U1 = max(0.0, U1)

        if U1 < 1e-6:
            Uex = 0.0
            Uey = 0.0
            phi_d = phi
            th_d = th
        else:
            Uex = (self.m / U1) * (-g1 * e_pos[0] + acc_used[0] - g2 * e_vel[0])
            Uey = (self.m / U1) * (-g1 * e_pos[1] + acc_used[1] - g2 * e_vel[1])
            sp = math.sin(yd)
            cp = math.cos(yd)
            asin_arg_phi = clip(Uex*sp - Uey*cp, -1.0, 1.0)
            phi_d = math.asin(asin_arg_phi)
            cpd = math.cos(phi_d)
            if abs(cpd) < self.min_f:
                th_d = 0.0
                if abs(cpd) > EPS:
                     rospy.logwarn_throttle(1.0,"cos(phi_d) near zero (%.3f) in theta_d calc.", cpd)
            else:
                 asin_arg_th = clip((Uex*cp + Uey*sp) / cpd, -1.0, 1.0)
                 th_d = math.asin(asin_arg_th)

        phi_d = clip(phi_d, -self.max_tilt, self.max_tilt)
        th_d = clip(th_d, -self.max_tilt, self.max_tilt)

        e_att = np.array([
            phi - phi_d,
            th - th_d,
            (psi - yd + math.pi) % (2*math.pi) - math.pi
        ])
        w_body = np.array([w3.x, w3.y, w3.z])
        w_des = np.array([0.0, 0.0, rd])
        e_rate = w_body - w_des

        U2 = self.Ix * (-g3*e_att[0] - g4*e_rate[0]) + (self.Iy - self.Iz) * w_body[1] * w_body[2]
        U3 = self.Iy * (-g3*e_att[1] - g4*e_rate[1]) + (self.Iz - self.Ix) * w_body[0] * w_body[2]
        U4 = self.Iz * (-g3*e_att[2] - g4*e_rate[2]) + (self.Ix - self.Iy) * w_body[0] * w_body[1]

        U = np.array([U1, U2, U3, U4])
        try:
            omega_sq = dot(self.invA, U)
        except Exception as e:
             rospy.logerr("Error in control allocation: %s", e)
             omega_sq = np.zeros(4)

        omega_sq = clip(omega_sq, 0, None)
        w_cmd = clip(rt(omega_sq), 0, self.w_max)

        m = Actuators()
        m.header.stamp = now
        m.angular_velocities = w_cmd.tolist()
        self.cmd_pub.publish(m)

        self.pubs["control/state"].publish(String(data=self.state.name))
        self.pubs["control/U"].publish(Float64MultiArray(data=U))
        self.pubs["control/omega_sq"].publish(Float64MultiArray(data=omega_sq))
        self.pubs["error/position"].publish(Point(*e_pos))
        self.pubs["error/velocity"].publish(Vector3(*e_vel))
        self.pubs["error/attitude_deg"].publish(Point(*(math.degrees(v) for v in e_att)))
        self.pubs["error/rates_deg_s"].publish(Vector3(*(math.degrees(v) for v in e_rate)))
        self.pubs["control/desired_position"].publish(Point(*tgt))
        self.pubs["control/desired_velocity"].publish(Vector3(*vd))
        self.pubs["control/desired_acceleration"].publish(Vector3(*acc_used))
        self.pubs["control/desired_attitude_deg"].publish(Point(*(math.degrees(v) for v in (phi_d, th_d, yd))))
        self.pubs["control/virtual_inputs"].publish(Point(Uex, Uey, 0.0))

        rospy.loginfo_throttle(
            1.0,
            "[%s] U1=%.2f U2=%.2f U3=%.2f U4=%.2f | AccNom(%.2f,%.2f,%.2f) AccSafe(%.2f,%.2f,%.2f)",
            self.state.name, U1, U2, U3, U4,
            ad_nom[0], ad_nom[1], ad_nom[2],
            acc_safe[0], acc_safe[1], acc_safe[2]
        )

    def shutdown(self):
        rospy.loginfo("Shutting down C3BF Controller. Sending zero motor speeds.")
        if hasattr(self, 'timer'):
             self.timer.shutdown()
        stop_msg = Actuators()
        stop_msg.angular_velocities = [0.0] * 4
        count = 0
        while count < 10 and not rospy.is_shutdown():
             if hasattr(self, 'cmd_pub') and self.cmd_pub.get_num_connections() > 0:
                  self.cmd_pub.publish(stop_msg)
             try:
                  rospy.sleep(0.01)
             except rospy.ROSTimeMovedBackwardsException:
                  pass
             except rospy.ROSInterruptException:
                  break
             count += 1
        rospy.loginfo("Zero commands sent attempt finished.")

if __name__ == "__main__":
    controller = None
    try:
        rospy.init_node("clf_c3bf_iris_controller", anonymous=True)
        controller = ClfIrisController()
        rospy.spin()
    except rospy.ROSInterruptException:
        rospy.loginfo("ROS interrupt received.")
    except Exception as e:
        rospy.logfatal("Unhandled exception in controller __main__: %s", e)
        import traceback
        traceback.print_exc()
    finally:
         if controller is not None and hasattr(controller, 'shutdown'):
              rospy.loginfo("Executing shutdown sequence in finally block.")
              controller.shutdown()
         else:
              rospy.loginfo("Controller object not available for shutdown in finally block.")
         rospy.loginfo("C3BF Controller node finished.")