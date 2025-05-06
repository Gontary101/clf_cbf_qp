#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import division
import rospy, math, numpy as np
from enum import Enum
from nav_msgs.msg import Odometry
from mav_msgs.msg import Actuators
from geometry_msgs.msg import Point, Vector3
from std_msgs.msg import Float64MultiArray, String
from tf.transformations import euler_from_quaternion
from cvxopt import matrix, solvers
import ast

solvers.options['show_progress'] = False

clip = np.clip
sign = np.sign
rt   = np.sqrt
def pget(n,d): return rospy.get_param("~"+n,d)

e3_world = np.array([0.0, 0.0, 1.0])
e3_body  = np.array([0.0, 0.0, 1.0])

class State(Enum):
    TAKEOFF=1; HOVER=2; TRAJ=3; LAND=4; IDLE=5
LOG_T = 1.0
DBG   = True

class ClfIrisController(object):
    def __init__(self):
        ns            = pget("namespace","iris")
        self.ns       = ns
        self.use_gz   = pget("use_model_states",False)
        self.xy_offset= None
        self.z_offset = None

        self.m        = pget("mass",1.5)
        self.g        = pget("gravity",9.81)
        self.Ix       = pget("I_x",0.0348)
        self.Iy       = pget("I_y",0.0459)
        self.Iz       = pget("I_z",0.0977)
        self.J_inv_diag = np.diag([1.0/self.Ix, 1.0/self.Iy, 1.0/self.Iz])
        self.J_mat    = np.diag([self.Ix, self.Iy, self.Iz])
        self.kf       = pget("motor_constant",8.54858e-06)
        self.km       = pget("moment_constant",1.3677728e-07)
        self.w_max    = pget("max_rot_velocity",838.0)
        self.min_f    = pget("min_thrust_factor",0.1)
        self.gc       = pget("gravity_comp_factor",1.022)
        self.max_tilt = math.radians(pget("max_tilt_angle_deg",30.0))
        self.r_drone  = pget("drone_radius", 0.5)

        self.d_start    = pget("helix_start_diameter",40.0)
        self.d_end      = pget("helix_end_diameter",15.0)
        self.height     = pget("helix_height",30.0)
        self.laps       = pget("helix_laps",4.0)
        self.omega_traj = pget("trajectory_omega",0.1)
        self.phase_offset = pget("phase_offset",0.0)
        self.remote_state_topic = pget("remote_state_topic","")
        self.remote_state = None
        if self.remote_state_topic:
            rospy.Subscriber(self.remote_state_topic,
                             String, self.cb_remote_state,
                             queue_size=1)
        self.invert_traj = pget("invert_trajectory",False)
        self.yaw_fix    = math.radians(pget("fixed_yaw_deg",0.0))

        self.r0         = 0.5 * self.d_start
        theta_tot      = self.laps * 2.0 * math.pi

        if abs(theta_tot) < 1e-6:
             rospy.logwarn("[%s] Total theta is near zero, trajectory params invalid.", self.ns)
             k_r_base = 0.0
             k_z_base = 0.0
        else:
             k_r_base       = (self.r0 - 0.5*self.d_end) / theta_tot
             k_z_base      = self.height / theta_tot

        self.k_r = (-1 if self.invert_traj else 1) * k_r_base
        self.k_z = (-1 if self.invert_traj else 1) * k_z_base

        tx = pget("takeoff_x", 0.0)
        ty = pget("takeoff_y", 0.0)
        th = pget("takeoff_height", 1.0)
        self.x_to, self.y_to, self.z_to = tx, ty, th

        def gains(pref, k1, k2, a1, a2):
            return [pget(pref+k,i) for k,i in
                    zip(("pos1","pos2","att1","att2"),(k1,k2,a1,a2))]
        self.g_take = gains("k_take",0.22,0.8,2.05,4.1)
        self.g_traj = gains("k_traj",1.5,5.5,16.0,32.0) # Using values from iris2 in launch file

        A = np.array([
            [self.kf]*4,
            [-0.22*self.kf,  0.20*self.kf,  0.22*self.kf, -0.20*self.kf],
            [-0.13*self.kf,  0.13*self.kf, -0.13*self.kf,  0.13*self.kf],
            [-self.km,       -self.km,       self.km,       self.km]
        ])
        self.invA = np.linalg.inv(A)

        self.beta   = pget("zcbf_beta", 1.0)
        self.a1     = pget("zcbf_a1", 0.2)
        self.a2     = pget("zcbf_a2", 1.0)
        self.gamma  = pget("zcbf_gamma", 2.4)
        self.kappa  = pget("zcbf_kappa", 1.0)
        self.a      = pget("zcbf_order_a", 0)
        obstacles_str = pget("dynamic_obstacles", "[]")
        default_obs = []
        try:
            obstacles_list = ast.literal_eval(obstacles_str)
            if not isinstance(obstacles_list, list):
                 rospy.logwarn("[%s] Parsed dynamic_obstacles is not a list. Using default empty list.", self.ns)
                 obstacles_list = default_obs
            elif obstacles_list and not all(isinstance(o, (list, tuple)) and len(o) == 10 and all(isinstance(n, (int, float)) for n in o) for o in obstacles_list):
                 rospy.logwarn("[%s] Invalid format in dynamic_obstacles list items. Expected list of [x,y,z,vx,vy,vz,ax,ay,az,r]. Using default empty list.", self.ns)
                 obstacles_list = default_obs

            self.obs = np.array(obstacles_list, dtype=float)
            if self.obs.size == 0:
                 self.obs = np.empty((0, 10), dtype=float)
            elif self.obs.ndim == 1:
                 if self.obs.shape[0] == 10:
                     self.obs = self.obs.reshape(1, 10)
                 else:
                     rospy.logwarn("[%s] Parsed single dynamic_obstacle does not have 10 elements. Using default empty list.", self.ns)
                     self.obs = np.empty((0, 10), dtype=float)
            elif self.obs.ndim == 2 and self.obs.shape[1] != 10:
                 rospy.logwarn("[%s] Parsed dynamic_obstacles does not have shape (N, 10). Using default empty list.", self.ns)
                 self.obs = np.empty((0, 10), dtype=float)

            if self.obs.size > 0:
                rospy.loginfo("[%s] Loaded %d dynamic obstacles.", self.ns, self.obs.shape[0])
            else:
                rospy.loginfo("[%s] No dynamic obstacles loaded.", self.ns)

        except (ValueError, SyntaxError) as e:
             rospy.logwarn("[%s] Error parsing dynamic_obstacles parameter '%s': %s. Using default empty list.", self.ns, obstacles_str, e)
             self.obs = np.empty((0, 10), dtype=float)
        except Exception as e:
             rospy.logwarn("[%s] Unexpected error processing dynamic_obstacles '%s': %s. Using default empty list.", self.ns, obstacles_str, e)
             self.obs = np.empty((0, 10), dtype=float)

        topic = lambda s: '~'+s
        self.cbf_pub = rospy.Publisher(topic("cbf/slack"),
                                       Float64MultiArray,
                                       queue_size=1)

        self.cmd_pub = rospy.Publisher('/{}/command/motor_speed'.format(ns), Actuators, queue_size=1)

        pubs = [
            ("control/state",String),("control/U",Float64MultiArray),
            ("control/omega_sq",Float64MultiArray),
            ("error/position",Point),("error/velocity",Vector3),
            ("error/attitude_deg",Point),("error/rates_deg_s",Vector3),
            ("control/desired_position",Point),
            ("control/desired_velocity",Vector3),
            ("control/desired_acceleration",Vector3),
            ("control/desired_attitude_deg",Point),
            ("control/virtual_inputs",Point),
        ]
        self.pubs = {n: rospy.Publisher(topic(n),t,queue_size=1)
                     for n,t in pubs}

        if self.use_gz:
            from gazebo_msgs.msg import ModelStates
            self.sub = rospy.Subscriber('/gazebo/model_states',
                                        ModelStates,self.cb_model,
                                        queue_size=5,buff_size=2**24)
        else:
            self.sub      = rospy.Subscriber('/{}/ground_truth/odometry'.format(ns),
                                        Odometry, self.cb_odom, queue_size=10)

        self.state      = State.TAKEOFF
        self.last       = None
        rate            = pget("control_rate",100.0)
        self.timer      = rospy.Timer(rospy.Duration(1.0/rate),
                                      self.loop,reset=True)
        self.t0_traj    = None
        self.hover_ok_t = None
        rospy.on_shutdown(self.shutdown)

    def _traj_ref_relative(self, t):
        omt = self.omega_traj*t + self.phase_offset
        r   = self.r0 - self.k_r*omt
        z   = self.k_z*omt
        c,s = math.cos, math.sin
        x_rel = r*c(omt)
        y_rel = r*s(omt)
        pos    = np.array([x_rel,y_rel,z])

        dr  = -self.k_r * self.omega_traj
        xp  = dr*c(omt) - r*s(omt)*self.omega_traj
        yp  = dr*s(omt) + r*c(omt)*self.omega_traj
        zp  = self.k_z * self.omega_traj
        vel = np.array([xp,yp,zp])

        a_x = -dr*s(omt)*self.omega_traj - (dr*s(omt)*self.omega_traj + r*c(omt)*self.omega_traj**2)
        a_y =  dr*c(omt)*self.omega_traj + (dr*c(omt)*self.omega_traj - r*s(omt)*self.omega_traj**2)
        a_z = 0.0
        acc = np.array([a_x, a_y, a_z])

        return pos, vel, acc, self.yaw_fix, 0.0

    def traj_ref(self, t):
        pos_rel, vel_rel, acc_rel, yaw_d, yaw_rate_d = self._traj_ref_relative(t)

        pos_abs = pos_rel.copy()
        if self.xy_offset is not None:
            pos_abs[0:2] += self.xy_offset
        if self.z_offset is not None:
            pos_abs[2]   += self.z_offset

        return pos_abs, vel_rel, acc_rel, yaw_d, yaw_rate_d

    def cb_odom(self,msg):
        self.last = msg

    def cb_model(self, msg):
        try:
            idx = -1
            for i, model_name in enumerate(msg.name):
                if model_name == self.ns:
                    idx = i
                    break
            if idx == -1:
                rospy.logwarn_throttle(5.0, "Controller '{self.ns}': Model name '{self.ns}' not found in /gazebo/model_states. Available models: {msg.name}")
                return
            o = Odometry()
            o.header.stamp = rospy.Time.now()
            o.header.frame_id = "world"
            o.child_frame_id = self.ns + "/base_link"
            o.pose.pose = msg.pose[idx]
            o.twist.twist = msg.twist[idx]
            self.last = o
        except Exception as e:
            rospy.logerr("Error in cb_model for {self.ns}: {e}")


    @staticmethod
    def R(phi,th,psi):
        c,s=math.cos,math.sin
        return np.array([
            [c(th)*c(psi), s(phi)*s(th)*c(psi)-c(phi)*s(psi), c(phi)*s(th)*c(psi)+s(phi)*s(psi)],
            [c(th)*s(psi), s(phi)*s(th)*s(psi)+c(phi)*c(psi), c(phi)*s(th)*s(psi)-s(phi)*c(psi)],
            [-s(th),       s(phi)*c(th),                      c(phi)*c(th)]])

    def cb_remote_state(self,msg):
        self.remote_state = msg.data

    def loop(self,_):
        if self.last is None: return
        now = rospy.Time.now()

        p = self.last.pose.pose.position
        q_ros = self.last.pose.pose.orientation
        v = self.last.twist.twist.linear
        w = self.last.twist.twist.angular
        x,y,z = p.x, p.y, p.z
        phi,th,psi = euler_from_quaternion(
            [q_ros.x,q_ros.y,q_ros.z,q_ros.w])
        R_mat = self.R(phi,th,psi)
        v_w = np.array([v.x, v.y, v.z])
        w_body = np.array([w.x, w.y, w.z])
        p_vec = np.array([x, y, z])

        if self.state in (State.TAKEOFF,State.HOVER):
            tgt = np.array([self.x_to, self.y_to, self.z_to])
            vd = ad_nom = np.zeros(3)
            yd,rd = self.yaw_fix, 0.0
            g1,g2,g3,g4 = self.g_take

            err_pos_vec = p_vec - tgt
            err_p = np.linalg.norm(err_pos_vec)
            err_v = np.linalg.norm(v_w)

            current_pos_thresh = pget("hover_pos_threshold",.15)
            current_vel_thresh = pget("hover_vel_threshold",.1)

            if (self.state==State.TAKEOFF
                and err_p < current_pos_thresh
                and err_v < current_vel_thresh):
                rospy.loginfo("[{self.ns}] Reached hover target. TRANSITIONING TO HOVER")
                self.state, self.hover_ok_t = State.HOVER, None

            if self.state==State.HOVER:
                if (err_p < current_pos_thresh
                    and err_v < current_vel_thresh):
                    if self.hover_ok_t is None:
                        self.hover_ok_t = now
                        rospy.loginfo("[{self.ns}] Hover stable, starting stabilization timer...")
                    elif (now - self.hover_ok_t >= rospy.Duration(pget("hover_stabilization_secs", 2.0))):
                        if (self.remote_state_topic and self.remote_state != "TRAJ"):
                            pass
                        else:
                            rospy.loginfo("[{self.ns}] Hover complete. TRANSITIONING TO TRAJ")
                            self.state = State.TRAJ
                            self.t0_traj = now

                            P_hover = np.array([self.x_to, self.y_to, self.z_to])
                            P_ref_start, _, _, _, _ = self._traj_ref_relative(t=0.0)
                            offset = P_hover - P_ref_start

                            self.xy_offset = offset[0:2]
                            self.z_offset  = offset[2]
                            rospy.loginfo("[{self.ns}] Calculated offsets: xy={self.xy_offset}, z={self.z_offset}")
                else:
                    if self.hover_ok_t is not None:
                         rospy.loginfo("[{self.ns}] Hover unstable, resetting stabilization timer.")
                    self.hover_ok_t = None

        elif self.state==State.TRAJ:
            if self.t0_traj is None:
                rospy.logwarn("[{self.ns}] Entered TRAJ state without t0_traj set. Reverting to HOVER.")
                self.state = State.HOVER
                tgt = np.array([self.x_to, self.y_to, self.z_to])
                vd = ad_nom = np.zeros(3)
                yd,rd = self.yaw_fix, 0.0
                g1,g2,g3,g4 = self.g_take
            else:
                posd, vd, ad_nom, yd, rd = self.traj_ref((now - self.t0_traj).to_sec())
                tgt = posd
                g1,g2,g3,g4 = self.g_traj

        else:
            tgt = p_vec.copy()
            vd = ad_nom = np.zeros(3)
            yd,rd = psi,0.0
            g1,g2,g3,g4 = self.g_take

        ex1 = p_vec - tgt
        ex2 = v_w - vd

        tof = math.cos(phi)*math.cos(th)
        if abs(tof)<self.min_f:
            tof = sign(tof or 1)*self.min_f

        tof_stable = tof + sign(tof or 1) * 1e-9

        U1_nom = (self.m/tof_stable)*(-g1*ex1[2] + ad_nom[2] - g2*ex2[2]) \
                 + self.m*self.g * self.gc / tof_stable
        U1_nom = max(0.0,U1_nom)

        if U1_nom < 1e-6:
            Uex = Uey = 0.0
        else:
            Uex = (self.m/U1_nom)*(-g1*ex1[0] + ad_nom[0] - g2*ex2[0])
            Uey = (self.m/U1_nom)*(-g1*ex1[1] + ad_nom[1] - g2*ex2[1])

        sp, cp = math.sin(yd), math.cos(yd)
        asin_arg_phi = clip(Uex*sp - Uey*cp, -1.0, 1.0)
        try:
            phi_d = math.asin(asin_arg_phi)
        except ValueError:
             phi_d = sign(asin_arg_phi) * math.pi / 2.0

        cpd = math.cos(phi_d)
        if abs(cpd) < self.min_f:
            theta_d = 0.0
        else:
            asin_arg_theta = clip((Uex*cp + Uey*sp) / cpd, -1.0, 1.0)
            try:
                 theta_d = math.asin(asin_arg_theta)
            except ValueError:
                 theta_d = sign(asin_arg_theta) * math.pi / 2.0

        phi_d   = clip(phi_d, -self.max_tilt, self.max_tilt)
        theta_d = clip(theta_d, -self.max_tilt, self.max_tilt)

        e_th = np.array([phi-phi_d,
                         th-theta_d,
                         (psi-yd+math.pi)%(2*math.pi) - math.pi])
        e_w  = (w_body - np.array([0.0,0.0,rd]))

        U2_nom = (self.Ix*(-g3*e_th[0] - g4*e_w[0])
                  - w_body[1]*w_body[2]*(self.Iy-self.Iz))
        U3_nom = (self.Iy*(-g3*e_th[1] - g4*e_w[1])
                  - w_body[0]*w_body[2]*(self.Iz-self.Ix))
        U4_nom = (self.Iz*(-g3*e_th[2] - g4*e_w[2])
                  - w_body[0]*w_body[1]*(self.Ix-self.Iy))

        U_nom = np.array([U1_nom, U2_nom, U3_nom, U4_nom])
        U = U_nom.copy()

        if self.state == State.TRAJ and self.obs.size > 0:
            G_cbf_list = []
            h_cbf_list = []

            for obs_row in self.obs:
                ox,oy,oz,vx,vy,vz,ax,ay,az,r_o = obs_row
                xo   = np.array([ox,oy,oz])
                Vo   = np.array([vx,vy,vz])
                Ao   = np.array([ax,ay,az])
                r_safe = r_o + self.r_drone

                r      = p_vec - xo
                q      = R_mat[:,2]
                r_dot  = v_w - Vo

                s      = np.dot(r, q)
                sigma  = -self.a1*np.arctan(self.a2*s)
                sig_p  = -self.a1*self.a2/(1+(self.a2*s)**2)
                sig_pp =  2.0 * self.a1 * (self.a2**3) * s / ((1.0 + (self.a2 * s)**2)**2)


                g_hat  = np.dot(r,r) - self.beta*r_safe**2 - sigma
                R_Omxe3 = R_mat.dot(np.cross(w_body, e3_body))
                dot_s = np.dot(r_dot, q) + np.dot(r, R_Omxe3)

                g_hat_d= 2.0*np.dot(r, r_dot) - sig_p * dot_s

                h_val  = self.gamma*g_hat + g_hat_d

                Gamma1 = (2.0*s - sig_p)/self.m

                r_b = np.dot(R_mat.T, r)
                cross_rb_e3 = np.cross(r_b, e3_body)
                Gamma2_vec = sig_p * np.dot(cross_rb_e3, self.J_inv_diag)

                term1 = self.gamma * g_hat_d
                term2 = 2.0 * np.dot(r_dot, r_dot)
                term3 = -2.0 * self.g * np.dot(r, e3_world)
                term4 = -2.0 * np.dot(r, Ao)
                term5 = -sig_pp * (dot_s ** 2)
                term6 = sig_p * self.g * q[2]
                term7 = -sig_p * (2.0 * np.dot(r_dot, R_Omxe3))

                Omega_cross_e3 = np.cross(w_body, e3_body)
                Omega_cross_Omega_cross_e3 = np.cross(w_body, Omega_cross_e3)
                R_Omega_cross_Omega_cross_e3 = np.dot(R_mat, Omega_cross_Omega_cross_e3)
                term8 = -sig_p * np.dot(r, R_Omega_cross_Omega_cross_e3)

                J_Omega = np.dot(self.J_mat, w_body)
                Omega_cross_JOmega = np.cross(w_body, J_Omega)
                xi = np.dot(Omega_cross_JOmega, self.J_inv_diag.T)
                xi_cross_e3 = np.cross(xi, e3_body)
                R_xi_cross_e3 = np.dot(R_mat, xi_cross_e3)
                term9 =  sig_p * np.dot(r, R_xi_cross_e3)

                Gamma3 = term1 + term2 + term3 + term4 + term5 + term6 + term7 + term8 + term9

                G_cbf_row  = np.hstack([-Gamma1, -Gamma2_vec])
                h_cbf_val  = Gamma3 + self.kappa*(h_val**(2*self.a+1))

                G_cbf_list.append(G_cbf_row)
                h_cbf_list.append(h_cbf_val)

            if G_cbf_list:
                cbf_G = np.vstack(G_cbf_list)
                cbf_h = np.array(h_cbf_list, dtype=float).reshape(-1,1)

                U1_max = 4 * self.kf * self.w_max**2
                box_G = np.array([[ 1., 0., 0., 0.]])
                box_h = np.array([[U1_max]])

                G_all = np.vstack([cbf_G, box_G])
                h_all = np.vstack([cbf_h, box_h])

                P = matrix(np.eye(4))
                q_qp = matrix(-U_nom)
                G = matrix(G_all)
                h_qp = matrix(h_all)

                try:
                    sol = solvers.qp(P, q_qp, G, h_qp)
                    if sol['status'] == 'optimal':
                        U  = np.array(sol['x']).flatten()
                        slack_val = h_qp - G * sol['x']
                        self.cbf_pub.publish(Float64MultiArray(data=list(slack_val)))
                    else:
                         rospy.logwarn_throttle(1.0,"[{self.ns}] ZCBF-QP non-optimal (status: {sol['status']}), using nominal U")
                         U = U_nom
                         self.cbf_pub.publish(Float64MultiArray(data=[0.0]*G_all.shape[0]))
                except ValueError as e:
                    rospy.logwarn_throttle(1.0, "[{self.ns}] ZCBF-QP infeasible ({e}), using nominal U")
                    U = U_nom
                    self.cbf_pub.publish(Float64MultiArray(data=[0.0]*G_all.shape[0]))
            else:
                 U = U_nom
                 if self.state == State.TRAJ:
                     self.cbf_pub.publish(Float64MultiArray(data=[]))
        else:
             U = U_nom
             if self.state == State.TRAJ:
                 self.cbf_pub.publish(Float64MultiArray(data=[]))

        w_sq = clip(np.dot(self.invA, U), 0, None)
        w_cmd= clip(rt(w_sq), 0, self.w_max)

        m = Actuators()
        m.header.stamp = now
        m.angular_velocities = w_cmd.tolist()
        self.cmd_pub.publish(m)

        self.pubs["control/state"].publish(String(data=self.state.name))
        self.pubs["control/U"].publish(Float64MultiArray(data=U))
        self.pubs["control/omega_sq"].publish(Float64MultiArray(data=w_sq))
        self.pubs["error/position"].publish(Point(*ex1))
        self.pubs["error/velocity"].publish(Vector3(*ex2))
        self.pubs["error/attitude_deg"].publish(Point(*(math.degrees(i) for i in e_th)))
        self.pubs["error/rates_deg_s"].publish(Vector3(*(math.degrees(i) for i in e_w)))
        self.pubs["control/desired_position"].publish(Point(*tgt))
        self.pubs["control/desired_velocity"].publish(Vector3(*vd))
        self.pubs["control/desired_acceleration"].publish(Vector3(*ad_nom))
        self.pubs["control/desired_attitude_deg"].publish(Point(*(math.degrees(i) for i in (phi_d,theta_d,yd))))
        self.pubs["control/virtual_inputs"].publish(Point(Uex,Uey,0.0))

        if DBG:
             err_p_norm = np.linalg.norm(ex1) if self.state != State.IDLE else 0.0
             err_v_norm = np.linalg.norm(ex2) if self.state != State.IDLE else 0.0
             rospy.loginfo_throttle(
                 LOG_T,
                 "[%s] State: %s | U: [%.2f, %.2f, %.2f, %.2f] | U_nom: [%.2f, %.2f, %.2f, %.2f] | Err P: %.2f | Err V: %.2f",
                 self.ns, self.state.name,
                 U[0], U[1], U[2], U[3],
                 U_nom[0], U_nom[1], U_nom[2], U_nom[3],
                 err_p_norm, err_v_norm)

    def shutdown(self):
        rospy.loginfo("[{self.ns}] Shutting down controller, sending zero commands.")
        stop = Actuators()
        stop.angular_velocities = [0.0]*4
        rate = rospy.Rate(100)
        for _ in range(10):
            self.cmd_pub.publish(stop)
            try:
                rate.sleep()
            except rospy.ROSInterruptException:
                break

if __name__=="__main__":
    rospy.init_node("zclf_iris_trajectory_controller", anonymous=True)
    controller = None # Define controller in broader scope for finally block
    try:
        controller = ClfIrisController()
        rospy.loginfo("[{controller.ns}] ZCBF CLF QP Controller Initialized.")
        rospy.spin()
    except rospy.ROSInterruptException:
        rospy.loginfo("ROS Interrupt received. Shutting down.")
    except Exception as e:
        rospy.logerr("Unhandled exception in controller: {e}", exc_info=True)
    finally:
        if controller is not None:
             controller.shutdown()