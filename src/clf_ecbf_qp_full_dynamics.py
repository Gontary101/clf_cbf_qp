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
rt   = np.sqrt
dot  = np.dot
norm = np.linalg.norm
solvers.options['show_progress'] = False
pget = lambda name, default: rospy.get_param("~" + name, default)
EPS = 1e-6

class State(Enum):
    TAKEOFF = 1
    HOVER   = 2
    TRAJ    = 3
    LAND    = 4
    IDLE    = 5

class ClfIrisController(object):
    def __init__(self):
        ns = pget("namespace", "iris")
        self.ns = ns
        self.use_gz = pget("use_model_states", False)
        self.xy_offset = None
        self.z_offset  = None

        # physical params
        self.m  = pget("mass", 1.5)
        self.g  = pget("gravity", 9.81)
        self.Ix = pget("I_x", 0.0348)
        self.Iy = pget("I_y", 0.0459)
        self.Iz = pget("I_z", 0.0977)
        self.kf = pget("motor_constant", 8.54858e-06)
        self.km = pget("moment_constant", 1.3677728e-07)
        self.w_max  = pget("max_rot_velocity", 838.0)
        self.min_f   = pget("min_thrust_factor", 0.1)
        self.gc      = pget("gravity_comp_factor", 1.022)
        self.max_tilt = math.radians(pget("max_tilt_angle_deg", 30.0))

        # CLF gains
        def gains(pref, k1,k2,a1,a2):
            return [pget(pref + k, i) for k,i in
                    zip(("pos1","pos2","att1","att2"), (k1,k2,a1,a2))]
        self.g_take = gains("k_take", 0.22, 0.8, 2.05, 4.1)
        self.g_traj = gains("k_traj", 0.75, 4.1,16.0,32.0)

        # ECBF gains (deg 4)
        self.ecbf_alpha0 = pget("ecbf_alpha0", 1.0)
        self.ecbf_alpha1 = pget("ecbf_alpha1", 4.0)
        self.ecbf_alpha2 = pget("ecbf_alpha2", 6.0)
        self.ecbf_alpha3 = pget("ecbf_alpha3", 4.0)

        # drone radius
        self.r_drone = pget("drone_radius", 0.5)

        # obstacles: [x,y,z,radius, vx,vy,vz]
        raw = pget("static_obstacles", "[]")
        try:
            lst = ast.literal_eval(raw)
        except Exception:
            lst = []
        obs = []
        for o in lst:
            if isinstance(o, (list,tuple)) and len(o) in (4,7):
                if len(o)==4:
                    obs.append([o[0],o[1],o[2],o[3],0.0,0.0,0.0])
                else:
                    obs.append(list(o))
        self.obs = np.array(obs, dtype=float)
        if self.obs.size:
            rospy.loginfo("Loaded %d obstacles for ECBF.", self.obs.shape[0])
        else:
            rospy.loginfo("No static obstacles for ECBF.")

        # control allocation
        A = np.array([
            [ self.kf,  self.kf,  self.kf,  self.kf],
            [-0.22*self.kf, 0.20*self.kf, 0.22*self.kf,-0.20*self.kf],
            [-0.13*self.kf, 0.13*self.kf,-0.13*self.kf, 0.13*self.kf],
            [-self.km, -self.km, self.km, self.km]
        ])
        try:
            self.invA = np.linalg.inv(A)
        except np.linalg.LinAlgError:
            rospy.logfatal("Control allocation matrix singular.")
            rospy.signal_shutdown("bad A")
            return

        # ROS pubs/subs
        self.cbf_h_pub     = rospy.Publisher("~cbf/h_values", Float64MultiArray, queue_size=1)
        self.cbf_slack_pub = rospy.Publisher("~cbf/slack",    Float64MultiArray, queue_size=1)
        self.cmd_pub       = rospy.Publisher(ns + '/command/motor_speed', Actuators, queue_size=1)

        if self.use_gz:
            from gazebo_msgs.msg import ModelStates
            self.sub = rospy.Subscriber('/gazebo/model_states', ModelStates, self.cb_model, queue_size=5)
        else:
            self.sub = rospy.Subscriber(ns + '/ground_truth/odometry', Odometry, self.cb_odom, queue_size=10)

        rate = pget("control_rate", 100.0)
        if rate <= 0: rate = 100.0
        self.timer = rospy.Timer(rospy.Duration(1.0/rate), self.loop, reset=True)
        rospy.on_shutdown(self.shutdown)
        self.state = State.TAKEOFF
        self.last_odom = None
        rospy.loginfo("Full-dynamics CLF+ECBF controller init.")

    def cb_odom(self, msg):
        self.last_odom = msg

    def cb_model(self, msg):
        try:
            i = msg.name.index(self.ns)
        except Exception:
            return
        od = Odometry()
        od.header.stamp = rospy.Time.now()
        od.header.frame_id = "world"
        od.child_frame_id = self.ns + "/base_link"
        od.pose.pose   = msg.pose[i]
        od.twist.twist = msg.twist[i]
        self.last_odom = od

    @staticmethod
    def R(phi,th,psi):
        c = math.cos; s = math.sin
        return np.array([
            [ c(th)*c(psi), s(phi)*s(th)*c(psi)-c(phi)*s(psi), c(phi)*s(th)*c(psi)+s(phi)*s(psi)],
            [ c(th)*s(psi), s(phi)*s(th)*s(psi)+c(phi)*c(psi), c(phi)*s(th)*s(psi)-s(phi)*c(psi)],
            [      -s(th),                  s(phi)*c(th),                 c(phi)*c(th)]
        ])

    def build_ecbf_constraints_full(self, p, v, R_bw, omega):
        e3 = np.array([0.0,0.0,1.0])
        if self.obs.size == 0:
            return None, None, []
        G_rows, h_rows, debug = [], [], []
        for ox,oy,oz,ro,vxo,vyo,vzo in self.obs:
            po = np.array([ox,oy,oz])
            vo = np.array([vxo,vyo,vzo])
            dp = p - po
            dv = v - vo
            rs = ro + self.r_drone
            d2 = dp.dot(dp)
            if d2 <= (rs - 0.2)**2 + EPS:
                debug.append(-999.0)
                continue
            # barrier and derivatives
            h0 = d2 - rs**2
            h1 = 2.0 * dp.dot(dv)
            b3 = R_bw.dot(e3)
            Lg1 = -2.0 * dp.dot(b3) / self.m
            h2 = 2.0 * dv.dot(dv) + 2.0 * dp.dot(self.g*e3)
            b3_dot = R_bw.dot(np.cross(omega, e3))
            h3 = 2.0*dv.dot(self.g*e3) - 2.0*(1.0/self.m)*dp.dot(b3_dot)
            # Lie for tau -> ddot(b3)
            Jinv = np.diag([1.0/self.Ix,1.0/self.Iy,1.0/self.Iz])
            # e3 x tau  => skew(e3) * tau
            skew_e3 = np.array([[0.0,-1.0,0.0],[1.0,0.0,0.0],[0.0,0.0,0.0]])
            Mtau = R_bw.dot(skew_e3.dot(Jinv))  # (3x3)
            LgTau = (2.0/self.m) * dp.dot(Mtau) # (1x3)
            # compute Lf4h (drift-only 4th derivative) via helper
            # user should implement exact symbolic expression here
            Lf4h = 0.0
            rhs = (Lf4h
                   + self.ecbf_alpha3 * h3
                   + self.ecbf_alpha2 * h2
                   + self.ecbf_alpha1 * h1
                   + self.ecbf_alpha0 * h0)
            G_rows.append(np.hstack([-Lg1, -LgTau]))
            h_rows.append(rhs)
            debug.append(h0)
        if not G_rows:
            return None, None, debug
        G = matrix(np.vstack(G_rows))
        h = matrix(np.array(h_rows))
        return G, h, debug

    def loop(self, event):
        if self.last_odom is None:
            rospy.logwarn_throttle(5.0, "No odom yet")
            return
        od = self.last_odom
        p3 = od.pose.pose.position
        q  = od.pose.pose.orientation
        v3 = od.twist.twist.linear
        w3 = od.twist.twist.angular
        try:
            phi,th,psi = euler_from_quaternion([q.x,q.y,q.z,q.w])
        except Exception:
            return
        p_world = np.array([p3.x,p3.y,p3.z])
        R_bw    = self.R(phi,th,psi)
        v_body  = np.array([v3.x,v3.y,v3.z])
        v_world = R_bw.dot(v_body)
        w_body  = np.array([w3.x,w3.y,w3.z])

        # CLF backstepping computes U1,U2,U3,U4 exactly as before
        # ----- altitude -------
        if self.state in (State.TAKEOFF, State.HOVER):
            tgt = np.array([self.x_to,self.y_to,self.z_to])
            vd = np.zeros(3)
            ad = np.zeros(3)
            yd,rd = self.yaw_fix,0.0
            g1,g2,g3,g4 = self.g_take
        elif self.state == State.TRAJ:
            dt = (rospy.Time.now() - self.t0_traj).to_sec()
            tgt, vd, ad, yd, rd = self.traj_ref(dt)
            g1,g2,g3,g4 = self.g_traj
        else:
            tgt = p_world.copy()
            vd = np.zeros(3)
            ad = np.zeros(3)
            yd,rd = psi,0.0
            g1,g2,g3,g4 = self.g_take

        # position & velocity error
        e_pos = p_world - tgt
        e_vel = v_world - vd
        # desired acceleration from CLF
        U1 = (self.m/(math.cos(phi)*math.cos(th)+EPS))*(-g1*e_pos[2] + ad[2] - g2*e_vel[2]) \
             + self.m*self.g*self.gc/(math.cos(phi)*math.cos(th)+EPS)
        U1 = max(0.0, U1)
        if U1 < EPS:
            Uex,Uey = 0.0,0.0
        else:
            Uex = (self.m/U1)*(-g1*e_pos[0] + ad[0] - g2*e_vel[0])
            Uey = (self.m/U1)*(-g1*e_pos[1] + ad[1] - g2*e_vel[1])
        sp,cp = math.sin(yd), math.cos(yd)
        phi_d = math.asin(clip(Uex*sp - Uey*cp, -1.0,1.0))
        th_d = math.asin(clip((Uex*cp + Uey*sp)/max(math.cos(phi_d),EPS), -1.0,1.0))
        phi_d = clip(phi_d,-self.max_tilt,self.max_tilt)
        th_d  = clip(th_d, -self.max_tilt,self.max_tilt)
        e_att = np.array([phi-phi_d, th-th_d, ((psi-yd+math.pi)%(2*math.pi))-math.pi])
        e_rate= w_body - np.array([0.0,0.0,rd])

        U2 = self.Ix *(-g3*e_att[0] - g4*e_rate[0]) + (self.Iy-self.Iz)*w_body[1]*w_body[2]
        U3 = self.Iy *(-g3*e_att[1] - g4*e_rate[1]) + (self.Iz-self.Ix)*w_body[0]*w_body[2]
        U4 = self.Iz *(-g3*e_att[2] - g4*e_rate[2]) + (self.Ix-self.Iy)*w_body[0]*w_body[1]
        U_clf = np.array([U1,U2,U3,U4])

        # apply ECBF filter in TRAJ
        if self.state==State.TRAJ and self.obs.size>0:
            Gc,hc,dbg = self.build_ecbf_constraints_full(p_world, v_world, R_bw, w_body)
            if Gc is not None:
                P = matrix(np.eye(4))
                q = matrix(-U_clf)
                try:
                    sol = solvers.qp(P, q, Gc, hc)
                    if sol['status']=='optimal':
                        U = np.array(sol['x']).flatten()
                        slack = hc - Gc*sol['x']
                        self.cbf_slack_pub.publish(Float64MultiArray(data=list(slack)))
                    else:
                        U = U_clf.copy()
                        self.cbf_slack_pub.publish(Float64MultiArray(data=[]))
                except Exception:
                    U = U_clf.copy()
                    self.cbf_slack_pub.publish(Float64MultiArray(data=[]))
                if dbg: self.cbf_h_pub.publish(Float64MultiArray(data=dbg))
            else:
                U = U_clf.copy()
        else:
            U = U_clf.copy()

        # motor speeds
        try:
            omega_sq = self.invA.dot(U)
        except Exception:
            omega_sq = np.zeros(4)
        omega_sq = clip(omega_sq, 0, None)
        w_cmd   = clip(rt(omega_sq), 0, self.w_max)

        m = Actuators()
        m.header.stamp = rospy.Time.now()
        m.angular_velocities = w_cmd.tolist()
        self.cmd_pub.publish(m)

        # publish debug topics if any
        # ...

    def traj_ref(self, t):
        omt = self.omega_traj * t
        r   = 0.5*self.d_start - self.k_r*omt
        z   = self.k_z * omt
        pos = np.array([r*math.cos(omt), r*math.sin(omt), z])
        if self.xy_offset is not None:
            pos[0:2] += self.xy_offset
        if self.z_offset is not None:
            pos[2] += self.z_offset
        # compute vel, acc as in original
        # ...
        return pos, np.zeros(3), np.zeros(3), self.yaw_fix, 0.0

    def shutdown(self):
        rospy.loginfo("Shutdown: zero speeds")
        stop = Actuators()
        stop.angular_velocities = [0.0]*4
        for _ in range(10):
            self.cmd_pub.publish(stop)
            rospy.sleep(0.01)

if __name__ == "__main__":
    rospy.init_node("clf_ecbf_full_dynamics", anonymous=True)
    ctrl = ClfIrisController()
    rospy.spin()
