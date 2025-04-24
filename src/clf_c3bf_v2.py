#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
CLF + upgraded C3BF safety-critical controller for the Iris quadrotor
=====================================================================
Implements Equation (16) of Tayal et al. (2024) with state-dependent
penalty gamma(chi) = ||v_rel|| and full Lie derivatives including cos(phi) gradient.
Launch-time parameters:
  namespace, use_model_states, mass, gravity, I_x, I_y, I_z,
  motor_constant, moment_constant, max_rot_velocity, min_thrust_factor,
  gravity_comp_factor, max_tilt_angle_deg,
  helix_start_diameter, helix_end_diameter, helix_height, helix_laps,
  trajectory_omega, fixed_yaw_deg,
  takeoff_x, takeoff_y, takeoff_height,
  hover_pos_threshold, hover_vel_threshold, hover_stabilization_secs,
  k_takepos1, k_takepos2, k_takeatt1, k_takeatt2,
  k_trajpos1, k_trajpos2, k_trajatt1, k_trajatt2,
  cbf_kappa, cbf_eps, drone_radius,
  static_obstacles, dynamic_obstacles,
  control_rate.
"""
from __future__ import division, print_function, absolute_import
import math, rospy, numpy as np
from enum import Enum
from nav_msgs.msg import Odometry
from mav_msgs.msg import Actuators
from geometry_msgs.msg import Point, Vector3
from std_msgs.msg import Float64MultiArray, String
from tf.transformations import euler_from_quaternion
from cvxopt import matrix, solvers
import ast

solvers.options['show_progress'] = False

# Utility functions
clip = np.clip
rt   = np.sqrt
sign = np.sign

def pget(name, default):
    return rospy.get_param('~' + name, default)

class State(Enum):
    TAKEOFF = 1
    HOVER   = 2
    TRAJ    = 3
    LAND    = 4
    IDLE    = 5

LOG_T = 1.0
DBG   = True

class C3bfClfIrisController(object):
    def __init__(self):
        # Parameters
        self.ns       = pget('namespace', 'iris')
        self.use_gz   = pget('use_model_states', False)
        # Physical properties
        self.m        = pget('mass', 1.5)
        self.g        = pget('gravity', 9.81)
        self.Ix       = pget('I_x', 0.0348)
        self.Iy       = pget('I_y', 0.0459)
        self.Iz       = pget('I_z', 0.0977)
        self.kf       = pget('motor_constant', 8.54858e-06)
        self.km       = pget('moment_constant', 1.3677728e-07)
        self.w_max    = pget('max_rot_velocity', 838.0)
        self.min_f    = pget('min_thrust_factor', 0.1)
        self.gc       = pget('gravity_comp_factor', 1.022)
        self.max_tilt = math.radians(pget('max_tilt_angle_deg', 30.0))
        # Helix trajectory parameters
        self.d_start    = pget('helix_start_diameter', 40.0)
        self.d_end      = pget('helix_end_diameter', 15.0)
        self.height     = pget('helix_height', 30.0)
        self.laps       = pget('helix_laps', 4.0)
        self.omega_traj = pget('trajectory_omega', 0.1)
        self.yaw_fix    = math.radians(pget('fixed_yaw_deg', 0.0))
        self.r0         = 0.5 * self.d_start
        theta_tot      = self.laps * 2.0 * math.pi
        self.k_r       = (self.r0 - 0.5 * self.d_end) / theta_tot
        self.k_z       = self.height / theta_tot
        # Takeoff point
        self.x_to = pget('takeoff_x', self.r0)
        self.y_to = pget('takeoff_y', 0.0)
        self.z_to = pget('takeoff_height', 3.0)
        # CLF gains
        def gains(pref):
            return [pget(pref + 'pos1', 0.75), pget(pref + 'pos2', 4.1),
                    pget(pref + 'att1', 16.0), pget(pref + 'att2', 32.0)]
        self.g_take = gains('k_take')
        self.g_traj = gains('k_traj')
        # Mixer inverse
        A = np.array([
            [self.kf, self.kf, self.kf, self.kf],
            [-0.22*self.kf, 0.20*self.kf, 0.22*self.kf, -0.20*self.kf],
            [-0.13*self.kf, 0.13*self.kf, -0.13*self.kf, 0.13*self.kf],
            [-self.km, -self.km, self.km, self.km]
        ])
        self.invA = np.linalg.inv(A)
        # CBF parameters
        self.kappa     = pget('cbf_kappa', 1.0)
        self.eps       = pget('cbf_eps', 1e-6)
        self.drone_r   = pget('drone_radius', 0.5)
        # Obstacles
        self.static_obs  = self._parse_static(pget('static_obstacles', '[]'))
        self.dynamic_obs = self._parse_dynamic(pget('dynamic_obstacles', '[]'))
        # Publishers and subscribers
        self.cbf_pub = rospy.Publisher('~cbf/slack', Float64MultiArray, queue_size=1)
        self.cmd_pub = rospy.Publisher(self.ns + '/command/motor_speed', Actuators, queue_size=1)
        pubs = [
            ('control/state', String), ('control/U', Float64MultiArray),
            ('control/omega_sq', Float64MultiArray), ('error/position', Point),
            ('error/velocity', Vector3), ('error/attitude_deg', Point),
            ('error/rates_deg_s', Vector3), ('control/desired_position', Point),
            ('control/desired_velocity', Vector3), ('control/desired_acceleration', Vector3),
            ('control/desired_attitude_deg', Point), ('control/virtual_inputs', Point)
        ]
        self.pubs = {n: rospy.Publisher('~'+n, t, queue_size=1) for n,t in pubs}
        if self.use_gz:
            from gazebo_msgs.msg import ModelStates
            rospy.Subscriber('/gazebo/model_states', ModelStates, self.cb_model,
                             queue_size=5, buff_size=2**24)
        else:
            rospy.Subscriber(self.ns + '/ground_truth/odometry', Odometry,
                             self.cb_odom, queue_size=10)
        # State
        self.state = State.TAKEOFF
        self.last  = None
        self.t0_traj = None
        self.hover_ok_t = None
        rate = pget('control_rate', 100.0)
        rospy.Timer(rospy.Duration(1.0/rate), self.loop)
        rospy.on_shutdown(self.shutdown)

    def _parse_static(self, s):
        try:
            lst = ast.literal_eval(s)
            ok  = []
            for o in lst:
                if isinstance(o, (list, tuple)) and len(o) == 4:
                    try:
                        ok.append([float(e) for e in o])
                    except (ValueError, TypeError):
                        continue
            return np.array(ok, dtype=float)
        except:
            return np.empty((0,4), dtype=float)

    def _parse_dynamic(self, s):
        try:
            lst = ast.literal_eval(s)
            ok  = []
            for o in lst:
                if isinstance(o, (list, tuple)) and len(o) == 7:
                    try:
                        ok.append([float(e) for e in o])
                    except (ValueError, TypeError):
                        continue
            return np.array(ok, dtype=float)
        except:
            return np.empty((0,7), dtype=float)

    def _parse_dynamic(self, s):
        try:
            lst = ast.literal_eval(s)
            return np.array([o for o in lst if len(o)==7], dtype=float)
        except:
            return np.empty((0,7))

    @staticmethod
    def R(phi, th, psi):
        c,s = math.cos, math.sin
        return np.array([
            [c(th)*c(psi), s(phi)*s(th)*c(psi)-c(phi)*s(psi), c(phi)*s(th)*c(psi)+s(phi)*s(psi)],
            [c(th)*s(psi), s(phi)*s(th)*s(psi)+c(phi)*c(psi), c(phi)*s(th)*s(psi)-s(phi)*c(psi)],
            [    -s(th),                 s(phi)*c(th),                 c(phi)*c(th)]
        ])

    def traj_ref(self, t):
        omt = self.omega_traj * t
        r   = self.r0 - self.k_r * omt
        z   = self.k_z * omt
        c,s = math.cos, math.sin
        pos = np.array([r*c(omt), r*s(omt), z])
        if self.xy_offset is not None: pos[:2] += self.xy_offset
        if self.z_offset  is not None: pos[2]  += self.z_offset
        dr = -self.k_r
        vel = np.array([dr*c(omt)-r*s(omt), dr*s(omt)+r*c(omt), self.k_z]) * self.omega_traj
        acc = np.array([2*self.k_r*s(omt)-r*c(omt), -2*self.k_r*c(omt)-r*s(omt), 0]) * (self.omega_traj**2)
        return pos, vel, acc, self.yaw_fix, 0.0

    def cb_odom(self, msg):
        self.last = msg

    def cb_model(self, msg):
        try:
            idx = msg.name.index(self.ns)
        except ValueError:
            return
        o = Odometry()
        o.header.stamp = rospy.Time.now()
        o.header.frame_id = 'world'
        o.child_frame_id = self.ns + '/base_link'
        o.pose.pose, o.twist.twist = msg.pose[idx], msg.twist[idx]
        self.last = o

    def _c3bf_qp(self, p_w, v_w, a_nom):
        P, q = matrix(np.eye(3)), matrix(-a_nom)
        G_list, h_list = [], []
        obs = []
        if self.static_obs.size: obs += [[ox,oy,oz,0,0,0,r] for ox,oy,oz,r in self.static_obs]
        if self.dynamic_obs.size: obs += self.dynamic_obs.tolist()
        if not obs: return a_nom, []
        for ox,oy,oz,vx,vy,vz,r in obs:
            p_rel = np.array([ox,oy,oz]) - p_w
            v_rel = np.array([vx,vy,vz]) - v_w
            n_p   = rt(np.dot(p_rel,p_rel)) + self.eps
            n_v   = rt(np.dot(v_rel,v_rel)) + self.eps
            r_s   = r + self.drone_r
            diff      = max(n_p**2 - r_s**2, 0.0)
            sqrt_diff = rt(diff)
            cos_phi   = sqrt_diff / n_p
            den       = sqrt_diff + self.eps
            dcos_dp   = (r_s**2) / (n_p**2 * den) * p_rel
            h_val     = np.dot(p_rel, v_rel) + n_p * n_v * cos_phi
            dh_dp     = (v_rel
                        + (p_rel/n_p) * n_v * cos_phi
                        + n_p * n_v * dcos_dp)
            dh_dv     = p_rel + n_p * (v_rel/n_v) * cos_phi
            Lf_h      = np.dot(dh_dp, v_rel)
            Lg_h      = -dh_dv
            G_list.append(matrix(Lg_h))
            h_list.append(Lf_h + self.kappa * h_val)
        G = matrix(np.vstack(G_list))
        h = matrix(np.array(h_list))
        try:
            sol = solvers.qp(P, q, G, h)
            if sol['status'] == 'optimal':
                a     = np.array(sol['x']).flatten()
                slack = (h - G*sol['x']).T.tolist()[0]
                return a, slack
            rospy.logwarn_throttle(1.0, 'C3BF-QP non-optimal (%s)', sol['status'])
        except ValueError:
            rospy.logwarn_throttle(1.0, 'C3BF-QP infeasible')
        return a_nom, [0.0]*len(h_list)

    def loop(self, _):
        if self.last is None: return
        now = rospy.Time.now()
        p_lin = self.last.pose.pose.position
        q     = self.last.pose.pose.orientation
        v_lin = self.last.twist.twist.linear
        w     = self.last.twist.twist.angular
        x,y,z = p_lin.x, p_lin.y, p_lin.z
        phi,th,psi = euler_from_quaternion([q.x,q.y,q.z,q.w])
        v_w = np.dot(self.R(phi,th,psi), [v_lin.x, v_lin.y, v_lin.z])
        # State machine
        if self.state in (State.TAKEOFF, State.HOVER):
            tgt    = np.array([self.x_to, self.y_to, self.z_to])
            vd     = ad_nom = np.zeros(3)
            yd,rd  = self.yaw_fix, 0.0
            g1,g2,g3,g4 = self.g_take
            err_z  = abs(z - tgt[2])
            err_v  = np.linalg.norm(v_w)
            if self.state==State.TAKEOFF and err_z<pget('hover_pos_threshold',0.15) and err_v<pget('hover_vel_threshold',0.1):
                rospy.loginfo('TRANSITIONING TO HOVER')
                self.state, self.hover_ok_t = State.HOVER, None
            if self.state==State.HOVER:
                if err_z<pget('hover_pos_threshold',0.15) and err_v<pget('hover_vel_threshold',0.1):
                    if self.hover_ok_t is None:
                        self.hover_ok_t = now
                    elif (now - self.hover_ok_t)>=rospy.Duration(pget('hover_stabilization_secs',2.0)):
                        rospy.loginfo('TRANSITIONING TO TRAJ')
                        self.state, self.t0_traj = State.TRAJ, now
                        self.xy_offset = np.array([self.x_to - self.r0, self.y_to])
                        self.z_offset  = self.z_to
                else:
                    self.hover_ok_t = None
        elif self.state==State.TRAJ:
            if self.t0_traj is None:
                self.state = State.HOVER; return
            posd, vd, ad_nom, yd, rd = self.traj_ref((now-self.t0_traj).to_sec())
            tgt = posd; g1,g2,g3,g4 = self.g_traj
        else:
            tgt    = np.array([x,y,z]); vd = ad_nom = np.zeros(3)
            yd,rd  = psi, 0.0; g1,g2,g3,g4 = self.g_take
        # Safety filter
        ad, slack = self._c3bf_qp(np.array([x,y,z]), v_w, ad_nom)
        self.cbf_pub.publish(Float64MultiArray(data=slack))
        # CLF control
        ex1    = np.array([x,y,z]) - tgt
        ex2    = v_w - vd
        tof    = math.cos(phi)*math.cos(th)
        if abs(tof) < self.min_f:
            tof = sign(tof or 1) * self.min_f
        U1     = (self.m/tof)*(-g1*ex1[2] + ad[2] - g2*ex2[2]) + self.m*self.g*self.gc/tof
        U1     = max(0.0, U1)
        if U1 < 1e-6:
            Uex = Uey = 0.0
        else:
            Uex = (self.m/U1)*(-g1*ex1[0] + ad[0] - g2*ex2[0])
            Uey = (self.m/U1)*(-g1*ex1[1] + ad[1] - g2*ex2[1])
        sp,cp = math.sin(yd), math.cos(yd)
        phi_d    = math.asin(clip(Uex*sp - Uey*cp, -1,1))
        cpd      = math.cos(phi_d)
        theta_d  = 0.0 if abs(cpd)<self.min_f else math.asin(clip((Uex*cp + Uey*sp)/cpd, -1,1))
        phi_d    = clip(phi_d, -self.max_tilt, self.max_tilt)
        theta_d  = clip(theta_d, -self.max_tilt, self.max_tilt)
        e_th     = np.array([phi-phi_d, th-theta_d, (psi-yd+math.pi)%(2*math.pi)-math.pi])
        e_w      = np.array([w.x,w.y,w.z]) - np.array([0.0,0.0,rd])
        U2       = self.Ix*(-g3*e_th[0] - g4*e_w[0]) - w.y*w.z*(self.Iy-self.Iz)
        U3       = self.Iy*(-g3*e_th[1] - g4*e_w[1]) - w.x*w.z*(self.Iz-self.Ix)
        U4       = self.Iz*(-g3*e_th[2] - g4*e_w[2]) - w.x*w.y*(self.Ix-self.Iy)
        U        = np.array([U1,U2,U3,U4])
        w_sq     = clip(np.dot(self.invA, U), 0, None)
        w_cmd    = clip(rt(w_sq), 0, self.w_max)
        m        = Actuators()
        m.header.stamp = now
        m.angular_velocities = w_cmd.tolist()
        self.cmd_pub.publish(m)
        # Telemetry
        self.pubs['control/state'].publish(String(self.state.name))
        self.pubs['control/U'].publish(Float64MultiArray(U))
        self.pubs['control/omega_sq'].publish(Float64MultiArray(w_sq))
        self.pubs['error/position'].publish(Point(*ex1))
        self.pubs['error/velocity'].publish(Vector3(*ex2))
        self.pubs['error/attitude_deg'].publish(Point(*(math.degrees(i) for i in e_th)))
        self.pubs['error/rates_deg_s'].publish(Vector3(*(math.degrees(i) for i in e_w)))
        self.pubs['control/desired_position'].publish(Point(*tgt))
        self.pubs['control/desired_velocity'].publish(Vector3(*vd))
        self.pubs['control/desired_acceleration'].publish(Vector3(*ad))
        self.pubs['control/desired_attitude_deg'].publish(Point(*(math.degrees(i) for i in (phi_d,theta_d,yd))))
        self.pubs['control/virtual_inputs'].publish(Point(Uex,Uey,0.0))
        if DBG:
            rospy.loginfo_throttle(LOG_T,
                '[%s] U1=%.2f U2=%.2f U3=%.2f U4=%.2f | ad=[%.2f %.2f %.2f]',
                self.state.name, U1,U2,U3,U4, ad[0],ad[1],ad[2])

    def shutdown(self):
        stop = Actuators()
        stop.angular_velocities = [0.0]*4
        for _ in range(10):
            self.cmd_pub.publish(stop)
            rospy.sleep(0.01)

if __name__ == '__main__':
    rospy.init_node('clf_c3bf_iris_trajectory_controller', anonymous=True)
    try:
        C3bfClfIrisController()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
