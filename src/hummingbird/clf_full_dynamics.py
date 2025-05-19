#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import division
import math
import numpy as np
import rospy

try:
    from cvxopt import matrix, solvers
    solvers.options['show_progress'] = False
    _HAVE_CVXOPT = True
except ImportError:
    rospy.logwarn("CVXOPT not available – CLF-QP falls back to a simple geometric PD controller (still stable but non-optimal).")
    _HAVE_CVXOPT = False


def _vee(R):
    return np.array([R[2, 1] - R[1, 2],
                     R[0, 2] - R[2, 0],
                     R[1, 0] - R[0, 1]]) / 2.0


def _rotmat_to_eul(R):
    theta = -math.asin(np.clip(R[2, 0], -1.0, 1.0))
    cth = math.cos(theta)
    if abs(cth) < 1e-6:
        phi = 0.0
        psi = math.atan2(-R[0, 1], R[1, 1])
    else:
        phi = math.atan2(R[2, 1] / cth, R[2, 2] / cth)
        psi = math.atan2(R[1, 0] / cth, R[0, 0] / cth)
    return phi, theta, psi


class CLFBackstepping(object):
    def __init__(self, model):
        self.model = model
        self.w_u     = rospy.get_param("~clf_qp_wu",     5.0)
        self.w_slack = rospy.get_param("~clf_qp_wslack", 1e3)
        self.alpha   = rospy.get_param("~clf_qp_alpha",  3.2)

    def compute(self, state, ref, gains):
        m, g = self.model.m, self.model.g
        J    = self.model.J_mat
        Jinv = self.model.J_inv_diag

        p     = state["p_vec"]
        v     = state["v_vec"]
        R     = state["R_mat"]
        Omega = state["omega_body"]
        phi, th, psi = state["phi"], state["th"], state["psi"]

        pd   = ref["tgt"]
        vd   = ref["vd"]
        ad   = ref["ad"]
        psi_d = ref["yd"]
        r_d   = ref["rd"]

        Kp, Kv, KR, Kw = gains

        e_p = p - pd
        e_v = v - vd

        e3 = np.array([0.0, 0.0, 1.0])
        a_cmd = ad - Kp * e_p - Kv * e_v + g * e3
        if np.linalg.norm(a_cmd) < 1e-6:
            b3_d = np.array([0.0, 0.0, 1.0])
        else:
            b3_d = a_cmd / np.linalg.norm(a_cmd)

        b1_des = np.array([math.cos(psi_d), math.sin(psi_d), 0.0])
        b2_d   = np.cross(b3_d, b1_des)
        if np.linalg.norm(b2_d) < 1e-6:
            b1_des = np.array([math.cos(psi_d + math.pi/2.0),
                               math.sin(psi_d + math.pi/2.0), 0.0])
            b2_d   = np.cross(b3_d, b1_des)
        b2_d /= np.linalg.norm(b2_d)
        b1_d  = np.cross(b2_d, b3_d)
        b1_d /= np.linalg.norm(b1_d)

        R_d = np.column_stack((b1_d, b2_d, b3_d))

        #    e_R = ½ (R_dᵀ R − Rᵀ R_d)ˇ
        e_R = 0.5 * _vee(R_d.T.dot(R) - R.T.dot(R_d))

        Omega_d_b = np.array([0.0, 0.0, r_d])
        e_Omega = Omega - R.T.dot(R_d.dot(Omega_d_b))

        V_trans = 0.5 * m * np.dot(e_v, e_v) + 0.5 * Kp * np.dot(e_p, e_p)
        V_rot   = 0.5 * np.dot(e_Omega, J.dot(e_Omega)) + 0.5 * KR * np.dot(e_R, e_R)
        V       = V_trans + V_rot

        f_ff = m * np.linalg.norm(a_cmd)
        tau_ff = np.zeros(3)
        u_ff   = np.hstack((f_ff, tau_ff))

        # Initialize desired angular velocity derivative (zero by default)
        Omega_d_b_dot = np.zeros(3)

        # ---------- translational Lie derivatives ----------
        g_vec       = g * np.array([0.0, 0.0, 1.0])
        Lf_V_trans  = m * e_v.dot(-g_vec - ad) + Kp * e_p.dot(e_v)

        Lg_V_f      =  -e_v.dot(R.dot(np.array([0.0, 0.0, 1.0])))

        rospy.loginfo_throttle(1.0, "Translational Lie derivatives - Lf_V_trans: %s, Lg_V_f: %s", Lf_V_trans, Lg_V_f)

        # ---------- rotational Lie derivatives ----------
        omega_cross_Jomega = np.cross(Omega, J.dot(Omega))
        Lf_V_rot = (KR * e_R.dot(e_Omega)
                   - e_Omega.dot(omega_cross_Jomega)
                   - e_Omega.dot(J.dot(R.T.dot(R_d.dot(Omega_d_b_dot)))))  

        rospy.loginfo_throttle(1.0, "Rotational Lie derivative - Lf_V_rot: %s", Lf_V_rot)

        # ---------- control input Jacobian ----------
        Lg_V_tau = e_Omega.copy()

        Lf_V = Lf_V_trans + Lf_V_rot
        Lg_V = np.hstack((Lg_V_f, Lg_V_tau))

        rospy.loginfo_throttle(1.0, "Combined Lie derivatives - Lf_V: %s, Lg_V: %s", Lf_V, Lg_V)

        H = np.diag([self.w_u]*4 + [self.w_slack])
        f_vec = np.zeros(5)
        f_vec[:4] = -np.diag(H)[:4] * u_ff

        # -------- 1️⃣ translational CLF -----------
        A_clf_tr = np.zeros((1,5))
        A_clf_tr[0,:4] = [Lg_V_f, 0, 0, 0]
        A_clf_tr[0,4]  = -1
        b_clf_tr = - (Lf_V_trans + self.alpha * V_trans)

        # -------- 2️⃣ rotational CLF --------------
        A_clf_rot = np.zeros((1,5))
        A_clf_rot[0,1:4] = Lg_V_tau               # only torques enter
        A_clf_rot[0,4]   = -1
        b_clf_rot = - (Lf_V_rot  + self.alpha * V_rot)

        # stack them
        A_clf = np.vstack((A_clf_tr, A_clf_rot))
        b_clf = np.hstack((b_clf_tr, b_clf_rot))

        f_min = self.model.min_f * m * g
        f_max = 4.0 * self.model.kf * (self.model.w_max ** 2)
        tau_lim = np.array([
            0.25 * f_max * self.model.arm_length,
            0.25 * f_max * self.model.arm_length,
            0.6  * f_max * self.model.km / self.model.kf
        ])

        G_u   = np.vstack((np.eye(4), -np.eye(4)))
        G_box = np.hstack((G_u, np.zeros((G_u.shape[0],1))))

        h_box = np.hstack((
            [f_max] + tau_lim.tolist(),
            [-f_min] + tau_lim.tolist()
        ))

        G_slack = np.zeros((1,5))
        G_slack[0,4] = -1.0
        h_slack = np.array([0.0])

        G_all = np.vstack((A_clf, G_box, G_slack))
        h_all = np.hstack((b_clf, h_box, h_slack))

        if _HAVE_CVXOPT:
            P = matrix(H, tc='d')
            q = matrix(f_vec, tc='d')
            G = matrix(G_all, tc='d')
            h = matrix(h_all, tc='d')
            try:
                sol = solvers.qp(P, q, G, h)
                u_opt = np.array(sol['x']).flatten()[:4]
            except Exception as ex:
                rospy.logwarn_throttle(1.0, "CLF-QP solve failed – %s", ex)
                u_opt = u_ff
        else:
            u_opt = u_ff.copy()
            tau_pd = -KR * e_R - Kw * e_Omega
            u_opt[1:4] = tau_pd

        u_opt[0] = np.clip(u_opt[0], f_min, f_max)
        u_opt[1:] = np.clip(u_opt[1:], -tau_lim, tau_lim)

        phi_d, theta_d, _ = _rotmat_to_eul(R_d)
        Uex = Uey = 0.0
        
        rospy.loginfo_throttle(1.0, "CLF-QP Control Outputs:")
        rospy.loginfo_throttle(1.0, "U_nom: %s", u_opt)
        rospy.loginfo_throttle(1.0, "Desired angles - phi_d: %.3f, theta_d: %.3f", phi_d, theta_d)
        rospy.loginfo_throttle(1.0, "Attitude errors - e_th: %s, e_w: %s", e_R, e_Omega)
        rospy.loginfo_throttle(1.0, "Position errors - ex1: %s, ex2: %s", e_p, e_v)
        rospy.loginfo_throttle(1.0, "Virtual inputs - Uex: %.3f, Uey: %.3f", Uex, Uey)

        return dict(U_nom=u_opt,
                    phi_d=phi_d, theta_d=theta_d,
                    e_th=e_R,     e_w=e_Omega,
                    ex1=e_p,      ex2=e_v,
                    Uex=Uex,      Uey=Uey)
