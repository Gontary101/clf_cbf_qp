#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import division
import math, numpy as np, rospy
from cvxopt import matrix, solvers
from std_msgs.msg import Float64, Float64MultiArray
from utils.dynamics_utils2 import e3_world, e3_body

solvers.options['show_progress'] = False
clip = np.clip


class ZCBFFilter(object):
    """Safety filter that enforces CBF constraints by solving a small QP."""

    def __init__(self, model, obstacles, params, cbf_pub=None):
        self.model = model
        self.obs   = obstacles
        self.beta  = params.get("zcbf_beta",   1.5)
        self.a1    = params.get("zcbf_a1",     1.5)
        self.a2    = params.get("zcbf_a2",     1.6)
        self.gamma = params.get("zcbf_gamma",  8.4)
        self.kappa = params.get("zcbf_kappa",  0.8)
        self.a     = params.get("zcbf_order_a", 0)
        self.pub   = cbf_pub   # may be None
        # --- low-pass filter setup (τ in seconds) ---
        self.tau           = params.get("zcbf_tau", 0.2)
        self._last_time    = rospy.get_time()
        self._U_nom_prev   = None
        self._U_out_prev   = None

        # Initialize publishers with absolute topic names
        #self.gamma1_pub = rospy.Publisher('/clf_iris_trajectory_controller/gamma1', Float64, queue_size=1)
        #self.gamma2_pub = rospy.Publisher('/clf_iris_trajectory_controller/gamma2', Float64MultiArray, queue_size=1)
        #self.gamma3_pub = rospy.Publisher('/clf_iris_trajectory_controller/gamma3', Float64, queue_size=1)
        # Wait for publishers to be ready
        rospy.sleep(0.1)  # Give ROS time to set up publishers

    # ----------------------------------------------------------------------
    def constraints(self, state):
        """
        Build Control-Barrier constraints **only**.
        They are solved upstream together with the CLF objective,
        so no optimisation happens here any more.
        Returns:
            G_all : (m × 4) ndarray
            h_all : (m × 1) ndarray
        """
        if self.obs.size == 0:
            return np.empty((0, 4)), np.empty((0, 1))

        m, g   = self.model.m, self.model.g
        J      = self.model.J_mat
        Jinv   = self.model.J_inv_diag
        kf, km = self.model.kf, self.model.km
        w_max  = self.model.w_max
        r_d    = self.model.r_drone
        # --- new: minimum collective thrust allowed ------------------------
        thr_min_fac = rospy.get_param("~min_thrust_factor", 0.10)  # same name as YAML
        U1_min      = thr_min_fac * m * g                         # [N]
        # --- new: ignore obstacles farther than this (for numerical health)
        cbf_range   = rospy.get_param("~cbf_active_range", 2.0)   # [m]

        p = state["p_vec"];  v = state["v_vec"]
        R = state["R_mat"];  Om = state["omega_body"]

        G_list, h_list = [], []

        for o in self.obs:
            # early-reject if far away
            if np.linalg.norm(state["p_vec"] - o[:3]) > cbf_range:
                continue
            ox, oy, oz, vx, vy, vz, ax, ay, az, r_o = o
            x_o, V_o, A_o = np.array([ox, oy, oz]), \
                            np.array([vx, vy, vz]), \
                            np.array([ax, ay, az])
            r_safe = r_o + r_d

            r       = p - x_o
            r_dot   = v - V_o
            q       = R[:, 2]
            s       = np.dot(r, q)

            sigma   = -self.a1 * math.atan(self.a2 * s)
            sig_p   = -self.a1 * self.a2 / (1.0 + (self.a2 * s) ** 2)
            sig_pp  =  2.0 * self.a1 * (self.a2 ** 2) * s / (1.0 + (self.a2 * s) ** 2) ** 2

            g_hat   = np.dot(r, r) - self.beta * (r_safe ** 2) - sigma
            R_Omxe3 = R.dot(np.cross(Om, e3_body))
            g_hat_d = 2.0 * np.dot(r, r_dot) - sig_p * (np.dot(r_dot, q) +
                                                         np.dot(r, R_Omxe3))
            h_val   = self.gamma * g_hat + g_hat_d

            # -------- gradient wrt U ----------------------------------------
            Gamma1      = (2.0 * s - sig_p) / m
            r_b         = np.dot(R.T, r)
            Gamma2_vec  = sig_p * np.dot(np.cross(r_b, e3_body), Jinv)

            # -------- higher‑order terms ------------------------------------
            dot_s = np.dot(r_dot, q) + np.dot(r, R_Omxe3)
            term1 = self.gamma * g_hat_d
            term2 = 2.0 * np.dot(r_dot, r_dot)
            term3 = -2.0 * g * np.dot(r, e3_world)
            term4 = -2.0 * np.dot(r, A_o)
            term5 = -sig_pp * (dot_s ** 2)
            term6 = sig_p * g * q[2]
            term7 = -sig_p * 2.0 * np.dot(r_dot, R_Omxe3)

            Om_cross_e3               = np.cross(Om, e3_body)
            Om_cross_Om_cross_e3      = np.cross(Om, Om_cross_e3)
            R_Om_cross2               = np.dot(R, Om_cross_Om_cross_e3)
            OmJ                       = np.dot(J, Om)
            Om_cross_JOm              = np.cross(Om, OmJ)
            xi                        = np.dot(Jinv, Om_cross_JOm)
            R_xi_cross_e3             = np.dot(R, np.cross(xi, e3_body))

            term8 = -sig_p * np.dot(r, R_Om_cross2)
            term9 =  sig_p * np.dot(r, R_xi_cross_e3)

            Gamma3 = (term1 + term2 + term3 + term4 +
                      term5 + term6 + term7 + term8 + term9)

            G_row   = np.hstack([Gamma1, Gamma2_vec])
            h_value = Gamma3 + self.kappa * (h_val ** (2 * self.a + 1))

            G_list.append(-G_row)        # inequality: −G U ≤ h
            h_list.append(h_value)

        # -------- input box constraint on collective thrust ------------------
        U1_max = 4.0 * kf * w_max ** 2
        #  U1_max  ≥ U₁ ≥ U1_min   →   [ 1 0 0 0]·U ≤ U1_max
        #                                [-1 0 0 0]·U ≤ –U1_min
        G_box  = np.array([[ 1., 0, 0, 0],
                           [-1., 0, 0, 0]])
        h_box  = np.array([[ U1_max],
                           [-U1_min]])

        G_cbf = np.vstack(G_list)               if G_list else np.empty((0, 4))
        h_cbf = np.array(h_list, dtype=float).reshape(-1, 1)

        G_all = np.vstack([G_cbf, G_box])
        h_all = np.vstack([h_cbf, h_box])

        return G_all, h_all

