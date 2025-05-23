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
    def __init__(self, model, obstacles, params, cbf_pub=None):
        self.model = model
        self.obs   = obstacles
        self.cbf_enter      = rospy.get_param("~cbf_active_range",    2.0)
        self._hyst_margin   = rospy.get_param("~cbf_hysteresis_margin", 0.5)
        self.cbf_exit       = self.cbf_enter - self._hyst_margin
        self._obs_active    = [False] * len(self.obs)
        self.a1    = params.get("zcbf_a1",     1.5)
        self.a2    = params.get("zcbf_a2",     1.6)
        self.gamma = params.get("zcbf_gamma",  8.4)
        self.kappa = params.get("zcbf_kappa",  0.8)
        self.a     = params.get("zcbf_order_a", 0)
        
        self.pub   = cbf_pub
        self.tau           = params.get("zcbf_tau", 0.2)
        self._last_time    = rospy.get_time()
        self._U_nom_prev   = None
        self._U_out_prev   = None
        self.s_min         = params.get("zcbf_s_min", -0.5)
        
        self.EPSILON = 1e-9 
        rospy.sleep(0.1)

    def constraints(self, state):
        if len(self.obs) != len(self._obs_active):
            rospy.logdebug_throttle(5.0, "ZCBF: Obstacle count changed from %d to %d. Reinitializing _obs_active.", len(self._obs_active), len(self.obs))
            self._obs_active = [False] * len(self.obs)

        if self.obs.size == 0:
            return np.empty((0, 4)), np.empty((0, 1))

        m, g   = self.model.m, self.model.g
        J      = self.model.J_mat
        Jinv   = self.model.J_inv_diag

        thr_min_fac = rospy.get_param("~min_thrust_factor", 0.10)
        U1_min      = thr_min_fac * m * g

        p = state["p_vec"];  v = state["v_vec"]
        R = state["R_mat"];  Om = state["omega_body"]

        G_list, h_list = [], []

        q = R[:, 2]
        R_Omxe3 = R.dot(np.cross(Om, e3_body))

        Om_cross_e3           = np.cross(Om, e3_body)
        Om_cross_Om_cross_e3  = np.cross(Om, Om_cross_e3)
        R_Om_cross2           = np.dot(R, Om_cross_Om_cross_e3)
        
        OmJ                   = np.dot(J, Om)
        Om_cross_JOm          = np.cross(Om, OmJ)
        xi                    = np.dot(Jinv, Om_cross_JOm)
        R_xi_cross_e3         = np.dot(R, np.cross(xi, e3_body))


        for i, o in enumerate(self.obs):
            ox, oy, oz = o[0], o[1], o[2]
            vx, vy, vz = o[3], o[4], o[5]
            ax, ay, az = o[6], o[7], o[8]
            obs_a, obs_b, obs_c, obs_n = o[9], o[10], o[11], o[12]

            x_o = np.array([ox, oy, oz])
            V_o = np.array([vx, vy, vz])
            A_o = np.array([ax, ay, az])

            dist_sq = np.sum((p - x_o)**2)
            if not self._obs_active[i]:
                if dist_sq > self.cbf_enter**2:
                    continue
                else:
                    self._obs_active[i] = True
            else:
                if dist_sq > self.cbf_exit**2:
                    self._obs_active[i] = False
                    continue
            
            r = p - x_o
            r_dot = v - V_o

            s = np.dot(r, q)
            sigma = -self.a1 * math.atan(self.a2 * s)
            
            sig_p_denominator = (1.0 + (self.a2 * s)**2)
            sig_p = -self.a1 * self.a2 / sig_p_denominator
            sig_pp = 2.0 * self.a1 * (self.a2**3) * s / (sig_p_denominator**2)

            val_ax_abs = abs(r[0]/obs_a)
            val_by_abs = abs(r[1]/obs_b)
            val_cz_abs = abs(r[2]/obs_c)

            Phi = np.power(val_ax_abs, obs_n) + \
                  np.power(val_by_abs, obs_n) + \
                  np.power(val_cz_abs, obs_n)

            g_hat = Phi - 1.0 - sigma

            grad_Phi = np.array([
                (obs_n/obs_a) * np.sign(r[0]) * np.power(val_ax_abs, obs_n-1),
                (obs_n/obs_b) * np.sign(r[1]) * np.power(val_by_abs, obs_n-1),
                (obs_n/obs_c) * np.sign(r[2]) * np.power(val_cz_abs, obs_n-1)
            ])

            dot_s_calc = np.dot(r_dot, q) + np.dot(r, R_Omxe3)
            
            g_hat_d = np.dot(grad_Phi, r_dot) - sig_p * dot_s_calc
            
            h_val = self.gamma * g_hat + g_hat_d
            
            Gamma1 = (np.dot(grad_Phi, q) - sig_p) / m
            
            r_b = np.dot(R.T, r)
            Gamma2_vec = sig_p * np.dot(np.cross(r_b, e3_body), Jinv)

            term1 = self.gamma * g_hat_d

            n_factor = obs_n * (obs_n - 1)
            H_xx = (n_factor / (obs_a**2)) * np.power(val_ax_abs + self.EPSILON if obs_n < 2 else val_ax_abs, obs_n - 2)
            H_yy = (n_factor / (obs_b**2)) * np.power(val_by_abs + self.EPSILON if obs_n < 2 else val_by_abs, obs_n - 2)
            H_zz = (n_factor / (obs_c**2)) * np.power(val_cz_abs + self.EPSILON if obs_n < 2 else val_cz_abs, obs_n - 2)
            term2_new = H_xx * r_dot[0]**2 + H_yy * r_dot[1]**2 + H_zz * r_dot[2]**2
            
            term3_new = -g * grad_Phi[2]

            term4_new = -np.dot(grad_Phi, A_o)

            term5 = -sig_pp * (dot_s_calc**2)

            term6 = sig_p * g * q[2]

            term_Ao_q_sig_p_new = sig_p * np.dot(A_o, q)

            term7 = -sig_p * 2.0 * np.dot(r_dot, R_Omxe3)
            
            term8 = -sig_p * np.dot(r, R_Om_cross2)
            
            term9 = sig_p * np.dot(r, R_xi_cross_e3)

            L_f_sq_g_hat = (term2_new + term3_new + term4_new + term5 + term6 + 
                            term_Ao_q_sig_p_new + term7 + term8 + term9)
            
            Gamma3_code = term1 + L_f_sq_g_hat
            
            current_G_row = np.hstack([Gamma1, Gamma2_vec])
            
            h_val_pow_exponent = 2 * self.a + 1
            if h_val_pow_exponent == 1:
                 h_val_pow = h_val
            else:
                 h_val_pow = np.sign(h_val) * np.power(abs(h_val), h_val_pow_exponent)


            current_h_value = Gamma3_code + self.kappa * h_val_pow

            G_list.append(-current_G_row)
            h_list.append(current_h_value)

        U1_max = 4.0 * self.model.kf * self.model.w_max**2

        G_box  = np.array([[ 1., 0, 0, 0],
                           [-1., 0, 0, 0]])
        h_box  = np.array([[ U1_max],
                           [-U1_min]])

        G_cbf = np.vstack(G_list) if G_list else np.empty((0, 4))
        h_cbf = np.array(h_list, dtype=float).reshape(-1, 1) if h_list else np.empty((0,1))

        G_all = np.vstack([G_cbf, G_box])
        h_all = np.vstack([h_cbf, h_box])
        
        return G_all, h_all