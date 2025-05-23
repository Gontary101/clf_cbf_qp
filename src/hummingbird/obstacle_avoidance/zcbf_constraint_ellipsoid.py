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
        self.cbf_enter = rospy.get_param("~cbf_active_range_ellipse", 4.0)
        self.cbf_exit  = self.cbf_enter - rospy.get_param(
                        "~cbf_hysteresis_margin_ellipse", 0.0)
        self._obs_active    = [False] * len(self.obs)
        self.a1    = params.get("a1",    rospy.get_param("~zcbf_a1_ellipse",    1.5))
        self.a2    = params.get("a2",    rospy.get_param("~zcbf_a2_ellipse",    1.6))
        self.gamma = params.get("gamma", rospy.get_param("~zcbf_gamma_ellipse",  8.4))
        self.kappa = params.get("kappa", rospy.get_param("~zcbf_kappa_ellipse",  0.8))
        self.a     = params.get("a", rospy.get_param("~zcbf_order_a_ellipse", 0))
        
        self.pub   = cbf_pub
        self._last_time    = rospy.get_time()
        self._U_nom_prev   = None
        self._U_out_prev   = None
        
        self.EPSILON = 1e-9 
        
        # Add debug logging initialization
        self._last_debug_log_time = rospy.get_time()
        self._debug_log_interval = 0.5  # Log every 0.5 seconds
        
        rospy.loginfo("[ZCBF_ELLIPSOID] Initialized with params: a1=%.3f, a2=%.3f, gamma=%.3f, kappa=%.3f, cbf_enter=%.3f, cbf_exit=%.3f", 
                     self.a1, self.a2, self.gamma, self.kappa, self.cbf_enter, self.cbf_exit)
        rospy.sleep(0.1)

    def constraints(self, state):
        current_time = rospy.get_time()
        should_log_debug = (current_time - self._last_debug_log_time) >= self._debug_log_interval
        
        if len(self.obs) != len(self._obs_active):
            rospy.logdebug_throttle(5.0, "ZCBF: Obstacle count changed from %d to %d. Reinitializing _obs_active.", len(self._obs_active), len(self.obs))
            self._obs_active = [False] * len(self.obs)

        if self.obs.size == 0:
            if should_log_debug:
                rospy.loginfo_throttle(self._debug_log_interval, "[ZCBF_DEBUG] No obstacles present, returning empty constraints")
                self._last_debug_log_time = current_time
            return np.empty((0, 4)), np.empty((0, 1))

        m, g   = self.model.m, self.model.g
        J      = self.model.J_mat
        Jinv   = self.model.J_inv_diag
        kf, km = self.model.kf, self.model.km
        w_max  = self.model.w_max
        
        thr_min_fac = rospy.get_param("~min_thrust_factor", 0.10)
        U1_min_val  = float(thr_min_fac) * m * g

        p_vec = state["p_vec"];  v_vec = state["v_vec"]
        R_mat = state["R_mat"];  Om_body = state["omega_body"]
        
        RT_mat = R_mat.T

        G_list, h_list = [], []

        q_vec = R_mat[:, 2]
        
        Om_cross_e3_val          = np.cross(Om_body, e3_body)
        R_Omxe3_val              = R_mat.dot(Om_cross_e3_val)
        
        Om_cross_Om_cross_e3_val = np.cross(Om_body, Om_cross_e3_val)
        R_Om_cross2_val          = R_mat.dot(Om_cross_Om_cross_e3_val)
        
        J_Om_val                 = J.dot(Om_body)
        Om_cross_JOm_val         = np.cross(Om_body, J_Om_val)
        xi_val                   = Jinv.dot(Om_cross_JOm_val) 
        
        xi_cross_e3_val          = np.cross(xi_val, e3_body)
        R_xi_cross_e3_val        = R_mat.dot(xi_cross_e3_val)

        cbf_enter_sq = self.cbf_enter**2
        cbf_exit_sq = self.cbf_exit**2
        
        a1_a2_val = self.a1 * self.a2
        
        # Debug logging for state
        if should_log_debug:
            rospy.loginfo_throttle(self._debug_log_interval, "[ZCBF_DEBUG] State: pos=[%.3f,%.3f,%.3f], vel=[%.3f,%.3f,%.3f], |Om|=%.3f", 
                                 p_vec[0], p_vec[1], p_vec[2], v_vec[0], v_vec[1], v_vec[2], np.linalg.norm(Om_body))
            rospy.loginfo_throttle(self._debug_log_interval, "[ZCBF_DEBUG] Total obstacles: %d, Active obstacles: %d", 
                                 len(self.obs), sum(self._obs_active))

        active_constraints_count = 0
        obs_status_info = []

        for i, o in enumerate(self.obs):
            ox, oy, oz = o[0], o[1], o[2]
            vx, vy, vz = o[3], o[4], o[5]
            ax, ay, az = o[6], o[7], o[8]
            obs_a, obs_b, obs_c, obs_n = o[9], o[10], o[11], o[12]
            
            # Extract rotation matrix (elements 13-21, reshaped to 3x3)
            if len(o) >= 22:
                R_obs_flat = o[13:22]
                R_obs = np.array(R_obs_flat).reshape(3, 3)   # local → world
                R_obs_T = R_obs.T
            else:
                # Fallback to identity matrix for backward compatibility
                R_obs = np.eye(3)
                R_obs_T = np.eye(3)

            x_o_vec = np.array([ox, oy, oz])
            V_o_vec = np.array([vx, vy, vz])
            A_o_vec = np.array([ax, ay, az])

            r_vec = p_vec - x_o_vec
            # distance in the *scaled* (ellipsoid) space
            r_local_tmp = R_obs.T.dot(r_vec)             # world → local

            # Detailed log for psi inputs
            if should_log_debug and i == 0: # Log for the first obstacle being processed
                rospy.loginfo_throttle(self._debug_log_interval, 
                                     "[ZCBF_PSI_DEBUG_OBS%d] p_vec: [%.3f,%.3f,%.3f], x_o_vec: [%.3f,%.3f,%.3f]",
                                     i, p_vec[0], p_vec[1], p_vec[2], x_o_vec[0], x_o_vec[1], x_o_vec[2])
                rospy.loginfo_throttle(self._debug_log_interval,
                                     "[ZCBF_PSI_DEBUG_OBS%d] r_vec: [%.3f,%.3f,%.3f], r_local_tmp: [%.3f,%.3f,%.3f]",
                                     i, r_vec[0], r_vec[1], r_vec[2], r_local_tmp[0], r_local_tmp[1], r_local_tmp[2])
                rospy.loginfo_throttle(self._debug_log_interval,
                                     "[ZCBF_PSI_DEBUG_OBS%d] axes: a=%.3f,b=%.3f,c=%.3f, n=%.1f",
                                     i, obs_a, obs_b, obs_c, obs_n)

            # dimensionless super‐ellipsoid norm
            psi = (
                  (abs(r_local_tmp[0]/obs_a))**obs_n
                + (abs(r_local_tmp[1]/obs_b))**obs_n
                + (abs(r_local_tmp[2]/obs_c))**obs_n
            ) ** (1.0/obs_n)

            # signed distance to super‐ellipsoid surface (in meters)
            dist_val = (psi - 1.0) * min(obs_a, obs_b, obs_c)

            # hysteresis thresholds (in meters)
            enter_dist = self.cbf_enter
            exit_dist  = self.cbf_exit
            
            # Log obstacle activation/deactivation
            was_active = self._obs_active[i]
            
            if should_log_debug and i == 0: # Log for the first obstacle being processed
                 rospy.loginfo_throttle(self._debug_log_interval,
                                     "[ZCBF_PSI_DEBUG_OBS%d] Calculated psi: %.4f, dist_val: %.4f",
                                     i, psi, dist_val)
            
            if not was_active:
                # stay inactive until we come within `enter_dist` meters
                if dist_val > enter_dist:
                    obs_status_info.append("Obs%d: INACTIVE (dist=%.3f > enter=%.3f)" % (i, dist_val, self.cbf_enter))
                    continue
                else:
                    self._obs_active[i] = True
                    rospy.loginfo("[ZCBF_DEBUG] Obstacle %d ACTIVATED at distance %.3f (enter_thresh=%.3f)", i, dist_val, self.cbf_enter)
            else:
                # deactivate once we exceed `exit_dist` meters
                if dist_val > exit_dist:
                    self._obs_active[i] = False
                    rospy.loginfo("[ZCBF_DEBUG] Obstacle %d DEACTIVATED at distance %.3f (exit_thresh=%.3f)", i, dist_val, self.cbf_exit)
                    obs_status_info.append("Obs%d: DEACTIVATED (dist=%.3f > exit=%.3f)" % (i, dist_val, self.cbf_exit))
                    continue
            
            active_constraints_count += 1
            r_dot_vec = v_vec - V_o_vec

            # Transform relative position to ellipsoid frame
            r_local = R_obs_T.dot(r_vec)
            r_dot_local = R_obs_T.dot(r_dot_vec)

            s_val = np.dot(r_vec, q_vec)
            a2_s_val = self.a2 * s_val
            
            sigma_calc_val = -self.a1 * math.atan(a2_s_val)
            
            sig_p_denominator_val = (1.0 + a2_s_val**2)
            sig_p_val = -a1_a2_val / sig_p_denominator_val
            sig_pp_val = -2.0 * sig_p_val * self.a2 * a2_s_val / sig_p_denominator_val

            # --- super‐ellipsoid L-p norm (p = n) ---
            val_ax_abs = abs(r_local[0] / obs_a)
            val_by_abs = abs(r_local[1] / obs_b)
            val_cz_abs = abs(r_local[2] / obs_c)

            pow_x = val_ax_abs ** obs_n
            pow_y = val_by_abs ** obs_n
            pow_z = val_cz_abs ** obs_n

            sum_p = pow_x + pow_y + pow_z
            Phi_calc_val = sum_p ** (1.0 / obs_n)            #  ⟨—   root !

            # now rescale into meters
            g_hat_raw = Phi_calc_val - 1.0 - sigma_calc_val
            g_hat_calc_val = g_hat_raw * min(obs_a, obs_b, obs_c)

            # Compute gradient in local frame
            inner_grad_x = (obs_n/obs_a) * np.sign(r_local[0]) * val_ax_abs ** (obs_n - 1)
            inner_grad_y = (obs_n/obs_b) * np.sign(r_local[1]) * val_by_abs ** (obs_n - 1)
            inner_grad_z = (obs_n/obs_c) * np.sign(r_local[2]) * val_cz_abs ** (obs_n - 1)

            # dΦ = (1/n) · sum_p^{1/n - 1} · d(sum_p)
            root_factor = (1.0/obs_n) * (sum_p ** (1.0/obs_n - 1.0))

            grad_Phi_x_local = root_factor * inner_grad_x
            grad_Phi_y_local = root_factor * inner_grad_y
            grad_Phi_z_local = root_factor * inner_grad_z
            grad_Phi_local = np.array([grad_Phi_x_local, grad_Phi_y_local, grad_Phi_z_local])
            
            # Transform gradient back to world frame
            grad_Phi_vec_val = R_obs.dot(grad_Phi_local)  # back to world

            dot_r_dot_q_val = np.dot(r_dot_vec, q_vec)
            dot_r_R_Omxe3_val = np.dot(r_vec, R_Omxe3_val)
            dot_s_calc_val = dot_r_dot_q_val + dot_r_R_Omxe3_val
            
            dot_grad_Phi_r_dot_val = np.dot(grad_Phi_vec_val, r_dot_vec)
            # derivative also lives in Φ-units → scale to meters
            g_hat_d_raw = dot_grad_Phi_r_dot_val - sig_p_val * dot_s_calc_val
            g_hat_d_calc_val = g_hat_d_raw * min(obs_a, obs_b, obs_c)
            
            h_internal_val = self.gamma * g_hat_calc_val + g_hat_d_calc_val
            
            dot_grad_Phi_q_val = np.dot(grad_Phi_vec_val, q_vec)
            Gamma1_calc_val = (dot_grad_Phi_q_val - sig_p_val) / m
            
            r_b_vec_val = RT_mat.dot(r_vec)
            
            cross_r_b_e3_val = np.cross(r_b_vec_val, e3_body)
            Gamma2_vec_calc_val = sig_p_val * np.dot(cross_r_b_e3_val, Jinv)

            term1_calc_val = self.gamma * g_hat_d_calc_val

            n_factor_val = obs_n * (obs_n - 1)
            common_obs_n_minus_2 = obs_n - 2

            # Compute Hessian terms in local frame
            base_H_xx_val = val_ax_abs + self.EPSILON if obs_n < 2 else val_ax_abs
            base_H_yy_val = val_by_abs + self.EPSILON if obs_n < 2 else val_by_abs
            base_H_zz_val = val_cz_abs + self.EPSILON if obs_n < 2 else val_cz_abs
            
            # Same inner Hessian but scaled by root_factor again
            H_xx_local = root_factor * (n_factor_val / (obs_a**2)) * base_H_xx_val ** common_obs_n_minus_2
            H_yy_local = root_factor * (n_factor_val / (obs_b**2)) * base_H_yy_val ** common_obs_n_minus_2
            H_zz_local = root_factor * (n_factor_val / (obs_c**2)) * base_H_zz_val ** common_obs_n_minus_2
            
            # Transform velocity to local frame for Hessian computation
            r_dot_local_sq = r_dot_local**2
            term2_new_calc_val = H_xx_local * r_dot_local_sq[0] + H_yy_local * r_dot_local_sq[1] + H_zz_local * r_dot_local_sq[2]
            
            term3_new_calc_val = -g * grad_Phi_vec_val[2]

            dot_grad_Phi_A_o_val = np.dot(grad_Phi_vec_val, A_o_vec)
            term4_new_calc_val = -dot_grad_Phi_A_o_val

            dot_s_calc_sq_val = dot_s_calc_val**2
            term5_calc_val = -sig_pp_val * dot_s_calc_sq_val

            term6_calc_val = sig_p_val * g * q_vec[2]

            dot_A_o_q_val = np.dot(A_o_vec, q_vec)
            term_Ao_q_sig_p_new_calc_val = sig_p_val * dot_A_o_q_val

            dot_r_dot_R_Omxe3_val = np.dot(r_dot_vec, R_Omxe3_val)
            term7_calc_val = -sig_p_val * 2.0 * dot_r_dot_R_Omxe3_val
            
            dot_r_R_Om_cross2_val = np.dot(r_vec, R_Om_cross2_val)
            term8_calc_val = -sig_p_val * dot_r_R_Om_cross2_val
            
            dot_r_R_xi_cross_e3_val = np.dot(r_vec, R_xi_cross_e3_val)
            term9_calc_val = sig_p_val * dot_r_R_xi_cross_e3_val

            L_f_sq_g_hat_calc_val = (term2_new_calc_val + term3_new_calc_val + term4_new_calc_val + term5_calc_val + term6_calc_val + 
                                     term_Ao_q_sig_p_new_calc_val + term7_calc_val + term8_calc_val + term9_calc_val)
            
            Gamma3_code_calc_val = term1_calc_val + L_f_sq_g_hat_calc_val
            
            current_G_row_val = np.hstack([Gamma1_calc_val, Gamma2_vec_calc_val])
            
            h_val_pow_exponent = 2 * self.a + 1
            if h_val_pow_exponent == 1:
                 h_val_pow_calc_val = h_internal_val
            else:
                 h_val_pow_calc_val = np.sign(h_internal_val) * np.power(abs(h_internal_val), h_val_pow_exponent)

            current_h_final_val = Gamma3_code_calc_val + self.kappa * h_val_pow_calc_val

            G_list.append(-current_G_row_val)
            h_list.append(current_h_final_val)
            
            # Detailed logging for active constraints
            if should_log_debug:
                obs_status_info.append("Obs%d: ACTIVE, dist=%.3f, g_hat=%.4f, g_hat_d=%.4f, h_internal=%.4f, constraint_h=%.4f" % 
                                     (i, dist_val, g_hat_calc_val, g_hat_d_calc_val, h_internal_val, current_h_final_val))
                rospy.loginfo_throttle(self._debug_log_interval, "[ZCBF_DEBUG] Obs%d detailed: Phi=%.4f, sigma=%.4f, s=%.4f, Gamma1=%.4f, |Gamma2|=%.4f", 
                                     i, Phi_calc_val, sigma_calc_val, s_val, Gamma1_calc_val, np.linalg.norm(Gamma2_vec_calc_val))

        U1_max_val = 4.0 * kf * w_max ** 2

        G_box_val  = np.array([[ 1., 0, 0, 0],
                               [-1., 0, 0, 0]])
        h_box_val  = np.array([[ float(U1_max_val)],
                               [-float(U1_min_val)]])

        G_cbf_val = np.vstack(G_list) if G_list else np.empty((0, 4))
        h_cbf_val = np.array(h_list, dtype=float).reshape(-1, 1) if h_list else np.empty((0,1))

        G_all_val = np.vstack([G_cbf_val, G_box_val])
        h_all_val = np.vstack([h_cbf_val, h_box_val])
        
        # Log constraint summary
        if should_log_debug:
            self._last_debug_log_time = current_time
            rospy.loginfo_throttle(self._debug_log_interval, "[ZCBF_DEBUG] Constraints summary: %d CBF + 2 box = %d total", 
                                 len(G_list), G_all_val.shape[0])
            rospy.loginfo_throttle(self._debug_log_interval, "[ZCBF_DEBUG] Thrust limits: min=%.3f, max=%.3f", U1_min_val, U1_max_val)
            
            for info in obs_status_info:
                rospy.loginfo_throttle(self._debug_log_interval, "[ZCBF_DEBUG] %s", info)
            
            if len(h_list) > 0:
                h_cbf_array = np.array(h_list)
                min_h = np.min(h_cbf_array)
                max_h = np.max(h_cbf_array)
                rospy.loginfo_throttle(self._debug_log_interval, "[ZCBF_DEBUG] CBF constraint values: min_h=%.4f, max_h=%.4f, active_count=%d", 
                                     min_h, max_h, active_constraints_count)
                
                # Log most restrictive constraint
                if min_h < 0.1:  # Close to being violated
                    min_idx = np.argmin(h_cbf_array)
                    rospy.logwarn_throttle(self._debug_log_interval, "[ZCBF_DEBUG] Most restrictive constraint %d with h=%.4f (close to violation!)", 
                                         min_idx, min_h)
        
        return G_all_val, h_all_val