#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import division
import math, numpy as np
import rospy

try:
    from cvxopt import matrix, solvers
    CVXOPT_AVAILABLE = True
    solvers.options['show_progress'] = False
except ImportError:
    CVXOPT_AVAILABLE = False
    rospy.logerr("CRITICAL: CVXOPT is not available. CLF-QP controller cannot function.")


def skew_symmetric(v):
    return np.array([[0, -v[2], v[1]],
                     [v[2], 0, -v[0]],
                     [-v[1], v[0], 0]])

def vee_operator(S_in):
    S = 0.5 * (S_in - S_in.T) # Ensure skew-symmetry due to potential numerical noise
    return np.array([S[2, 1], S[0, 2], S[1, 0]])


class CLF_QP_Controller(object):
    def __init__(self, model, clf_qp_params=None):
        self.model = model
        self.m = model.m
        self.g_eff = model.g * model.gc # Effective gravity
        self.I = model.J_mat
        self.I_inv = np.linalg.inv(self.I)

        # For '+' configuration, arm_length is used for both lx and ly
        # If your drone has different lx, ly, you'd need to add those to DroneModel
        # Based on DroneModel.invA, it uses a single L for arm_length.
        # Rotor 0: front, 1: right, 2: back, 3: left (standard + config mapping)
        # invA implies:
        # M_roll (around x) = L*kf*(w1^2 - w3^2) -> L* (f1 - f3) if f1 is right, f3 is left
        # M_pitch (around y) = L*kf*(w2^2 - w0^2) -> L* (f2 - f0) if f2 is back, f0 is front
        # M_yaw (around z) = km * (w0^2 - w1^2 + w2^2 - w3^2)
        # The provided invA in DroneModel:
        # U[1] (Mx) = L*kf * w_sq[1] - L*kf * w_sq[3]  (rotor 1 vs 3 for roll)
        # U[2] (My) = -L*kf * w_sq[0] + L*kf * w_sq[2] (rotor 0 vs 2 for pitch)
        # This means:
        # Rotor 0: front (negative pitch moment)
        # Rotor 1: right (positive roll moment)
        # Rotor 2: back (positive pitch moment)
        # Rotor 3: left (negative roll moment)
        # So, d_tau_df for [f0, f1, f2, f3] (front, right, back, left):
        self.lx = 0.17 # Used for pitch moments (front/back rotors)
        self.ly = 0.17 # Used for roll moments (left/right rotors)

        self.c_tau = model.km / model.kf

        self.d_tau_df_list = [
            np.array([0.0,      -self.lx,  self.c_tau]),  # Rotor 0 (front)
            np.array([self.ly,   0.0,     -self.c_tau]),  # Rotor 1 (right)
            np.array([0.0,       self.lx,  self.c_tau]),  # Rotor 2 (back)
            np.array([-self.ly,  0.0,     -self.c_tau])   # Rotor 3 (left)
        ]


        if clf_qp_params is None:
            rospy.logwarn("CLF_QP_Controller using default QP parameters.")
            clf_qp_params = {}

        self.k_p = clf_qp_params.get('k_p', 8.0)
        self.k_R = clf_qp_params.get('k_R', 10.0)
        self.gamma_clf = clf_qp_params.get('gamma_clf', 10.0)
        self.rho_slack = clf_qp_params.get('rho_slack', 2000.0)
        self.f_max_per_rotor = model.kf * (model.w_max**2)
        self.f_min_per_rotor = clf_qp_params.get('f_min_per_rotor', 0.01 * self.f_max_per_rotor)

        self.u_ff_nominal = np.array([self.m * self.g_eff / 4.0] * 4)
        self.last_successful_f_sol = np.copy(self.u_ff_nominal)

        self.prev_R_d = None
        self.prev_Omega_d_body = None
        self.prev_t_R_d_secs = None # Use seconds directly
        self.dt_min_diff = 1e-4

        if not CVXOPT_AVAILABLE:
            rospy.logerr("CVXOPT is required for CLF_QP_Controller but not found. Controller will not operate correctly.")
            # Consider raising an exception here to halt initialization if CVXOPT is critical
            # raise ImportError("CVXOPT is required for CLF_QP_Controller.")


    def _compute_R_d_derivatives(self, R_d_current, t_current_secs):
        Omega_d_body = np.zeros(3)
        dot_Omega_d_body = np.zeros(3)

        if self.prev_R_d is not None and self.prev_t_R_d_secs is not None:
            dt = t_current_secs - self.prev_t_R_d_secs
            if dt > self.dt_min_diff:
                dot_R_d_approx = (R_d_current - self.prev_R_d) / dt
                Omega_d_hat_current = R_d_current.T @ dot_R_d_approx
                Omega_d_body = vee_operator(Omega_d_hat_current) # vee_operator ensures skew

                if self.prev_Omega_d_body is not None:
                    dot_Omega_d_body = (Omega_d_body - self.prev_Omega_d_body) / dt
                self.prev_Omega_d_body = np.copy(Omega_d_body)
        
        self.prev_R_d = np.copy(R_d_current)
        self.prev_t_R_d_secs = t_current_secs
        return Omega_d_body, dot_Omega_d_body


    def compute(self, state, ref, _gains_ignored):
        if not CVXOPT_AVAILABLE:
            rospy.logerr_throttle(1.0, "CVXOPT not available. CLF-QP cannot compute control. Returning zero thrusts.")
            # Return a U_nom that results in zero motor speeds for safety
            # This requires knowing how motor allocation works.
            # If U_nom=[0,0,0,0] leads to zero speeds, that's fine.
            # Otherwise, determine what U_nom leads to zero f_i.
            # For now, assume U_nom=[0,0,0,0] is safe.
            return dict(U_nom=np.zeros(4), f_ind=np.zeros(4),
                        phi_d=0, theta_d=0,
                        e_th=np.zeros(3), e_w=np.zeros(3),
                        ex1=np.zeros(3), ex2=np.zeros(3),
                        Uex=0.0, Uey=0.0,
                        R_d=np.eye(3), V_clf=0, LfV_clf=0, LgV_clf=np.zeros(4), delta_clf=0)

        p_curr = state["p_vec"]
        v_curr = state["v_vec"]
        R_curr = state["R_mat"]
        omega_b_curr = state["omega_body"]
        
        # Pass trajectory time from main loop if available, otherwise use rospy.Time.now()
        # For consistency with trajectory generator, main.py should pass its (now - self.t0_traj).to_sec()
        # Assuming 't_traj' is passed in 'ref' or a global time is used.
        # If not, this numerical differentiation might have jitter.
        # For now, using rospy.Time.now() as a fallback if t_traj not in ref.
        t_curr_secs = ref.get("t_traj_secs", rospy.Time.now().to_sec())


        p_des = ref["tgt"]
        v_des = ref["vd"]
        a_des = ref["ad"]
        psi_des = ref["yd"]

        e_p = p_curr - p_des
        e_v = v_curr - v_des

        g_world_vec = np.array([0, 0, self.g_eff])
        F_thrust_des_world = self.m * (a_des + g_world_vec)

        norm_F_thrust_des = np.linalg.norm(F_thrust_des_world)
        if norm_F_thrust_des < 1e-6:
            b3_d = R_curr[:, 2] 
        else:
            b3_d = F_thrust_des_world / norm_F_thrust_des

        x_c_world = np.array([math.cos(psi_des), math.sin(psi_des), 0.0])
        
        b2_d_intermediate = np.cross(b3_d, x_c_world)
        norm_b2_d_intermediate = np.linalg.norm(b2_d_intermediate)

        if norm_b2_d_intermediate < 1e-6: # b3_d is (anti)parallel to x_c_world
            # This typically happens if b3_d is vertical, or if b3_d is horizontal and aligned with psi_des
            # If b3_d is vertical, x_c_world is suitable for b1_d's projection.
            # b1_d = x_c_world - dot(x_c_world, b3_d)*b3_d (Gram-Schmidt on x_c_world wrt b3_d)
            # Since x_c_world is on xy plane, if b3_d is vertical, dot is 0, so b1_d = x_c_world
            if abs(b3_d[2]) > 0.999 : # b3_d is nearly vertical
                b1_d = x_c_world # x_c_world is already unit normal
                b2_d = np.cross(b3_d, b1_d)
            else: # b3_d is horizontal and aligned with psi_des. Need a different vector for cross product.
                  # e.g., cross b3_d with world z-axis to get a b2_d, then b1_d = b2_d x b3_d
                b2_d_alt = np.cross(b3_d, np.array([0,0,1.0]))
                if np.linalg.norm(b2_d_alt) < 1e-6: # b3_d was vertical, this case should have been caught
                    b2_d_alt = np.cross(b3_d, np.array([1,0,0.0])) # Fallback if b3_d was along z
                b2_d = b2_d_alt / np.linalg.norm(b2_d_alt)
                b1_d = np.cross(b2_d, b3_d)
        else:
            b2_d = b2_d_intermediate / norm_b2_d_intermediate
            b1_d = np.cross(b2_d, b3_d)

        R_d = np.column_stack((b1_d, b2_d, b3_d))

        Omega_d_body, dot_Omega_d_body = self._compute_R_d_derivatives(R_d, t_curr_secs)

        e_R_mat_term = R_d.T @ R_curr
        e_R = 0.5 * vee_operator(e_R_mat_term - e_R_mat_term.T)
        e_Omega = omega_b_curr - R_curr.T @ R_d @ Omega_d_body

        Psi_term = 0.5 * np.trace(np.eye(3) - e_R_mat_term)
        V_val = 0.5 * e_v.dot(e_v) + \
                0.5 * self.k_p * e_p.dot(e_p) + \
                self.k_R * Psi_term + \
                0.5 * e_Omega.T @ self.I @ e_Omega

        LfV_pos_drift = self.k_p * e_p.dot(e_v) + e_v.dot(g_world_vec - a_des)
        LfV_att_tr_term = self.k_R * e_R.dot(e_Omega)
        
        term_Omega_cross_IOmega = np.cross(omega_b_curr, self.I @ omega_b_curr)
        term_Omega_hat_RTRdOmega_d = skew_symmetric(omega_b_curr) @ R_curr.T @ R_d @ Omega_d_body
        term_RTRd_dot_Omega_d = R_curr.T @ R_d @ dot_Omega_d_body
        LfV_e_Omega_drift = e_Omega.T @ ( -term_Omega_cross_IOmega + self.I @ (term_Omega_hat_RTRdOmega_d - term_RTRd_dot_Omega_d) )
        LfV = LfV_pos_drift + LfV_att_tr_term + LfV_e_Omega_drift

        e_z_body_frame = np.array([0,0,1.0])
        term_trans_LgV = (-1.0 / self.m) * (e_v.dot(R_curr @ e_z_body_frame))
        
        LgV = np.zeros(4)
        for i in range(4):
            LgV[i] = term_trans_LgV + e_Omega.dot(self.d_tau_df_list[i])

        H_qp = np.eye(5)
        H_qp[4,4] = self.rho_slack
        q_qp = np.zeros(5)
        q_qp[:4] = -self.u_ff_nominal

        A_clf = np.hstack((LgV, -1.0)).reshape(1, 5)
        b_clf = np.array([-self.gamma_clf * V_val - LfV])

        A_bounds = np.zeros((8, 5))
        b_bounds = np.zeros(8)
        for i in range(4):
            A_bounds[i, i] = 1.0; b_bounds[i] = self.f_max_per_rotor
            A_bounds[i+4, i] = -1.0; b_bounds[i+4] = -self.f_min_per_rotor

        A_slack = np.array([[0,0,0,0, -1.0]]); b_slack = np.array([0.0])
        A_ineq = np.vstack((A_clf, A_bounds, A_slack))
        b_ineq = np.concatenate((b_clf, b_bounds, b_slack))

        f_sol = np.copy(self.last_successful_f_sol)
        delta_sol = 0.0

        try:
            P_cvx, q_cvx = matrix(H_qp), matrix(q_qp)
            G_cvx, h_cvx = matrix(A_ineq), matrix(b_ineq)
            solution = solvers.qp(P_cvx, q_cvx, G_cvx, h_cvx)
            if solution['status'] == 'optimal':
                x_qp_sol = np.array(solution['x']).flatten()
                f_sol = x_qp_sol[:4]
                delta_sol = x_qp_sol[4]
                self.last_successful_f_sol = np.copy(f_sol)
            else:
                rospy.logwarn_throttle(1.0, f"CLF-QP solver status: {solution['status']}. Using last good/nominal.")
        except ValueError:
            rospy.logwarn_throttle(1.0, "CLF-QP infeasible. Using last good/nominal.")
        except Exception as e:
            rospy.logwarn_throttle(1.0, f"CLF-QP solver error: {e}. Using last good/nominal.")
        
        f_sol = np.clip(f_sol, self.f_min_per_rotor, self.f_max_per_rotor)

        F_total_actual = np.sum(f_sol)
        Mx_actual = self.ly * (f_sol[1] - f_sol[3]) # Rotor 1 (right +ve), Rotor 3 (left -ve)
        My_actual = self.lx * (f_sol[2] - f_sol[0]) # Rotor 2 (back +ve), Rotor 0 (front -ve)
        Mz_actual = self.c_tau * (f_sol[0] - f_sol[1] + f_sol[2] - f_sol[3])
        U_nom_out = np.array([F_total_actual, Mx_actual, My_actual, Mz_actual])

        theta_d_rd = math.asin(np.clip(-R_d[2,0], -1.0, 1.0))
        cos_theta_d_rd = math.cos(theta_d_rd)
        if abs(cos_theta_d_rd) > 1e-4:
            phi_d_rd = math.atan2(R_d[2,1] / cos_theta_d_rd, R_d[2,2] / cos_theta_d_rd)
        else:
            phi_d_rd = math.atan2(-R_d[1,2], R_d[1,1]) if R_d[0,0] > 0 else math.atan2(R_d[1,2], -R_d[1,1])


        e_th_approx = np.array([state["phi"] - phi_d_rd,
                                state["th"] - theta_d_rd,
                                (state["psi"] - ref["yd"] + math.pi) % (2 * math.pi) - math.pi])
        
        return dict(U_nom=U_nom_out, f_ind=f_sol,
                    phi_d=phi_d_rd, theta_d=theta_d_rd,
                    e_th=e_th_approx, e_w=e_Omega,
                    ex1=e_p, ex2=e_v,
                    Uex=0.0, Uey=0.0,
                    R_d=R_d, V_clf=V_val, LfV_clf=LfV, LgV_clf=LgV, delta_clf=delta_sol)


class CLFBackstepping(object):
    def __init__(self, model):
        self.model = model

    def compute(self, state, ref, gains):
        m, g, gc = self.model.m, self.model.g, self.model.gc
        min_f_factor, max_tilt = self.model.min_f, self.model.max_tilt
        Ix, Iy, Iz = self.model.Ix, self.model.Iy, self.model.Iz
        kf, km     = self.model.kf, self.model.km

        g1, g2, g3, g4 = gains

        p = state["p_vec"]; v = state["v_vec"]
        phi, th, psi = state["phi"], state["th"], state["psi"]
        omega_b      = state["omega_body"]

        tgt, vd, ad = ref["tgt"], ref["vd"], ref["ad"]
        yd, rd      = ref["yd"],  ref["rd"]

        ex1 = p - tgt
        ex2 = v - vd
        
        tof = math.cos(phi) * math.cos(th)
        if abs(tof) < 1e-3: 
            tof = np.sign(tof) * 1e-3 if tof != 0 else 1e-3

        U1_nom = (m / tof) * (-g1 * ex1[2] + ad[2] - g2 * ex2[2]) + m * g * gc / tof
        U1_nom = max(0.0, U1_nom)

        if U1_nom < 1e-6:
            Uex = Uey = 0.0
        else:
            Uex = (m / U1_nom) * (-g1 * ex1[0] + ad[0] - g2 * ex2[0])
            Uey = (m / U1_nom) * (-g1 * ex1[1] + ad[1] - g2 * ex2[1])

        sp_yd, cp_yd = math.sin(yd), math.cos(yd)
        
        phi_d_arg = -Uey * cp_yd + Uex * sp_yd
        phi_d = math.asin(np.clip(phi_d_arg, -1.0, 1.0))

        cpd = math.cos(phi_d)
        theta_d_arg_num = Uex * cp_yd + Uey * sp_yd
        
        if abs(cpd) < 1e-3: 
            theta_d = np.sign(theta_d_arg_num) * math.pi / 2.0 if theta_d_arg_num != 0 else 0.0
        else:
            theta_d_val_to_asin = np.clip(theta_d_arg_num / cpd, -1.0, 1.0)
            theta_d = math.asin(theta_d_val_to_asin)
        
        phi_d   = np.clip(phi_d,   -max_tilt, max_tilt)
        theta_d = np.clip(theta_d, -max_tilt, max_tilt)

        yaw_error = (psi - yd + math.pi) % (2 * math.pi) - math.pi
        e_th = np.array([phi - phi_d, th  - theta_d, yaw_error])
        
        omega_d_body = np.array([0.0, 0.0, rd])
        e_w  = omega_b - omega_d_body

        U2_nom = Ix * (-g3 * e_th[0] - g4 * e_w[0]) - omega_b[1] * omega_b[2] * (Iy - Iz)
        U3_nom = Iy * (-g3 * e_th[1] - g4 * e_w[1]) - omega_b[0] * omega_b[2] * (Iz - Ix)
        U4_nom = Iz * (-g3 * e_th[2] - g4 * e_w[2]) - omega_b[0] * omega_b[1] * (Ix - Iy)
        
        U_nom = np.array([U1_nom, U2_nom, U3_nom, U4_nom])

        return dict(U_nom=U_nom, phi_d=phi_d, theta_d=theta_d,
                    e_th=e_th, e_w=e_w, ex1=ex1, ex2=ex2,
                    Uex=Uex, Uey=Uey)