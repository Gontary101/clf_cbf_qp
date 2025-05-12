#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
C3BF (Control‑Lyapunov / 3‑Dimensional Barrier Function) safety filter.

•  Builds zero‑slack CBF constraints on the **translational force** F = m(a+g).
•  Solves a small convex QP   min‖F−F_nom‖²   subject to the CBF
   and a collective–thrust upper bound F_z ≤ Fz_max.
•  Converts the resulting safe acceleration back to full 4‑DOF inputs
   by calling the nominal CLF back‑stepping controller a second time.

Interface is identical to zcbf_filter.ZCBFFilter, so you can hot‑swap it by
changing one import line in main.py.
"""

from __future__ import division
import math, numpy as np, rospy
from cvxopt import matrix, solvers
from std_msgs.msg import Float64MultiArray

from dynamics_utils import pget
from clf_backstepping import CLFBackstepping

solvers.options['show_progress'] = False

# ---------------------------------------------------------------------------

EPS = 1e-2          # tiny number to avoid division by zero
clip = np.clip
rt   = np.sqrt
dot  = np.dot
norm = np.linalg.norm


class C3BFFilter(object):

    # --------------------------------------------------------------- init --
    def __init__(self, model, obstacles, _unused=None, cbf_pub=None):
        """
        Parameters
        ----------
        model : dynamics_utils.DroneModel
        obstacles : ndarray
            Either (N, 4)  →  [x, y, z, r]
            or     (N, 7)  →  [x, y, z, r, vx, vy, vz]
        cbf_pub : rospy.Publisher or None
            If given, publishes the residuals (h‑values / slacks) each cycle.
        """
        self.model = model
        self.pub   = cbf_pub

        # -------- parameters (can be tuned from the launch file) ----------
        self.gamma      = pget("cbf_gamma",        500.0)
        self.Fz_max     = pget("cbf_Fz_max",       18.0)  
        self.Fz_min     = -10          # N
        #self.r_drone    = model.r_drone

        # -------- obstacle list ------------------------------------------
        obs_arr_input = np.asarray(obstacles, dtype=float)
        if obs_arr_input.ndim == 2 and obs_arr_input.shape[1] in (4, 7, 10):
            # accept (N,10) from dynamic‑obstacle format → keep first 7
            if obs_arr_input.shape[1] == 10:
                obs_arr = obs_arr_input[:, :7]
                self.obs = np.zeros((obs_arr_input.shape[0], 7))
                self.obs[:, 0:3] = obs_arr_input[:, 0:3]  # x, y, z (cols 0, 1, 2)
                self.obs[:, 3]   = obs_arr_input[:, 9]    # r_o (obstacle radius from 10th column, index 9)
                self.obs[:, 4:7] = obs_arr_input[:, 3:6]  # vx, vy, vz (from 4th, 5th, 6th columns, indices 3, 4, 5)
            elif obs_arr_input.shape[1] == 4: # [x,y,z,r]
                rospy.loginfo("C3BFFilter: Padding (N,4) obstacles with zero velocities.")
                zpad = np.zeros((obs_arr_input.shape[0], 3))
                self.obs = np.hstack([obs_arr_input, zpad]) # obs_arr_input already has r in 4th col
            elif obs_arr_input.shape[1] == 7: # Already in [x,y,z,r,vx,vy,vz]
                rospy.loginfo("C3BFFilter: Using (N,7) obstacles as is.")
                self.obs = obs_arr_input
        else:
            rospy.logwarn("C3BFFilter: unexpected obstacle array shape %s – "
                        "filter disabled.", obs_arr_input.shape)
            self.obs = np.empty((0, 7))

    # Ensure r_drone is available for _build_constraints
        self.r_drone = self.model.r_drone # Make sure this is correctly loaded in DroneModel


        # -------- helper controller to get torques after filtering -------
        self.clf = CLFBackstepping(model)

    # --------------------------------------------------------------------
    def _build_constraints(self, p_w, v_w):
        """
        Return (G,h,h_debug) for constraints  G F ≤ h.
        h_debug lists raw h_i values for optional plotting.
        """
        if self.obs.size == 0:
            return None, None, []

        m, g, r_d = self.model.m, self.model.g, self.r_drone
        G_rows, h_vals, h_dbg = [], [], []

        for ox, oy, oz, r_o, vx_o, vy_o, vz_o in self.obs:
            p_rel = p_w - np.array([ox, oy, oz])
            v_rel = v_w - np.array([vx_o, vy_o, vz_o])
            r_s   = r_o + r_d

            d2    = dot(p_rel, p_rel)
            d     = rt(d2)
            if d <= r_s + EPS:         
                h_dbg.append(-999.0)
                continue

            s2    = dot(v_rel, v_rel)
            s     = rt(s2)
            rho   = rt(max(EPS**2, d2 - r_s**2))
            p_dot_v = dot(p_rel, v_rel)

            # C3BF formulae (same as original monolith) -------------------
            h_i   = p_dot_v + s * rho
            h_dbg.append(h_i)

            safe_div = s / (rho + EPS)
            Lfh  = s2 + safe_div * p_dot_v
            Lgh  = p_rel + v_rel * (rho / (s + EPS))
            Lf_bar = Lfh - g * Lgh[2]
            Lg_bar = Lgh / m                           # ∂h/∂F

            G_rows.append(-Lg_bar)                    # −Lg F ≤ …
            h_vals.append(Lf_bar + self.gamma * h_i)  # … ≤ RHS

        if not G_rows:
            return None, None, h_dbg

        # collective thrust upper bound (no slack variable here)
        G_rows.append([0.0, 0.0,  1.0])
        h_vals.append(self.Fz_max)
        G_rows.append(np.array([0.0, 0.0, -1.0]))
        h_vals.append(self.Fz_min)
        return matrix(np.vstack(G_rows)), matrix(np.array(h_vals)), h_dbg

    # ----------------------------------------------------------------------
    def filter(self, mode, U_nom, state, clf_out):
        """
        Parameters
        ----------
        mode : str     – controller state name ("TRAJ" enables the filter)
        U_nom : ndarray(4,)  – nominal [F,Mx,My,Mz] from CLF
        state  : dict   – same dict that main.py passes to other filters
        clf_out: dict   – intermediates from first CLF call (unused here)

        Returns
        -------
        U_safe : ndarray(4,)
        slack  : None or cvxopt matrix – residuals (for diagnostics)
        """
        if mode != "TRAJ" or self.obs.size == 0:
            if self.pub:
                self.pub.publish(Float64MultiArray(data=[]))
            return U_nom, None

        m, g  = self.model.m, self.model.g
        p_w   = state['p_vec']
        v_w   = state['v_vec']
        ad_nom= state['ad_nom']      # nominal acceleration from main
        gains = state['gains']
        ref   = state['ref']

        F_nom = m * (ad_nom + np.array([0.0, 0.0, g]))

        G, h, h_dbg = self._build_constraints(p_w, v_w)
        if G is None:                               # no active constraints
            if self.pub:
                self.pub.publish(Float64MultiArray(data=[]))
            return U_nom, None

        P = matrix(np.eye(3))
        q = matrix(-F_nom)

        try:
            sol = solvers.qp(P, q, G, h)
            if sol['status'] != 'optimal':
                raise ValueError("status=%s" % sol['status'])
            F_safe = np.array(sol['x']).flatten()
            slack  = h - G * sol['x']
            
        except Exception as e:
            rospy.logwarn_throttle(1.0,
                "C3BF QP failed (%s) – using nominal.", e)
            if self.pub:
                self.pub.publish(Float64MultiArray(data=[]))
            return U_nom, None

        # ------------------------------------------------------------------
        ad_safe  = F_safe / m - np.array([0.0, 0.0, g])
        ref_safe = ref.copy()
        ref_safe['ad'] = ad_safe
        clf_out2 = self.clf.compute(state, ref_safe, gains)
        U_safe   = clf_out2['U_nom']

        # Publish residuals (and raw h_i for plotting, if desired)
        if self.pub:
            # concatenate slack and h_i vectors: first residuals, then h_i list
            all_vals = list(np.asarray(slack).flatten()) + h_dbg
            self.pub.publish(Float64MultiArray(data=all_vals))

        return U_safe, slack
