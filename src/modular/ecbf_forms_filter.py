#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Exponential / High‑Order CBF filter with 3 different α‑forms.

Solves a small QP on translational **force** + slack:
    min ‖F – F_nom‖² + ρ δ²
subject to CBF, thrust and slack constraints,
then feeds the resulting safe acceleration into a secondary
CLF computation to obtain full [F,Mx,My,Mz] control inputs.

"""

from __future__ import division
import math, numpy as np, rospy
from cvxopt import matrix, solvers
from std_msgs.msg import Float64MultiArray

from dynamics_utils import pget
from clf_backstepping import CLFBackstepping

solvers.options['show_progress'] = False
clip = np.clip


class ECBFFormsFilter(object):
    """
    Exponential / High‑Order CBF filter (three α‑forms) that matches the
    constructor and `filter()` signature of :class:`zcbf_filter.ZCBFFilter`.

    >>> from ecbf_forms_filter import ECBFFormsFilter as SAFETYFilter

    is therefore the *only* change required in *main.py*.
    """

    def __init__(self, model, obstacles, _params=None, cbf_pub=None):
        # keep reference to the physical model
        self.model = model

        # -------- obstacle shape normalisation ----------------------------
        arr = np.asarray(obstacles, dtype=float)
        if arr.ndim == 2 and arr.shape[1] in (4, 10):
            self.obs = arr
        else:
            rospy.logwarn("ECBFFormsFilter: unexpected obstacle shape %s – "
                          "filter disabled.", arr.shape)
            self.obs = np.empty((0, 4))

        self.pub = cbf_pub

        # gains / options (read once, constant afterwards)
        self.cbf_form      = pget("cbf_form", 2)      # 1, 2, or 3
        self.p1            = pget("cbf_p1", 0.1)
        self.p2            = pget("cbf_p2", 1.2)
        self.k0            = pget("cbf_k0", 12.0)
        self.k1            = pget("cbf_k1", 28.0)
        self.slack_penalty = pget("cbf_slack_penalty", 300.0)

        self.Fz_min  = model.m * model.g * model.min_f
        self.Fz_max  = 4.0 * model.kf * (model.w_max ** 2)
        self.tan_tilt= math.tan(model.max_tilt)
        self.clf     = CLFBackstepping(model)

    # ----------------------------------------------------------------------
    def filter(self, mode, U_nom, state, clf_out):
        """Same 4‑argument signature as ZCBFFilter for plug‑and‑play use."""
        if mode != "TRAJ" or self.obs.size == 0:
            if self.pub:
                self.pub.publish(Float64MultiArray(data=[]))
            return U_nom, None

        m, g   = self.model.m, self.model.g
        p_vec  = state['p_vec']
        v_vec  = state['v_vec']
        ad_nom = state['ad_nom']
        gains  = state['gains']
        ref    = state['ref']

        # desired force in world frame (includes weight)
        F_nom = m * (ad_nom + np.array([0.0, 0.0, g]))

        # ---------- QP matrices -------------------------------------------
        rho  = self.slack_penalty
        P    = matrix(np.diag([1.0, 1.0, 1.0, rho]))
        q    = matrix(np.hstack((-F_nom, [0.0])))

        G_rows, h_vals = [], []
        r_drone = self.model.r_drone

        for row in self.obs:
            # Compatible with both (N,4) and (N,10) formats.
            ox, oy, oz = row[:3]
            r          = row[9] if row.shape[0] >= 10 else row[3]
            r_safe = r + r_drone
            pr  = p_vec - np.array([ox, oy, oz])
            Br  = pr.dot(pr) - r_safe ** 2            # h0
            Bd  = 2.0 * pr.dot(v_vec)                 # h0_dot
            v2  = v_vec.dot(v_vec)
            g_prz = g * pr[2]

            if self.cbf_form == 1:        # α2 = k2 √ψ1
                psi1      = Bd + self.p1 * Br
                sqrt_psi1 = math.sqrt(max(psi1, 1e-9))
                psi_rhs   = (2.0 * v2 - 2.0 * g_prz
                             + self.p1 * Bd
                             + self.p2 * sqrt_psi1)

            elif self.cbf_form == 3:      # quadratic / quadratic
                psi1     = Bd + self.p1 * (Br ** 2)
                psi_rhs  = (2.0 * v2 - 2.0 * g_prz
                            + 2.0 * self.p1 * Br * Bd
                            + self.p2 * (psi1 ** 2))
            else:                         # default Form 2: α2 = k1 ψ1
                psi_rhs  = (2.0 * v2 - 2.0 * g_prz
                            + self.k1 * Bd
                            + self.k0 * Br)

            # inequality:  −prᵀ F + δ ≤ ½ m ψ_rhs
            G_rows.append(np.hstack((-pr, [1.0])))
            h_vals.append(0.5 * m * psi_rhs)

        # thrust bounds & slack positivity
        G_rows.extend([[0.0, 0.0,  1.0, 0.0],       # Fz ≤ Fz_max
                       [0.0, 0.0, -1.0, 0.0],       # Fz ≥ Fz_min
                       [0.0, 0.0,  0.0, -1.0]])     # δ ≥ 0
        h_vals.extend([ self.Fz_max,
                       -self.Fz_min,
                        0.0])

        G = matrix(np.vstack(G_rows))
        h = matrix(np.array(h_vals))

        try:
            sol = solvers.qp(P, q, G, h)
            if sol['status'] != 'optimal':
                raise ValueError("status=%s" % sol['status'])
            vec   = np.array(sol['x']).flatten()
            F_safe, delta = vec[:3], vec[3]
            slack = h - G * sol['x']
        except Exception as e:
            rospy.logwarn_throttle(1.0,
                "ECBF QP failed (%s) – falling back to nominal.", e)
            if self.pub:
                self.pub.publish(Float64MultiArray(data=[0.0]*len(h_vals)))
            return U_nom, None

        # safe acceleration & recomputed full control -----------------------
        ad_safe = F_safe / m - np.array([0.0, 0.0, g])
        ref_safe = ref.copy(); ref_safe['ad'] = ad_safe
        clf_out2 = self.clf.compute(state, ref_safe, gains)
        U_safe   = clf_out2["U_nom"]

        if self.pub:
            self.pub.publish(Float64MultiArray(
                data=list(np.asarray(slack).flatten())))

        return U_safe, slack
