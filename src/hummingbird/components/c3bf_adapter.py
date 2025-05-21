#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import division
import numpy as np
from cvxopt import matrix, solvers
from std_msgs.msg import Float64MultiArray
import rospy
from obstacle_avoidance.C3BF_constraint import C3BFFilter as ConstraintOnlyC3BFFilter
from clf_backstepping import CLFBackstepping

solvers.options['show_progress'] = False

class C3BFFilterAdapter(ConstraintOnlyC3BFFilter):
    """
    Wraps the existing constraint-only C3BFFilter and adds:
      1) a 3-DOF force‐space QP,
      2) a re-invoke of the CLF controller on F_safe → a_safe,
      3) publishes slack exactly like the adapter.
    """
    def filter(self, mode, U_nom, state, clf_out):
        # only active in TRAJ
        if mode != "TRAJ":
            return U_nom, None

        # get raw G/h from the parent
        G_all, h_all = super().constraints(state)
        # no constraints → pass through
        if G_all.size == 0:
            if self.pub: 
                self.pub.publish(Float64MultiArray(data=[]))
            return U_nom, None

        # build world‐force QP:   minimize ||F – m(a_nom + g e3)||²
        m = self.model.m
        g = self.model.g
        ad_nom = state['ad_nom']
        F_nom = m * (ad_nom + np.array([0.0, 0.0, g]))

        P = matrix(np.eye(3))
        q = matrix(-F_nom)

        # parent G_all is shape (n,4): [ G_F  0 ]^T U ≤ h
        # we need only the first 3 columns
        Gf = matrix(G_all[:, :3])
        hf = matrix(h_all)

        sol = solvers.qp(P, q, Gf, hf)
        if sol['status'] != 'optimal':
            rospy.logwarn_throttle(1.0, "C3BF‐Adapter QP failed (%s)", sol['status'])
            if self.pub:
                self.pub.publish(Float64MultiArray(data=[]))
            return U_nom, None

        F_safe = np.array(sol['x']).flatten()
        slack   = np.asarray(hf - Gf * sol['x']).flatten()
        # publish slack (if desired)
        if self.pub:
            self.pub.publish(Float64MultiArray(data=list(slack)))

        # now re-invoke CLF‐backstepping on the “safe” accel
        a_safe = F_safe/m - np.array([0.0, 0.0, g])
        ref_safe = state['ref'].copy()
        ref_safe['ad'] = a_safe

        clf2 = CLFBackstepping(self.model).compute(state, ref_safe, state['gains'])
        U_safe = clf2['U_nom']
        return U_safe, slack
