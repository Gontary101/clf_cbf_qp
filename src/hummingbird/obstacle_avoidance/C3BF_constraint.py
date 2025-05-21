#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division
import numpy as np, rospy
from utils.dynamics_utils2 import pget

EPS = 1e-3

class C3BFFilter(object):
    def __init__(self, model, obstacles, _unused=None, cbf_pub=None):
        self.model = model
        self.pub   = cbf_pub
        self.gamma = pget("cbf_gamma", 500.0)
        # only enforce CBF when obstacles come within this distance
        self.cbf_enter = pget("cbf_active_range", 2.0)
        self.r_drone  = self.model.r_drone
        obs = np.asarray(obstacles, dtype=float)
        if obs.ndim == 2 and obs.shape[1] in (4,7,10):
            if obs.shape[1] == 10:
                radii = obs[:, 9:10]    
                vels  = obs[:, 3:6]       
                self.obs = np.hstack([obs[:,0:3], radii, vels])
            elif obs.shape[1] == 4:
                rospy.loginfo("C3BFFilter: padding (N,4) with zero velocities")
                zpad = np.zeros((obs.shape[0],3))
                self.obs = np.hstack([obs, zpad])
            else:
                rospy.loginfo("C3BFFilter: using (N,7) obstacles as is")
                self.obs = obs
        else:
            rospy.logwarn("C3BFFilter: unexpected obstacle shape %s; disabling", obs.shape)
            self.obs = np.empty((0,7))

    def constraints(self, state):
        if self.obs.size == 0:
            return np.empty((0, 4)), np.empty((0,))

        p_w = state['p_vec']
        v_w = state['v_vec']
        m   = self.model.m
        g   = self.model.g

        G_rows = []
        h_vals = []

        for ox, oy, oz, r_o, vx, vy, vz in self.obs:
            p_rel = p_w - np.array([ox, oy, oz])
            dist  = np.dot(p_rel, p_rel)
            # skip any obstacle outside the active range
            if dist > self.cbf_enter:
                continue

            # safe-radius test: if inside the union of radii, skip
            v_rel = v_w - np.array([vx, vy, vz])
            r_safe = r_o + self.r_drone
            d2      = np.dot(p_rel, p_rel)
            if d2 <= (r_safe + EPS)**2:
                continue

            s    = np.dot(v_rel, v_rel)
            pdv  = np.dot(p_rel, v_rel)
            rho = np.sqrt(max(EPS**2, d2 - r_safe**2))
            h_i   = pdv + s * rho
            Lgh   = p_rel + v_rel * (rho/(s+EPS))
            Lfh   = s + (s/(rho+EPS))*pdv - g * p_rel[2]
            LgF   = Lgh / m
            Gf = -LgF
            G_rows.append(Gf)
            h_vals.append(Lfh + self.gamma * h_i)

        if not G_rows:
            return np.empty((0, 4)), np.empty((0,))
        G_all = np.vstack(G_rows)
        h_all = np.array(h_vals)
        return G_all, h_all
