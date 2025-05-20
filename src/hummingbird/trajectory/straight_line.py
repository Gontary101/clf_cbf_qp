#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import division
import math, numpy as np, rospy, ast
from utils.dynamics_utils2 import pget

class StraightLineTrajectory:

    def __init__(self):
        try:
            p0_str = pget("trajectory_start_point", "[0.0, 0.0, 3.0]")
            self.p0_initial = np.array(ast.literal_eval(p0_str), dtype=float)
            if self.p0_initial.shape != (3,):
                raise ValueError("start_point must be a 3-element list/tuple.")
        except (ValueError, SyntaxError) as e:
            rospy.logwarn("Invalid trajectory_start_point '%s': %s. Using [0, 0, 3].", p0_str, e)
            self.p0_initial = np.array([0.0, 0.0, 3.0])

        try:
            pf_str = pget("trajectory_end_point", "[10.0, 10.0, 5.0]")
            self.pf_initial = np.array(ast.literal_eval(pf_str), dtype=float)
            if self.pf_initial.shape != (3,):
                raise ValueError("end_point must be a 3-element list/tuple.")
        except (ValueError, SyntaxError) as e:
            rospy.logwarn("Invalid trajectory_end_point '%s': %s. Using [10, 10, 5].", pf_str, e)
            self.pf_initial = np.array([10.0, 10.0, 5.0])

        self.p0 = self.p0_initial.copy()
        self.pf = self.pf_initial.copy()

        self.T = pget("trajectory_duration", 20.0)
        if self.T <= 1e-6:
            rospy.logwarn("Trajectory duration T=%.3f too small. Setting to 1.0.", self.T)
            self.T = 1.0

        self.yaw_fix_initial = math.radians(pget("fixed_yaw_deg", 0.0))
        self.yaw_fix = self.yaw_fix_initial

        self.xy_offset = None
        self.z_offset = None

        self._update_trajectory_params()

    def _update_trajectory_params(self):
        """Recalculates delta, velocity, and desired yaw based on current p0, pf."""
        self.delta_p = self.pf - self.p0
        delta_norm = np.linalg.norm(self.delta_p)

        if delta_norm > 1e-6:
            self.v_const = self.delta_p / self.T
            self.psi_d = math.atan2(self.delta_p[1], self.delta_p[0])
        else:
            rospy.logwarn("Start and end points are the same. Velocity will be zero.")
            self.v_const = np.zeros(3)
            initial_delta = self.pf_initial - self.p0_initial
            if np.linalg.norm(initial_delta) > 1e-6:
                 self.psi_d = math.atan2(initial_delta[1], initial_delta[0])
            else:
                 self.psi_d = self.yaw_fix_initial

        self.xy_offset = None
        self.z_offset = None
        rospy.loginfo("Trajectory params updated: p0=[%.1f,%.1f,%.1f], pf=[%.1f,%.1f,%.1f], psi_d=%.1f deg",
                      self.p0[0], self.p0[1], self.p0[2],
                      self.pf[0], self.pf[1], self.pf[2],
                      math.degrees(self.psi_d))

    def reverse(self):
        """Swaps start and end points and recalculates trajectory parameters."""
        rospy.loginfo("Reversing trajectory.")
        p0_old = self.p0.copy()
        self.p0 = self.pf.copy()
        self.pf = p0_old
        self._update_trajectory_params()

    def ref(self, t):
        """Calculates reference at time t along the current segment."""
        frac = t / self.T

        if frac >= 1.0:
            pos_base = self.pf
            vel = np.zeros(3)
            acc = np.zeros(3)
        elif frac < 0.0:
             pos_base = self.p0
             vel = np.zeros(3)
             acc = np.zeros(3)
        else:
            pos_base = self.p0 + self.delta_p * frac
            vel = self.v_const
            acc = np.zeros(3)

        pos = pos_base.copy()
        if self.xy_offset is not None:
            pos[:2] += self.xy_offset
        if self.z_offset is not None:
            pos[2] += self.z_offset

        psi_d = self.psi_d
        rd = 0.0

        return pos, vel, acc, psi_d, rd