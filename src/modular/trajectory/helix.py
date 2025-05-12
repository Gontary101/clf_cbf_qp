#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import division
import math, numpy as np
from dynamics_utils import pget

class HelixTrajectory:
    """Helix reference generator (world frame)."""

    def __init__(self):
        # parameters (can all be overridden via ROS params)
        self.d_start    = pget("helix_start_diameter", 40.0)
        self.d_end      = pget("helix_end_diameter",   15.0)
        self.height     = pget("helix_height",         30.0)
        self.laps       = pget("helix_laps",           4.0)
        self.omega      = pget("trajectory_omega",     0.07)
        self.yaw_fix    = math.radians(pget("fixed_yaw_deg", 0.0))

        # precompute constants
        self.r0         = 0.5 * self.d_start
        theta_tot       = self.laps * 2.0 * math.pi
        self.k_r        = (self.r0 - 0.5 * self.d_end) / theta_tot
        self.k_z        = self.height / theta_tot

        # offsets (to be set by controller, e.g. after hover)
        self.xy_offset  = None
        self.z_offset   = None

    def ref(self, t):
        """Return (pos, vel, acc, psi_d, rd) at time t."""
        omt = self.omega * t
        r   = self.r0 - self.k_r * omt
        z   = self.k_z * omt
        c, s = math.cos(omt), math.sin(omt)

        # position
        pos = np.array([r * c, r * s, z])
        if self.xy_offset is not None:
            pos[:2] += self.xy_offset
        if self.z_offset is not None:
            pos[2]  += self.z_offset

        # velocity
        dr = -self.k_r
        xp = dr * c - r * s
        yp = dr * s + r * c
        zp = self.k_z
        vel = np.array([xp, yp, zp]) * self.omega

        # acceleration
        a0 =  2 * self.k_r * s - r * c
        a1 = -2 * self.k_r * c - r * s
        acc = np.array([a0, a1, 0.0]) * (self.omega ** 2)

        # heading and curvature
        psi_d = math.atan2(vel[1], vel[0])
        denom = vel[0]**2 + vel[1]**2
        rd = (vel[0]*acc[1] - vel[1]*acc[0]) / denom if denom > 1e-6 else 0.0

        return pos, vel, acc, psi_d, rd
