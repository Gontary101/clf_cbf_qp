#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import division
import math, numpy as np, rospy

clip = np.clip
rt   = np.sqrt

def pget(name, default):
    """Private parameter fetch helper (`~param_name`)."""
    return rospy.get_param("~" + name, default)


# World / body z‑axes (handy constants shared by several modules)
e3_world = np.array([0.0, 0.0, 1.0])
e3_body  = np.array([0.0, 0.0, 1.0])


def rotation_matrix(phi, th, psi):
    """Body-to-world rotation from ZYX Euler angles."""
    c, s = math.cos, math.sin
    return np.array([
        [c(th) * c(psi),
         s(phi) * s(th) * c(psi) - c(phi) * s(psi),
         c(phi) * s(th) * c(psi) + s(phi) * s(psi)],
        [c(th) * s(psi),
         s(phi) * s(th) * s(psi) + c(phi) * c(psi),
         c(phi) * s(th) * s(psi) - s(phi) * c(psi)],
        [-s(th),
         s(phi) * c(th),
         c(phi) * c(th)]
    ])


class DroneModel(object):
    """Physical parameters + allocation logic for a quadrotor."""

    def __init__(self):
        # -------- basic parameters -----------------------------------------
        self.m  = pget("mass",   1.5)
        self.g  = pget("gravity", 9.81)
        self.Ix = pget("I_x",    0.0348)
        self.Iy = pget("I_y",    0.0459)
        self.Iz = pget("I_z",    0.0977)

        self.kf = pget("motor_constant",   8.54858e-06)
        self.km = pget("moment_constant",  1.3677728e-07)
        self.w_max      = pget("max_rot_velocity",    838.0)
        self.min_f      = pget("min_thrust_factor",     0.1)
        self.gc         = pget("gravity_comp_factor",   1.022)
        self.max_tilt   = math.radians(pget("max_tilt_angle_deg", 30.0))
        self.r_drone    = pget("drone_radius", 0.5)

        # -------- inertia & mixer ------------------------------------------
        self.J_mat      = np.diag([self.Ix, self.Iy, self.Iz])
        self.J_inv_diag = np.diag([1.0 / self.Ix,
                                   1.0 / self.Iy,
                                   1.0 / self.Iz])

        A = np.array([
            [self.kf] * 4,
            [-0.22 * self.kf,  0.20 * self.kf,  0.22 * self.kf, -0.20 * self.kf],
            [-0.13 * self.kf,  0.13 * self.kf, -0.13 * self.kf,  0.13 * self.kf],
            [-self.km,        -self.km,         self.km,         self.km]
        ])
        self.invA = np.linalg.inv(A)

    # ----------------------------------------------------------------------
    def thrust_torques_to_motor_speeds(self, U):
        """
        Convert [F, Mx, My, Mz] to commanded motor speeds (rad/s).
        Returns (w_cmd, w_squared).
        """
        w_sq  = clip(np.dot(self.invA, U), 0.0, None)
        w_cmd = clip(rt(w_sq), 0.0, self.w_max)
        return w_cmd, w_sq
