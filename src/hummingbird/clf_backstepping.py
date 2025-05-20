#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import division
import math, numpy as np
import rospy

class CLFBackstepping(object):
    def __init__(self, model):
        self.model = model

    def compute(self, state, ref, gains):
        m, g, gc = self.model.m, self.model.g, self.model.gc
        min_f, max_tilt = self.model.min_f, self.model.max_tilt
        Ix, Iy, Iz = self.model.Ix, self.model.Iy, self.model.Iz
        kf, km     = self.model.kf, self.model.km

        g1, g2, g3, g4 = gains

        p = state["p_vec"];      v = state["v_vec"]
        phi, th, psi = state["phi"], state["th"], state["psi"]
        omega_b      = state["omega_body"]

        tgt, vd, ad = ref["tgt"], ref["vd"], ref["ad"]
        yd, rd      = ref["yd"],  ref["rd"]

        ex1 = p - tgt
        ex2 = v - vd

        tof = math.cos(phi) * math.cos(th)
        if abs(tof) < min_f:
            tof = (1 if tof >= 0 else -1) * min_f

        U1_nom = (m / tof) * (-g1 * ex1[2] + ad[2] - g2 * ex2[2]) + m * g * gc / tof
        U1_nom = max(0.0, U1_nom)

        if U1_nom < 1e-6:
            Uex = Uey = 0.0
        else:
            Uex = (m / U1_nom) * (-g1 * ex1[0] + ad[0] - g2 * ex2[0])
            Uey = (m / U1_nom) * (-g1 * ex1[1] + ad[1] - g2 * ex2[1])

        sp, cp = math.sin(yd), math.cos(yd)
        try:
            phi_d = math.asin(np.clip(Uex * sp - Uey * cp, -1.0, 1.0))
        except ValueError:
            phi_d = math.pi / 2.0 * (1 if (Uex * sp - Uey * cp) >= 0 else -1)

        cpd = math.cos(phi_d)
        try:
            theta_d = 0.0 if abs(cpd) < min_f else math.asin(
                np.clip((Uex * cp + Uey * sp) / cpd, -1.0, 1.0))
        except ValueError:
            theta_d = math.pi / 2.0 * (1 if (Uex * cp + Uey * sp) >= 0 else -1)\
                      if abs(cpd) >= min_f else 0.0

        phi_d   = np.clip(phi_d,   -max_tilt, max_tilt)
        theta_d = np.clip(theta_d, -max_tilt, max_tilt)

        e_th = np.array([phi - phi_d,
                         th  - theta_d,
                         (psi - yd + math.pi) % (2 * math.pi) - math.pi])
        e_w  = omega_b - np.array([0.0, 0.0, rd])

        U2_nom = Ix * (-g3 * e_th[0] - g4 * e_w[0]) \
                 - omega_b[1] * omega_b[2] * (Iy - Iz)
        U3_nom = Iy * (-g3 * e_th[1] - g4 * e_w[1]) \
                 - omega_b[0] * omega_b[2] * (Iz - Ix)
        U4_nom = Iz * (-g3 * e_th[2] - g4 * e_w[2]) \
                 - omega_b[0] * omega_b[1] * (Ix - Iy)
        U4_nom = np.clip(U4_nom, -0.75 * U1_nom * km / kf,
                                   0.75 * U1_nom * km / kf)

        U_nom = np.array([U1_nom, U2_nom, U3_nom, U4_nom])

        return dict(U_nom=U_nom, phi_d=phi_d, theta_d=theta_d,
                    e_th=e_th, e_w=e_w, ex1=ex1, ex2=ex2,
                    Uex=Uex, Uey=Uey)