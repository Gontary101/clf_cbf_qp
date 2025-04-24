#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division
import math
# Removed: from typing import List

import numpy as np
import rospy
from cvxopt import matrix
from cvxopt import solvers
from geometry_msgs.msg import Point, Vector3
from nav_msgs.msg import Odometry
from std_msgs.msg import Float64MultiArray

solvers.options["show_progress"] = False  # silent QP

# -----------------------------------------------------------------------------
# Helper
# -----------------------------------------------------------------------------

def pget(name, default):
    return rospy.get_param("~" + name, default)


# -----------------------------------------------------------------------------
# CBF / CLF filter
# -----------------------------------------------------------------------------

class CbfFilter:
    """Safety filter that corrects planar virtual inputs (u_x, u_y)."""

    def __init__(self):
        # ---------------- parameters --------------------------------------
        self.obs = np.array(pget("obstacle_list", []), dtype=float)
        if self.obs.size == 0:
            rospy.logfatal("~obstacle_list parameter is empty – nothing to avoid")
            raise SystemExit

        self.r_safe = float(pget("r_safe", 0.5))
        self.k1 = float(pget("k1", 2.0))
        self.k2 = float(pget("k2", 2.0))
        self.c_clf = float(pget("clf_gain", 4.0))
        self.lmb_slack = float(pget("slack_weight", 1000.0))
        self.lookahead = float(pget("lookahead_margin", 1.0))
        self.m = float(pget("mass", 1.5))
        self.rate = float(pget("rate", 100.0))
        self.use_ref_topics = bool(pget("use_ref_topics", True))
        self.g_val = float(pget("gravity", 9.81))

        # ---------------- dynamic state -----------------------------------
        self.p = np.zeros(2)
        self.v = np.zeros(2)
        self.u_nom = np.zeros(2)
        self.U1 = None  # total thrust
        self.have_state = False
        self.have_u = False

        # references for the CLF
        self.p_ref = np.zeros(2)
        self.v_ref = np.zeros(2)

        # ---------------- pubs / subs ------------------------------------
        self.pub_u = rospy.Publisher("~u_safe", Point, queue_size=1)
        self.pub_dbg = rospy.Publisher("~qp_debug", Float64MultiArray, queue_size=1)

        rospy.Subscriber("~odom", Odometry, self.cb_odom, queue_size=10)
        rospy.Subscriber("~u_nom", Point, self.cb_unom, queue_size=10)
        rospy.Subscriber(
            "/clf_iris_trajectory_controller/control/U", Float64MultiArray,
            self.cb_thrust,
            queue_size=1,
        )

        if self.use_ref_topics:
            rospy.Subscriber("~p_ref", Point, self.cb_p_ref, queue_size=10)
            rospy.Subscriber("~v_ref", Vector3, self.cb_v_ref, queue_size=10)

        self.timer = rospy.Timer(rospy.Duration(1.0 / self.rate), self.loop)

        rospy.loginfo(
            "CBF+CLF filter ready: %d obs • r_safe=%.2f • clf=%.2f • lookahead=%.2f "
            "• slack=%.0f • k1=%.2f • k2=%.2f",
            len(self.obs),
            self.r_safe,
            self.c_clf,
            self.lookahead,
            self.lmb_slack,
            self.k1,
            self.k2,
        )
        self.have_p_ref = False
        self.have_v_ref = False

    # ------------------------------------------------------------------
    # ROS callbacks
    # ------------------------------------------------------------------
    def cb_odom(self, msg):
        self.p[:] = (msg.pose.pose.position.x, msg.pose.pose.position.y)
        self.v[:] = (msg.twist.twist.linear.x, msg.twist.twist.linear.y)
        self.have_state = True

    def cb_unom(self, msg):
        self.u_nom[:] = (msg.x, msg.y)
        self.have_u = True

    def cb_thrust(self, msg):
        self.U1 = msg.data[0] if msg.data else None

    def cb_p_ref(self, msg):
        self.p_ref[:] = (msg.x, msg.y)
        self.have_p_ref = True  # unchanged flag if v_ref still missing

    def cb_v_ref(self, msg):
        self.v_ref[:] = (msg.x, msg.y)
        self.have_v_ref = True
    
    def have_ref(self):
        return self.have_p_ref and self.have_v_ref
    # ------------------------------------------------------------------
    # QP construction
    # ------------------------------------------------------------------
    def build_qp(self):
        # --- nominal acceleration --------------------------------------
        U1 = self.U1 if self.U1 and self.U1 > 0.0 else self.m * self.g_val
        a_nom = (U1 / self.m) * self.u_nom  # a_nom = (U1/m) u_nom

        # decision: [a_x, a_y, δ]
        H = np.diag([1.0, 1.0, 2.0 * self.lmb_slack])
        f = np.hstack((-a_nom, 0.0))

        G = []
        h= []

        # ---------------- CLF row (tracking errors) ---------------------
        if self.have_ref() and self.c_clf > 0.0:
            e_p = self.p - self.p_ref
            e_v = self.v - self.v_ref
            V = e_p.dot(e_p) + e_v.dot(e_v)
            G.append([2.0 * e_v[0], 2.0 * e_v[1], -1.0])
            h.append(-2.0 * e_p.dot(e_v) - self.c_clf * V)

        # ---------------- ECBF rows (correct sign) ----------------------
        # 2 d^T a + 2 v^T v + 2 k1 Lfh + k2 h >= 0  →  -2 d^T a <= -(2 v^T v + 2 k1 Lfh + k2 h)
        for ox, oy, rad in self.obs:
            d = self.p - np.array((ox, oy))
            h_val = d.dot(d) - (rad + self.r_safe) ** 2
            Lfh = 2.0 * d.dot(self.v)
            rhs = -(
                2.0 * self.v.dot(self.v)
                + 2.0 * self.k1 * Lfh
                + self.k2 * h_val
            )
            G.append([-2.0 * d[0], -2.0 * d[1], 0.0])
            h.append(rhs)

        # ---------------- acceleration bounds --------------------------
        a_max = self.g_val * math.tan(math.radians(pget("max_tilt_angle_deg", 30.0)))
        for ex, ey in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            G.append([ex, ey, 0.0])
            h.append(a_max)

        # ---------------- slack positivity -----------------------------
        G.append([0.0, 0.0, -1.0])  # -δ ≤ 0  ⇒  δ ≥ 0
        h.append(0.0)

        #convert to cvxopt matrices
        G_np = np.asarray(G, dtype=float)
        h_np = np.asarray(h, dtype=float)
        m_rows, n_cols = G_np.shape  # n_cols must be 3
        G_cvx = matrix(G_np, (m_rows, n_cols), "d")  # ensure correct dimensions/order
        h_cvx = matrix(h_np, (m_rows, 1), "d")
        H_cvx = matrix(H, tc="d")
        f_cvx = matrix(f, tc="d")
        # ---------------- solve ---------------------------------------
        sol = solvers.qp(H_cvx, f_cvx, G_cvx, h_cvx)
        z = np.array(sol["x"]).flatten()
        a_safe, delta = z[:2], z[2]
        u_safe = (self.m / U1) * a_safe
        return u_safe, delta

    # ------------------------------------------------------------------
    # Timer loop
    # ------------------------------------------------------------------
    def loop(self, _event):
        if not (self.have_state and self.have_u):
            return

        # quick exit if every obstacle is farther than lookahead
        dist = np.linalg.norm(self.p - self.obs[:, :2], axis=1) - (
            self.obs[:, 2] + self.r_safe
        )
        if dist.min() > self.lookahead:
            self.pub_u.publish(Point(self.u_nom[0], self.u_nom[1], 0.0))
            # Changed: [*self.u_nom, -1.0] to list(self.u_nom) + [-1.0]
            self.pub_dbg.publish(Float64MultiArray(data=list(self.u_nom) + [-1.0]))
            return

        try:
            u_safe, delta = self.build_qp()
        except Exception as exc:
            rospy.logwarn("QP failed (%s) – forwarding u_nom", exc)
            u_safe, delta = self.u_nom.copy(), -1.0

        self.pub_u.publish(Point(u_safe[0], u_safe[1], 0.0))
        self.pub_dbg.publish(Float64MultiArray(data=list(u_safe) + [delta]))


# ----------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------

if __name__ == "__main__":
    rospy.init_node("cbf_filter")
    CbfFilter()
    rospy.spin()