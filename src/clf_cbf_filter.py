#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
2D ECBF+CLF safety filter for planar obstacle avoidance
---------------------------------------------------
*   Publishes filtered virtual inputs only when an obstacle is inside a
    configurable look-ahead distance (skips the QP otherwise).
*   Adds a CLF constraint to maintain trajectory tracking.
"""
from __future__ import division
import rospy, numpy as np, math
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Point
from std_msgs.msg import Float64MultiArray
try:
    from cvxopt import matrix, solvers
    solvers.options['show_progress'] = False
except ImportError:
    raise ImportError("cvxopt is required")

def pget(name, default): return rospy.get_param("~"+name, default)

class CbfFilter(object):
    def __init__(self):
        # parameters --------------------------------------------------------
        self.obs       = np.array(pget("obstacle_list", []), dtype=float)
        if self.obs.size == 0:
            rospy.logfatal("~obstacle_list parameter is empty"); raise SystemExit
        self.r_safe    = float(pget("r_safe", 0.5))
        self.k1, self.k2 = map(float, (pget("k1", 2.0), pget("k2", 2.0)))
        self.c_clf     = float(pget("clf_gain", 4.0))
        self.lmb_slack = float(pget("slack_weight", 1000))
        self.lookahead = float(pget("lookahead_margin", 1.0))
        rate           = pget("rate", 100.0)

        # state -------------------------------------------------------------
        self.p = np.zeros(2)
        self.v = np.zeros(2)
        self.u_nom = np.zeros(2)
        self.have_state = False
        self.have_u     = False

        # pubs / subs -------------------------------------------------------
        self.pub_u   = rospy.Publisher("~u_safe",  Point,             queue_size=1)
        self.pub_dbg = rospy.Publisher("~qp_debug",Float64MultiArray, queue_size=1)
        rospy.Subscriber("~odom", Odometry, self.cb_odom, queue_size=10)
        rospy.Subscriber("~u_nom", Point,     self.cb_unom, queue_size=10)
        self.timer = rospy.Timer(rospy.Duration(1.0/rate), self.loop)

        rospy.loginfo("CBF+CLF filter ready: %d obs, r_safe=%.2f, clf=%.2f, lookahead=%.2f, slack=%.2f, k1=%.2f, k2=%.2f",
                      len(self.obs), self.r_safe, self.c_clf, self.lookahead, self.lmb_slack, self.k1, self.k2)

    def cb_odom(self, msg):
        self.p[:] = [msg.pose.pose.position.x, msg.pose.pose.position.y]
        self.v[:] = [msg.twist.twist.linear.x, msg.twist.twist.linear.y]
        self.have_state = True

    def cb_unom(self, msg):
        self.u_nom[:] = [msg.x, msg.y]
        self.have_u   = True

    def build_qp(self):
        # Cost: ½‖u − u_nom‖² + λ δ²
        H = np.diag([1.0, 1.0, 2.0*self.lmb_slack])
        f = np.hstack([-self.u_nom, 0.0])

        G_list, h_list = [], []

        # CLF constraint: 2 v^T u - δ ≤ -2 p^T v - c_clf*V
        V = 2*(self.p.dot(self.p) + self.v.dot(self.v))
        a_clf = np.hstack([2.0*self.v, -1.0])
        b_clf = -2.0*self.p.dot(self.v) - self.c_clf*V
        G_list.append(a_clf)
        h_list.append(b_clf)

        # CBF constraints ---------------------------------------------------
        for ox, oy, rad in self.obs:
            d     = self.p - np.array([ox, oy])
            h_val = d.dot(d) - (rad + self.r_safe)**2
            Lfh   = 2.0*d.dot(self.v)
            S     = 2.0*self.v.dot(self.v) +self.k1*Lfh + self.k2*h_val
            # -2 d^T u ≤ -S
            G_list.append(np.hstack([-2.0*d, 0.0]))
            h_list.append(S)

        # input bound: |u_x|,|u_y| ≤ a_max from max_tilt
        """
        
        phi_max = math.radians(pget("max_tilt_angle_deg",30.0))
        g_val   = pget("gravity",9.81)
        a_max   = g_val * math.tan(phi_max)
        # u_x ≤ a_max
        G_list.append([ 1.0,  0.0, 0.0]); h_list.append( a_max)
        # -u_x ≤ a_max
        G_list.append([-1.0,  0.0, 0.0]); h_list.append( a_max)
        # u_y ≤ a_max
        G_list.append([ 0.0,  1.0, 0.0]); h_list.append( a_max)
        # -u_y ≤ a_max
        G_list.append([ 0.0, -1.0, 0.0]); h_list.append( a_max)
        # δ ≥ 0  ⇒   [0 0 -1] · z ≤ 0
        """
        G_list.append([0.0, 0.0, -1.0])
        h_list.append(0.0)

        G = np.vstack(G_list)
        h = np.array(h_list)

        sol = solvers.qp(matrix(H), matrix(f), matrix(G), matrix(h))
        z   = np.array(sol['x']).flatten()
        return z[:2], z[2]

    def loop(self, _):
        if not (self.have_state and self.have_u): return
        # skip if no obstacle close
        dist = np.linalg.norm(self.p - self.obs[:,:2],axis=1) - (self.obs[:,2]+self.r_safe)
        if np.min(dist) > self.lookahead:
            self.pub_u.publish(Point(self.u_nom[0],self.u_nom[1],0.0))
            self.pub_dbg.publish(Float64MultiArray(data=[self.u_nom[0],self.u_nom[1],-1.0]))
            return
        # solve QP
        try:
            u_safe,delta = self.build_qp()
        except Exception as e:
            rospy.logwarn("QP failed (%s), fallback to u_nom",e)
            u_safe,delta = self.u_nom.copy(), -1.0
        self.pub_u.publish(Point(u_safe[0],u_safe[1],0.0))
        self.pub_dbg.publish(Float64MultiArray(data=[u_safe[0],u_safe[1],delta]))

if __name__=='__main__':
    rospy.init_node('cbf_filter')
    CbfFilter(); rospy.spin()

    
    
    
#as a summary : 
"""
Model:
  ˙p = v,      ˙v = u.

CLF:
  V(e) = [e_p;e_v]^T P [e_p;e_v],  P≻0
  ˙V = 2 e_p^T e_v + 2 e_v^T u
  ⇒ 2 e_p^T e_v + 2 e_v^T u ≤ -c_clf V

CBF (obs i):
  h_i(p)=∥p-o_i∥^2-(r_i+r_safe)^2
  ˙h_i   = 2 d_i^T v
  ¨h_i   = 2 v^T v + 2 d_i^T u
  ψ_i    = ¨h_i + k1 ˙h_i + k2 h_i ≥0
         = 2 d_i^T u + 2∥v∥^2 + 2 k1 d_i^T v + k2 h_i ≥0

QP:
  min_{u,δ≥0}  ∥u-u_nom∥^2 + λ_δ δ^2
  s.t.
    2 e_p^T e_v + 2 e_v^T u + c_clf V ≤ δ
    -2 d_i^T u ≤ 2∥v∥^2 + 2 k1 d_i^T v + k2 h_i ,  ∀i
    δ ≥ 0
"""