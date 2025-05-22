import numpy as np
import rospy
from tf.transformations import euler_from_quaternion
from utils.dynamics_utils2 import rotation_matrix

class StateEstimator(object):
    def __init__(self):
        # state for LPF on omega_b
        self.prev_omega_b = np.zeros(3)
        # same alpha for all channels
        self.omega_filter_alpha = rospy.get_param("~omega_filter_alpha", 0.5)
        # state for LPF on position, velocity and Euler angles
        self.prev_p_vec = np.zeros(3)
        self.prev_v_world = np.zeros(3)
        self.prev_euler = np.zeros(3)

    def process_odometry(self, odom_msg, use_gz_sim):
        p = odom_msg.pose.pose.position
        q = odom_msg.pose.pose.orientation
        v = odom_msg.twist.twist.linear
        w = odom_msg.twist.twist.angular

        phi, th, psi = euler_from_quaternion((q.x, q.y, q.z, q.w))
        R_mat = rotation_matrix(phi, th, psi)
        p_vec = np.array([p.x, p.y, p.z])

        # get raw measurements
        if use_gz_sim:
            v_world = np.array([v.x, v.y, v.z])
            omega_b = np.dot(R_mat.T, np.array([w.x, w.y, w.z]))
        else:
            v_body = np.array([v.x, v.y, v.z])
            v_world = np.dot(R_mat, v_body)
            omega_b = np.array([w.x, w.y, w.z])
        alpha = self.omega_filter_alpha

        
        # 1) filter position
        p_vec = alpha * self.prev_p_vec + (1.0 - alpha) * p_vec
        self.prev_p_vec = p_vec

        

        return {
            "p_vec": p_vec,
            "v_world": v_world,
            "phi": phi,
            "th": th,
            "psi": psi,
            "omega_b": omega_b,   
            "R_mat": R_mat
        }