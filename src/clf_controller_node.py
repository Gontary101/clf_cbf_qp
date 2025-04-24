#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rospy
import math
import numpy as np
# Use Odometry or ModelStates? Let's use Odometry for better practice so for now i have a problem with odometry, changing to perfect states (as in modelStates works pretty well, probably an issue with the IMU or other sensors, did we even include them?) -> i need to look into this and maybe add the modules into the drone, since they're already in my workspace.
# from gazebo_msgs.msg import ModelStates
from nav_msgs.msg import Odometry
from mav_msgs.msg import Actuators # Use Actuators for commands
from std_msgs.msg import Float32
from tf.transformations import euler_from_quaternion
from std_msgs.msg import Bool

class ClfIrisController(object):
    def __init__(self):
        rospy.loginfo("Initializing CLF Controller for Iris")

        # --- Parameters ---
        self.namespace = rospy.get_param("~namespace", "iris")
        self.use_model_states = rospy.get_param("~use_model_states", False) # Option to use ModelStates

        # Iris Dynamics (Load from parameters or use XACRO values)
        self.mass = rospy.get_param("~mass", 1.5)  # kg
        self.gravity = rospy.get_param("~gravity", 9.81)   # m/s^2
        self.I_x = rospy.get_param("~I_x", 0.0347563)  # kg*m^2
        self.I_y = rospy.get_param("~I_y", 0.0458929)  # kg*m^2
        self.I_z = rospy.get_param("~I_z", 0.0977)     # kg*m^2

        # Iris Motor/Prop Constants (from XACRO)
        self.k_f = rospy.get_param("~motor_constant", 8.54858e-06)
        self.k_m = rospy.get_param("~moment_constant", 0.016 * self.k_f) # Using RotorS relation moment_constant * k_f
        self.max_rot_velocity = rospy.get_param("~max_rot_velocity", 838.0) # rad/s
        self.motor_count = 4
        self.prev_enabled = False
        # Arm lengths
        l_fx = rospy.get_param("~arm_length_front_x", 0.13)
        l_bx = rospy.get_param("~arm_length_back_x", 0.13)
        l_fy = rospy.get_param("~arm_length_front_y", 0.22)
        l_by = rospy.get_param("~arm_length_back_y", 0.20)
        self.l_x_arm = (l_fx + l_bx) / 2.0
        self.l_y_arm = (l_fy + l_by) / 2.0

        # Control targets
        self.target_z = rospy.get_param("~target_altitude", 1.5)  # meters
        self.target_yaw = rospy.get_param("~target_yaw_deg", 0.0) * math.pi / 180.0 # radians

        # Control gains (Used only for Yaw PD)
        self.kp_yaw = rospy.get_param("~kp_yaw", 2.0)
        self.kd_yaw = rospy.get_param("~kd_yaw", 1.0)
        # Note: Altitude/Roll/Pitch gains k1=1, k2=2 are implicit in the CLF formulas used

        # --- Precompute Mixer Matrix Inverse ---
        A = np.array([
            [self.k_f, self.k_f, self.k_f, self.k_f],
            [-self.l_y_arm*self.k_f, self.l_y_arm*self.k_f, self.l_y_arm*self.k_f, -self.l_y_arm*self.k_f],
            [self.l_x_arm*self.k_f, -self.l_x_arm*self.k_f, self.l_x_arm*self.k_f, -self.l_x_arm*self.k_f],
            [-self.k_m, -self.k_m, self.k_m, self.k_m]
        ])
        try:
            self.inv_A = np.linalg.inv(A)
            rospy.loginfo("Mixer matrix computed successfully.")
        except np.linalg.LinAlgError:
            rospy.logerr("Mixer matrix A is singular. Check parameters.")
            self.inv_A = None

        # --- Publishers ---
        self.command_pub = rospy.Publisher(
            self.namespace + '/command/motor_speed', Actuators, queue_size=1)

        # Optional: Debug publishers
        self.U1Pub = rospy.Publisher('~control/U1', Float32, queue_size=1)
        self.U2Pub = rospy.Publisher('~control/U2', Float32, queue_size=1)
        self.U3Pub = rospy.Publisher('~control/U3', Float32, queue_size=1)
        self.U4Pub = rospy.Publisher('~control/U4', Float32, queue_size=1)
        self.err_zPub = rospy.Publisher('~error/z', Float32, queue_size=1)
        self.err_yawPub = rospy.Publisher('~error/yaw_deg', Float32, queue_size=1)
        


        # --- Subscriber ---
        self.enabled = False
        self.enable_sub = rospy.Subscriber('clf_hover_enable', Bool, self.enable_callback)
        self.latest_state = None
        if self.use_model_states:
            from gazebo_msgs.msg import ModelStates # Import only if needed
            self.state_sub = rospy.Subscriber(
                '/gazebo/model_states', ModelStates, self.model_state_callback, queue_size=1)
            rospy.loginfo("Subscribing to /gazebo/model_states")
        else:
            self.state_sub = rospy.Subscriber(
                self.namespace + '/ground_truth/odometry', Odometry, self.odometry_callback, queue_size=1)
            rospy.loginfo("Subscribing to %s/ground_truth/odometry", self.namespace)


        # --- Control Loop Timer ---
        control_rate = rospy.get_param("~control_rate", 50.0) # Hz
        self.control_timer = rospy.Timer(rospy.Duration(1.0 / control_rate), self.control_loop)

        rospy.loginfo("CLF Controller Initialized for namespace: %s", self.namespace)

    # --- Callback Functions ---
    
    
    def enable_callback(self, msg):
        """Log only when enable state changes."""
        if msg.data != self.enabled:  # Log only if new state differs
            self.enabled = msg.data
            if self.enabled:
                rospy.loginfo("Hover controller enabled")
            else:
                rospy.loginfo("Hover controller disabled")
        
        
    def odometry_callback(self, msg):
        """Stores the latest state from Odometry."""
        self.latest_state = msg

    def model_state_callback(self, msg):
        """Stores the latest state from ModelStates, converting to Odometry-like structure."""
        try:
            # Find the index corresponding to the namespace (model name)
            # Assumes model name in Gazebo matches the namespace
            ind = msg.name.index(self.namespace)
            # Create a pseudo-Odometry message (can't create the real type easily here)
            odom_like = lambda: None # Create empty object
            odom_like.header = rospy.Header() # Add header later if needed
            odom_like.header.stamp = rospy.Time.now()
            odom_like.header.frame_id = self.namespace + "/odom" # Guess frame
            odom_like.child_frame_id = self.namespace + "/base_link" # Guess frame
            odom_like.pose = lambda: None
            odom_like.pose.pose = msg.pose[ind]
            odom_like.twist = lambda: None
            odom_like.twist.twist = msg.twist[ind]
            self.latest_state = odom_like # Store the converted state
        except ValueError:
            self.latest_state = None
            rospy.logwarn_throttle(5.0, "Model '%s' not found in /gazebo/model_states.", self.namespace)

    # --- Control Loop ---
    def control_loop(self, event):
        """Main control loop."""
        if self.latest_state is None or self.inv_A is None:
            rospy.logwarn_throttle(2.0, "State not received or mixer invalid. Skipping control loop.")
            return

        # --- State Extraction (from Odometry or pseudo-Odometry) ---
        z = self.latest_state.pose.pose.position.z
        q_orientation = self.latest_state.pose.pose.orientation
        orientation_list = [q_orientation.x, q_orientation.y, q_orientation.z, q_orientation.w]
        (phi, theta, psi) = euler_from_quaternion(orientation_list) # roll, pitch, yaw

        # Compute world vertical velocity
        if self.use_model_states:
            world_vz = self.latest_state.twist.twist.linear.z  # World frame from ModelStates
        else:
            # Body frame to world frame transformation
            v_bx = self.latest_state.twist.twist.linear.x
            v_by = self.latest_state.twist.twist.linear.y
            v_bz = self.latest_state.twist.twist.linear.z
            s_theta = math.sin(theta)
            c_theta = math.cos(theta)
            s_phi = math.sin(phi)
            c_phi = math.cos(phi)
            s_psi = math.sin(psi)
            c_psi = math.cos(psi)
            R20 = -s_theta * c_psi + c_theta * s_phi * s_psi
            R21 = -s_theta * s_psi - c_theta * s_phi * c_psi
            R22 = c_theta * c_phi
            world_vz = R20 * v_bx + R21 * v_by + R22 * v_bz
        p = self.latest_state.twist.twist.angular.x # roll rate
        q = self.latest_state.twist.twist.angular.y # pitch rate
        r = self.latest_state.twist.twist.angular.z # yaw rate

        # --- Calculate Errors ---
        z_d = self.target_z
        psi_d = self.target_yaw
        phi_d = 0.0
        theta_d = 0.0
        vz_d = 0.0
        p_d = 0.0
        q_d = 0.0
        r_d = 0.0

        e_z1 = z - z_d
        e_z2 = world_vz - vz_d
        e_phi1 = phi - phi_d
        e_phi2 = p - p_d
        e_theta1 = theta - theta_d
        e_theta2 = q - q_d
        e_psi1 = psi - psi_d
        e_psi1 = (e_psi1 + np.pi) % (2 * np.pi) - np.pi # Normalize
        e_psi2 = r - r_d

        # --- CLF Control Law ---
        cos_phi = math.cos(phi)
        cos_theta = math.cos(theta)
        thrust_orientation_factor = cos_phi * cos_theta
        # Add safety check for near-singular orientation
        if abs(thrust_orientation_factor) < 0.1:
             thrust_orientation_factor = np.sign(thrust_orientation_factor) * 0.1
             rospy.logwarn_throttle(1.0, "Near-singular orientation detected, capping thrust factor.")

        # Altitude Control (U1 = Total Thrust) - Implicit gains k1=1, k2=2
        U1 = (self.mass / thrust_orientation_factor) * (-e_z1 - 2 * e_z2 + self.gravity)
        U1 = max(0.0, U1)

        # Roll Control (U2 = Roll Torque) - Implicit gains k1=1, k2=2
        roll_coupling = q * r * (self.I_y - self.I_z)
        U2 = self.I_x * (-e_phi1 - 2 * e_phi2) - roll_coupling # Removed division by Ix, as U2 IS the torque

        # Pitch Control (U3 = Pitch Torque) - Implicit gains k1=1, k2=2
        pitch_coupling = p * r * (self.I_z - self.I_x)
        U3 = self.I_y * (e_theta1 + 2 * e_theta2) + pitch_coupling # Removed division by Iy

        # Yaw Control (U4 = Yaw Torque) - PD Control
        # yaw_coupling = p * q * (self.I_x - self.I_y) # Optional
        U4 = self.I_z * (-self.kp_yaw * e_psi1 - self.kd_yaw * e_psi2) # Removed coupling

        # --- Mixer ---
        U = np.array([U1, U2, U3, U4])
        omega_sq = np.dot(self.inv_A, U)
        omega_sq = np.maximum(omega_sq, 0.0) # Ensure non-negative
        omega = np.sqrt(omega_sq)

        # --- Saturate ---
        omega = np.minimum(omega, self.max_rot_velocity)
        # NOTE: No minimum speed enforced here, unlike original clf.py. Motors can go to 0.

        # --- Publish ---
        if self.enabled:
            actuator_msg = Actuators()
            actuator_msg.header.stamp = rospy.Time.now()
            actuator_msg.angular_velocities = omega.tolist() # Order 0,1,2,3 -> FR,BL,FL,BR
            self.command_pub.publish(actuator_msg)

            # --- Publish Debug Info ---
            self.U1Pub.publish(Float32(U1))
            self.U2Pub.publish(Float32(U2))
            self.U3Pub.publish(Float32(U3))
            self.U4Pub.publish(Float32(U4))
            self.err_zPub.publish(Float32(e_z1))
            self.err_yawPub.publish(Float32(math.degrees(e_psi1)))

    def shutdown_hook(self):
        """Send zero commands on shutdown."""
        rospy.loginfo("CLF Controller shutting down. Sending zero motor speeds.")
        if self.command_pub:
             # Publish multiple times to ensure it gets through
             for _ in range(5):
                 self.publish_zero_command()
                 rospy.sleep(0.01)


if __name__ == '__main__':
    try:
        rospy.init_node("clf_iris_controller", anonymous=True)
        controller = ClfIrisController()
        rospy.on_shutdown(controller.shutdown_hook) # Register shutdown hook
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
    except Exception as e:
        rospy.logerr("CLF Controller encountered an error: %s", e)
        import traceback
        traceback.print_exc()
