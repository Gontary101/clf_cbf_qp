#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rospy
import math
import numpy as np
from nav_msgs.msg import Odometry
from mav_msgs.msg import Actuators
from std_msgs.msg import Float32, Float64MultiArray # Added for debug
from geometry_msgs.msg import Point # Added for debug
from tf.transformations import euler_from_quaternion

# Add a flag for enabling/disabling detailed logging easily
ENABLE_DETAILED_LOGGING = True
LOG_THROTTLE_SECS = 1.0 # Log every N seconds

class ClfIrisController(object):
    def __init__(self):
        rospy.loginfo("Initializing CLF Sequential Position Controller for Iris (Absolute Axis Threshold)")
        self.namespace = rospy.get_param("~namespace", "iris")
        self.use_model_states = rospy.get_param("~use_model_states", False) # model states or odometry

        # Iris Dynamics
        self.mass = rospy.get_param("~mass", 1.5)  # kg
        self.gravity = rospy.get_param("~gravity", 9.81)   # m/s^2
        self.I_x = rospy.get_param("~I_x", 0.0347563)  # kg*m^2
        self.I_y = rospy.get_param("~I_y", 0.0458929)  # kg*m^2
        self.I_z = rospy.get_param("~I_z", 0.0977)     # kg*m^2

        # Iris Motor/Prop Constants
        self.k_f = rospy.get_param("~motor_constant", 8.54858e-06)
        self.k_m = rospy.get_param("~moment_constant", 0.016 * self.k_f)
        self.max_rot_velocity = rospy.get_param("~max_rot_velocity", 838.0) # rad/s
        self.motor_count = 4

        # Arm lengths
        l_fx = rospy.get_param("~arm_length_front_x", 0.13)
        l_bx = rospy.get_param("~arm_length_back_x", 0.13)
        l_fy = rospy.get_param("~arm_length_front_y", 0.22)
        l_by = rospy.get_param("~arm_length_back_y", 0.20)
        self.l_x_arm = (l_fx + l_bx) / 2.0
        self.l_y_arm = (l_fy + l_by) / 2.0
        rospy.loginfo("Using original arm lengths: lx=%.4f, ly=%.4f", self.l_x_arm, self.l_y_arm)

        # --- Control Targets ---
        # Target A (Initial Target)
        self.target_A_x = rospy.get_param("~target_A_x", 0.0) # meters (East)
        self.target_A_y = rospy.get_param("~target_A_y", 0.0) # meters (North)
        self.target_A_z = rospy.get_param("~target_A_z", 1.5) # meters (Up)

        # Target B (Second Target)
        self.target_B_x = rospy.get_param("~target_B_x", 1.0) # meters (East)
        self.target_B_y = rospy.get_param("~target_B_y", 1.0) # meters (North)
        self.target_B_z = rospy.get_param("~target_B_z", 1.5) # meters (Up)

        # Shared Yaw Target
        self.target_yaw = rospy.get_param("~target_yaw_deg", 0.0) * math.pi / 180.0 # radians

        # --- Waypoint Switching Logic ---
        # Threshold for switching from A to B. Drone switches when |ex|, |ey|, AND |ez| are ALL below this value.
        self.waypoint_switch_threshold = rospy.get_param("~waypoint_switch_threshold", 0.1) # meters
        self.current_target_state = "NAVIGATING_TO_A" # States: "NAVIGATING_TO_A", "NAVIGATING_TO_B"
        # Removed target_A_magnitude calculation

        # Active target used in control loop calculations
        self.active_target_x = self.target_A_x
        self.active_target_y = self.target_A_y
        self.active_target_z = self.target_A_z
        rospy.loginfo("Target A: [%.2f, %.2f, %.2f]", self.target_A_x, self.target_A_y, self.target_A_z)
        rospy.loginfo("Target B: [%.2f, %.2f, %.2f]", self.target_B_x, self.target_B_y, self.target_B_z)
        rospy.loginfo("Target Yaw: %.1f deg", math.degrees(self.target_yaw))
        rospy.loginfo("Waypoint Switch Threshold: %.3f m (absolute error on each axis)", self.waypoint_switch_threshold) # Updated log message

        # Desired accelerations & rates (zero for setpoint tracking)
        self.x_dd, self.y_dd, self.z_dd = 0.0, 0.0, 0.0
        self.phi_dd, self.theta_dd, self.psi_dd = 0.0, 0.0, 0.0
        self.p_d, self.q_d, self.r_d = 0.0, 0.0, 0.0

        # CLF Gains
        self.k_pos1 = rospy.get_param("~k_pos1", 1.0)
        self.k_pos2 = rospy.get_param("~k_pos2", 2.0)
        self.k_att1 = rospy.get_param("~k_att1", 1.0)
        self.k_att2 = rospy.get_param("~k_att2", 2.0)
        rospy.loginfo("Gains: Kp1=%.2f, Kp2=%.2f, Ka1=%.2f, Ka2=%.2f",
                      self.k_pos1, self.k_pos2, self.k_att1, self.k_att2)

        # Safety limits
        self.max_tilt_angle = rospy.get_param("~max_tilt_angle_deg", 30.0) * math.pi / 180.0 # rad
        self.min_thrust_factor = rospy.get_param("~min_thrust_factor", 0.1)

        # Gravity compensation adjustment
        self.gravity_compensation_factor = rospy.get_param("~gravity_comp_factor", 1.0)

        # Precompute Mixer Matrix Inverse
        A = np.array([
            [self.k_f, self.k_f, self.k_f, self.k_f],
            [-0.22 * self.k_f, 0.20 * self.k_f, 0.22 * self.k_f, -0.20 * self.k_f],
            [0.13 * self.k_f, -0.13 * self.k_f, 0.13 * self.k_f, -0.13 * self.k_f],
            [-self.k_m, -self.k_m, self.k_m, self.k_m]
        ])
        try:
            self.inv_A = np.linalg.inv(A)
            rospy.loginfo("Original Mixer matrix computed successfully.")
        except np.linalg.LinAlgError:
            rospy.logerr("Original Mixer matrix A is singular. Check parameters.")
            self.inv_A = None

        # --- Publishers ---
        self.command_pub = rospy.Publisher(self.namespace + '/command/motor_speed', Actuators, queue_size=1)
        self.U_pub = rospy.Publisher('~control/U', Float64MultiArray, queue_size=1)
        self.errors_pos_pub = rospy.Publisher('~error/position', Point, queue_size=1)
        self.errors_vel_pub = rospy.Publisher('~error/velocity', Point, queue_size=1)
        self.errors_att_pub = rospy.Publisher('~error/attitude_deg', Point, queue_size=1)
        self.errors_rate_pub = rospy.Publisher('~error/rates_deg_s', Point, queue_size=1)
        self.desired_att_pub = rospy.Publisher('~control/desired_attitude_deg', Point, queue_size=1)
        self.virtual_inputs_pub = rospy.Publisher('~control/virtual_inputs', Point, queue_size=1)
        self.active_target_pub = rospy.Publisher('~control/active_target', Point, queue_size=1)

        # --- Subscriber ---
        self.latest_state = None
        if self.use_model_states:
            from gazebo_msgs.msg import ModelStates
            self.state_sub = rospy.Subscriber('/gazebo/model_states', ModelStates, self.model_state_callback, queue_size=1)
            rospy.loginfo("Subscribing to /gazebo/model_states")
        else:
            self.state_sub = rospy.Subscriber(self.namespace + '/ground_truth/odometry', Odometry, self.odometry_callback, queue_size=1)
            rospy.loginfo("Subscribing to %s/ground_truth/odometry", self.namespace)

        # --- Control Loop Timer ---
        control_rate = rospy.get_param("~control_rate", 100.0) # Hz
        self.control_timer = rospy.Timer(rospy.Duration(1.0 / control_rate), self.control_loop)

        rospy.loginfo("CLF Sequential Controller Initialized for namespace: %s", self.namespace)


    # --- Callbacks and Helper Functions (remain unchanged) ---
    def odometry_callback(self, msg):
        self.latest_state = msg

    def model_state_callback(self, msg):
        try:
            try:
                ind = msg.name.index(self.namespace)
            except ValueError:
                 try:
                      ind = msg.name.index(self.namespace + '/')
                 except ValueError:
                      self.latest_state = None
                      rospy.logwarn_throttle(5.0, "Model '%s' or '%s/' not found in /gazebo/model_states. Available models: %s",
                                             self.namespace, self.namespace, ", ".join(msg.name))
                      return

            odom_like = Odometry()
            odom_like.header.stamp = rospy.Time.now()
            odom_like.header.frame_id = "world"
            odom_like.child_frame_id = self.namespace + "/base_link"
            odom_like.pose.pose = msg.pose[ind]
            odom_like.twist.twist = msg.twist[ind]
            self.latest_state = odom_like
        except Exception as e:
            rospy.logerr("Error in model_state_callback: %s", e)
            self.latest_state = None

    def get_rotation_matrix(self, phi, theta, psi):
        cphi, sphi = math.cos(phi), math.sin(phi)
        cth, sth = math.cos(theta), math.sin(theta)
        cpsi, spsi = math.cos(psi), math.sin(psi)
        R = np.array([
            [cth*cpsi, sphi*sth*cpsi - cphi*spsi, cphi*sth*cpsi + sphi*spsi],
            [cth*spsi, sphi*sth*spsi + cphi*cpsi, cphi*sth*spsi - sphi*cpsi],
            [-sth,     sphi*cth,                 cphi*cth]
        ])
        return R

    # --- Control Loop ---
    def control_loop(self, event):
        if self.latest_state is None or self.inv_A is None:
            rospy.logwarn_throttle(5.0, "State not received or mixer invalid. Skipping control loop.")
            self.publish_zero_command()
            return

        # --- State Extraction ---
        pos = self.latest_state.pose.pose.position
        q_orientation = self.latest_state.pose.pose.orientation
        vel_body = self.latest_state.twist.twist.linear
        omega_body = self.latest_state.twist.twist.angular

        x, y, z = pos.x, pos.y, pos.z
        (phi, theta, psi) = euler_from_quaternion([q_orientation.x, q_orientation.y, q_orientation.z, q_orientation.w])
        p, q, r = omega_body.x, omega_body.y, omega_body.z

        # Transform Body Velocity to World Velocity
        R_body_to_world = self.get_rotation_matrix(phi, theta, psi)
        vel_world_vec = np.dot(R_body_to_world, [vel_body.x, vel_body.y, vel_body.z])
        world_vx, world_vy, world_vz = vel_world_vec[0], vel_world_vec[1], vel_world_vec[2]

        # --- Calculate Errors w.r.t ACTIVE Target ---
        ex1 = x - self.active_target_x
        ey1 = y - self.active_target_y
        ez1 = z - self.active_target_z
        ex2 = world_vx - 0.0 # Desired velocity is 0
        ey2 = world_vy - 0.0
        ez2 = world_vz - 0.0

        if self.current_target_state == "NAVIGATING_TO_A":
            # Check if absolute error on ALL axes is below the threshold
            if (abs(ex1) < self.waypoint_switch_threshold and
                abs(ey1) < self.waypoint_switch_threshold and
                abs(ez1) < self.waypoint_switch_threshold):

                rospy.loginfo("Near Target A: All axis errors (|x|=%.3f, |y|=%.3f, |z|=%.3f) < threshold %.3f m. Switching to Target B.",
                              abs(ex1), abs(ey1), abs(ez1), self.waypoint_switch_threshold) # Updated log message

                self.current_target_state = "NAVIGATING_TO_B"
                self.active_target_x = self.target_B_x
                self.active_target_y = self.target_B_y
                self.active_target_z = self.target_B_z
                # Re-calculate errors instantly for the new target B
                ex1 = x - self.active_target_x
                ey1 = y - self.active_target_y
                ez1 = z - self.active_target_z
                # Velocity errors remain the same (target vel is 0)

        # Publish the currently active target
        self.active_target_pub.publish(Point(self.active_target_x, self.active_target_y, self.active_target_z))

        # --- Positional Control (U1, Uex, Uey) ---
        cos_phi, cos_theta = math.cos(phi), math.cos(theta)
        thrust_orientation_factor = cos_phi * cos_theta
        if abs(thrust_orientation_factor) < self.min_thrust_factor:
             cap_sign = np.sign(thrust_orientation_factor) if thrust_orientation_factor != 0 else 1.0
             thrust_orientation_factor = cap_sign * self.min_thrust_factor
             rospy.logwarn_throttle(1.0, "Near-singular orientation, capping thrust factor to %.2f", thrust_orientation_factor)

        gravity_feedforward = self.gravity * self.gravity_compensation_factor
        U1 = (self.mass / thrust_orientation_factor) * (-self.k_pos1 * ez1 + self.z_dd - self.k_pos2 * ez2 + gravity_feedforward)
        U1 = max(0.0, U1)

        if U1 < 1e-6:
             Uex, Uey = 0.0, 0.0
        else:
             Uex = (self.mass / U1) * (-self.k_pos1 * ex1 + self.x_dd - self.k_pos2 * ex2)
             Uey = (self.mass / U1) * (-self.k_pos1 * ey1 + self.y_dd - self.k_pos2 * ey2)

        # --- Calculate Desired Attitude (phi_d, theta_d) ---
        psi_d = self.target_yaw
        sin_psi_d, cos_psi_d = math.sin(psi_d), math.cos(psi_d)

        phi_d_arg = Uex * sin_psi_d - Uey * cos_psi_d
        phi_d_arg_clipped = np.clip(phi_d_arg, -1.0, 1.0)
        phi_d = math.asin(phi_d_arg_clipped)

        cos_phi_d = math.cos(phi_d)
        if abs(cos_phi_d) < 1e-6:
            theta_d = 0.0
        else:
            theta_d_arg = (Uex * cos_psi_d + Uey * sin_psi_d) / cos_phi_d
            theta_d_arg_clipped = np.clip(theta_d_arg, -1.0, 1.0)
            theta_d = math.asin(theta_d_arg_clipped)

        phi_d_sat = np.clip(phi_d, -self.max_tilt_angle, self.max_tilt_angle)
        theta_d_sat = np.clip(theta_d, -self.max_tilt_angle, self.max_tilt_angle)

        # --- Attitude Errors ---
        e_phi1 = phi - phi_d_sat
        e_theta1 = theta - theta_d_sat
        e_psi1 = psi - psi_d
        e_psi1 = (e_psi1 + np.pi) % (2 * np.pi) - np.pi # Normalize yaw error

        e_phi2 = p - self.p_d
        e_theta2 = q - self.q_d
        e_psi2 = r - self.r_d

        # --- Attitude Control (U2, U3, U4) ---
        roll_coupling = q * r * (self.I_y - self.I_z)
        pitch_coupling = p * r * (self.I_z - self.I_x)
        yaw_coupling = p * q * (self.I_x - self.I_y)

        U2 = self.I_x * (-self.k_att1 * e_phi1 + self.phi_dd - self.k_att2 * e_phi2) - roll_coupling
        U3 = self.I_y * (self.k_att1 * e_theta1 - self.theta_dd + self.k_att2 * e_theta2) - pitch_coupling # Sign based on previous script version
        U4 = self.I_z * (-self.k_att1 * e_psi1 + self.psi_dd - self.k_att2 * e_psi2) - yaw_coupling

        # --- Logging ---
        if ENABLE_DETAILED_LOGGING:
             rospy.loginfo_throttle(LOG_THROTTLE_SECS,
                  "\n--- CLF DEBUG (State: %s) ---\n"
                  "Active Target: x=%.2f, y=%.2f, z=%.2f\n"
                  "State (World): x=%.2f, y=%.2f, z=%.2f | vx=%.2f, vy=%.2f, vz=%.2f\n"
                  "Att (deg):   phi=%.1f, th=%.1f, psi=%.1f | p=%.1f, q=%.1f, r=%.1f\n"
                  "Errors Pos:  ex=%.2f, ey=%.2f, ez=%.2f\n"
                  "Errors Vel:  ex2=%.2f, ey2=%.2f, ez2=%.2f\n"
                  "Errors Att(d): ephi=%.1f, eth=%.1f, epsi=%.1f\n"
                  "Errors Rate(d):ep=%.1f, eq=%.1f, er=%.1f\n"
                  "Desired Att(d): phid=%.1f, thd=%.1f (Sat: phid=%.1f, thd=%.1f)\n"
                  "Virtual In:  Uex=%.3f, Uey=%.3f\n"
                  "Control Out: U1=%.2f, U2=%.3f, U3=%.3f, U4=%.3f\n"
                  "-----------------",
                  self.current_target_state,
                  self.active_target_x, self.active_target_y, self.active_target_z,
                  x, y, z, world_vx, world_vy, world_vz,
                  math.degrees(phi), math.degrees(theta), math.degrees(psi), math.degrees(p), math.degrees(q), math.degrees(r),
                  ex1, ey1, ez1,
                  ex2, ey2, ez2,
                  math.degrees(e_phi1), math.degrees(e_theta1), math.degrees(e_psi1),
                  math.degrees(e_phi2), math.degrees(e_theta2), math.degrees(e_psi2),
                  math.degrees(phi_d), math.degrees(theta_d), math.degrees(phi_d_sat), math.degrees(theta_d_sat),
                  Uex, Uey,
                  U1, U2, U3, U4
             )

        # --- Mixer Logic ---
        U = np.array([U1, U2, U3, U4])
        try:
            omega_sq = np.dot(self.inv_A, U)
        except (ValueError, TypeError) as e:
             rospy.logerr_throttle(1.0, "Mixer dot product failed. U shape: %s, inv_A type: %s. Error: %s", U.shape, type(self.inv_A), e)
             self.publish_zero_command()
             return

        # --- Saturation and Command Publishing ---
        omega_sq = np.maximum(omega_sq, 0.0)
        omega = np.sqrt(omega_sq)
        omega = np.minimum(omega, self.max_rot_velocity)

        actuator_msg = Actuators()
        actuator_msg.header.stamp = rospy.Time.now()
        actuator_msg.angular_velocities = omega.tolist() if isinstance(omega, np.ndarray) else [0.0] * self.motor_count
        self.command_pub.publish(actuator_msg)

        # --- Publish Debug Info ---
        u_msg = Float64MultiArray(data=U.tolist())
        self.U_pub.publish(u_msg)
        self.errors_pos_pub.publish(Point(ex1, ey1, ez1))
        self.errors_vel_pub.publish(Point(ex2, ey2, ez2))
        self.errors_att_pub.publish(Point(math.degrees(e_phi1), math.degrees(e_theta1), math.degrees(e_psi1)))
        self.errors_rate_pub.publish(Point(math.degrees(e_phi2), math.degrees(e_theta2), math.degrees(e_psi2)))
        self.desired_att_pub.publish(Point(math.degrees(phi_d_sat), math.degrees(theta_d_sat), math.degrees(psi_d)))
        self.virtual_inputs_pub.publish(Point(Uex, Uey, 0.0))


    # --- publish_zero_command and shutdown_hook (remain unchanged) ---
    def publish_zero_command(self):
        actuator_msg = Actuators()
        actuator_msg.header.stamp = rospy.Time.now()
        actuator_msg.angular_velocities = [0.0] * self.motor_count
        if hasattr(self, 'command_pub') and self.command_pub:
            self.command_pub.publish(actuator_msg)
        else:
             rospy.logwarn_throttle(5.0, "Cannot publish zero command, command_pub not initialized.")

    def shutdown_hook(self):
        rospy.loginfo("CLF Controller shutting down. Sending zero motor speeds.")
        for _ in range(5):
             self.publish_zero_command()
             try:
                 rospy.sleep(0.01)
             except rospy.ROSInterruptException:
                 rospy.loginfo("Shutdown interrupted sleep.")
                 break


if __name__ == '__main__':
    try:
        rospy.init_node("clf_iris_sequential_position_controller", anonymous=True)
        controller = ClfIrisController()
        rospy.on_shutdown(controller.shutdown_hook)
        rospy.spin()
    except rospy.ROSInterruptException:
        rospy.loginfo("ROS interrupt received. Shutting down.")
    except Exception as e:
        rospy.logfatal("CLF Controller encountered a critical error: %s", e)
        import traceback
        traceback.print_exc()
    finally:
        if 'controller' in locals() and controller:
             rospy.loginfo("Attempting final zero command in finally block.")
             controller.publish_zero_command()