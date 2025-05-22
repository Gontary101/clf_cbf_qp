#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import division
import numpy as np
import rospy
from nav_msgs.msg import Odometry
from mav_msgs.msg import Actuators
from std_msgs.msg import Float64MultiArray

import utils.dynamics_utils2 as dyn
from utils.dynamics_utils2 import pget
# Adjusted imports based on the subtask
from utils.obstacle_parser import parse_obstacles, GazeboObstacleProcessor
from components.trajectory_module_straight import TrajectoryModuleStraight
from components import (
    StateEstimator, FlightStateManager, ControlAllocator, SafetyManager, ActuationHandler, TelemetryPublisher
)
# State and TrajectoryModule are not directly used here anymore, replaced by TrajectoryModuleStraight
# from components import State, TrajectoryModule 
from gazebo_msgs.msg import ModelStates # Moved here for clarity if use_gz block changes
import ast # For parsing all_drone_namespaces


class Controller(object):
    """
    Refactored controller for Hummingbird drone executing straight-line trajectories
    with ZCBF-based obstacle avoidance and online time scaling.

    This controller integrates various components for state estimation,
    flight state management, trajectory tracking, obstacle avoidance (static and dynamic),
    and control actuation. It is designed to be configurable via ROS parameters.
    """
    def __init__(self):
        self.ns = pget("namespace", "hummingbird")
        self.use_gz = pget("use_model_states", False)

        self.model = dyn.DroneModel()

        # Trajectory Module
        self.trajectory_module = TrajectoryModuleStraight()

        # State Estimator
        self.state_estimator = StateEstimator()

        # Flight State Manager
        initial_takeoff_x = pget("takeoff_x", self.trajectory_module.get_p0()[0])
        initial_takeoff_y = pget("takeoff_y", self.trajectory_module.get_p0()[1])
        initial_takeoff_z = pget("takeoff_z", self.trajectory_module.get_p0()[2])
        
        g_take = [pget("k_take{}".format(n), dflt) for n, dflt in zip(("pos1", "pos2", "att1", "att2"), (0.22, 0.8, 2.05, 4.1))]
        g_traj = [pget("k_traj{}".format(n), dflt) for n, dflt in zip(("pos1", "pos2", "att1", "att2"), (0.75, 4.1, 16.00, 32.0))]
        
        self.flight_state_manager = FlightStateManager(initial_takeoff_x, initial_takeoff_y, initial_takeoff_z, g_take, g_traj, self.trajectory_module)

        # Control Allocator
        self.control_allocator = ControlAllocator(self.model)

        # Obstacle Parser Setup
        self.static_obstacles = parse_obstacles(lambda param_name, default_val: pget(param_name, default_val))
        rospy.loginfo_once("[{}] Loaded {} static obstacles.".format(self.ns, self.static_obstacles.shape[0]))

        # Setup for multi-drone awareness: list of all drone namespaces.
        # This is used by GazeboObstacleProcessor to identify other drones as dynamic obstacles.
        try:
            all_ns_str = pget("all_drone_namespaces", "[]")
            self.all_drone_namespaces = ast.literal_eval(all_ns_str)
            if not isinstance(self.all_drone_namespaces, list):
                raise ValueError("all_drone_namespaces is not a list")
            rospy.loginfo("[{}] Aware of drones: {}".format(self.ns, self.all_drone_namespaces))
        except (ValueError, SyntaxError) as e:
            rospy.logerr("[{}] Invalid all_drone_namespaces parameter '{}': {}".format(self.ns, all_ns_str, e))
            rospy.signal_shutdown("Configuration Error")
            return # Important to stop initialization if this fails
        
        self.gazebo_obstacle_processor = GazeboObstacleProcessor(
            ns=self.ns, 
            all_drone_namespaces=self.all_drone_namespaces, 
            pget_func=lambda param_name, default_val: pget(param_name, default_val)
        )
        self.combined_obstacles = np.empty((0, 10), dtype=float)

        # Safety Manager
        cbf_params = dict(
            beta=pget("zcbf_beta", 1.5),
            a1=pget("zcbf_a1", 1.5),
            a2=pget("zcbf_a2", 1.6),
            gamma=pget("zcbf_gamma", 8.4),
            kappa=pget("zcbf_kappa", 0.8),
            order_a=pget("zcbf_order_a", 0)
        )
        self.cbf_slack_pub = rospy.Publisher("~cbf/slack", Float64MultiArray, queue_size=1)
        self.safety_manager = SafetyManager(self.model, self.static_obstacles, cbf_params, self.cbf_slack_pub)

        # Actuation Handler
        motor_cmd_topic = pget("motor_command_topic", "command/motor_speed")
        self.cmd_pub = rospy.Publisher(motor_cmd_topic, Actuators, queue_size=1)
        self.actuation_handler = ActuationHandler(self.model, self.cmd_pub)

        # Telemetry Publisher
        self.telemetry_publisher = TelemetryPublisher()

        # Time scaling and phase tracking attributes
        self.prev_phase_name = None
        self.t_traj = 0.0  # Current scaled trajectory time
        self.t_last = rospy.Time.now().to_sec() # Used for main loop dt, also for initial t_last_scale_update
        self.t_last_scale_update = self.t_last # Initialize with current time
        
        # Logging attributes
        self.LOG_T = pget("loop_log_period", 0.5)
        self.DBG = pget("debug_logging_enabled", True) # Assuming True enables detailed logs if used

        self.last_odom_msg = None

        # Subscribers
        if self.use_gz:
            model_states_topic = pget("model_states_topic", "/gazebo/model_states")
            self.sub = rospy.Subscriber(model_states_topic,
                                        ModelStates, self.cb_model_combined, # Changed to combined callback
                                        queue_size=5, buff_size=2**24)
        else:
            odometry_topic_suffix = pget("odometry_topic_suffix", "/ground_truth/odometry")
            self.sub = rospy.Subscriber(self.ns + odometry_topic_suffix,
                                        Odometry, self.cb_odom, queue_size=10)
        
        # Timer
        self.control_rate = pget("control_rate", 200.0) # Store control_rate as instance variable
        self.timer = rospy.Timer(rospy.Duration(1.0 / self.control_rate),
                                 self.loop, reset=True)

        rospy.loginfo("[{}] Refactored controller initialized with control rate {}Hz and log period {}s.".format(self.ns, self.control_rate, self.LOG_T))
        rospy.on_shutdown(self.shutdown)

    def cb_odom(self, msg):
        self.last_odom_msg = msg

    def cb_model_combined(self, msg):
        """
        Callback for /gazebo/model_states. Updates ego drone odometry from the message
        and processes other models for obstacle avoidance using GazeboObstacleProcessor.
        """
        # First, handle ego drone state update (copied from main_helix.py's cb_model)
        try:
            idx = msg.name.index(self.ns)
        except ValueError:
            try: # Try with a trailing slash if the namespace was registered like that in Gazebo
                idx = msg.name.index(self.ns + "/")
            except ValueError:
                rospy.logwarn_throttle(5.0, "[{}] Own state not found in /gazebo/model_states".format(self.ns))
                return
        
        o = Odometry()
        o.header.stamp = rospy.Time.now() # Use current time for fresher state
        o.header.frame_id = "world" 
        o.child_frame_id = self.ns + "/base_link"
        o.pose.pose = msg.pose[idx]
        o.twist.twist = msg.twist[idx]
        self.last_odom_msg = o

        # Then, process for obstacles
        if self.gazebo_obstacle_processor:
            self.gazebo_obstacle_processor.process_model_states_msg(msg)

    # cb_model is removed as its functionality for ego drone is now in cb_model_combined
    # and obstacle processing is handled by gazebo_obstacle_processor.

    def loop(self, event):
        """
        Main control loop, executed at `control_rate`.
        Handles state estimation, obstacle processing, flight logic, online time scaling,
        control computation (nominal and safety-filtered), actuation, and telemetry.
        """
        # a. Initial checks (odom, time)
        if self.last_odom_msg is None:
            rospy.logwarn_throttle(1.0, "[{}] No odometry message received yet.".format(self.ns))
            return
        
        now = rospy.Time.now()
        now_sec = now.to_sec()

        # b. State Estimation
        current_kinematics = self.state_estimator.process_odometry(self.last_odom_msg, self.use_gz)
        current_pos_numpy = current_kinematics["p_vec"]

        # c. Obstacle Aggregation
        # Get obstacles from Gazebo processor (dynamic environment + other drones)
        if self.use_gz and hasattr(self, 'gazebo_obstacle_processor'): # Ensure processor exists
            env_and_other_drone_obs = self.gazebo_obstacle_processor.get_combined_obstacles(current_pos_numpy)
        else:
            env_and_other_drone_obs = np.empty((0, 10), dtype=float)
        
        # Combine with static obstacles
        parts = [obs_array for obs_array in (self.static_obstacles, env_and_other_drone_obs) if obs_array.size > 0]
        if parts:
            self.combined_obstacles = np.vstack(parts)
            # Optional: apply a final sort by distance and limit
            max_active_total = int(pget("zcbf_max_active_spheres", 5)) 
            if self.combined_obstacles.shape[0] > max_active_total:
                dists_sq = np.sum((self.combined_obstacles[:, :3] - current_pos_numpy)**2, axis=1)
                indices = np.argsort(dists_sq)[:max_active_total]
                self.combined_obstacles = self.combined_obstacles[indices]
        else:
            self.combined_obstacles = np.empty((0, 10), dtype=float)
        
        # Update safety manager's view of obstacles
        if hasattr(self.safety_manager, 'zcbf'):
             self.safety_manager.zcbf.obs = self.combined_obstacles
        else:
            # This case implies SafetyManager might not be fully ZCBF based, or zcbf object is nested.
            # For this refactoring, we assume zcbf is a direct attribute or needs a dedicated update method.
            # If safety_manager.update_obstacles(self.combined_obstacles) is the API, use that.
            # Based on main_helix.py, direct assignment is common.
            rospy.logwarn_throttle(5.0, "[{}] ZCBF object not found directly on safety_manager. Obstacles not updated in ZCBF.".format(self.ns))


        # d. Flight State Management
        refs_fsm, current_gains = self.flight_state_manager.update_state(now, current_kinematics)
        current_phase_name = refs_fsm["state_name"]
        t_traj_fsm = refs_fsm["t_traj_secs"]

        # e. Online Time Scaling (if TRAJ)
        if current_phase_name == "TRAJ" and self.prev_phase_name != "TRAJ":
            self.t_traj = t_traj_fsm  # Reset/sync scaled time with FSM time on new entry
            self.t_last_scale_update = now_sec
            rospy.loginfo("[{}] TRAJ phase entered. Initial t_traj: {:.2f}s".format(self.ns, self.t_traj))

        if current_phase_name == "TRAJ":
            dt_real = now_sec - self.t_last_scale_update
            if dt_real < 0: # Should not happen if time moves forward
                dt_real = 0 

            # Get the previously desired position based on the current scaled trajectory time
            pos_d_prev_scaling, _, _, _, _ = self.trajectory_module.get_reference(self.t_traj)
            p_act = current_kinematics["p_vec"]
            # Calculate the tracking error: Euclidean distance between current and previously desired position
            tracking_error = np.linalg.norm(p_act - pos_d_prev_scaling)

            sigma = 1.0  # Default scaling factor: no slowdown
            k_e_time_scale = pget("time_scale_k", 0.5) # Gain for error-based slowdown
            
            # Calculate error-based slowdown: sigma decreases as tracking_error increases
            # This means the trajectory time t_traj advances slower if the drone lags.
            sigma_error = 1.0 / (1.0 + k_e_time_scale * tracking_error)
            sigma = min(sigma, sigma_error) # Apply the error-based slowdown

            # Obstacle influence on sigma:
            # If obstacles are close, the previously calculated error-based slowdown (sigma_error)
            # becomes more critical. The logic doesn't add a separate slowdown factor for obstacles,
            # but rather makes the existing error-based one more relevant if obstacles are within time_scale_dist.
            time_scale_dist = pget("time_scale_dist", 0.8) # Distance threshold to consider obstacles for time scaling
            if self.combined_obstacles.size > 0:
                centers = self.combined_obstacles[:, :3]
                radii = self.combined_obstacles[:, 9]  # Assuming radius is the 10th element (index 9)
                safety_radius = self.model.r_drone  # Using drone's physical radius
                
                # Calculate distances from drone's surface to obstacles' surfaces
                dist_to_obs_surface = np.linalg.norm(centers - p_act[None,:], axis=1) - (radii + safety_radius)
                
                # If any obstacle surface is within the time_scale_dist, the current sigma (already reflecting tracking error)
                # is maintained. This implies that the slowdown due to tracking error is considered sufficient
                # to handle situations where obstacles are nearby. No additional explicit slowdown factor for obstacles
                # is applied here beyond the error-based one.
                if np.any(dist_to_obs_surface < time_scale_dist):
                    # If a more aggressive slowdown specifically due to obstacles was desired,
                    # it could be implemented here by further reducing sigma, e.g.:
                    # sigma_obs_specific = 0.5 # or some other factor
                    # sigma = min(sigma, sigma_obs_specific)
                    pass # Current sigma (error-based) is used as is.

            # Advance the scaled trajectory time: dt_real is wall-clock time elapsed in this loop iteration.
            # If sigma is < 1, t_traj advances slower than wall-clock time.
            self.t_traj += sigma * dt_real
            self.t_last_scale_update = now_sec

            # Get trajectory references based on the new scaled trajectory time
            posd, vd, ad, yd, rd = self.trajectory_module.get_reference(self.t_traj)
            reference_signals = {"tgt": posd, "vd": vd, "ad": ad, "yd": yd, "rd": rd, 
                                 "t_traj_secs": self.t_traj, "state_name": current_phase_name, "sigma": sigma}
        else:
            reference_signals = refs_fsm
            reference_signals["sigma"] = 1.0 # No scaling outside TRAJ
            self.t_traj = refs_fsm["t_traj_secs"]  # Keep t_traj synced with FSM when not scaling
        
        self.prev_phase_name = current_phase_name
        self.t_last = now_sec # Update t_last for the next iteration's dt_real in TRAJ phase if entered immediately

        # f. Control Allocation (Nominal Control)
        clf_output_dict = self.control_allocator.compute_nominal_control(current_kinematics, reference_signals, current_gains)
        U_nominal_clf = clf_output_dict["U_nom"]

        # g. Safety Filtering
        # Ensure current_gains is a dict if filter_control expects one (it does, via nominal_control_object.gains)
        # current_gains from FSM is a list [kp,kd,ka,ko]. ControlAllocator uses a dict.
        # This was an oversight in previous steps, FSM provides list, ControlAllocator.compute_nominal_control takes list.
        # SafetyManager.filter_control takes gains_dict.
        # Let's assume safety_manager can handle the list or it's converted internally.
        # For now, passing current_gains as is. If it needs dict, it must be converted.
        # The original main_helix passes the list from FSM to safety_manager.filter_control.
        U_filtered, _ = self.safety_manager.filter_control(current_phase_name, U_nominal_clf, current_kinematics, current_gains, reference_signals)

        # h. Actuation
        _, w_sq_final = self.actuation_handler.generate_and_publish_motor_commands(U_filtered, now)

        # i. Telemetry
        self.telemetry_publisher.publish_telemetry(current_kinematics, current_phase_name, U_filtered, w_sq_final, clf_output_dict, reference_signals, U_nominal_clf)

        # j. Throttled Logging (Example)
        if self.DBG: # Check if debug logging is enabled
            # Calculate time passed since last log based on control rate and counter
            # This ensures logging happens approximately every self.LOG_T seconds
            if not hasattr(self, 'log_event_counter'):
                self.log_event_counter = 0
            
            self.log_event_counter += 1
            
            # Assuming control_rate is fetched correctly and available
            # control_rate = pget("control_rate", 500.0) # This is already fetched in __init__
            # To access it here, it should be self.control_rate or fetched again. Let's assume it's self.control_rate
            # For simplicity, let's ensure control_rate is an attribute or re-fetch. (Done: self.control_rate in __init__)
            
            if (self.log_event_counter * (1.0 / self.control_rate)) >= self.LOG_T:
                self.log_event_counter = 0 # Reset counter
                p = current_kinematics.get("p_vec", np.zeros(3))
                pd = reference_signals.get("tgt", np.zeros(3))
                sigma_val = reference_signals.get("sigma", 1.0)
                # Using self.LOG_T for the throttle period ensures it's configurable
                rospy.loginfo_throttle(self.LOG_T, 
                    "[%s] t_traj: %.2f (s=%.2f) | P: [%.2f,%.2f,%.2f] -> Pd: [%.2f,%.2f,%.2f] | Uf: [%.2f,%.2f,%.2f,%.2f]",
                    current_phase_name, self.t_traj, sigma_val,
                    p[0], p[1], p[2], pd[0], pd[1], pd[2],
                    U_filtered[0], U_filtered[1], U_filtered[2], U_filtered[3]
                )

    def shutdown(self):
        # TODO: Implement a more robust shutdown if necessary, 
        # for now, try to send stop commands like in main_helix.py
        # This might require self.actuation_handler to be initialized.
        # If self.actuation_handler is not available, this will error.
        # For now, keeping it simple.
        rospy.loginfo("[{}] Shutting down refactored controller. Sending stop commands.".format(self.ns))
        rospy.loginfo("[{}] Shutting down refactored controller. Sending stop commands.".format(self.ns))
        if hasattr(self, 'actuation_handler') and self.actuation_handler is not None:
            for _ in range(10): # Send multiple times to ensure it's received
                self.actuation_handler.send_single_stop_command()
                rospy.sleep(0.01)
        else:
            rospy.loginfo("[{}] Actuation handler not initialized, cannot send stop commands.".format(self.ns))


if __name__ == "__main__":
    rospy.init_node("clf_hummingbird_refactored_controller", anonymous=True)
    # Loginfo moved to after Controller initialization for cleaner startup log
    # rospy.loginfo("Starting clf_hummingbird_refactored_controller.") 
    try:
        c = Controller()
        rospy.loginfo("[{}] Refactored controller started successfully.".format(c.ns if hasattr(c, 'ns') else 'hummingbird'))
        rospy.spin()
    except rospy.ROSInterruptException:
        rospy.loginfo("ROS interrupt received. Shutting down.")
    except Exception as e: # Catch other init errors
        rospy.logfatal("Failed to initialize controller: {}".format(e))
        # No explicit rospy.signal_shutdown here as it might already be shutting down or error is fatal
