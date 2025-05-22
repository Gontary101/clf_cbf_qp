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
from utils.obstacle_parser import parse_obstacles, GazeboObstacleProcessor
from components.trajectory_module_straight import TrajectoryModuleStraight
from components import (
    StateEstimator, FlightStateManager, ControlAllocator, SafetyManager, ActuationHandler, TelemetryPublisher
)
from gazebo_msgs.msg import ModelStates
import ast


class Controller(object):
    def __init__(self):
        self.ns = pget("namespace", "hummingbird")
        self.use_gz = pget("use_model_states", False)

        self.model = dyn.DroneModel()

        self.trajectory_module = TrajectoryModuleStraight()

        self.state_estimator = StateEstimator()

        initial_takeoff_x = pget("takeoff_x", self.trajectory_module.get_p0()[0])
        initial_takeoff_y = pget("takeoff_y", self.trajectory_module.get_p0()[1])
        initial_takeoff_z = pget("takeoff_z", self.trajectory_module.get_p0()[2])
        
        g_take = [pget("k_take{}".format(n), dflt) for n, dflt in zip(("pos1", "pos2", "att1", "att2"), (0.22, 0.8, 2.05, 4.1))]
        g_traj = [pget("k_traj{}".format(n), dflt) for n, dflt in zip(("pos1", "pos2", "att1", "att2"), (0.75, 4.1, 16.00, 32.0))]
        
        self.flight_state_manager = FlightStateManager(initial_takeoff_x, initial_takeoff_y, initial_takeoff_z, g_take, g_traj, self.trajectory_module)

        self.control_allocator = ControlAllocator(self.model)

        self.static_obstacles = parse_obstacles(lambda param_name, default_val: pget(param_name, default_val))
        rospy.loginfo_once("[{}] Loaded {} static obstacles.".format(self.ns, self.static_obstacles.shape[0]))

        try:
            all_ns_str = pget("all_drone_namespaces", "[]")
            self.all_drone_namespaces = ast.literal_eval(all_ns_str)
            if not isinstance(self.all_drone_namespaces, list):
                raise ValueError("all_drone_namespaces is not a list")
            rospy.loginfo("[{}] Aware of drones: {}".format(self.ns, self.all_drone_namespaces))
        except (ValueError, SyntaxError) as e:
            rospy.logerr("[{}] Invalid all_drone_namespaces parameter '{}': {}".format(self.ns, all_ns_str, e))
            rospy.signal_shutdown("Configuration Error")
            return
        
        self.gazebo_obstacle_processor = GazeboObstacleProcessor(
            ns=self.ns, 
            all_drone_namespaces=self.all_drone_namespaces, 
            pget_func=lambda param_name, default_val: pget(param_name, default_val)
        )
        self.combined_obstacles = np.empty((0, 10), dtype=float)

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

        motor_cmd_topic = pget("motor_command_topic", "command/motor_speed")
        self.cmd_pub = rospy.Publisher(motor_cmd_topic, Actuators, queue_size=1)
        self.actuation_handler = ActuationHandler(self.model, self.cmd_pub)

        self.telemetry_publisher = TelemetryPublisher()

        self.prev_phase_name = None
        self.t_traj = 0.0
        self.t_last = rospy.Time.now().to_sec()
        self.t_last_scale_update = self.t_last
        
        self.LOG_T = pget("loop_log_period", 0.5)
        self.DBG = pget("debug_logging_enabled", True)

        self.last_odom_msg = None

        if self.use_gz:
            model_states_topic = pget("model_states_topic", "/gazebo/model_states")
            self.sub = rospy.Subscriber(model_states_topic,
                                        ModelStates, self.cb_model_combined,
                                        queue_size=5, buff_size=2**24)
        else:
            odometry_topic_suffix = pget("odometry_topic_suffix", "/ground_truth/odometry")
            self.sub = rospy.Subscriber(self.ns + odometry_topic_suffix,
                                        Odometry, self.cb_odom, queue_size=10)
        
        self.control_rate = pget("control_rate", 200.0)
        self.timer = rospy.Timer(rospy.Duration(1.0 / self.control_rate),
                                 self.loop, reset=True)

        rospy.loginfo("[{}] Refactored controller initialized with control rate {}Hz and log period {}s.".format(self.ns, self.control_rate, self.LOG_T))
        rospy.on_shutdown(self.shutdown)

    def cb_odom(self, msg):
        self.last_odom_msg = msg

    def cb_model_combined(self, msg):
        try:
            idx = msg.name.index(self.ns)
        except ValueError:
            try:
                idx = msg.name.index(self.ns + "/")
            except ValueError:
                rospy.logwarn_throttle(5.0, "[{}] Own state not found in /gazebo/model_states".format(self.ns))
                return
        
        o = Odometry()
        o.header.stamp = rospy.Time.now()
        o.header.frame_id = "world" 
        o.child_frame_id = self.ns + "/base_link"
        o.pose.pose = msg.pose[idx]
        o.twist.twist = msg.twist[idx]
        self.last_odom_msg = o

        if self.gazebo_obstacle_processor:
            self.gazebo_obstacle_processor.process_model_states_msg(msg)


    def loop(self, event):
        if self.last_odom_msg is None:
            rospy.logwarn_throttle(1.0, "[{}] No odometry message received yet.".format(self.ns))
            return
        
        now = rospy.Time.now()
        now_sec = now.to_sec()

        current_kinematics = self.state_estimator.process_odometry(self.last_odom_msg, self.use_gz)
        current_pos_numpy = current_kinematics["p_vec"]

        if self.use_gz and hasattr(self, 'gazebo_obstacle_processor'):
            env_and_other_drone_obs = self.gazebo_obstacle_processor.get_combined_obstacles(current_pos_numpy)
        else:
            env_and_other_drone_obs = np.empty((0, 10), dtype=float)
        
        parts = [obs_array for obs_array in (self.static_obstacles, env_and_other_drone_obs) if obs_array.size > 0]
        if parts:
            self.combined_obstacles = np.vstack(parts)
            max_active_total = int(pget("zcbf_max_active_spheres", 5)) 
            if self.combined_obstacles.shape[0] > max_active_total:
                dists_sq = np.sum((self.combined_obstacles[:, :3] - current_pos_numpy)**2, axis=1)
                indices = np.argsort(dists_sq)[:max_active_total]
                self.combined_obstacles = self.combined_obstacles[indices]
        else:
            self.combined_obstacles = np.empty((0, 10), dtype=float)
        
        if hasattr(self.safety_manager, 'zcbf'):
             self.safety_manager.zcbf.obs = self.combined_obstacles
        else:
            rospy.logwarn_throttle(5.0, "[{}] ZCBF object not found directly on safety_manager. Obstacles not updated in ZCBF.".format(self.ns))


        refs_fsm, current_gains = self.flight_state_manager.update_state(now, current_kinematics)
        current_phase_name = refs_fsm["state_name"]
        t_traj_fsm = refs_fsm["t_traj_secs"]

        if current_phase_name == "TRAJ" and self.prev_phase_name != "TRAJ":
            self.t_traj = t_traj_fsm
            self.t_last_scale_update = now_sec
            rospy.loginfo("[{}] TRAJ phase entered. Initial t_traj: {:.2f}s".format(self.ns, self.t_traj))

        if current_phase_name == "TRAJ":
            dt_real = now_sec - self.t_last_scale_update
            if dt_real < 0:
                dt_real = 0 

            pos_d_prev_scaling, _, _, _, _ = self.trajectory_module.get_reference(self.t_traj)
            p_act = current_kinematics["p_vec"]
            tracking_error = np.linalg.norm(p_act - pos_d_prev_scaling)

            # --- only slow down when *actually* near an obstacle
            sigma = 1.0
            time_scale_dist = pget("time_scale_dist", 0.8)
            k_e_time_scale = pget("time_scale_k", 0.5)

            if self.combined_obstacles.size > 0:
                centers = self.combined_obstacles[:, :3]
                radii   = self.combined_obstacles[:, 9]
                safety_r = self.model.r_drone
                d2surf = np.linalg.norm(centers - p_act[None,:], axis=1) - (radii + safety_r)

                # if *any* obstacle within the clearance, apply time‐scale
                if np.any(d2surf < time_scale_dist):
                    sigma = 1.0 / (1.0 + k_e_time_scale * tracking_error)
                    sigma = max(sigma, pget("time_scale_min", 0.4))

            self.t_traj += sigma * dt_real
            self.t_last_scale_update = now_sec

            posd, vd, ad, yd, rd = self.trajectory_module.get_reference(self.t_traj)
            reference_signals = {"tgt": posd, "vd": vd, "ad": ad, "yd": yd, "rd": rd, 
                                 "t_traj_secs": self.t_traj, "state_name": current_phase_name, "sigma": sigma}
        else:
            reference_signals = refs_fsm
            reference_signals["sigma"] = 1.0
            self.t_traj = refs_fsm["t_traj_secs"]
        
        self.prev_phase_name = current_phase_name
        self.t_last = now_sec

        clf_output_dict = self.control_allocator.compute_nominal_control(current_kinematics, reference_signals, current_gains)
        U_nominal_clf = clf_output_dict["U_nom"]

        U_filtered, _ = self.safety_manager.filter_control(current_phase_name, U_nominal_clf, current_kinematics, current_gains, reference_signals)

        _, w_sq_final = self.actuation_handler.generate_and_publish_motor_commands(U_filtered, now)

        self.telemetry_publisher.publish_telemetry(current_kinematics, current_phase_name, U_filtered, w_sq_final, clf_output_dict, reference_signals, U_nominal_clf)

        if self.DBG:
            if not hasattr(self, 'log_event_counter'):
                self.log_event_counter = 0
            
            self.log_event_counter += 1
            
            
            if (self.log_event_counter * (1.0 / self.control_rate)) >= self.LOG_T:
                self.log_event_counter = 0
                p = current_kinematics.get("p_vec", np.zeros(3))
                pd = reference_signals.get("tgt", np.zeros(3))
                sigma_val = reference_signals.get("sigma", 1.0)
                rospy.loginfo_throttle(self.LOG_T, 
                    "[%s] t_traj: %.2f (s=%.2f) | P: [%.2f,%.2f,%.2f] -> Pd: [%.2f,%.2f,%.2f] | Uf: [%.2f,%.2f,%.2f,%.2f]",
                    current_phase_name, self.t_traj, sigma_val,
                    p[0], p[1], p[2], pd[0], pd[1], pd[2],
                    U_filtered[0], U_filtered[1], U_filtered[2], U_filtered[3]
                )

    def shutdown(self):
        rospy.loginfo("[{}] Shutting down refactored controller. Sending stop commands.".format(self.ns))
        rospy.loginfo("[{}] Shutting down refactored controller. Sending stop commands.".format(self.ns))
        if hasattr(self, 'actuation_handler') and self.actuation_handler is not None:
            for _ in range(10):
                self.actuation_handler.send_single_stop_command()
                rospy.sleep(0.01)
        else:
            rospy.loginfo("[{}] Actuation handler not initialized, cannot send stop commands.".format(self.ns))


if __name__ == "__main__":
    rospy.init_node("clf_hummingbird_refactored_controller", anonymous=True)
    try:
        c = Controller()
        rospy.loginfo("[{}] Refactored controller started successfully.".format(c.ns if hasattr(c, 'ns') else 'hummingbird'))
        rospy.spin()
    except rospy.ROSInterruptException:
        rospy.loginfo("ROS interrupt received. Shutting down.")
    except Exception as e:
        rospy.logfatal("Failed to initialize controller: {}".format(e))
