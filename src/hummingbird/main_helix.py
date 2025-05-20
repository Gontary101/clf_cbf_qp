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
from utils.obstacle_parser import parse_obstacles

from components import (
    State, TrajectoryModule, StateEstimator, FlightStateManager,
    ControlAllocator, SafetyManager, ActuationHandler, TelemetryPublisher
)

LOG_T = 1.0
DBG = True

class Controller(object):
    def __init__(self):
        self.ns = pget("namespace", "hummingbird")
        self.use_gz = pget("use_model_states", False)

        self.model = dyn.DroneModel()

        self.trajectory_module = TrajectoryModule()
        
        initial_takeoff_x = pget("takeoff_x", self.trajectory_module.r0)
        initial_takeoff_y = pget("takeoff_y", 0.0)
        initial_takeoff_z = pget("takeoff_height", 3.0)

        def get_gains_tuple(tag, k1, k2, a1, a2):
            return [pget("%s%s" % (tag, n), dflt)
                    for n, dflt in zip(("pos1", "pos2", "att1", "att2"),
                                       (k1, k2, a1, a2))]
        
        g_take = get_gains_tuple("k_take", 0.22, 0.8, 2.05, 4.1)
        g_traj = get_gains_tuple("k_traj", 0.75, 4.1, 16.00, 32.0)

        self.flight_state_manager = FlightStateManager(
            initial_takeoff_x, initial_takeoff_y, initial_takeoff_z,
            g_take, g_traj, self.trajectory_module
        )
        
        self.state_estimator = StateEstimator()
        self.control_allocator = ControlAllocator(self.model)

        obstacles = parse_obstacles(pget)
        cbf_params = dict(
            beta=pget("zcbf_beta", 1.5),
            a1=pget("zcbf_a1", 1.5),
            a2=pget("zcbf_a2", 1.6),
            gamma=pget("zcbf_gamma", 8.4),
            kappa=pget("zcbf_kappa", 0.8),
            order_a=pget("zcbf_order_a", 0)
        )
        self.cbf_slack_pub = rospy.Publisher("~cbf/slack", Float64MultiArray, queue_size=1)
        self.safety_manager = SafetyManager(self.model, obstacles, cbf_params, self.cbf_slack_pub)
        
        self.cmd_pub = rospy.Publisher(self.ns + "/command/motor_speed", Actuators, queue_size=1)
        self.actuation_handler = ActuationHandler(self.model, self.cmd_pub)
        
        self.telemetry_publisher = TelemetryPublisher()

        self.last_odom_msg = None
        
        if self.use_gz:
            from gazebo_msgs.msg import ModelStates
            self.sub = rospy.Subscriber("/gazebo/model_states",
                                        ModelStates, self.cb_model,
                                        queue_size=5, buff_size=2**24)
        else:
            self.sub = rospy.Subscriber(self.ns + "/ground_truth/odometry",
                                        Odometry, self.cb_odom, queue_size=10)

        control_rate = pget("control_rate", 500.0)
        self.timer = rospy.Timer(rospy.Duration(1.0 / control_rate),
                                 self.loop, reset=True)
        rospy.on_shutdown(self.shutdown)

    def cb_odom(self, msg):
        self.last_odom_msg = msg

    def cb_model(self, msg):
        try:
            idx = msg.name.index(self.ns)
        except ValueError:
            try:
                idx = msg.name.index(self.ns + "/")
            except ValueError:
                return
        
        o = Odometry()
        o.header.stamp = rospy.Time.now()
        o.header.frame_id = "world"
        o.child_frame_id = self.ns + "/base_link"
        o.pose.pose, o.twist.twist = msg.pose[idx], msg.twist[idx]
        self.last_odom_msg = o

    def loop(self, event):
        if self.last_odom_msg is None:
            return
        
        current_time = rospy.Time.now()

        current_kinematics = self.state_estimator.process_odometry(self.last_odom_msg, self.use_gz)
        
        reference_signals, current_gains = self.flight_state_manager.update_state(
            current_time, current_kinematics
        )
        
        clf_output_dict = self.control_allocator.compute_nominal_control(
            current_kinematics, reference_signals, current_gains
        )
        U_nominal_clf = clf_output_dict["U_nom"]

        U_filtered, _ = self.safety_manager.filter_control(
            self.flight_state_manager.get_current_state_name(),
            U_nominal_clf,
            current_kinematics, 
            current_gains,
            reference_signals 
        )
        
        _, w_sq_final = self.actuation_handler.generate_and_publish_motor_commands(
            U_filtered, current_time
        )
        
        self.telemetry_publisher.publish_telemetry(
            current_kinematics,
            self.flight_state_manager.get_current_state_name(),
            U_filtered,
            w_sq_final,
            clf_output_dict,
            reference_signals,
            U_nominal_clf
        )

        if DBG:
            rospy.loginfo_throttle(LOG_T,
                "[%s] U=[%.2f %.2f %.2f %.2f]  |  Nom=[%.2f %.2f %.2f %.2f]",
                self.flight_state_manager.get_current_state_name(),
                U_filtered[0], U_filtered[1], U_filtered[2], U_filtered[3],
                U_nominal_clf[0], U_nominal_clf[1], U_nominal_clf[2], U_nominal_clf[3])

    def shutdown(self):
        for _ in range(10):
            self.actuation_handler.send_single_stop_command()
            rospy.sleep(0.01)

if __name__ == "__main__":
    rospy.init_node("clf_hummingbird_trajectory_controller", anonymous=True)
    try:
        Controller()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass