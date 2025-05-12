#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import division
import rospy
import csv
import os
import math
from datetime import datetime
from std_msgs.msg import String, Float64MultiArray
from geometry_msgs.msg import Point, Vector3
from nav_msgs.msg import Odometry # Assuming we might want raw odometry too

class CsvRecorder(object):
    def __init__(self):
        rospy.init_node('csv_data_recorder', anonymous=True)

        # --- Parameters ---
        self.output_dir = rospy.get_param('~output_dir', '/tmp/ros_logs')
        base_filename = rospy.get_param('~filename', 'flight_data')
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.filename = os.path.join(self.output_dir, '{}_{}.csv'.format(base_filename, timestamp))
        self.log_period = rospy.get_param('~log_period', 0.5) # Log every 0.5 seconds by default

        # --- Controller Node Name (adjust if different) ---
        # It's slightly better to get this from a param if possible,
        # but hardcoding based on your rostopic list is okay for now.
        self.controller_node_name = "/clf_iris_trajectory_controller" # <-- From rostopic list

        # --- State Storage ---
        self.latest_data = {
            'timestamp': 0.0,
            'control_state': '',
            'pos_x': float('nan'), 'pos_y': float('nan'), 'pos_z': float('nan'),
            'vel_x': float('nan'), 'vel_y': float('nan'), 'vel_z': float('nan'),
            'orient_phi_rad': float('nan'), 'orient_theta_rad': float('nan'), 'orient_psi_rad': float('nan'),
            'rate_x_rad_s': float('nan'), 'rate_y_rad_s': float('nan'), 'rate_z_rad_s': float('nan'),
            'U1_final': float('nan'), 'U2_final': float('nan'), 'U3_final': float('nan'), 'U4_final': float('nan'),
            'U1_nom': float('nan'), 'U2_nom': float('nan'), 'U3_nom': float('nan'), 'U4_nom': float('nan'),
            'omega_sq_1': float('nan'), 'omega_sq_2': float('nan'), 'omega_sq_3': float('nan'), 'omega_sq_4': float('nan'),
            'err_pos_x': float('nan'), 'err_pos_y': float('nan'), 'err_pos_z': float('nan'),
            'err_vel_x': float('nan'), 'err_vel_y': float('nan'), 'err_vel_z': float('nan'),
            'err_att_phi_deg': float('nan'), 'err_att_theta_deg': float('nan'), 'err_att_psi_deg': float('nan'),
            'err_rate_x_deg_s': float('nan'), 'err_rate_y_deg_s': float('nan'), 'err_rate_z_deg_s': float('nan'),
            'des_pos_x': float('nan'), 'des_pos_y': float('nan'), 'des_pos_z': float('nan'),
            'des_vel_x': float('nan'), 'des_vel_y': float('nan'), 'des_vel_z': float('nan'),
            'des_acc_x': float('nan'), 'des_acc_y': float('nan'), 'des_acc_z': float('nan'),
            'des_att_phi_deg': float('nan'), 'des_att_theta_deg': float('nan'), 'des_att_psi_deg': float('nan'),
            'virt_inp_Uex': float('nan'), 'virt_inp_Uey': float('nan'),
            'cbf_slack_0': float('nan') # Example for first slack variable
        }
        self.csv_headers = self.latest_data.keys()
        self.csv_headers.sort() # Ensure consistent column order

        # --- File Handling ---
        try:
            if not os.path.exists(self.output_dir):
                os.makedirs(self.output_dir)
            rospy.loginfo('Recording data to: {}'.format(self.filename))
            self.csv_file = open(self.filename, 'wb') # Use 'wb' for python 2.7 csv
            self.csv_writer = csv.writer(self.csv_file)
            self.csv_writer.writerow(self.csv_headers)
            rospy.loginfo('Logging data every {} seconds'.format(self.log_period))
        except IOError as e:
            rospy.logerr('Failed to open CSV file: {}'.format(e))
            rospy.signal_shutdown('File error')
            return

        # --- Subscribers (Using Absolute Paths) ---
        # Construct the full topic path using the controller's node name
        def get_topic(sub_topic):
            return self.controller_node_name + "/" + sub_topic

        rospy.Subscriber(get_topic("control/state"), String, self._control_state_cb)
        rospy.Subscriber(get_topic("state/position"), Point, self._state_pos_cb)
        rospy.Subscriber(get_topic("state/velocity_world"), Vector3, self._state_vel_cb)
        rospy.Subscriber(get_topic("state/orientation_eul"), Point, self._state_orient_cb)
        rospy.Subscriber(get_topic("state/rates_body"), Vector3, self._state_rates_cb)
        rospy.Subscriber(get_topic("control/U"), Float64MultiArray, self._control_U_cb)
        rospy.Subscriber(get_topic("control/U_nominal"), Float64MultiArray, self._control_U_nom_cb)
        rospy.Subscriber(get_topic("control/omega_sq"), Float64MultiArray, self._control_omega_sq_cb)
        rospy.Subscriber(get_topic("error/position"), Point, self._error_pos_cb)
        rospy.Subscriber(get_topic("error/velocity"), Vector3, self._error_vel_cb)
        rospy.Subscriber(get_topic("error/attitude_deg"), Point, self._error_att_cb)
        rospy.Subscriber(get_topic("error/rates_deg_s"), Vector3, self._error_rates_cb)
        rospy.Subscriber(get_topic("control/desired_position"), Point, self._des_pos_cb)
        rospy.Subscriber(get_topic("control/desired_velocity"), Vector3, self._des_vel_cb)
        rospy.Subscriber(get_topic("control/desired_acceleration"), Vector3, self._des_acc_cb)
        rospy.Subscriber(get_topic("control/desired_attitude_deg"), Point, self._des_att_cb)
        rospy.Subscriber(get_topic("control/virtual_inputs"), Point, self._virt_inp_cb)
        rospy.Subscriber(get_topic("cbf/slack"), Float64MultiArray, self._cbf_slack_cb) # Note: CBF path might be different

        # --- Timer for logging ---
        self.log_timer = rospy.Timer(rospy.Duration(self.log_period), self._log_data_callback)

        # --- Shutdown Hook ---
        rospy.on_shutdown(self.shutdown)

    def _update_data(self, key, value):
        """Safely update the latest data dictionary."""
        if key in self.latest_data:
            self.latest_data[key] = value

    # --- Callback Functions (No changes needed below this line) ---
    def _control_state_cb(self, msg):
        self._update_data('control_state', msg.data)

    def _state_pos_cb(self, msg):
        self._update_data('pos_x', msg.x)
        self._update_data('pos_y', msg.y)
        self._update_data('pos_z', msg.z)

    def _state_vel_cb(self, msg):
        self._update_data('vel_x', msg.x)
        self._update_data('vel_y', msg.y)
        self._update_data('vel_z', msg.z)

    def _state_orient_cb(self, msg):
        self._update_data('orient_phi_rad', msg.x)
        self._update_data('orient_theta_rad', msg.y)
        self._update_data('orient_psi_rad', msg.z)

    def _state_rates_cb(self, msg):
        self._update_data('rate_x_rad_s', msg.x)
        self._update_data('rate_y_rad_s', msg.y)
        self._update_data('rate_z_rad_s', msg.z)

    def _control_U_cb(self, msg):
        if len(msg.data) >= 4:
            self._update_data('U1_final', msg.data[0])
            self._update_data('U2_final', msg.data[1])
            self._update_data('U3_final', msg.data[2])
            self._update_data('U4_final', msg.data[3])

    def _control_U_nom_cb(self, msg):
        if len(msg.data) >= 4:
            self._update_data('U1_nom', msg.data[0])
            self._update_data('U2_nom', msg.data[1])
            self._update_data('U3_nom', msg.data[2])
            self._update_data('U4_nom', msg.data[3])

    def _control_omega_sq_cb(self, msg):
         if len(msg.data) >= 4:
            self._update_data('omega_sq_1', msg.data[0])
            self._update_data('omega_sq_2', msg.data[1])
            self._update_data('omega_sq_3', msg.data[2])
            self._update_data('omega_sq_4', msg.data[3])

    def _error_pos_cb(self, msg):
        self._update_data('err_pos_x', msg.x)
        self._update_data('err_pos_y', msg.y)
        self._update_data('err_pos_z', msg.z)

    def _error_vel_cb(self, msg):
        self._update_data('err_vel_x', msg.x)
        self._update_data('err_vel_y', msg.y)
        self._update_data('err_vel_z', msg.z)

    def _error_att_cb(self, msg):
        self._update_data('err_att_phi_deg', msg.x)
        self._update_data('err_att_theta_deg', msg.y)
        self._update_data('err_att_psi_deg', msg.z)

    def _error_rates_cb(self, msg):
        self._update_data('err_rate_x_deg_s', msg.x)
        self._update_data('err_rate_y_deg_s', msg.y)
        self._update_data('err_rate_z_deg_s', msg.z)

    def _des_pos_cb(self, msg):
        self._update_data('des_pos_x', msg.x)
        self._update_data('des_pos_y', msg.y)
        self._update_data('des_pos_z', msg.z)

    def _des_vel_cb(self, msg):
        self._update_data('des_vel_x', msg.x)
        self._update_data('des_vel_y', msg.y)
        self._update_data('des_vel_z', msg.z)

    def _des_acc_cb(self, msg):
        self._update_data('des_acc_x', msg.x)
        self._update_data('des_acc_y', msg.y)
        self._update_data('des_acc_z', msg.z)

    def _des_att_cb(self, msg):
        self._update_data('des_att_phi_deg', msg.x)
        self._update_data('des_att_theta_deg', msg.y)
        self._update_data('des_att_psi_deg', msg.z)

    def _virt_inp_cb(self, msg):
        self._update_data('virt_inp_Uex', msg.x)
        self._update_data('virt_inp_Uey', msg.y)

    def _cbf_slack_cb(self, msg):
        if len(msg.data) > 0:
            self._update_data('cbf_slack_0', msg.data[0])

    def _log_data_callback(self, event=None):
        """Called periodically by the timer to write data."""
        self._write_csv_row()

    def _write_csv_row(self):
        """Gathers the latest data and writes it to the CSV file."""
        if self.csv_writer:
            self.latest_data['timestamp'] = rospy.Time.now().to_sec()
            row = [self.latest_data.get(h, '') for h in self.csv_headers]
            try:
                self.csv_writer.writerow(row)
            except Exception as e:
                rospy.logerr_throttle(5.0, "Error writing CSV row: {}".format(e))


    def shutdown(self):
        """Closes the CSV file properly."""
        rospy.loginfo('Shutting down CSV recorder.')
        if hasattr(self, 'log_timer') and self.log_timer:
            self.log_timer.shutdown()
        if hasattr(self, 'csv_file') and self.csv_file:
            self.csv_file.close()
            rospy.loginfo('CSV file closed: {}'.format(self.filename))

# ------------------------------------------------------------------- main ---
if __name__ == "__main__":
    try:
        recorder = CsvRecorder()
        rospy.spin()
    except rospy.ROSInterruptException:
        rospy.loginfo("CSV recorder interrupted.")
    except Exception as e:
        rospy.logerr("Unhandled exception in CSV recorder: {}".format(e))
        if 'recorder' in locals() and recorder:
             recorder.shutdown()