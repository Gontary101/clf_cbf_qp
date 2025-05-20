#!/usr/bin/env python

# Python 2.7 Compatibility
from __future__ import print_function

import rospy
import matplotlib
try:
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    matplotlib_available = True
except ImportError:
    rospy.logerr("Matplotlib or Agg backend not found. Please install python-matplotlib.")
    matplotlib_available = False
except Exception as e:
    rospy.logerr("Error importing or setting backend for Matplotlib: %s", e)
    matplotlib_available = False

# --- Required Message Types ---
from geometry_msgs.msg import Pose
from std_msgs.msg import Float64MultiArray # For Thrust data U=[U1,U2,U3,U4]
from std_msgs.msg import Float32MultiArray # For omega_sq data
# --- End Message Types ---

import time
import os
import math
import numpy as np

class ConvergencePlotter(object):

    def __init__(self):
        """
        Initializes the ConvergencePlotter node.
        Sets up parameters, subscribers, data storage, and shutdown hook.
        """
        if not matplotlib_available:
            rospy.logerr("Matplotlib not available. Node cannot function. Exiting.")
            return

        rospy.loginfo("Initializing Convergence Plotter Node...")

        # --- Parameters ---
        self.pose_topic = rospy.get_param("~pose_topic", "/iris/ground_truth/pose")
        self.thrust_topic = rospy.get_param("~thrust_topic", "/clf_iris_position_controller/control/U")
        self.omega_sq_topic = rospy.get_param("~omega_sq_topic", "/control/omega_sq")

        self.target_x = rospy.get_param("~target_x", 0.0)
        self.target_y = rospy.get_param("~target_y", 0.0)
        self.target_z = rospy.get_param("~target_z", 0.0)
        self.settling_threshold_percent = rospy.get_param("~settling_threshold_percent", 2.0)

        self.min_thrust = rospy.get_param("~min_thrust", 0.0) # Min U1
        self.max_thrust = rospy.get_param("~max_thrust", 25.0) # Max U1
        default_max_omega_sq = 70000.0
        default_max_omega = math.sqrt(default_max_omega_sq) if default_max_omega_sq >= 0 else 0.0
        self.min_omega = rospy.get_param("~min_omega", 0.0)
        self.max_omega = rospy.get_param("~max_omega", default_max_omega)

        self.plot_save_dir = rospy.get_param("~plot_save_dir", os.path.join(os.path.expanduser('~'), 'ros_plots'))

        rospy.loginfo("--- Plotter Configuration ---")
        rospy.loginfo(" Subscribing to Pose: %s (Type: %s)", self.pose_topic, Pose.__name__)
        rospy.loginfo(" Subscribing to Thrust: %s (Type: %s)", self.thrust_topic, Float64MultiArray.__name__)
        rospy.loginfo(" Subscribing to Omega^2: %s (Type: %s)", self.omega_sq_topic, Float32MultiArray.__name__)
        rospy.loginfo(" Target Pose: [x=%.3f, y=%.3f, z=%.3f]", self.target_x, self.target_y, self.target_z)
        rospy.loginfo(" Settling threshold: %.1f%%", self.settling_threshold_percent)
        rospy.loginfo(" Plotting Limits: Thrust U1=[%.1f, %.1f], Omega=[%.1f, %.1f] rad/s",
                      self.min_thrust, self.max_thrust, self.min_omega, self.max_omega)
        rospy.loginfo(" Plot save directory: %s", self.plot_save_dir)
        rospy.loginfo("-----------------------------")

        self.pose_abs_timestamps = []
        self.x_data = []
        self.y_data = []
        self.z_data = []
        self.thrust_abs_timestamps = []
        self.thrust_data = [] # U1
        self.omega_sq_abs_timestamps = []
        self.omega_sq_data = []

        self.start_time = None # Absolute time of the first pose message

        try:
            self.pose_subscriber = rospy.Subscriber(self.pose_topic, Pose, self.pose_callback, queue_size=10)
            self.thrust_subscriber = rospy.Subscriber(self.thrust_topic, Float64MultiArray, self.thrust_callback, queue_size=10)
            self.omega_sq_subscriber = rospy.Subscriber(self.omega_sq_topic, Float32MultiArray, self.omega_sq_callback, queue_size=10)
            rospy.loginfo("Subscribers created successfully.")
        except Exception as e:
            rospy.logerr("Failed to create subscriber: %s", e)
            rospy.signal_shutdown("Subscriber creation failed")
            return

        rospy.on_shutdown(self.generate_plots)
        rospy.loginfo("Convergence Plotter node initialized. Waiting for data...")

    def pose_callback(self, msg):
        current_time_sec = rospy.Time.now().to_sec()
        if self.start_time is None:
            self.start_time = current_time_sec
            rospy.loginfo("First pose message received. Starting timer at %.3f.", self.start_time)
        try:
            self.pose_abs_timestamps.append(current_time_sec)
            self.x_data.append(msg.position.x)
            self.y_data.append(msg.position.y)
            self.z_data.append(msg.position.z)
        except AttributeError as e:
             rospy.logerr_throttle(10, "AttributeError extracting pose data: %s.", e)

    def thrust_callback(self, msg):
        current_time_sec = rospy.Time.now().to_sec()
        try:
            if not msg.data:
                 rospy.logwarn_throttle(5.0, "Received empty data array on thrust topic %s", self.thrust_topic)
                 return
            thrust_U1 = msg.data[0]
            self.thrust_abs_timestamps.append(current_time_sec)
            self.thrust_data.append(thrust_U1)
        except IndexError:
            rospy.logerr_throttle(5.0, "IndexError extracting thrust data[0] on topic %s.", self.thrust_topic)
        except Exception as e:
            rospy.logerr_throttle(5.0, "Error processing thrust message on topic %s: %s", self.thrust_topic, e)

    def omega_sq_callback(self, msg):
        current_time_sec = rospy.Time.now().to_sec()
        try:
            omega_sq_values = list(msg.data)
            if len(omega_sq_values) != 4:
                 rospy.logwarn_throttle(10, "Received omega_sq data length %d, expected 4.", len(omega_sq_values))
            self.omega_sq_abs_timestamps.append(current_time_sec)
            self.omega_sq_data.append(omega_sq_values)
        except Exception as e:
            rospy.logerr_throttle(10, "Error processing omega_sq message data: %s", e)


    def calculate_metrics(self, data, target, timestamps):
        # (Function unchanged)
        metrics = {'overshoot_percent': 0.0, 'settling_time': None}
        if not data or not timestamps: return metrics
        error = [val - target for val in data]
        initial_error = error[0] if error else 0
        abs_initial_error = abs(initial_error)
        max_error = 0.0; min_error = 0.0
        if error: max_error = max(error); min_error = min(error)
        overshoot = 0.0
        if initial_error > 0 and min_error < 0: overshoot = abs(min_error)
        elif initial_error < 0 and max_error > 0: overshoot = abs(max_error)
        elif initial_error == 0: overshoot = max(abs(max_error), abs(min_error))
        if abs_initial_error > 1e-6: metrics['overshoot_percent'] = (overshoot / abs_initial_error) * 100.0
        else:
             if abs(target) > 1e-6: metrics['overshoot_percent'] = (overshoot / abs(target)) * 100.0
             else: metrics['overshoot_value'] = overshoot
        if abs(target) > 1e-6: settling_band = abs(target * (self.settling_threshold_percent / 100.0))
        elif abs_initial_error > 1e-6: settling_band = abs(initial_error * (self.settling_threshold_percent / 100.0))
        else: settling_band = 0.01
        settled = False; settling_time_val = None
        for i in range(len(error) - 1, -1, -1):
            if abs(error[i]) > settling_band:
                if i + 1 < len(timestamps): settling_time_val = timestamps[i+1]; settled = True
                else: settled = False
                break
        else:
            if timestamps: settling_time_val = timestamps[0]; settled = True
        if settled: metrics['settling_time'] = settling_time_val
        else: metrics['settling_time'] = float('inf')
        return metrics


    def generate_plots(self):
        """Generates and saves plots when the node is shutting down."""
        rospy.loginfo("Shutdown signal received. Generating plots...")

        if not matplotlib_available:
            rospy.logerr("Matplotlib not available.")
            return
        if self.start_time is None:
             rospy.logwarn("No pose data received (start_time not set). Cannot generate plots.")
             return
        if not self.pose_abs_timestamps:
             rospy.logwarn("Pose timestamp list is empty. Skipping plots.")
             return

        rospy.loginfo("Plotting reference time (t=0) established at %.3f", self.start_time)

        # --- Filter and Make Timestamps Relative ---
        pose_rel_timestamps, pose_x_filt, pose_y_filt, pose_z_filt = [], [], [], []
        for i, t_abs in enumerate(self.pose_abs_timestamps):
            if t_abs >= self.start_time:
                pose_rel_timestamps.append(t_abs - self.start_time)
                pose_x_filt.append(self.x_data[i])
                pose_y_filt.append(self.y_data[i])
                pose_z_filt.append(self.z_data[i])

        thrust_rel_timestamps, thrust_data_filt = [], []
        for i, t_abs in enumerate(self.thrust_abs_timestamps):
            if t_abs >= self.start_time:
                thrust_rel_timestamps.append(t_abs - self.start_time)
                thrust_data_filt.append(self.thrust_data[i])

        omega_sq_rel_timestamps, omega_sq_data_filt = [], []
        for i, t_abs in enumerate(self.omega_sq_abs_timestamps):
            if t_abs >= self.start_time:
                omega_sq_rel_timestamps.append(t_abs - self.start_time)
                omega_sq_data_filt.append(self.omega_sq_data[i])

        if not pose_rel_timestamps:
            rospy.logwarn("No pose data remains after filtering by start_time. Skipping plots.")
            return

        # --- Directory and Filename Setup ---
        try:
            if not os.path.exists(self.plot_save_dir): os.makedirs(self.plot_save_dir)
        except Exception as e:
            rospy.logerr("Error creating plot directory '%s': %s. Saving locally.", self.plot_save_dir, e)
            self.plot_save_dir = "."
        timestamp_str = time.strftime("%Y%m%d_%H%M%S")
        base_filename = os.path.join(self.plot_save_dir, timestamp_str)

        def save_plot(fig, name_suffix):
            plot_filename = base_filename + name_suffix
            try:
                fig.savefig(plot_filename)
                rospy.loginfo("Saved plot to: %s", plot_filename)
            except Exception as e: rospy.logerr("Error saving plot '%s': %s", plot_filename, e)
            finally: plt.close(fig)

        # --- Plotting: Position vs Time ---
        # (Code unchanged - uses filtered data)
        rospy.loginfo("Plotting Position data (%d points)...", len(pose_rel_timestamps))
        try:
            metrics_x = self.calculate_metrics(pose_x_filt, self.target_x, pose_rel_timestamps)
            metrics_y = self.calculate_metrics(pose_y_filt, self.target_y, pose_rel_timestamps)
            metrics_z = self.calculate_metrics(pose_z_filt, self.target_z, pose_rel_timestamps)
            rospy.loginfo("--- Position Metrics ---")
            rospy.loginfo(" X: Overshoot=%.2f%%, Settling Time=%.3fs", metrics_x['overshoot_percent'], metrics_x['settling_time'] if metrics_x['settling_time'] is not None else -1.0)
            rospy.loginfo(" Y: Overshoot=%.2f%%, Settling Time=%.3fs", metrics_y['overshoot_percent'], metrics_y['settling_time'] if metrics_y['settling_time'] is not None else -1.0)
            rospy.loginfo(" Z: Overshoot=%.2f%%, Settling Time=%.3fs", metrics_z['overshoot_percent'], metrics_z['settling_time'] if metrics_z['settling_time'] is not None else -1.0)

            fig1, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True, figsize=(12, 8))
            fig1.suptitle('Position Convergence vs. Time (Relative)', fontsize=16)
            # X Plot
            target_x_val = self.target_x
            if abs(target_x_val) > 1e-6: settling_band_x = abs(target_x_val * (self.settling_threshold_percent / 100.0))
            elif pose_x_filt and abs(pose_x_filt[0] - target_x_val) > 1e-6: settling_band_x = abs((pose_x_filt[0] - target_x_val) * (self.settling_threshold_percent / 100.0))
            else: settling_band_x = 0.01
            ax1.plot(pose_rel_timestamps, pose_x_filt, label='X Position')
            ax1.axhline(target_x_val, color='r', linestyle='--', label='Target (%.2f)' % target_x_val)
            ax1.axhline(target_x_val + settling_band_x, color='g', linestyle=':', alpha=0.5, label='Settling (%.1f%%)' % self.settling_threshold_percent)
            ax1.axhline(target_x_val - settling_band_x, color='g', linestyle=':', alpha=0.5)
            if metrics_x['settling_time'] is not None and metrics_x['settling_time'] != float('inf'): ax1.axvline(metrics_x['settling_time'], color='m', linestyle='-.', label='Settle T (%.2fs)' % metrics_x['settling_time'])
            ax1.set_ylabel('X Position'); ax1.legend(); ax1.grid(True)
            # Y Plot
            target_y_val = self.target_y
            if abs(target_y_val) > 1e-6: settling_band_y = abs(target_y_val * (self.settling_threshold_percent / 100.0))
            elif pose_y_filt and abs(pose_y_filt[0] - target_y_val) > 1e-6: settling_band_y = abs((pose_y_filt[0] - target_y_val) * (self.settling_threshold_percent / 100.0))
            else: settling_band_y = 0.01
            ax2.plot(pose_rel_timestamps, pose_y_filt, label='Y Position')
            ax2.axhline(target_y_val, color='r', linestyle='--', label='Target (%.2f)' % target_y_val)
            ax2.axhline(target_y_val + settling_band_y, color='g', linestyle=':', alpha=0.5)
            ax2.axhline(target_y_val - settling_band_y, color='g', linestyle=':', alpha=0.5)
            if metrics_y['settling_time'] is not None and metrics_y['settling_time'] != float('inf'): ax2.axvline(metrics_y['settling_time'], color='m', linestyle='-.', label='Settle T (%.2fs)' % metrics_y['settling_time'])
            ax2.set_ylabel('Y Position'); ax2.legend(); ax2.grid(True)
            # Z Plot
            target_z_val = self.target_z
            if abs(target_z_val) > 1e-6: settling_band_z = abs(target_z_val * (self.settling_threshold_percent / 100.0))
            elif pose_z_filt and abs(pose_z_filt[0] - target_z_val) > 1e-6: settling_band_z = abs((pose_z_filt[0] - target_z_val) * (self.settling_threshold_percent / 100.0))
            else: settling_band_z = 0.01
            ax3.plot(pose_rel_timestamps, pose_z_filt, label='Z Position')
            ax3.axhline(target_z_val, color='r', linestyle='--', label='Target (%.2f)' % target_z_val)
            ax3.axhline(target_z_val + settling_band_z, color='g', linestyle=':', alpha=0.5)
            ax3.axhline(target_z_val - settling_band_z, color='g', linestyle=':', alpha=0.5)
            if metrics_z['settling_time'] is not None and metrics_z['settling_time'] != float('inf'): ax3.axvline(metrics_z['settling_time'], color='m', linestyle='-.', label='Settle T (%.2fs)' % metrics_z['settling_time'])
            ax3.set_ylabel('Z Position'); ax3.set_xlabel('Time (seconds)'); ax3.legend(); ax3.grid(True)

            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            save_plot(fig1, "_1_position.png")
        except Exception as e:
            rospy.logerr("Error during position plotting: %s", e)
            import traceback; traceback.print_exc()

        # --- Plotting: Error vs Time ---
        # (Code unchanged - uses filtered data)
        rospy.loginfo("Plotting Error data...")
        try:
            error_x = [x - self.target_x for x in pose_x_filt]
            error_y = [y - self.target_y for y in pose_y_filt]
            error_z = [z - self.target_z for z in pose_z_filt]
            fig2, (ax1e, ax2e, ax3e) = plt.subplots(3, 1, sharex=True, figsize=(12, 8))
            fig2.suptitle('Error Convergence vs. Time', fontsize=16)
            ax1e.plot(pose_rel_timestamps, error_x, label='X Error'); ax1e.axhline(0, color='r', linestyle='--')
            ax1e.set_ylabel('X Error'); ax1e.legend(); ax1e.grid(True)
            ax2e.plot(pose_rel_timestamps, error_y, label='Y Error'); ax2e.axhline(0, color='r', linestyle='--')
            ax2e.set_ylabel('Y Error'); ax2e.legend(); ax2e.grid(True)
            ax3e.plot(pose_rel_timestamps, error_z, label='Z Error'); ax3e.axhline(0, color='r', linestyle='--')
            ax3e.set_ylabel('Z Error'); ax3e.set_xlabel('Time (seconds)'); ax3e.legend(); ax3e.grid(True)
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            save_plot(fig2, "_2_error.png")
        except Exception as e:
            rospy.logerr("Error during error plotting: %s", e)
            import traceback; traceback.print_exc()

        # --- Plotting: Thrust vs Time ---
        # **** MODIFIED: Check if data was received AT ALL vs. filtered out ****
        if not self.thrust_abs_timestamps:
            rospy.logwarn("No thrust data was ever received on topic %s.", self.thrust_topic)
        elif not thrust_rel_timestamps:
            rospy.logwarn("No thrust data remains after filtering by start_time (%.3f). All received thrust data occurred before the first pose message.", self.start_time)
        else:
            rospy.loginfo("Plotting Thrust U1 data (%d points)...", len(thrust_rel_timestamps))
            try:
                fig3, ax_thrust = plt.subplots(1, 1, figsize=(12, 5))
                fig3.suptitle('Commanded Total Thrust (U1) vs. Time', fontsize=16)
                ax_thrust.plot(thrust_rel_timestamps, thrust_data_filt, label='Commanded Thrust (U1)')
                ax_thrust.axhline(self.max_thrust, color='r', linestyle='--', alpha=0.7, label='Max Limit (%.1f N)' % self.max_thrust)
                ax_thrust.axhline(self.min_thrust, color='r', linestyle=':', alpha=0.7, label='Min Limit (%.1f N)' % self.min_thrust)
                ax_thrust.set_ylabel('Thrust U1 (N)'); ax_thrust.set_xlabel('Time (seconds)')
                ax_thrust.legend(); ax_thrust.grid(True)
                plt.tight_layout(rect=[0, 0.03, 1, 0.95])
                save_plot(fig3, "_3_thrust.png")
            except Exception as e:
                rospy.logerr("Error during thrust plotting: %s", e)
                import traceback; traceback.print_exc()

        # --- Plotting: Omega (Angular Velocity) vs Time ---
        # **** MODIFIED: Check if data was received AT ALL vs. filtered out ****
        if not self.omega_sq_abs_timestamps:
            rospy.logwarn("No omega_sq data was ever received on topic %s.", self.omega_sq_topic)
        elif not omega_sq_rel_timestamps:
             rospy.logwarn("No omega_sq data remains after filtering by start_time (%.3f). All received omega_sq data occurred before the first pose message.", self.start_time)
        else:
            rospy.loginfo("Plotting Omega (Angular Velocity) data (%d points)...", len(omega_sq_rel_timestamps))
            try:
                omega_data_filt = []
                neg_omega_sq_found = False
                for sq_vals in omega_sq_data_filt:
                    velocities = []
                    for sq_val in sq_vals:
                        if sq_val < 0:
                            velocities.append(0.0)
                            if not neg_omega_sq_found:
                                rospy.logwarn_once("Found negative omega_sq value(s). Plotting omega as 0.")
                                neg_omega_sq_found = True
                        else:
                            velocities.append(math.sqrt(sq_val))
                    omega_data_filt.append(velocities)

                fig4, ax_omega = plt.subplots(1, 1, figsize=(12, 5))
                fig4.suptitle('Commanded Rotor Angular Velocities vs. Time', fontsize=16)

                try:
                    omega_array_filt = np.array(omega_data_filt)
                    num_rotors = omega_array_filt.shape[1] if len(omega_array_filt.shape) > 1 and omega_array_filt.shape[1] > 0 else 0
                    if num_rotors == 0 and len(omega_array_filt) > 0: rospy.logwarn("Filtered Omega data seems malformed.")
                    for i in range(num_rotors):
                         ax_omega.plot(omega_sq_rel_timestamps, omega_array_filt[:, i], label='Omega Rotor %d' % (i+1))
                except Exception as np_err:
                     rospy.logwarn("Could not process filtered omega data as numpy array (%s). Plotting via list.", np_err)
                     num_rotors = len(omega_data_filt[0]) if omega_data_filt else 0
                     for i in range(num_rotors):
                          rotor_i_data = [item[i] for item in omega_data_filt if len(item)>i]
                          ax_omega.plot(omega_sq_rel_timestamps, rotor_i_data, label='Omega Rotor %d' % (i+1))

                ax_omega.axhline(self.max_omega, color='r', linestyle='--', alpha=0.7, label='Max Limit (%.1f rad/s)' % self.max_omega)
                ax_omega.axhline(self.min_omega, color='r', linestyle=':', alpha=0.7, label='Min Limit (%.1f rad/s)' % self.min_omega)
                ax_omega.set_ylabel('Angular Velocity (rad/s)')
                ax_omega.set_xlabel('Time (seconds)')
                ax_omega.legend(); ax_omega.grid(True)
                plt.tight_layout(rect=[0, 0.03, 1, 0.95])
                save_plot(fig4, "_4_omega.png")

            except Exception as e:
                rospy.logerr("Error during omega plotting: %s", e)
                import traceback; traceback.print_exc()

        rospy.loginfo("Plot generation finished.")


# --- Main Execution ---
# (Unchanged)
if __name__ == '__main__':
    try:
        rospy.init_node('convergence_plotter', anonymous=True)
        plotter = ConvergencePlotter()
        rospy.spin()
    except rospy.ROSInterruptException:
        rospy.loginfo("ROS node interrupted.")
    except Exception as e:
        rospy.logerr("An unhandled exception occurred in main: %s", e)
        import traceback; traceback.print_exc()
    finally:
        if matplotlib_available: plt.close('all')
        rospy.loginfo("Convergence plotter node shutting down.")