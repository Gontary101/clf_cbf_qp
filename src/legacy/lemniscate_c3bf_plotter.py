#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import unicode_literals
from __future__ import division
import rospy, os, math, datetime, numpy as np
import matplotlib; matplotlib.use('Agg') # Use Agg backend for non-GUI environments
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from nav_msgs.msg import Odometry
from std_msgs.msg import Float64MultiArray, String
from geometry_msgs.msg import Point, Vector3
from tf.transformations import euler_from_quaternion
import ast

# Helper functions
clip = np.clip
pget = lambda n,d: rospy.get_param("~"+n,d)
# Savitzky-Golay filter wrapper with check for sufficient data points
FILT = lambda x,w,o: savgol_filter(x,w,o) if isinstance(x, np.ndarray) and x.ndim > 0 and len(x) >= w else x

class Plotter(object):
    def __init__(self):
        # --- Get Parameters ---
        # Topics
        self.topics = dict(
            odom   = pget("odom_topic","/iris/ground_truth/odometry"),
            state  = pget("state_topic","/clf_iris_trajectory_controller/control/state")
        )
        # Trajectory Params (keep for calculating desired path)
        self.d_start = pget("helix_start_diameter",40.0)
        self.d_end   = pget("helix_end_diameter",15.0)
        self.height  = pget("helix_height",30.0)
        self.laps    = pget("helix_laps", 4.0) # Match controller default
        self.omega_traj = pget("trajectory_omega",0.07) # Match controller default
        # Plotting Params
        self.save_dir= pget("plot_save_dir",".")
        self.fw_o,self.fp_o = pget("filter_window_odom",51),pget("filter_polyorder_odom",3)
        self.use_filt= pget("use_filtering",True)
        self.run_T   = pget("run_duration_secs",200.0)
        # Get controller params to display on plot
        self.controller_ns = "/clf_iris_trajectory_controller" # Namespace of the controller node
        # Fetch parameters with error handling for missing params
        try:
            self.k_trajpos1 = rospy.get_param(self.controller_ns + "/k_trajpos1")
        except KeyError:
            rospy.logwarn("[Plotter] Parameter %s/k_trajpos1 not found, using default N/A.", self.controller_ns)
            self.k_trajpos1 = "N/A"
        try:
            self.k_trajpos2 = rospy.get_param(self.controller_ns + "/k_trajpos2")
        except KeyError:
            rospy.logwarn("[Plotter] Parameter %s/k_trajpos2 not found, using default N/A.", self.controller_ns)
            self.k_trajpos2 = "N/A"
        try:
            self.cbf_gamma = rospy.get_param(self.controller_ns + "/cbf_gamma")
        except KeyError:
            rospy.logwarn("[Plotter] Parameter %s/cbf_gamma not found, using default N/A.", self.controller_ns)
            self.cbf_gamma = "N/A"
        try:
            self.drone_radius = rospy.get_param(self.controller_ns + "/drone_radius")
        except KeyError:
            rospy.logwarn("[Plotter] Parameter %s/drone_radius not found, using default N/A.", self.controller_ns)
            self.drone_radius = "N/A"


        # Precompute helix params
        self.r0      = 0.5*self.d_start
        theta_tot   = self.laps*2.0*math.pi
        # Prevent division by zero if laps is zero
        self.k_r    = (self.r0 - 0.5*self.d_end)/theta_tot if abs(theta_tot) > 1e-6 else 0.0
        self.k_z    = self.height/theta_tot if abs(theta_tot) > 1e-6 else 0.0

        # Parse obstacles (accepts 4 or 7 element lists)
        # Get obstacles from the controller node's parameter server
        obstacles_str = rospy.get_param(self.controller_ns + "/static_obstacles", "[]")
        self.obstacles = []
        try:
            parsed_obstacles = ast.literal_eval(obstacles_str)
            if isinstance(parsed_obstacles, list):
                for i, obs in enumerate(parsed_obstacles):
                    is_valid = False
                    if isinstance(obs, (list, tuple)):
                        if len(obs) == 4 and all(isinstance(n, (int, float)) for n in obs):
                            # Add zero velocity for static obstacles
                            self.obstacles.append(list(obs) + [0.0, 0.0, 0.0])
                            is_valid = True
                        elif len(obs) == 7 and all(isinstance(n, (int, float)) for n in obs):
                            self.obstacles.append(list(obs))
                            is_valid = True
                    if not is_valid:
                         rospy.logwarn("[Plotter] Invalid obstacle format in %s/static_obstacles at index %d: %s. Expected [x,y,z,r] or [x,y,z,r,vx,vy,vz]. Skipping.", self.controller_ns, i, str(obs))
            else:
                rospy.logwarn("[Plotter] Could not parse %s/static_obstacles, expected list format. Got: %s", self.controller_ns, obstacles_str)
        except (ValueError, SyntaxError) as e:
             rospy.logwarn("[Plotter] Error parsing %s/static_obstacles: '%s': %s", self.controller_ns, obstacles_str, e)
        except Exception as e:
             rospy.logwarn("[Plotter] Unexpected error processing %s/static_obstacles: '%s': %s", self.controller_ns, obstacles_str, e)

        if self.obstacles:
            rospy.loginfo("[Plotter] Loaded %d obstacles for plotting.", len(self.obstacles))
        else:
            rospy.loginfo("[Plotter] No valid obstacles loaded for plotting.")


        # Internal state
        self.t0   = None
        self.xy0  = None
        self.z0   = None
        # Simplified data storage
        self.data = {'t':[],'x':[],'y':[],'z':[]}
        self.rec  = False
        self.done = False

        # Subscribers
        rospy.Subscriber(self.topics['state'],  String,   self.cb_state, queue_size=2)
        rospy.Subscriber(self.topics['odom'],   Odometry, self.cb_odom,  queue_size=200)
        rospy.on_shutdown(self.shutdown)
        rospy.loginfo("Trajectory Plotter ready - waiting for TRAJECTORY phase.")

    def cb_state(self,msg):
        if msg.data=="TRAJ" and not self.rec:
            self.rec=True
            rospy.loginfo("[Plotter] TRAJECTORY phase detected - recording starts now.")

    def cb_odom(self,msg):
        if not self.rec or self.done: return
        if self.t0 is None:
            self.t0 = msg.header.stamp
            p = msg.pose.pose.position
            # Align desired path based on first actual position
            ref_x_start = self.r0
            ref_y_start = 0.0
            ref_z_start = 0.0
            self.xy0 = np.array([p.x - ref_x_start, p.y - ref_y_start])
            self.z0  = p.z - ref_z_start
            rospy.loginfo("[Plotter] First odom received. t0=%.2f, xy0=[%.2f, %.2f], z0=%.2f",
                          self.t0.to_sec(), self.xy0[0], self.xy0[1], self.z0)

        t_sec = (msg.header.stamp - self.t0).to_sec()
        self.data['t'].append(t_sec)
        p=msg.pose.pose.position
        self.data['x'].append(p.x)
        self.data['y'].append(p.y)
        self.data['z'].append(p.z) # Keep Z for potential future use, but won't plot

        if t_sec>=self.run_T: self.shutdown()

    @staticmethod
    def R(phi,th,psi): # Keep for potential future use if velocity needed
        c,s = math.cos,math.sin
        return np.array([
            [c(th)*c(psi),  s(phi)*s(th)*c(psi)-c(phi)*s(psi),  c(phi)*s(th)*c(psi)+s(phi)*s(psi)],
            [c(th)*s(psi),  s(phi)*s(th)*s(psi)+c(phi)*c(psi),  c(phi)*s(th)*s(psi)-s(phi)*c(psi)],
            [   -s(th),                     s(phi)*c(th),                    c(phi)*c(th)]
        ])

    def desired(self,t):
        if self.xy0 is None or self.z0 is None: # Check if initial offset is calculated
             rospy.logwarn_throttle(1.0, "[Plotter] Initial offset not set, cannot calculate desired path.")
             t_np = np.asarray(t)
             # Return empty arrays of the correct shape if t is not empty
             shape = (len(t_np), 3) if t_np.ndim > 0 and len(t_np) > 0 else (0, 3)
             return np.zeros(shape), None # Return zeros for position

        t_np = np.asarray(t)
        psit = self.omega_traj * t_np
        r = self.r0 - self.k_r*psit
        x = r*np.cos(psit) + self.xy0[0]
        y = r*np.sin(psit) + self.xy0[1]
        z = self.k_z*psit + self.z0 # Calculate Z for completeness
        pos = np.vstack([x,y,z]).T
        # Handle scalar input case
        if t_np.ndim == 0: # If t was a scalar
             pos = pos.reshape(1,3)

        # Velocity calculation removed as it's not plotted
        return pos, None # Return None for velocity

    def shutdown(self):
        if self.done: return
        self.done=True
        if self.t0 is None or not self.data['t']:
            rospy.logwarn("[Plotter] No data captured - nothing to plot."); return

        t   = np.array(self.data['t'])
        # Make sure data is numpy array before processing
        X = np.array(self.data['x'])
        Y = np.array(self.data['y'])
        Z = np.array(self.data['z']) # Keep Z for reference

        # Regenerate desired path
        Pd_calc, _ = self.desired(t) # Ignore velocity
        if Pd_calc is None or Pd_calc.shape[0] != len(t):
             rospy.logerr("[Plotter] Failed to generate desired path for plotting.")
             return
        Pd = Pd_calc

        # --- Filtering (Position Only) ---
        if self.use_filt:
            w = self.fw_o
            if len(X) >= w: X = FILT(X, w, self.fp_o)
            if len(Y) >= w: Y = FILT(Y, w, self.fp_o)
            # No need to filter Z if not plotting it

        # --- Calculate RMS Error (XY only) ---
        # Ensure dimensions match before subtraction
        if X.shape == Pd[:,0].shape and Y.shape == Pd[:,1].shape:
            rms_xy =np.sqrt(np.mean((X-Pd[:,0])**2+(Y-Pd[:,1])**2))
            rospy.loginfo("[Plotter] RMS XY position error: %.3f m", rms_xy)
        else:
            rospy.logwarn("[Plotter] Dimension mismatch between actual and desired path, cannot calculate RMS error.")


        # --- Prepare Save Path ---
        ts  = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        base= os.path.join(self.save_dir,ts)
        if not os.path.isdir(self.save_dir):
             try: os.makedirs(self.save_dir)
             except OSError as e: rospy.logerr("[Plotter] Failed to create plot save directory %s: %s", self.save_dir, e); return

        # --- Generate 2D Top-Down Plot ---
        fig2d, ax2d = plt.subplots(figsize=(10, 10)) # Slightly larger figure

        # Initialize plot limits with drone data (check if data exists)
        min_x_2d, max_x_2d = (np.min(X), np.max(X)) if len(X) > 0 else (0,0)
        min_y_2d, max_y_2d = (np.min(Y), np.max(Y)) if len(Y) > 0 else (0,0)
        try: # Handle case where desired path might be empty
             if Pd.shape[0] > 0:
                  min_x_2d = min(min_x_2d, np.min(Pd[:,0])); max_x_2d = max(max_x_2d, np.max(Pd[:,0]))
                  min_y_2d = min(min_y_2d, np.min(Pd[:,1])); max_y_2d = max(max_y_2d, np.max(Pd[:,1]))
        except ValueError:
             pass # Keep initial limits if Pd is empty or calculation failed

        obstacle_circle_legend_added = False
        kinematic_traj_legend_added = False
        kinematic_start_legend_added = False
        kinematic_end_legend_added = False

        # Plot Obstacles and Kinematic Trajectories
        if self.obstacles:
            rospy.loginfo("[Plotter] Plotting %d obstacles in 2D.", len(self.obstacles))
            # Need time vector t to calculate kinematic path
            if len(t) == 0:
                 rospy.logwarn("[Plotter] Time vector is empty, cannot plot kinematic obstacle trajectories.")
            else:
                for obs_idx, obs in enumerate(self.obstacles):
                    try:
                        # Plot obstacle circle representation (static or kinematic)
                        cx, cy, _, r = map(float, obs[0:4])
                        label_circ = 'Obstacle Repr.' if not obstacle_circle_legend_added else ""
                        circle = plt.Circle((cx, cy), r, color='grey', alpha=0.4, label=label_circ)
                        ax2d.add_patch(circle)
                        obstacle_circle_legend_added = True
                        min_x_2d = min(min_x_2d, cx - r); max_x_2d = max(max_x_2d, cx + r)
                        min_y_2d = min(min_y_2d, cy - r); max_y_2d = max(max_y_2d, cy + r)

                        # Check if kinematic (has non-zero XY velocity)
                        if not (abs(obs[4]) < 1e-6 and abs(obs[5]) < 1e-6):
                             ox, oy, _, _, vx, vy, _ = map(float, obs[:7])
                             Ox = ox + vx * t # Calculate path over time t
                             Oy = oy + vy * t
                             label_traj = 'Kin. Obs. Traj.' if not kinematic_traj_legend_added else ""
                             ax2d.plot(Ox, Oy, 'm:', linewidth=1.5, label=label_traj) # Magenta dotted
                             kinematic_traj_legend_added = True
                             label_start = 'Kin. Start' if not kinematic_start_legend_added else ""
                             ax2d.plot(ox, oy, 'ms', markersize=6, label=label_start) # Magenta square
                             kinematic_start_legend_added = True
                             # Plot end marker only if trajectory is not empty
                             if len(Ox) > 0:
                                  label_end = 'Kin. End' if not kinematic_end_legend_added else ""
                                  ax2d.plot(Ox[-1], Oy[-1], 'm^', markersize=7, label=label_end) # Magenta triangle
                                  kinematic_end_legend_added = True
                                  min_x_2d = min(min_x_2d, np.min(Ox)); max_x_2d = max(max_x_2d, np.max(Ox))
                                  min_y_2d = min(min_y_2d, np.min(Oy)); max_y_2d = max(max_y_2d, np.max(Oy))

                    except Exception as e:
                        rospy.logwarn("[Plotter] Error processing obstacle %s for 2D plot: %s", str(obs), e)

        # Plot Drone Trajectories (check if data exists)
        if len(X) > 0: ax2d.plot(X, Y, 'b-', linewidth=1.5, label='Actual Traj.')
        if Pd.shape[0] > 0: ax2d.plot(Pd[:, 0], Pd[:, 1], 'r--', linewidth=1.5, label='Desired Traj.')

        # --- Finalize 2D Plot ---
        ax2d.set_xlabel('X (m)'); ax2d.set_ylabel('Y (m)')
        ax2d.set_title('Top-Down Trajectory View')
        ax2d.grid(True); ax2d.legend(fontsize='small', loc='best')
        ax2d.set_aspect('equal', adjustable='box')

        # Set final limits
        range_x_2d = max(max_x_2d - min_x_2d, 1.0) # Ensure min range
        range_y_2d = max(max_y_2d - min_y_2d, 1.0) # Ensure min range
        max_range_2d = max(range_x_2d, range_y_2d) * 1.1 # Add padding
        mid_x_2d = (max_x_2d + min_x_2d) / 2.0
        mid_y_2d = (max_y_2d + min_y_2d) / 2.0
        ax2d.set_xlim(mid_x_2d - max_range_2d / 2.0, mid_x_2d + max_range_2d / 2.0)
        ax2d.set_ylim(mid_y_2d - max_range_2d / 2.0, mid_y_2d + max_range_2d / 2.0)

        # --- Add Parameter Text ---
        try:
            # Attempt to format numbers, handle potential "N/A" strings or other types
            kp1_str = "{:.2f}".format(float(self.k_trajpos1)) if isinstance(self.k_trajpos1, (int, float)) else str(self.k_trajpos1)
            kp2_str = "{:.2f}".format(float(self.k_trajpos2)) if isinstance(self.k_trajpos2, (int, float)) else str(self.k_trajpos2)
            gamma_str = "{:.1f}".format(float(self.cbf_gamma)) if isinstance(self.cbf_gamma, (int, float)) else str(self.cbf_gamma)
            dr_str = "{:.2f}".format(float(self.drone_radius)) if isinstance(self.drone_radius, (int, float)) else str(self.drone_radius)

            param_text = "Gains: k_trajpos1={}, k_trajpos2={}\nCBF: gamma={}, drone_radius={} m".format(
                kp1_str, kp2_str, gamma_str, dr_str
            )
        except (ValueError, TypeError) as e:
             rospy.logwarn("[Plotter] Could not format all parameters for display: %s", e)
             param_text = "Error displaying parameters"

        # Add text box below the plot
        plt.text(0.5, -0.12, param_text, ha='center', va='top', transform=ax2d.transAxes, # Adjusted y slightly
                 fontsize=9, bbox=dict(boxstyle='round,pad=0.5', fc='wheat', alpha=0.5))

        # Adjust layout to prevent text overlap
        plt.subplots_adjust(bottom=0.2) # May need slight adjustment

        # Save the 2D plot
        plot_filename = base + "_traj2d_topdown.png"
        try:
            plt.savefig(plot_filename);
            rospy.loginfo("[Plotter] Saved 2D plot: %s", plot_filename)
        except Exception as e:
             rospy.logerr("[Plotter] Failed to save plot %s: %s", plot_filename, e)
        finally:
            plt.close(fig2d) # Ensure figure is closed even if save fails


        rospy.loginfo("[Plotter] Plot generation complete.")

if __name__=="__main__":
    rospy.init_node("trajectory_plotter_node",anonymous=True)
    try:
         Plotter(); rospy.spin()
    except rospy.ROSInterruptException:
         rospy.loginfo("[Plotter] Node interrupted.")
    except Exception as e:
         rospy.logerr("[Plotter] Unhandled exception in Plotter node: %s", e)
         import traceback
         traceback.print_exc()