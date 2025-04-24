#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import unicode_literals
from __future__ import division
import rospy, os, math, datetime, numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from nav_msgs.msg import Odometry
from std_msgs.msg import Float64MultiArray, String
from geometry_msgs.msg import Point, Vector3
from tf.transformations import euler_from_quaternion
import ast

clip = np.clip
norm = np.linalg.norm
pget = lambda n,d: rospy.get_param("~"+n,d)
FILT = lambda x,w,o: savgol_filter(x,w,o) if len(x)>=w else x
EPS = 1e-6

class Plotter(object):
    def __init__(self):
        # --- Topic and General Parameters ---
        self.topics = dict(
            odom   = pget("odom_topic","/iris/ground_truth/odometry"),
            omega  = pget("omega_sq_topic","/clf_iris_straight_line_controller/control/omega_sq"),
            thrust = pget("thrust_topic","/clf_iris_straight_line_controller/control/U"),
            state  = pget("state_topic","/clf_iris_straight_line_controller/control/state")
        )
        self.save_dir= pget("plot_save_dir",".")
        self.w_lim   = (pget("min_omega",0.),pget("max_omega",838.))
        self.fw_o,self.fp_o = pget("filter_window_odom",51),pget("filter_polyorder_odom",3)
        self.fw_w,self.fp_w = pget("filter_window_omega",31),pget("filter_polyorder_omega",3)
        self.fw_t,self.fp_t = pget("filter_window_thrust",31),pget("filter_polyorder_thrust",3)
        self.use_filt= pget("use_filtering",True)
        self.run_T   = pget("run_duration_secs",200.0) # Max duration to record after TRAJ starts

        # --- Straight Line Trajectory Parameters ---
        self.takeoff_x = pget("takeoff_x", 0.0)
        self.takeoff_y = pget("takeoff_y", 0.0)
        self.takeoff_height = pget("takeoff_height", 3.0)
        self.z_final = pget("z_final", 20.0)
        self.slope_deg = pget("slope_deg", 60.0)
        self.traj_speed = pget("traj_speed", 1.0)

        # --- Calculate Trajectory Details ---
        self.traj_start_pos = np.array([self.takeoff_x, self.takeoff_y, self.takeoff_height])
        self.delta_z = self.z_final - self.takeoff_height
        self.traj_vector = np.zeros(3)
        self.traj_length = 0.0
        self.total_traj_time = 0.0
        self.traj_velocity_vector = np.zeros(3)

        if self.delta_z <= EPS:
            rospy.logwarn("Plotter: Final Z (%.2f) not above takeoff Z (%.2f). Desired path is just hover.", self.z_final, self.takeoff_height)
            self.traj_vector = np.zeros(3)
            self.traj_length = 0.0
            self.total_traj_time = float('inf') # Effectively infinite time (hover)
            self.traj_velocity_vector = np.zeros(3)
        else:
            slope_rad = math.radians(self.slope_deg)
            if abs(slope_rad - math.pi/2.0) < EPS:
                delta_xy = 0.0
            elif abs(math.tan(slope_rad)) < EPS:
                rospy.logwarn("Plotter: Slope angle is too close to zero. Assuming vertical ascent.")
                delta_xy = 0.0
            else:
                delta_xy = self.delta_z / math.tan(slope_rad)

            # Assume movement primarily in X direction from start point
            delta_x = delta_xy
            delta_y = 0.0 # No movement in Y for this trajectory definition
            self.traj_vector = np.array([delta_x, delta_y, self.delta_z])
            self.traj_length = norm(self.traj_vector)

            if self.traj_speed <= EPS:
                rospy.logwarn("Plotter: Trajectory speed is zero or negative (%.2f). Cannot calculate time.", self.traj_speed)
                self.total_traj_time = float('inf')
                self.traj_velocity_vector = np.zeros(3)
            elif self.traj_length < EPS:
                 rospy.logwarn("Plotter: Trajectory length is near zero (%.3f). Desired path is just hover.", self.traj_length)
                 self.total_traj_time = float('inf')
                 self.traj_velocity_vector = np.zeros(3)
            else:
                self.total_traj_time = self.traj_length / self.traj_speed
                self.traj_velocity_vector = self.traj_speed * (self.traj_vector / self.traj_length)
                rospy.loginfo("Plotter: Calculated desired trajectory. Start: [%.2f, %.2f, %.2f], End: [%.2f, %.2f, %.2f], Speed: %.2f, Duration: %.2f s",
                              self.traj_start_pos[0], self.traj_start_pos[1], self.traj_start_pos[2],
                              self.traj_start_pos[0] + self.traj_vector[0],
                              self.traj_start_pos[1] + self.traj_vector[1],
                              self.traj_start_pos[2] + self.traj_vector[2],
                              self.traj_speed, self.total_traj_time)

        # --- Obstacle Parsing ---
        obstacles_str = pget("static_obstacles", "[]")
        self.obstacles = []
        try:
            parsed_obstacles = ast.literal_eval(obstacles_str)
            if isinstance(parsed_obstacles, list):
                for i, obs in enumerate(parsed_obstacles):
                    is_valid = False
                    if isinstance(obs, (list, tuple)):
                        if len(obs) == 4 and all(isinstance(n, (int, float)) for n in obs):
                            self.obstacles.append(list(obs) + [0.0, 0.0, 0.0]) # Add zero velocity
                            is_valid = True
                        elif len(obs) == 7 and all(isinstance(n, (int, float)) for n in obs):
                            self.obstacles.append(list(obs))
                            is_valid = True
                    if not is_valid:
                         rospy.logwarn("Invalid obstacle format in static_obstacles at index %d: %s. Skipping.", i, str(obs))
            else:
                rospy.logwarn("Could not parse static_obstacles parameter, expected list format. Got: %s", obstacles_str)
        except (ValueError, SyntaxError) as e:
             rospy.logwarn("Error parsing static_obstacles parameter '%s': %s", obstacles_str, e)
        except Exception as e:
             rospy.logwarn("Unexpected error processing static_obstacles '%s': %s", obstacles_str, e)

        if self.obstacles:
            rospy.loginfo("Loaded %d obstacles for plotting.", len(self.obstacles))
        else:
            rospy.loginfo("No valid obstacles loaded for plotting.")

        # --- Data Storage and State ---
        self.t0   = None # Timestamp of first odom message *after* TRAJ state starts
        self.data = {'t':[],'x':[],'y':[],'z':[],'vx':[],'vy':[],'vz':[],
                     'w_t':[],'w':[],'u_t':[],'u1':[]}
        self.rec  = False # Are we in the TRAJ state?
        self.done = False # Has plotting finished?

        # --- Subscribers ---
        rospy.Subscriber(self.topics['state'],  String,            self.cb_state,  queue_size=2)
        rospy.Subscriber(self.topics['odom'],   Odometry,          self.cb_odom,   queue_size=200)
        rospy.Subscriber(self.topics['omega'],  Float64MultiArray, self.cb_omega,  queue_size=200)
        rospy.Subscriber(self.topics['thrust'], Float64MultiArray, self.cb_thrust, queue_size=200)
        rospy.on_shutdown(self.shutdown)
        rospy.loginfo("Trajectory Plotter ready - waiting for TRAJ state.")

    def cb_state(self,msg):
        if msg.data=="TRAJ" and not self.rec:
            self.rec=True
            self.t0 = None # Reset t0 when TRAJ starts to get relative time
            rospy.loginfo("TRAJ state detected - recording starts now.")
        # Optional: Stop recording if state changes away from TRAJ?
        # elif msg.data != "TRAJ" and self.rec:
        #    rospy.loginfo("Exited TRAJ state, stopping recording.")
        #    self.shutdown() # Trigger plotting

    def cb_odom(self,msg):
        if not self.rec or self.done: return
        if self.t0 is None:
            self.t0 = msg.header.stamp # Record time of first odom message in TRAJ state

        t_sec = (msg.header.stamp - self.t0).to_sec()
        self.data['t'].append(t_sec)
        p=msg.pose.pose.position; q=msg.pose.pose.orientation
        v=msg.twist.twist.linear
        try:
             R=self.R(*euler_from_quaternion([q.x,q.y,q.z,q.w]))
             vw=np.dot(R,[v.x,v.y,v.z]) # Velocity in world frame
        except ValueError:
             rospy.logwarn_throttle(5.0, "Invalid quaternion in odom, using zero velocity.")
             vw = np.zeros(3)

        self.data['x'].append(p.x)
        self.data['y'].append(p.y)
        self.data['z'].append(p.z)
        self.data['vx'].append(vw[0])
        self.data['vy'].append(vw[1])
        self.data['vz'].append(vw[2])

        # Stop recording after specified duration or if trajectory should be finished
        # Add a small buffer to total_traj_time check
        if t_sec >= self.run_T or (self.total_traj_time != float('inf') and t_sec > self.total_traj_time + 2.0):
            rospy.loginfo("Stopping recording. Duration limit reached or trajectory time exceeded.")
            self.shutdown()

    def cb_omega(self,msg):
        if self.rec and not self.done and self.t0 is not None:
            # Use current time relative to t0 for omega/thrust as they might not be perfectly synced with odom
            t_rel = (rospy.Time.now() - self.t0).to_sec()
            self.data['w_t'].append(t_rel)
            self.data['w'].append(np.array(msg.data))

    def cb_thrust(self,msg):
         if self.rec and not self.done and self.t0 is not None:
            t_rel = (rospy.Time.now() - self.t0).to_sec()
            self.data['u_t'].append(t_rel)
            self.data['u1'].append(msg.data[0] if msg.data else 0.0)

    @staticmethod
    def R(phi,th,psi):
        c,s = math.cos,math.sin
        return np.array([
            [c(th)*c(psi),  s(phi)*s(th)*c(psi)-c(phi)*s(psi),  c(phi)*s(th)*c(psi)+s(phi)*s(psi)],
            [c(th)*s(psi),  s(phi)*s(th)*s(psi)+c(phi)*c(psi),  c(phi)*s(th)*s(psi)-s(phi)*c(psi)],
            [   -s(th),                     s(phi)*c(th),                    c(phi)*c(th)]
        ])

    def desired(self, t):
        """Calculates the desired position and velocity for a given time vector t."""
        t_np = np.asarray(t)
        pos = np.zeros((len(t_np), 3))
        vel = np.zeros((len(t_np), 3))

        # Handle cases where trajectory is just hovering
        if self.total_traj_time == float('inf') or self.traj_length < EPS:
            pos[:, :] = self.traj_start_pos # Stay at start position
            vel[:, :] = 0.0                 # Zero velocity
            return pos, vel

        # Calculate position and velocity for each time point
        for i, ti in enumerate(t_np):
            if ti < 0:
                pos[i, :] = self.traj_start_pos
                vel[i, :] = 0.0
            elif ti >= self.total_traj_time:
                pos[i, :] = self.traj_start_pos + self.traj_vector
                vel[i, :] = 0.0 # Stop at the end
            else:
                pos[i, :] = self.traj_start_pos + ti * self.traj_velocity_vector
                vel[i, :] = self.traj_velocity_vector # Constant velocity during trajectory

        return pos, vel

    def shutdown(self):
        if self.done: return
        self.done=True
        if self.t0 is None or not self.data['t']:
            rospy.logwarn("No data captured during TRAJ state - nothing to plot."); return

        rospy.loginfo("Plotting started...")
        t   = np.array(self.data['t'])
        X,Y,Z = map(np.array,(self.data['x'],self.data['y'],self.data['z']))
        Vx,Vy,Vz = map(np.array,(self.data['vx'],self.data['vy'],self.data['vz']))

        # Regenerate desired path using the actual time vector 't' from odom data
        Pd, Vd = self.desired(t)
        if Pd.shape[0] != len(t):
             rospy.logerr("Mismatch between desired path length (%d) and time vector length (%d)! Aborting plot.", Pd.shape[0], len(t))
             return

        # --- Filtering ---
        if self.use_filt:
            w = self.fw_o
            if len(X) >= w: X = FILT(X, w, self.fp_o)
            if len(Y) >= w: Y = FILT(Y, w, self.fp_o)
            if len(Z) >= w: Z = FILT(Z, w, self.fp_o)
            if len(Vx) >= w: Vx = FILT(Vx, w, self.fp_o)
            if len(Vy) >= w: Vy = FILT(Vy, w, self.fp_o)
            if len(Vz) >= w: Vz = FILT(Vz, w, self.fp_o)

        # --- RMS Error Calculation ---
        # Ensure comparison happens only during the intended trajectory time if finite
        if self.total_traj_time != float('inf'):
            valid_idx = t <= self.total_traj_time
            if np.any(valid_idx):
                 rms=np.sqrt(np.mean((X[valid_idx]-Pd[valid_idx,0])**2+(Y[valid_idx]-Pd[valid_idx,1])**2+(Z[valid_idx]-Pd[valid_idx,2])**2))
                 rospy.loginfo("RMS 3-D position error (during trajectory): %.3f m",rms)
            else:
                 rospy.loginfo("No data points within the calculated trajectory duration to compute RMS error.")
        else:
             # If infinite duration (hover), calculate RMS over the whole recorded period
             rms=np.sqrt(np.mean((X-Pd[:,0])**2+(Y-Pd[:,1])**2+(Z-Pd[:,2])**2))
             rospy.loginfo("RMS 3-D position error (hover): %.3f m",rms)


        # --- File Naming and Directory ---
        ts  = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        base= os.path.join(self.save_dir,ts)
        if not os.path.isdir(self.save_dir):
             try: os.makedirs(self.save_dir)
             except OSError as e: rospy.logerr("Failed to create plot save directory %s: %s", self.save_dir, e); return

        # --- Position Error Plot ---
        plt.figure(figsize=(12,6))
        plt.plot(t,X-Pd[:,0],label='ex');plt.plot(t,Y-Pd[:,1],label='ey');plt.plot(t,Z-Pd[:,2],label='ez')
        plt.title("Position Tracking Error");plt.legend();plt.grid();plt.xlabel('t (s)');plt.ylabel('Error (m)')
        plt.savefig(base+"_pos_err.png");plt.close()

        # --- Position Tracking Plot ---
        fig_pos,(ax1_pos,ax2_pos,ax3_pos)=plt.subplots(3,1,figsize=(12,10),sharex=True)
        for ax,act,des,lbl in zip((ax1_pos,ax2_pos,ax3_pos),(X,Y,Z),(Pd[:,0],Pd[:,1],Pd[:,2]),('X','Y','Z')):
            ax.plot(t,act,'b-',label='Actual '+lbl);ax.plot(t,des,'r--',label='Desired '+lbl)
            ax.set_ylabel(lbl+' (m)');ax.grid();ax.legend()
        ax3_pos.set_xlabel('t (s)');fig_pos.suptitle("Position Tracking");fig_pos.tight_layout(rect=[0, 0.03, 1, 0.97])
        plt.savefig(base+"_pos_track.png");plt.close()

        # --- Velocity Tracking Plot ---
        fig_vel,(ax1_vel,ax2_vel,ax3_vel)=plt.subplots(3,1,figsize=(12,10),sharex=True)
        for ax,act,des,lbl in zip((ax1_vel,ax2_vel,ax3_vel),(Vx,Vy,Vz),(Vd[:,0],Vd[:,1],Vd[:,2]),('Vx','Vy','Vz')):
            ax.plot(t,act,'b-',label='Actual '+lbl);ax.plot(t,des,'r--',label='Desired '+lbl)
            ax.set_ylabel(lbl+' (m/s)');ax.grid();ax.legend()
        ax3_vel.set_xlabel('t (s)');fig_vel.suptitle("Velocity Tracking");fig_vel.tight_layout(rect=[0, 0.03, 1, 0.97])
        plt.savefig(base+"_vel_track.png");plt.close()

        # --- 3D Trajectory Plot ---
        from mpl_toolkits.mplot3d import Axes3D
        fig3d=plt.figure(figsize=(8,8));ax3d=fig3d.add_subplot(111,projection='3d')

        min_x_3d, max_x_3d = np.min(X), np.max(X)
        min_y_3d, max_y_3d = np.min(Y), np.max(Y)
        min_z_3d, max_z_3d = np.min(Z), np.max(Z)
        min_x_3d = min(min_x_3d, np.min(Pd[:,0])); max_x_3d = max(max_x_3d, np.max(Pd[:,0]))
        min_y_3d = min(min_y_3d, np.min(Pd[:,1])); max_y_3d = max(max_y_3d, np.max(Pd[:,1]))
        min_z_3d = min(min_z_3d, np.min(Pd[:,2])); max_z_3d = max(max_z_3d, np.max(Pd[:,2]))

        if self.obstacles:
            rospy.loginfo("Plotting %d obstacles in 3D.", len(self.obstacles))
            u = np.linspace(0, 2 * np.pi, 20)
            v = np.linspace(0, np.pi, 10)
            for obs in self.obstacles:
                try:
                    cx, cy, cz, r = map(float, obs[0:4])
                    x_surf = cx + r * np.outer(np.cos(u), np.sin(v))
                    y_surf = cy + r * np.outer(np.sin(u), np.sin(v))
                    z_surf = cz + r * np.outer(np.ones(np.size(u)), np.cos(v))
                    ax3d.plot_surface(x_surf, y_surf, z_surf, color='grey', alpha=0.4, linewidth=0, antialiased=False, rstride=1, cstride=1)
                    min_x_3d = min(min_x_3d, cx - r); max_x_3d = max(max_x_3d, cx + r)
                    min_y_3d = min(min_y_3d, cy - r); max_y_3d = max(max_y_3d, cy + r)
                    min_z_3d = min(min_z_3d, cz - r); max_z_3d = max(max_z_3d, cz + r)
                except Exception as e:
                     rospy.logwarn("Error processing obstacle %s for 3D plotting: %s. Skipping.", str(obs), e)

        ax3d.plot(X,Y,Z,'b-',label='Actual Trajectory');
        ax3d.plot(Pd[:,0],Pd[:,1],Pd[:,2],'r--',label='Desired Trajectory')
        ax3d.set_xlabel('X (m)');ax3d.set_ylabel('Y (m)');ax3d.set_zlabel('Z (m)');ax3d.grid(); ax3d.legend()
        ax3d.set_title('3D Trajectory')

        range_x_3d = max_x_3d - min_x_3d
        range_y_3d = max_y_3d - min_y_3d
        range_z_3d = max_z_3d - min_z_3d
        max_range_3d = max(range_x_3d, range_y_3d, range_z_3d, 1.0) * 1.1
        mid_x_3d = (max_x_3d + min_x_3d) / 2.0
        mid_y_3d = (max_y_3d + min_y_3d) / 2.0
        mid_z_3d = (max_z_3d + min_z_3d) / 2.0
        ax3d.set_xlim(mid_x_3d - max_range_3d / 2.0, mid_x_3d + max_range_3d / 2.0)
        ax3d.set_ylim(mid_y_3d - max_range_3d / 2.0, mid_y_3d + max_range_3d / 2.0)
        ax3d.set_zlim(mid_z_3d - max_range_3d / 2.0, mid_z_3d + max_range_3d / 2.0)

        plt.savefig(base+"_traj3d.png");plt.close(fig3d)

        # --- 2D Top-Down Plot ---
        fig2d, ax2d = plt.subplots(figsize=(8, 8))

        min_x_2d, max_x_2d = np.min(X), np.max(X)
        min_y_2d, max_y_2d = np.min(Y), np.max(Y)
        min_x_2d = min(min_x_2d, np.min(Pd[:,0])); max_x_2d = max(max_x_2d, np.max(Pd[:,0]))
        min_y_2d = min(min_y_2d, np.min(Pd[:,1])); max_y_2d = max(max_y_2d, np.max(Pd[:,1]))

        obstacle_circle_legend_added = False
        kinematic_traj_legend_added = False
        kinematic_start_legend_added = False
        kinematic_end_legend_added = False

        if self.obstacles:
            rospy.loginfo("Plotting %d obstacles in 2D.", len(self.obstacles))
            for obs_idx, obs in enumerate(self.obstacles):
                try:
                    cx, cy, _, r = map(float, obs[0:4])
                    label_circ = 'Obstacle Representation' if not obstacle_circle_legend_added else ""
                    circle = plt.Circle((cx, cy), r, color='grey', alpha=0.5, label=label_circ)
                    ax2d.add_patch(circle)
                    obstacle_circle_legend_added = True
                    min_x_2d = min(min_x_2d, cx - r); max_x_2d = max(max_x_2d, cx + r)
                    min_y_2d = min(min_y_2d, cy - r); max_y_2d = max(max_y_2d, cy + r)

                    if not (abs(obs[4]) < EPS and abs(obs[5]) < EPS and abs(obs[6]) < EPS):
                         ox, oy, oz, r, vx, vy, vz = map(float, obs)
                         Ox = ox + vx * t
                         Oy = oy + vy * t
                         label_traj = 'Kinematic Obstacle Traj.' if not kinematic_traj_legend_added else ""
                         ax2d.plot(Ox, Oy, 'm:', label=label_traj)
                         kinematic_traj_legend_added = True
                         label_start = 'Kin. Obs. Start' if not kinematic_start_legend_added else ""
                         ax2d.plot(ox, oy, 'mx', markersize=8, label=label_start)
                         kinematic_start_legend_added = True
                         label_end = 'Kin. Obs. End' if not kinematic_end_legend_added else ""
                         ax2d.plot(Ox[-1], Oy[-1], 'm^', markersize=8, label=label_end)
                         kinematic_end_legend_added = True
                         min_x_2d = min(min_x_2d, np.min(Ox)); max_x_2d = max(max_x_2d, np.max(Ox))
                         min_y_2d = min(min_y_2d, np.min(Oy)); max_y_2d = max(max_y_2d, np.max(Oy))
                except Exception as e:
                    rospy.logwarn("Error processing obstacle %s for 2D plotting: %s. Skipping.", str(obs), e)

        ax2d.plot(X, Y, 'b-', label='Actual Trajectory')
        ax2d.plot(Pd[:, 0], Pd[:, 1], 'r--', label='Desired Trajectory')

        ax2d.set_xlabel('X (m)'); ax2d.set_ylabel('Y (m)')
        ax2d.set_title('Top-Down Trajectory View')
        ax2d.grid(True); ax2d.legend(fontsize='small')
        ax2d.set_aspect('equal', adjustable='box')

        range_x_2d = max_x_2d - min_x_2d
        range_y_2d = max_y_2d - min_y_2d
        max_range_2d = max(range_x_2d, range_y_2d, 1.0) * 1.1
        mid_x_2d = (max_x_2d + min_x_2d) / 2.0
        mid_y_2d = (max_y_2d + min_y_2d) / 2.0
        ax2d.set_xlim(mid_x_2d - max_range_2d / 2.0, mid_x_2d + max_range_2d / 2.0)
        ax2d.set_ylim(mid_y_2d - max_range_2d / 2.0, mid_y_2d + max_range_2d / 2.0)

        plt.savefig(base + "_traj2d_topdown.png"); plt.close(fig2d)

        # --- Omega Plot ---
        if self.data['w_t'] and self.data['w']:
            Tw = np.array(self.data['w_t'])
            W_raw = np.stack(self.data['w'])
            # Filter based on main odom time 't' to keep data relevant
            valid_w_indices = (Tw >= t[0]) & (Tw <= t[-1])
            Tw_valid = Tw[valid_w_indices]
            W = np.sqrt(np.maximum(W_raw[valid_w_indices,:], 0))

            if self.use_filt and W.shape[0] >= self.fw_w:
                try:
                     for i in range(W.shape[1]): W[:,i]=FILT(W[:,i], self.fw_w, self.fp_w)
                except ValueError as e:
                     rospy.logwarn("Filtering omega failed (maybe too few points?): %s", e)

            if Tw_valid.size > 0 and W.size > 0:
                plt.figure(figsize=(12,6))
                for i in range(W.shape[1]): plt.plot(Tw_valid, W[:,i], label='M'+str(i+1))
                plt.ylim(*self.w_lim); plt.xlabel('t (s)'); plt.ylabel('Motor Speed ω (rad/s)')
                plt.title("Motor Speeds"); plt.legend(); plt.grid()
                plt.savefig(base+"_omega.png"); plt.close()
            else:
                 rospy.logwarn("Not enough valid omega data points to plot.")
        else:
             rospy.logwarn("No omega data recorded.")

        # --- Thrust Plot ---
        if self.data['u_t'] and self.data['u1']:
            Tu = np.array(self.data['u_t'])
            U_raw = np.array(self.data['u1'])
            valid_u_indices = (Tu >= t[0]) & (Tu <= t[-1])
            Tu_valid = Tu[valid_u_indices]
            U = U_raw[valid_u_indices]

            if self.use_filt and U.size >= self.fw_t:
                try:
                     U=FILT(U, self.fw_t, self.fp_t)
                except ValueError as e:
                     rospy.logwarn("Filtering thrust failed (maybe too few points?): %s", e)

            if Tu_valid.size > 0 and U.size > 0:
                plt.figure(figsize=(12,4));plt.plot(Tu_valid,U)
                plt.xlabel('t (s)');plt.ylabel('Total Thrust U1 (N)');plt.grid()
                plt.title("Total Thrust Command");
                plt.savefig(base+"_thrust.png");plt.close()
            else:
                 rospy.logwarn("Not enough valid thrust data points to plot.")
        else:
             rospy.logwarn("No thrust data recorded.")

        rospy.loginfo("Plots saved to %s_*",base)

if __name__=="__main__":
    rospy.init_node("trajectory_plotter_node",anonymous=True)
    try:
         Plotter(); rospy.spin()
    except rospy.ROSInterruptException:
         rospy.loginfo("Plotter node interrupted.")
    except Exception as e:
         rospy.logerr("Unhandled exception in Plotter node: %s", e)
         import traceback
         traceback.print_exc()