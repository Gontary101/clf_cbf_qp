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
pget = lambda n,d: rospy.get_param("~"+n,d)
FILT = lambda x,w,o: savgol_filter(x,w,o) if len(x)>=w else x

class Plotter(object):
    def __init__(self):
        self.topics = dict(
            odom   = pget("odom_topic","/iris/ground_truth/odometry"),
            omega  = pget("omega_sq_topic","/clf_iris_trajectory_controller/control/omega_sq"),
            thrust = pget("thrust_topic","/clf_iris_trajectory_controller/control/U"),
            state  = pget("state_topic","/clf_iris_trajectory_controller/control/state")
        )
        self.d_start = pget("helix_start_diameter",40.0)
        self.d_end   = pget("helix_end_diameter",15.0)
        self.height  = pget("helix_height",30.0)
        self.laps    = pget("helix_laps",10.0) # Note: default was 4 in controller? Check consistency
        self.omega_traj = pget("trajectory_omega",0.1) # Note: default was 0.1 in controller
        self.save_dir= pget("plot_save_dir",".")
        self.w_lim   = (pget("min_omega",0.),pget("max_omega",838.))
        self.fw_o,self.fp_o = pget("filter_window_odom",51),pget("filter_polyorder_odom",3)
        self.fw_w,self.fp_w = pget("filter_window_omega",31),pget("filter_polyorder_omega",3)
        self.fw_t,self.fp_t = pget("filter_window_thrust",31),pget("filter_polyorder_thrust",3)
        self.use_filt= pget("use_filtering",True)
        self.run_T   = pget("run_duration_secs",200.0)

        self.r0      = 0.5*self.d_start
        theta_tot   = self.laps*2.0*math.pi
        self.k_r    = (self.r0 - 0.5*self.d_end)/theta_tot
        self.k_z    = self.height/theta_tot

        # Parse obstacles (accepts 4 or 7 element lists)
        obstacles_str = pget("static_obstacles", "[]")
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
                         rospy.logwarn("Invalid obstacle format in static_obstacles at index %d: %s. Expected [x,y,z,r] or [x,y,z,r,vx,vy,vz]. Skipping.", i, str(obs))
            else:
                rospy.logwarn("Could not parse static_obstacles parameter, expected list format. Got: %s", obstacles_str)
        except (ValueError, SyntaxError) as e:
             rospy.logwarn("Error parsing static_obstacles parameter '%s': %s", obstacles_str, e)
        except Exception as e:
             rospy.logwarn("Unexpected error processing static_obstacles '%s': %s", obstacles_str, e)

        if self.obstacles:
            rospy.loginfo("Loaded %d obstacles for plotting (static and kinematic).", len(self.obstacles))
        else:
            rospy.loginfo("No valid obstacles loaded for plotting.")


        self.t0   = None
        self.xy0  = None
        self.z0   = None
        self.data = {'t':[],'x':[],'y':[],'z':[],'vx':[],'vy':[],'vz':[],
                     'w_t':[],'w':[],'u_t':[],'u1':[]}
        self.rec  = False
        self.done = False

        rospy.Subscriber(self.topics['state'],  String,            self.cb_state,  queue_size=2)
        rospy.Subscriber(self.topics['odom'],   Odometry,          self.cb_odom,   queue_size=200)
        rospy.Subscriber(self.topics['omega'],  Float64MultiArray, self.cb_omega,  queue_size=200)
        rospy.Subscriber(self.topics['thrust'], Float64MultiArray, self.cb_thrust, queue_size=200)
        rospy.on_shutdown(self.shutdown)
        rospy.loginfo("Trajectory Plotter ready - waiting for TRAJECTORY phase.")

    def cb_state(self,msg):
        if msg.data=="TRAJ" and not self.rec:
            self.rec=True
            rospy.loginfo("TRAJECTORY phase detected - recording starts now.")

    def cb_odom(self,msg):
        if not self.rec or self.done: return
        if self.t0 is None:
            self.t0   = msg.header.stamp
            p       = msg.pose.pose.position
            # Calculate initial offsets based on first odom relative to theoretical start
            # This helps align desired path if drone doesn't start exactly at (r0, 0, 0) relative
            ref_x_start = self.r0 # Theoretical start X relative to center
            ref_y_start = 0.0    # Theoretical start Y relative to center
            ref_z_start = 0.0    # Theoretical start Z relative to center
            self.xy0 = np.array([p.x - ref_x_start, p.y - ref_y_start])
            self.z0  = p.z - ref_z_start
        t_sec = (msg.header.stamp - self.t0).to_sec()
        self.data['t'].append(t_sec)
        p=msg.pose.pose.position; q=msg.pose.pose.orientation
        v=msg.twist.twist.linear
        try:
             R=self.R(*euler_from_quaternion([q.x,q.y,q.z,q.w]))
             vw=np.dot(R,[v.x,v.y,v.z])
        except ValueError: # Handle potential quaternion issues
             rospy.logwarn_throttle(5.0, "Invalid quaternion in odom, using zero velocity.")
             vw = np.zeros(3)

        self.data['x'].append(p.x)
        self.data['y'].append(p.y)
        self.data['z'].append(p.z)
        self.data['vx'].append(vw[0])
        self.data['vy'].append(vw[1])
        self.data['vz'].append(vw[2])
        if t_sec>=self.run_T: self.shutdown()

    def cb_omega(self,msg):
        if self.rec and not self.done and self.t0 is not None: # Ensure t0 is set
            self.data['w_t'].append((rospy.Time.now() - self.t0).to_sec()) # Store relative time
            self.data['w'].append(np.array(msg.data))

    def cb_thrust(self,msg):
         if self.rec and not self.done and self.t0 is not None: # Ensure t0 is set
            self.data['u_t'].append((rospy.Time.now() - self.t0).to_sec()) # Store relative time
            self.data['u1'].append(msg.data[0] if msg.data else 0.0)

    @staticmethod
    def R(phi,th,psi):
        c,s = math.cos,math.sin
        return np.array([
            [c(th)*c(psi),  s(phi)*s(th)*c(psi)-c(phi)*s(psi),  c(phi)*s(th)*c(psi)+s(phi)*s(psi)],
            [c(th)*s(psi),  s(phi)*s(th)*s(psi)+c(phi)*c(psi),  c(phi)*s(th)*s(psi)-s(phi)*c(psi)],
            [   -s(th),                     s(phi)*c(th),                    c(phi)*c(th)]
        ])

    def desired(self,t):
        # Ensure t is treated as a numpy array for vectorized calculations
        t_np = np.asarray(t)
        psit = self.omega_traj * t_np
        r = self.r0 - self.k_r*psit
        x = r*np.cos(psit) + (self.xy0[0] if self.xy0 is not None else 0.0)
        y = r*np.sin(psit) + (self.xy0[1] if self.xy0 is not None else 0.0)
        z = self.k_z*psit     + (self.z0 if self.z0 is not None else 0.0)
        dr   = -self.k_r # Constant
        xp   = dr*np.cos(psit) - r*np.sin(psit)
        yp   = dr*np.sin(psit) + r*np.cos(psit)
        zp   = self.k_z # Constant
        vel  = np.vstack([xp,yp,zp]).T * self.omega_traj # Stack columns then multiply
        pos = np.vstack([x,y,z]).T # Stack columns
        # Handle scalar input case
        if pos.shape == (3,): pos = pos.reshape(1,3)
        if vel.shape == (3,): vel = vel.reshape(1,3)
        return pos, vel

    def shutdown(self):
        if self.done: return
        self.done=True
        if self.t0 is None or not self.data['t']:
            rospy.logwarn("No data captured - nothing to plot."); return

        t   = np.array(self.data['t'])
        X,Y,Z = map(np.array,(self.data['x'],self.data['y'],self.data['z']))
        Vx,Vy,Vz = map(np.array,(self.data['vx'],self.data['vy'],self.data['vz']))

        # Regenerate desired path using the actual time vector 't'
        # Ensure desired returns consistent shapes
        Pd_list, Vd_list = [], []
        Pd_calc, Vd_calc = self.desired(t)
        if Pd_calc.shape[0] != len(t):
             rospy.logwarn("Mismatch between desired path length and time vector!")
             # Fallback or error handling needed here
             return
        Pd = Pd_calc
        Vd = Vd_calc

        # --- Filtering ---
        if self.use_filt:
            # Ensure enough data points for filter window
            w = self.fw_o
            if len(X) >= w: X = FILT(X, w, self.fp_o)
            if len(Y) >= w: Y = FILT(Y, w, self.fp_o)
            if len(Z) >= w: Z = FILT(Z, w, self.fp_o)
            if len(Vx) >= w: Vx = FILT(Vx, w, self.fp_o)
            if len(Vy) >= w: Vy = FILT(Vy, w, self.fp_o)
            if len(Vz) >= w: Vz = FILT(Vz, w, self.fp_o)
        else:
            # Ensure variables exist even if filtering is off
            pass

        rms=np.sqrt(np.mean((X-Pd[:,0])**2+(Y-Pd[:,1])**2+(Z-Pd[:,2])**2))
        rospy.loginfo("RMS 3-D position error: %.3f m",rms)

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

        # Initialize plot limits with drone data
        min_x_3d, max_x_3d = np.min(X), np.max(X)
        min_y_3d, max_y_3d = np.min(Y), np.max(Y)
        min_z_3d, max_z_3d = np.min(Z), np.max(Z)
        min_x_3d = min(min_x_3d, np.min(Pd[:,0])); max_x_3d = max(max_x_3d, np.max(Pd[:,0]))
        min_y_3d = min(min_y_3d, np.min(Pd[:,1])); max_y_3d = max(max_y_3d, np.max(Pd[:,1]))
        min_z_3d = min(min_z_3d, np.min(Pd[:,2])); max_z_3d = max(max_z_3d, np.max(Pd[:,2]))

        # Plot obstacles and update limits
        if self.obstacles:
            rospy.loginfo("Plotting %d obstacles in 3D.", len(self.obstacles))
            u = np.linspace(0, 2 * np.pi, 20)
            v = np.linspace(0, np.pi, 10)
            for obs in self.obstacles:
                try:
                    # All obstacles have 7 elements now (static ones have zero vel)
                    cx, cy, cz, r = map(float, obs[0:4])
                    x_surf = cx + r * np.outer(np.cos(u), np.sin(v))
                    y_surf = cy + r * np.outer(np.sin(u), np.sin(v))
                    z_surf = cz + r * np.outer(np.ones(np.size(u)), np.cos(v))
                    ax3d.plot_surface(x_surf, y_surf, z_surf, color='grey', alpha=0.4, linewidth=0, antialiased=False, rstride=1, cstride=1)
                    # Update limits based on obstacle position and radius
                    min_x_3d = min(min_x_3d, cx - r); max_x_3d = max(max_x_3d, cx + r)
                    min_y_3d = min(min_y_3d, cy - r); max_y_3d = max(max_y_3d, cy + r)
                    min_z_3d = min(min_z_3d, cz - r); max_z_3d = max(max_z_3d, cz + r)
                except Exception as e:
                     rospy.logwarn("Error processing obstacle %s for 3D plotting: %s. Skipping.", str(obs), e)

        ax3d.plot(X,Y,Z,'b-',label='Actual Trajectory');
        ax3d.plot(Pd[:,0],Pd[:,1],Pd[:,2],'r--',label='Desired Trajectory')
        ax3d.set_xlabel('X (m)');ax3d.set_ylabel('Y (m)');ax3d.set_zlabel('Z (m)');ax3d.grid(); ax3d.legend()
        ax3d.set_title('3D Trajectory')

        # Set equal aspect ratio for 3D plot
        range_x_3d = max_x_3d - min_x_3d
        range_y_3d = max_y_3d - min_y_3d
        range_z_3d = max_z_3d - min_z_3d
        max_range_3d = max(range_x_3d, range_y_3d, range_z_3d, 1.0) * 1.1 # Ensure min range
        mid_x_3d = (max_x_3d + min_x_3d) / 2.0
        mid_y_3d = (max_y_3d + min_y_3d) / 2.0
        mid_z_3d = (max_z_3d + min_z_3d) / 2.0
        ax3d.set_xlim(mid_x_3d - max_range_3d / 2.0, mid_x_3d + max_range_3d / 2.0)
        ax3d.set_ylim(mid_y_3d - max_range_3d / 2.0, mid_y_3d + max_range_3d / 2.0)
        ax3d.set_zlim(mid_z_3d - max_range_3d / 2.0, mid_z_3d + max_range_3d / 2.0)

        plt.savefig(base+"_traj3d.png");plt.close(fig3d)

        # --- 2D Top-Down Plot ---
        fig2d, ax2d = plt.subplots(figsize=(8, 8))

        # Initialize plot limits with drone data
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
                    # Plot obstacle circle representation (static or kinematic)
                    cx, cy, _, r = map(float, obs[0:4])
                    label_circ = 'Obstacle Representation' if not obstacle_circle_legend_added else ""
                    circle = plt.Circle((cx, cy), r, color='grey', alpha=0.5, label=label_circ)
                    ax2d.add_patch(circle)
                    obstacle_circle_legend_added = True
                    # Update limits based on circle
                    min_x_2d = min(min_x_2d, cx - r); max_x_2d = max(max_x_2d, cx + r)
                    min_y_2d = min(min_y_2d, cy - r); max_y_2d = max(max_y_2d, cy + r)

                    # Check if kinematic (has velocity info)
                    if not (abs(obs[4]) < 1e-6 and abs(obs[5]) < 1e-6 and abs(obs[6]) < 1e-6):
                         ox, oy, oz, r, vx, vy, vz = map(float, obs)
                         # Calculate kinematic trajectory for the duration 't'
                         Ox = ox + vx * t
                         Oy = oy + vy * t
                         # Plot kinematic trajectory
                         label_traj = 'Kinematic Obstacle Traj.' if not kinematic_traj_legend_added else ""
                         ax2d.plot(Ox, Oy, 'm:', label=label_traj) # Magenta dotted line
                         kinematic_traj_legend_added = True
                         # Plot start marker
                         label_start = 'Kin. Obs. Start' if not kinematic_start_legend_added else ""
                         ax2d.plot(ox, oy, 'mx', markersize=8, label=label_start)
                         kinematic_start_legend_added = True
                         # Plot end marker
                         label_end = 'Kin. Obs. End' if not kinematic_end_legend_added else ""
                         ax2d.plot(Ox[-1], Oy[-1], 'm^', markersize=8, label=label_end)
                         kinematic_end_legend_added = True
                         # Update limits based on trajectory
                         min_x_2d = min(min_x_2d, np.min(Ox)); max_x_2d = max(max_x_2d, np.max(Ox))
                         min_y_2d = min(min_y_2d, np.min(Oy)); max_y_2d = max(max_y_2d, np.max(Oy))

                except Exception as e:
                    rospy.logwarn("Error processing obstacle %s for 2D plotting: %s. Skipping.", str(obs), e)

        # Plot drone trajectories
        ax2d.plot(X, Y, 'b-', label='Actual Trajectory')
        ax2d.plot(Pd[:, 0], Pd[:, 1], 'r--', label='Desired Trajectory')

        # Finalize 2D plot
        ax2d.set_xlabel('X (m)'); ax2d.set_ylabel('Y (m)')
        ax2d.set_title('Top-Down Trajectory View')
        ax2d.grid(True); ax2d.legend(fontsize='small') # Adjust legend size if needed
        ax2d.set_aspect('equal', adjustable='box')

        # Set final limits for 2D plot
        range_x_2d = max_x_2d - min_x_2d
        range_y_2d = max_y_2d - min_y_2d
        max_range_2d = max(range_x_2d, range_y_2d, 1.0) * 1.1 # Ensure min range, add padding
        mid_x_2d = (max_x_2d + min_x_2d) / 2.0
        mid_y_2d = (max_y_2d + min_y_2d) / 2.0
        ax2d.set_xlim(mid_x_2d - max_range_2d / 2.0, mid_x_2d + max_range_2d / 2.0)
        ax2d.set_ylim(mid_y_2d - max_range_2d / 2.0, mid_y_2d + max_range_2d / 2.0)

        plt.savefig(base + "_traj2d_topdown.png"); plt.close(fig2d)

        # --- Omega Plot ---
        if self.data['w']:
            # Ensure time vector aligns, use relative time
            Tw = np.array(self.data['w_t'])
            # Filter out potential outliers or times outside main traj time
            valid_w_indices = (Tw >= t[0]) & (Tw <= t[-1])
            Tw_valid = Tw[valid_w_indices]
            W_raw = np.stack(self.data['w'])
            W = np.sqrt(np.maximum(W_raw[valid_w_indices,:], 0)) # Use only valid indices

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

        # --- Thrust Plot ---
        if self.data['u1']:
            # Ensure time vector aligns, use relative time
            Tu = np.array(self.data['u_t'])
            valid_u_indices = (Tu >= t[0]) & (Tu <= t[-1])
            Tu_valid = Tu[valid_u_indices]
            U_raw = np.array(self.data['u1'])
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