#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import unicode_literals
from __future__ import division

import ctypes
ctypes.CDLL('libX11.so', ctypes.RTLD_GLOBAL).XInitThreads()

import rospy, os, math, datetime, numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import matplotlib.widgets as mwidgets
import signal, tempfile, pickle
try:
    _orig_press   = mwidgets.SpanSelector.on_key_press
    _orig_release = mwidgets.SpanSelector.on_key_release

    def _safe_on_key_press(self, event):
        try:
            return _orig_press(self, event)
        except UnicodeDecodeError:
            return  # drop any non-ASCII key

    def _safe_on_key_release(self, event):
        try:
            return _orig_release(self, event)
        except UnicodeDecodeError:
            return

    mwidgets.SpanSelector.on_key_press   = _safe_on_key_press
    mwidgets.SpanSelector.on_key_release = _safe_on_key_release
except Exception:
    pass
from matplotlib.widgets import SpanSelector
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
        self.laps    = pget("helix_laps",10.0)
        self.omega_traj = pget("trajectory_omega",0.2)
        self.save_dir= pget("plot_save_dir",".")
        self.w_lim   = (pget("min_omega",0.),pget("max_omega",838.))
        self.fw_o,self.fp_o = pget("filter_window_odom",51),pget("filter_polyorder_odom",3)
        self.fw_w,self.fp_w = pget("filter_window_omega",31),pget("filter_polyorder_omega",3)
        self.fw_t,self.fp_t = pget("filter_window_thrust",31),pget("filter_polyorder_thrust",3)
        self.use_filt= pget("use_filtering",True)
        self.run_T   = pget("run_duration_secs",400.0)
        self.takeoff_duration = pget("takeoff_duration", 5.0)  # Duration of takeoff phase in seconds
        self.hover_duration = pget("hover_duration", 5.0)      # Duration of hover phase in seconds

        self.r0      = 0.5*self.d_start
        theta_tot   = self.laps*2.0*math.pi
        self.k_r    = (self.r0 - 0.5*self.d_end)/theta_tot
        self.k_z    = self.height/theta_tot

        obstacles_str = pget("static_obstacles", "[]")
        self.obstacles = []
        try:
            parsed_obstacles = ast.literal_eval(obstacles_str)
            if isinstance(parsed_obstacles, list):
                for obs in parsed_obstacles:
                    if isinstance(obs, (list, tuple)) and len(obs) == 4 and all(isinstance(n, (int, float)) for n in obs):
                        self.obstacles.append(obs)
                    else:
                         rospy.logwarn("Invalid obstacle format in static_obstacles: %s. Expected [x,y,z,radius]. Skipping.", str(obs))
            else:
                rospy.logwarn("Could not parse static_obstacles parameter, expected list format. Got: %s", obstacles_str)
        except (ValueError, SyntaxError) as e:
             rospy.logwarn("Error parsing static_obstacles parameter '%s': %s", obstacles_str, e)
        except Exception as e:
             rospy.logwarn("Unexpected error processing static_obstacles '%s': %s", obstacles_str, e)

        if self.obstacles:
            rospy.loginfo("Loaded %d static obstacles for plotting.", len(self.obstacles))

        self.t0   = None
        self.xy0  = None
        self.z0   = None
        self.data = {'t':[],'x':[],'y':[],'z':[],'vx':[],'vy':[],'vz':[],
                     'w_t':[],'w':[],'u_t':[],'u1':[], 'u2':[], 'u3':[], 'u4':[]}
        self.rec  = False
        self.done = False

        rospy.Subscriber(self.topics['state'],  String,            self.cb_state,  queue_size=2)
        rospy.Subscriber(self.topics['odom'],   Odometry,          self.cb_odom,   queue_size=200)
        rospy.Subscriber(self.topics['omega'],  Float64MultiArray, self.cb_omega,  queue_size=200)
        rospy.Subscriber(self.topics['thrust'], Float64MultiArray, self.cb_thrust, queue_size=200)
        # we will call shutdown() ourselves, so that the window
        # outlives any other node crashing
        # rospy.on_shutdown(self.shutdown)
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
            self.xy0 = np.array([p.x - self.r0, p.y])
            self.z0  = p.z
        t = (msg.header.stamp - self.t0).to_sec()
        self.data['t'].append(t)
        p=msg.pose.pose.position; q=msg.pose.pose.orientation
        v=msg.twist.twist.linear
        R=self.R(*euler_from_quaternion([q.x,q.y,q.z,q.w]))
        vw=np.dot(R,[v.x,v.y,v.z])
        self.data['x'].append(p.x)
        self.data['y'].append(p.y)
        self.data['z'].append(p.z)
        self.data['vx'].append(vw[0])
        self.data['vy'].append(vw[1])
        self.data['vz'].append(vw[2])
        if t>=self.run_T: self.shutdown()

    def cb_omega(self,msg):
        if self.rec and not self.done:
            self.data['w_t'].append(rospy.Time.now().to_sec())
            self.data['w'].append(np.array(msg.data))

    def cb_thrust(self,msg):
        if self.rec and not self.done:
            self.data['u_t'].append(rospy.Time.now().to_sec())
            if msg.data and len(msg.data) == 4:
                self.data['u1'].append(msg.data[0])
                self.data['u2'].append(msg.data[1])
                self.data['u3'].append(msg.data[2])
                self.data['u4'].append(msg.data[3])
            else:
                rospy.logwarn_throttle(10, "Thrust/Torque message received with unexpected data length or empty: %s", str(msg.data))
                self.data['u1'].append(0.0)
                self.data['u2'].append(0.0)
                self.data['u3'].append(0.0)
                self.data['u4'].append(0.0)

    @staticmethod
    def R(phi,th,psi):
        c,s = math.cos,math.sin
        return np.array([
            [c(th)*c(psi),  s(phi)*s(th)*c(psi)-c(phi)*s(psi),  c(phi)*s(th)*c(psi)+s(phi)*s(psi)],
            [c(th)*s(psi),  s(phi)*s(th)*s(psi)+c(phi)*c(psi),  c(phi)*s(th)*s(psi)-s(phi)*c(psi)],
            [   -s(th),                     s(phi)*c(th),                    c(phi)*c(th)]
        ])

    def desired(self,t):
        psit = self.omega_traj * t
        r = self.r0 - self.k_r*psit
        x = r*math.cos(psit) + self.xy0[0]
        y = r*math.sin(psit) + self.xy0[1]
        z = self.k_z*psit     + self.z0
        dr   = -self.k_r
        xp   = dr*math.cos(psit) - r*math.sin(psit)
        yp   = dr*math.sin(psit) + r*math.cos(psit)
        zp   = self.k_z
        vel  = np.array([xp,yp,zp]) * self.omega_traj
        return np.array([x,y,z]), vel

    def shutdown(self):
        if self.done: return
        self.done=True
        if self.t0 is None or not self.data['t']:
            rospy.logwarn("No data captured - nothing to plot."); return

        # Ensure matplotlib and plt are properly imported
        import matplotlib
        matplotlib.use('agg')  # Use non-interactive backend for static plots
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D

        t   = np.array(self.data['t'])
        X,Y,Z = map(np.array,(self.data['x'],self.data['y'],self.data['z']))
        Vx,Vy,Vz = map(np.array,(self.data['vx'],self.data['vy'],self.data['vz']))

        Pd,Vd = [],[]
        for ti in t:
            p_,v_ = self.desired(ti)
            Pd.append(p_); Vd.append(v_)
        Pd,Vd = np.stack(Pd), np.stack(Vd)

        if self.use_filt:
            X=FILT(X,self.fw_o,self.fp_o); Y=FILT(Y,self.fw_o,self.fp_o); Z=FILT(Z,self.fw_o,self.fp_o)
            Vx=FILT(Vx,self.fw_o,self.fp_o);Vy=FILT(Vy,self.fw_o,self.fp_o);Vz=FILT(Vz,self.fw_o,self.fp_o)

        rms=np.sqrt(np.mean((X-Pd[:,0])**2+(Y-Pd[:,1])**2+(Z-Pd[:,2])**2))
        rospy.loginfo("RMS 3-D position error: %.3f m",rms)

        ts  = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        base= os.path.join(self.save_dir,ts)
        if not os.path.isdir(self.save_dir): os.makedirs(self.save_dir)

        plt.figure(figsize=(12,6))
        plt.plot(t,X-Pd[:,0],label='ex');plt.plot(t,Y-Pd[:,1],label='ey');plt.plot(t,Z-Pd[:,2],label='ez')
        plt.legend();plt.grid();plt.xlabel('t (s)');plt.ylabel('error (m)')
        plt.savefig(base+"_pos_err.png");plt.close()

        fig,(ax1,ax2,ax3)=plt.subplots(3,1,figsize=(12,10),sharex=True)
        for ax,act,des,lbl in zip((ax1,ax2,ax3),(X,Y,Z),(Pd[:,0],Pd[:,1],Pd[:,2]),('X','Y','Z')):
            ax.plot(t,act,'b-',label='act '+lbl);ax.plot(t,des,'r--',label='des '+lbl)
            ax.set_ylabel(lbl);ax.grid();ax.legend()
        ax3.set_xlabel('t (s)');fig.tight_layout()
        plt.savefig(base+"_pos_track.png");plt.close()

        fig,(ax1,ax2,ax3)=plt.subplots(3,1,figsize=(12,10),sharex=True)
        for ax,act,des,lbl in zip((ax1,ax2,ax3),(Vx,Vy,Vz),(Vd[:,0],Vd[:,1],Vd[:,2]),('Vx','Vy','Vz')):
            ax.plot(t,act,'b-',label='act '+lbl);ax.plot(t,des,'r--',label='des '+lbl)
            ax.set_ylabel(lbl);ax.grid();ax.legend()
        ax3.set_xlabel('t (s)');fig.tight_layout()
        plt.savefig(base+"_vel_track.png");plt.close()

        fig3d=plt.figure(figsize=(8,8));ax3d=fig3d.add_subplot(111,projection='3d')

        min_x_3d, max_x_3d = np.min(np.concatenate((X, Pd[:,0]))), np.max(np.concatenate((X , Pd[:,0])))
        min_y_3d, max_y_3d = np.min(np.concatenate((Y, Pd[:,1]))), np.max(np.concatenate((Y, Pd[:,1])))
        min_z_3d, max_z_3d = np.min(np.concatenate((Z, Pd[:,2]))), np.max(np.concatenate((Z, Pd[:,2])))

        if self.obstacles:
            rospy.loginfo("Plotting %d obstacles.", len(self.obstacles))
            u = np.linspace(0, 2 * np.pi, 20)
            v = np.linspace(0, np.pi, 10)
            for obs in self.obstacles:
                try:
                    cx, cy, cz, r = map(float, obs)
                    x_surf = cx + r * np.outer(np.cos(u), np.sin(v))
                    y_surf = cy + r * np.outer(np.sin(u), np.sin(v))
                    z_surf = cz + r * np.outer(np.ones(np.size(u)), np.cos(v))
                    ax3d.plot_surface(x_surf, y_surf, z_surf, color='grey', alpha=0.4, linewidth=0, antialiased=False, rstride=1, cstride=1)
                    min_x_3d = min(min_x_3d, cx - r)
                    max_x_3d = max(max_x_3d, cx + r)
                    min_y_3d = min(min_y_3d, cy - r)
                    max_y_3d = max(max_y_3d, cy + r)
                    min_z_3d = min(min_z_3d, cz - r)
                    max_z_3d = max(max_z_3d, cz + r)
                except (TypeError, ValueError) as e:
                     rospy.logwarn("Error processing obstacle %s for 3D plotting: %s. Skipping.", str(obs), e)
        else:
            rospy.loginfo("No obstacles defined or loaded to plot.")

        ax3d.plot(X,Y,Z,'b-',label='Actual Trajectory');
        ax3d.plot(Pd[:,0],Pd[:,1],Pd[:,2],'r--',label='Desired Trajectory')
        # Add starting position marker (red cross)
        ax3d.scatter(X[0], Y[0], Z[0], c='red', marker='x', s=100, linewidth=3, label='Starting position')
        # Add ending position marker (green cross)
        ax3d.scatter(X[-1], Y[-1], Z[-1], c='green', marker='x', s=100, linewidth=3, label='Ending position')
        ax3d.set_xlabel('X (m)');ax3d.set_ylabel('Y (m)');ax3d.set_zlabel('Z (m)');ax3d.grid(); ax3d.legend()

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

        fig2d, ax2d = plt.subplots(figsize=(8, 8))
        min_x_2d, max_x_2d = np.min(np.concatenate((X, Pd[:,0]))), np.max(np.concatenate((X, Pd[:,0])))
        min_y_2d, max_y_2d = np.min(np.concatenate((Y, Pd[:,1]))), np.max(np.concatenate((Y, Pd[:,1])))

        obstacle_legend_added = False
        if self.obstacles:
            for obs in self.obstacles:
                try:
                    cx, cy, _, r = map(float, obs)
                    label = 'Obstacle' if not obstacle_legend_added else ""
                    circle = plt.Circle((cx, cy), r, color='grey', alpha=0.5, label=label)
                    ax2d.add_patch(circle)
                    obstacle_legend_added = True
                    min_x_2d = min(min_x_2d, cx - r)
                    max_x_2d = max(max_x_2d, cx + r)
                    min_y_2d = min(min_y_2d, cy - r)
                    max_y_2d = max(max_y_2d, cy + r)
                except (TypeError, ValueError) as e:
                    rospy.logwarn("Error processing obstacle %s for 2D plotting: %s. Skipping.", str(obs), e)

        ax2d.plot(X, Y, 'b-', label='Actual Trajectory')
        ax2d.plot(Pd[:, 0], Pd[:, 1], 'r--', label='Desired Trajectory')
        # Add starting position marker (red cross)
        ax2d.scatter(X[0], Y[0], c='red', marker='x', s=100, linewidth=3, label='Starting position')
        # Add ending position marker (green cross)
        ax2d.scatter(X[-1], Y[-1], c='green', marker='x', s=100, linewidth=3, label='Ending position')

        ax2d.set_xlabel('X (m)'); ax2d.set_ylabel('Y (m)')
        ax2d.set_title('Top-Down Trajectory View')
        ax2d.grid(True); ax2d.legend()
        ax2d.set_aspect('equal', adjustable='box')

        range_x_2d = max_x_2d - min_x_2d
        range_y_2d = max_y_2d - min_y_2d
        max_range_2d = max(range_x_2d, range_y_2d, 1.0) * 1.1

        mid_x_2d = (max_x_2d + min_x_2d) / 2.0
        mid_y_2d = (max_y_2d + min_y_2d) / 2.0

        ax2d.set_xlim(mid_x_2d - max_range_2d / 2.0, mid_x_2d + max_range_2d / 2.0)
        ax2d.set_ylim(mid_y_2d - max_range_2d / 2.0, mid_y_2d + max_range_2d / 2.0)

        plt.savefig(base + "_traj2d_topdown.png"); plt.close(fig2d)


        if self.data['w']:
            Tw=np.array(self.data['w_t'])-self.t0.to_sec()
            W=np.sqrt(np.maximum(np.stack(self.data['w']),0))
            if self.use_filt and W.shape[0]>=self.fw_w:
                for i in range(W.shape[1]):W[:,i]=FILT(W[:,i],self.fw_w,self.fp_w)
            
            # Create figure with non-interactive backend for saving
            fig, ax = plt.subplots(figsize=(12,6))
            for i in range(W.shape[1]):
                ax.plot(Tw, W[:,i], label='M'+str(i+1))
            
            ax.set_ylim(*self.w_lim)
            ax.set_xlabel('t (s)')
            ax.set_ylabel('omega (rad/s)')
            ax.grid(True)
            
            takeoff_end = self.takeoff_duration
            hover_end = takeoff_end + self.hover_duration
            ax.axvspan(0, takeoff_end, color='green',  alpha=0.1, label='Takeoff Phase')
            ax.axvspan(takeoff_end, hover_end, color='yellow', alpha=0.1, label='Hover Phase')
            ax.axvspan(hover_end, max(Tw), color='blue', alpha=0.1, label='Trajectory Phase')
            
            handles, labels = ax.get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            ax.legend(by_label.values(), by_label.keys())
            
            # Save the non-interactive version
            plt.savefig(base + "_omega.png")
            plt.close(fig)
            
            # Create a temporary Python script for interactive plotting with TkAgg backend
            fd, script_path = tempfile.mkstemp(suffix='.py')
            with os.fdopen(fd, 'w') as f:
                f.write('''#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import matplotlib
matplotlib.use('TkAgg')  # Use TkAgg for interactive plotting
import matplotlib.pyplot as plt
from matplotlib.widgets import SpanSelector
import signal
import os
import sys
import pickle

# Load data from pickle file
with open(sys.argv[1], 'rb') as f:
    data = pickle.load(f)
    
Tw = data['Tw']
W = data['W']
w_lim = data['w_lim']
takeoff_duration = data['takeoff_duration']
hover_duration = data['hover_duration']

# Create interactive plot
fig, ax = plt.subplots(figsize=(12,6))
for i in range(W.shape[1]):
    ax.plot(Tw, W[:,i], label='M'+str(i+1))

ax.set_ylim(*w_lim)
ax.set_xlabel('t (s)')
ax.set_ylabel('omega (rad/s)')
ax.grid(True)

takeoff_end = takeoff_duration
hover_end = takeoff_end + hover_duration
ax.axvspan(0, takeoff_end, color='green',  alpha=0.1, label='Takeoff Phase')
ax.axvspan(takeoff_end, hover_end, color='yellow', alpha=0.1, label='Hover Phase')
ax.axvspan(hover_end, max(Tw), color='blue', alpha=0.1, label='Trajectory Phase')

handles, labels = ax.get_legend_handles_labels()
by_label = dict(zip(labels, handles))
ax.legend(by_label.values(), by_label.keys())

def onselect(xmin, xmax):
    # skip zero-width spans (prevents singular x-lim warnings)
    if abs(xmax - xmin) < 1e-6:
        return
    lo, hi = sorted((xmin, xmax))
    ax.set_xlim(lo, hi)
    fig.canvas.draw_idle()

# horizontal drag to zoom
span = SpanSelector(ax, onselect, 'horizontal',
                    useblit=True,
                    rectprops=dict(alpha=0.3, facecolor='orange'))

# Ignore signals that might interrupt the plot
for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
    signal.signal(sig, signal.SIG_IGN)

plt.show()
''')
            os.chmod(script_path, 0o755)
            
            # Save data to pickle file
            fd, data_path = tempfile.mkstemp(suffix='.pkl')
            os.close(fd)
            plot_data = {
                'Tw': Tw,
                'W': W,
                'w_lim': self.w_lim,
                'takeoff_duration': self.takeoff_duration,
                'hover_duration': self.hover_duration
            }
            with open(data_path, 'wb') as f:
                pickle.dump(plot_data, f)
            
            # before we show, ignore the Ctrl-C / SIGTERM from roslaunch
            for sig in (signal.SIGINT, signal.SIGTERM):
                signal.signal(sig, signal.SIG_IGN)

            # == DETACH & SHOW in a new session ==
            # Double-fork so this plt.show() lives in its own process group
            try:
                pid = os.fork()
                if pid > 0:
                    # parent returns immediately and will be killed by roslaunch
                    return
            except OSError:
                pass

            # first child: become session leader
            os.setsid()
            try:
                pid2 = os.fork()
                if pid2 > 0:
                    # intermediate child exits
                    os._exit(0)
            except OSError:
                pass

            # grandchild: truly detached, ignore kill signals
            for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
                signal.signal(sig, signal.SIG_IGN)

            # Execute the interactive plot script
            os.execl(script_path, script_path, data_path)

        if self.data['u_t']:
            Tu = np.array(self.data['u_t']) - self.t0.to_sec()
            U1 = np.array(self.data['u1'])
            U2 = np.array(self.data['u2'])
            U3 = np.array(self.data['u3'])
            U4 = np.array(self.data['u4'])

            if self.use_filt and len(Tu) >= self.fw_t:
                U1 = FILT(U1, self.fw_t, self.fp_t)
                U2 = FILT(U2, self.fw_t, self.fp_t)
                U3 = FILT(U3, self.fw_t, self.fp_t)
                U4 = FILT(U4, self.fw_t, self.fp_t)

            fig, axs = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
            fig.suptitle('Control Inputs vs Time')

            # U1 - Thrust
            axs[0, 0].plot(Tu, U1, label='U1')
            axs[0, 0].set_ylabel('U1 (N)')
            axs[0, 0].grid(True)
            axs[0, 0].legend()

            # U2 - Roll Torque
            axs[0, 1].plot(Tu, U2, label='U2')
            axs[0, 1].set_ylabel('U2 (Nm)')
            axs[0, 1].grid(True)
            axs[0, 1].legend()

            # U3 - Pitch Torque
            axs[1, 0].plot(Tu, U3, label='U3')
            axs[1, 0].set_ylabel('U3 (Nm)')
            axs[1, 0].set_xlabel('t (s)')
            axs[1, 0].grid(True)
            axs[1, 0].legend()

            # U4 - Yaw Torque
            axs[1, 1].plot(Tu, U4, label='U4')
            axs[1, 1].set_ylabel('U4 (Nm)')
            axs[1, 1].set_xlabel('t (s)')
            axs[1, 1].grid(True)
            axs[1, 1].legend()

            fig.tight_layout(rect=[0, 0.03, 1, 0.95])
            plt.savefig(base + "_control_inputs.png")
            plt.close(fig)

        rospy.loginfo("Plots saved to %s_*",base)

if __name__=="__main__":
    rospy.init_node("trajectory_plotter_node",anonymous=True)
    plotter = Plotter()

    # keep looping until data collection is done (or ROS shuts down)
    rate = rospy.Rate(10)
    try:
        while not plotter.done:
            rate.sleep()
    except rospy.exceptions.ROSInterruptException:
        # ROS is shutting down; just continue to plotting
        pass

    # now that we're out of the loop, bring up the interactive plot
    plotter.shutdown()