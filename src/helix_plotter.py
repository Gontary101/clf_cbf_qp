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
        self.laps    = pget("helix_laps",10.0)
        self.omega_traj = pget("trajectory_omega",0.2)
        self.save_dir= pget("plot_save_dir",".")
        self.w_lim   = (pget("min_omega",0.),pget("max_omega",838.))
        self.fw_o,self.fp_o = pget("filter_window_odom",51),pget("filter_polyorder_odom",3)
        self.fw_w,self.fp_w = pget("filter_window_omega",31),pget("filter_polyorder_omega",3)
        self.fw_t,self.fp_t = pget("filter_window_thrust",31),pget("filter_polyorder_thrust",3)
        self.use_filt= pget("use_filtering",True)
        self.run_T   = pget("run_duration_secs",250.0)

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

        from mpl_toolkits.mplot3d import Axes3D
        fig3d=plt.figure(figsize=(8,8));ax3d=fig3d.add_subplot(111,projection='3d')

        min_x_3d, max_x_3d = np.min(np.concatenate((X, Pd[:,0]))), np.max(np.concatenate((X, Pd[:,0])))
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
        ax3d.set_xlabel('X');ax3d.set_ylabel('Y');ax3d.set_zlabel('Z');ax3d.grid(); ax3d.legend()

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
            plt.figure(figsize=(12,6))
            for i in range(W.shape[1]):plt.plot(Tw,W[:,i],label='M'+str(i+1))
            plt.ylim(*self.w_lim);plt.xlabel('t (s)');plt.ylabel('ω (rad/s)')
            plt.legend();plt.grid()
            plt.savefig(base+"_omega.png");plt.close()

        if self.data['u1']:
            Tu=np.array(self.data['u_t'])-self.t0.to_sec()
            U=np.array(self.data['u1'])
            if self.use_filt:U=FILT(U,self.fw_t,self.fp_t)
            plt.figure(figsize=(12,4));plt.plot(Tu,U)
            plt.xlabel('t (s)');plt.ylabel('U1 (N)');plt.grid()
            plt.savefig(base+"_thrust.png");plt.close()

        rospy.loginfo("Plots saved to %s_*",base)

if __name__=="__main__":
    rospy.init_node("trajectory_plotter_node",anonymous=True)
    Plotter(); rospy.spin()