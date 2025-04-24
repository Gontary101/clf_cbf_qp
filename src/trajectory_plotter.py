#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import division
import rospy, os, math, datetime, numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from scipy.signal import savgol_filter
from nav_msgs.msg import Odometry
from std_msgs.msg import Float64MultiArray, String
from geometry_msgs.msg import Point, Vector3
from tf.transformations import euler_from_quaternion

# --------------------------------------------------------------------------
clip   = np.clip
pget   = lambda n,d: rospy.get_param("~"+n,d)
FILT   = lambda x,w,o: savgol_filter(x,w,o) if len(x)>=w else x

class Plotter(object):
    def __init__(self):
        # Params ------------------------------------------------------------
        self.topics = dict(
            odom   = pget("odom_topic"  ,"/iris/ground_truth/odometry"),
            omega  = pget("omega_sq_topic" ,"/clf_iris_trajectory_controller/control/omega_sq"),
            thrust = pget("thrust_topic"   ,"/clf_iris_trajectory_controller/control/U"),
            state  = pget("state_topic"    ,"/clf_iris_trajectory_controller/control/state")
        )
        self.a, self.b  = pget("ellipse_a",31.), pget("ellipse_b",21.)
        self.z0         = pget("trajectory_z",2.)
        self.omega      = pget("trajectory_omega",.1)
        self.save_dir   = pget("plot_save_dir",".")
        self.w_lim      = (pget("min_omega",0.), pget("max_omega",838.))
        # filter windows (must be odd)
        self.fw_o, self.fp_o = pget("filter_window_odom",51), pget("filter_polyorder_odom",3)
        self.fw_w, self.fp_w = pget("filter_window_omega",31),pget("filter_polyorder_omega",3)
        self.fw_t, self.fp_t = pget("filter_window_thrust",31),pget("filter_polyorder_thrust",3)
        self.use_filt   = pget("use_filtering",True)
        self.run_T      = pget("run_duration_secs",150.)

        # Obstacles: list of (x, y, radius)
        self.obstacles = [
            (-30.87, -5.79, 0.50),   # small cylinder outer ellipse
            (-26.18, -11.22, 0.50),  # small box middle ellipse
            (-21.39, 12.84, 0.50),   # small sphere inner ellipse
            (-31.00, 0.00, 1.00),
            (0.0,  21,  0.80),
            (0.0,  -21,  0.80) # large cylinder middle ellipse
        ]

        # Data --------------------------------------------------------------
        self.t0   = None          # ROS time at first TRAJECTORY odom
        self.data = dict(
            t=[], x=[], y=[], z=[], vx=[], vy=[], vz=[],
            w_t=[], w =[],            # ω² time & values
            u_t=[], u1=[]             # thrust time & U1
        )
        self.rec  = False         # start recording
        self.done = False

        # Subscribers ------------------------------------------------------
        self.subs = []
        self.subs.append(rospy.Subscriber(self.topics['state'], String, self.cb_state, queue_size=2))
        self.subs.append(rospy.Subscriber(self.topics['odom'], Odometry, self.cb_odom, queue_size=200))
        self.subs.append(rospy.Subscriber(self.topics['omega'], Float64MultiArray, self.cb_omega, queue_size=200))
        self.subs.append(rospy.Subscriber(self.topics['thrust'], Float64MultiArray, self.cb_thrust, queue_size=200))
        rospy.on_shutdown(self.shutdown)
        rospy.loginfo("Trajectory Plotter ready – waiting for TRAJECTORY phase.")

    # ---------------------------- callbacks --------------------------------
    def cb_state(self, msg):
        if msg.data in ("TRAJ","TRAJECTORY") and not self.rec:
            self.rec = True
            rospy.loginfo("TRAJECTORY phase detected – recording starts now.")

    def cb_odom(self, msg):
        if not self.rec or self.done:
            return
        if self.t0 is None:
            self.t0 = msg.header.stamp
        t = (msg.header.stamp - self.t0).to_sec()
        self.data['t'].append(t)
        # position
        p = msg.pose.pose.position
        self.data['x'].append(p.x)
        self.data['y'].append(p.y)
        self.data['z'].append(p.z)
        # velocity in world frame
        q = msg.pose.pose.orientation
        v_body = msg.twist.twist.linear
        R = self.R(*euler_from_quaternion([q.x, q.y, q.z, q.w]))
        v_w = np.dot(R, [v_body.x, v_body.y, v_body.z])
        self.data['vx'].append(v_w[0])
        self.data['vy'].append(v_w[1])
        self.data['vz'].append(v_w[2])
        # stop at run_T
        if t >= self.run_T:
            self.shutdown()

    def cb_omega(self, msg):
        if self.rec and not self.done:
            now_sec = (rospy.Time.now() - self.t0).to_sec() if self.t0 else 0.0
            self.data['w_t'].append(now_sec)
            self.data['w'].append(np.array(msg.data))

    def cb_thrust(self, msg):
        if self.rec and not self.done:
            now_sec = (rospy.Time.now() - self.t0).to_sec() if self.t0 else 0.0
            self.data['u_t'].append(now_sec)
            self.data['u1'].append(msg.data[0] if msg.data else 0.0)

    # -------------------------- maths helpers ------------------------------
    @staticmethod
    def R(phi, th, psi):
        c, s = math.cos, math.sin
        return np.array([
            [ c(th)*c(psi), s(phi)*s(th)*c(psi)-c(phi)*s(psi), c(phi)*s(th)*c(psi)+s(phi)*s(psi)],
            [ c(th)*s(psi), s(phi)*s(th)*s(psi)+c(phi)*c(psi), c(phi)*s(th)*s(psi)-s(phi)*c(psi)],
            [-s(th),        s(phi)*c(th),                   c(phi)*c(th)]
        ])

    def desired(self, t):
        th = self.omega * t
        c, s = math.cos, math.sin
        pos = np.array([ self.a*c(th), self.b*s(th), self.z0])
        vel = np.array([-self.a*s(th), self.b*c(th), 0]) * self.omega
        return pos, vel

    # ---------------------------- plotting ---------------------------------
    def shutdown(self):
        if self.done:
            return
        self.done = True
        # Prevent further callbacks from modifying data during plotting
        for sub in self.subs:
            sub.unregister()

        rospy.loginfo("Plotter: generating figures …")
        if self.t0 is None or not self.data['t']:
            rospy.logwarn("No data captured – nothing to plot.")
            return

        # Convert to arrays
        t  = np.array(self.data['t'])
        X  = np.array(self.data['x']); Y  = np.array(self.data['y']); Z  = np.array(self.data['z'])
        Vx = np.array(self.data['vx']); Vy = np.array(self.data['vy']); Vz = np.array(self.data['vz'])

        # Desired reference over time
        Pd, Vd = [], []
        for ti in t:
            p_d, v_d = self.desired(ti)
            Pd.append(p_d); Vd.append(v_d)
        Pd, Vd = np.stack(Pd), np.stack(Vd)

        # Optionally filter position/velocity
        if self.use_filt:
            X = FILT(X, self.fw_o, self.fp_o)
            Y = FILT(Y, self.fw_o, self.fp_o)
            Z = FILT(Z, self.fw_o, self.fp_o)
            Vx = FILT(Vx, self.fw_o, self.fp_o)
            Vy = FILT(Vy, self.fw_o, self.fp_o)
            Vz = FILT(Vz, self.fw_o, self.fp_o)

        # RMS 3D error
        rms = np.sqrt(np.mean((X - Pd[:,0])**2 + (Y - Pd[:,1])**2 + (Z - Pd[:,2])**2))
        rospy.loginfo("RMS 3-D position error: %.3f m", rms)

        # Prepare save directory
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        base = os.path.join(self.save_dir, ts)
        if not os.path.isdir(self.save_dir):
            try:
                os.makedirs(self.save_dir)
            except OSError as e:
                 rospy.logerr("Could not create save directory {}: {}".format(self.save_dir, e))
                 return # Abort if directory cannot be created

        # 1) Position error over time
        plt.figure(figsize=(12,6))
        plt.plot(t, X-Pd[:,0], label='ex')
        plt.plot(t, Y-Pd[:,1], label='ey')
        plt.plot(t, Z-Pd[:,2], label='ez')
        plt.legend(); plt.grid(); plt.xlabel('t (s)'); plt.ylabel('error (m)')
        plt.title('Position tracking error')
        plt.savefig(base + "_pos_err.png")
        plt.close()

        # 2) Position tracking X, Y, Z
        fig, (ax1, ax2, ax3) = plt.subplots(3,1, figsize=(12,10), sharex=True)
        for ax, act, des, lbl in zip((ax1, ax2, ax3), (X, Y, Z), (Pd[:,0], Pd[:,1], Pd[:,2]), ('X','Y','Z')):
            ax.plot(t, act, 'b-', label='actual '+lbl)
            ax.plot(t, des, 'r--', label='desired '+lbl)
            ax.set_ylabel(lbl+' (m)'); ax.grid(); ax.legend()
        ax3.set_xlabel('t (s)')
        fig.tight_layout(); plt.savefig(base + "_pos_track.png"); plt.close()

        # 3) Velocity tracking Vx, Vy, Vz
        fig, (ax1, ax2, ax3) = plt.subplots(3,1, figsize=(12,10), sharex=True)
        for ax, act, des, lbl in zip((ax1, ax2, ax3), (Vx, Vy, Vz), (Vd[:,0], Vd[:,1], Vd[:,2]), ('Vx','Vy','Vz')):
            ax.plot(t, act, 'b-', label='actual '+lbl)
            ax.plot(t, des, 'r--', label='desired '+lbl)
            ax.set_ylabel(lbl+' (m/s)'); ax.grid(); ax.legend()
        ax3.set_xlabel('t (s)')
        fig.tight_layout(); plt.savefig(base + "_vel_track.png"); plt.close()

        # 4) 3D trajectory
        try:
            from mpl_toolkits.mplot3d import Axes3D
            fig = plt.figure(figsize=(8,8)); ax = fig.add_subplot(111, projection='3d')
            ax.plot(X, Y, Z, 'b-', label='actual')
            ax.plot(Pd[:,0], Pd[:,1], Pd[:,2], 'r--', label='desired')
            ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z'); ax.legend(); ax.grid()
            # equalize
            limx = ax.get_xlim3d(); limy = ax.get_ylim3d(); limz = ax.get_zlim3d()
            mr = 0.5*max(limx[1]-limx[0], limy[1]-limy[0], limz[1]-limz[0])
            mid = [(limx[1]+limx[0])*0.5, (limy[1]+limy[0])*0.5, (limz[1]+limz[0])*0.5]
            ax.set_xlim(mid[0]-mr, mid[0]+mr); ax.set_ylim(mid[1]-mr, mid[1]+mr); ax.set_zlim(mid[2]-mr, mid[2]+mr)
            plt.title('3-D trajectory'); plt.savefig(base + "_traj3d.png"); plt.close()
        except ImportError:
            rospy.logwarn("Could not import mpl_toolkits.mplot3d, skipping 3D plot.")
        except Exception as e:
            rospy.logerr("Error during 3D plotting: {}".format(e))
            plt.close() # Ensure plot is closed even if error occurs

        # 5) Motor speeds and thrust
        if self.data['w'] and self.data['w_t']:
            Tw = np.array(self.data['w_t'])
            # Ensure W is 2D even if only one sample exists
            W_raw = np.stack(self.data['w']) if len(self.data['w']) > 1 else np.array(self.data['w'])
            W = np.sqrt(np.maximum(W_raw, 0))
            if self.use_filt and W.shape[0] >= self.fw_w:
                for i in range(W.shape[1]):
                    W[:,i] = FILT(W[:,i], self.fw_w, self.fp_w)
            plt.figure(figsize=(12,6))
            # Handle case where W might still be 1D if only one motor or one timestep
            if W.ndim == 1:
                 plt.plot(Tw, W, label='M1')
            else:
                for i in range(W.shape[1]):
                    plt.plot(Tw, W[:,i], label='M'+str(i+1))
            plt.ylim(*self.w_lim); plt.xlabel('t (s)'); plt.ylabel(u'ω (rad/s)') # Use unicode literal
            plt.title('Motor speeds'); plt.grid(); plt.legend()
            plt.savefig(base + "_omega.png"); plt.close()
        if self.data['u1'] and self.data['u_t']:
            Tu = np.array(self.data['u_t'])
            U = np.array(self.data['u1'])
            if self.use_filt and len(U) >= self.fw_t: # Check length before filtering
                U = FILT(U, self.fw_t, self.fp_t)
            plt.figure(figsize=(12,4))
            plt.plot(Tu, U); plt.xlabel('t (s)'); plt.ylabel('U1 (N)')
            plt.title('Total thrust'); plt.grid(); plt.savefig(base + "_thrust.png"); plt.close()

        # 6) Top-down 2D trajectory with obstacles
        plt.figure(figsize=(8,8))
        ax = plt.gca() # Use plt.gca() for clarity
        theta = np.linspace(0, 2*math.pi, 200)
        xe = self.a * np.cos(theta); ye = self.b * np.sin(theta)
        ax.plot(xe, ye, 'r--', label='Desired ellipse')
        ax.plot(X, Y, 'b-', label='Actual path')
        for ox, oy, r in self.obstacles:
            # Ensure radius is positive before plotting
            if r > 0:
                circ = Circle((ox, oy), r, facecolor='none', edgecolor='k', linewidth=1.5, label='_nolegend_') # Add label to avoid auto-legend
                ax.add_patch(circ)
            else:
                 rospy.logwarn("Skipping obstacle with non-positive radius: ({}, {}, {})".format(ox, oy, r))
        ax.set_aspect('equal', adjustable='box') # Use 'adjustable'
        ax.set_xlabel('X (m)'); ax.set_ylabel('Y (m)')
        ax.set_title('Top-down trajectory and obstacles')
        ax.legend(); ax.grid(True)
        plt.savefig(base + "_traj2d.png"); plt.close()

        rospy.loginfo("Plots saved to %s_*.png", base)

# --------------------------------------------------------------------------
if __name__ == "__main__":
    try:
        rospy.init_node("trajectory_plotter_node", anonymous=True)
        plotter = Plotter()
        rospy.spin()
    except rospy.ROSInterruptException:
        rospy.loginfo("Trajectory plotter node interrupted.")
    except Exception as e:
        rospy.logerr("Unhandled exception in trajectory_plotter: %s", e)
    finally:
        # Ensure shutdown/plotting is called even if spin is interrupted early
        if 'plotter' in locals() and hasattr(plotter, 'shutdown') and not plotter.done:
             rospy.loginfo("Calling shutdown routine due to node termination.")
             plotter.shutdown()