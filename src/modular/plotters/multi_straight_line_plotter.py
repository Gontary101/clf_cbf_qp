#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import unicode_literals
from __future__ import division
import rospy, os, math, datetime, numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.signal import savgol_filter
from gazebo_msgs.msg import ModelStates
from std_msgs.msg import Float64MultiArray, String
import ast

pget = lambda n,d: rospy.get_param("~"+n,d)
FILT = lambda x,w,o: savgol_filter(x,w,o) if len(x)>=w and w > 0 and o < w and w % 2 != 0 else x

class MultiDronePlotter(object):
    def __init__(self):
        all_ns_str = pget("all_drone_namespaces", "[]")
        try:
            self.all_drone_namespaces = ast.literal_eval(all_ns_str)
            is_py2 = hasattr(__builtins__, 'basestring')
            if not isinstance(self.all_drone_namespaces, list) or \
               not all(isinstance(ns, basestring if is_py2 else str) for ns in self.all_drone_namespaces):
                raise ValueError("all_drone_namespaces must be a list of strings.")
        except (ValueError, SyntaxError) as e:
            rospy.logerr("Plotter: Invalid all_drone_namespaces parameter '%s': %s. Shutting down.", all_ns_str, e)
            rospy.signal_shutdown("Config error: all_drone_namespaces")
            return
        
        if not self.all_drone_namespaces:
            rospy.logerr("Plotter: No drone namespaces provided in 'all_drone_namespaces'. Shutting down.")
            rospy.signal_shutdown("Config error: no drone namespaces")
            return
        rospy.loginfo("Plotter tracking drones: %s", self.all_drone_namespaces)

        self.drone_data = {ns: self._init_drone_data_dict() for ns in self.all_drone_namespaces}
        self.drone_trajectories = {}
        
        common_duration_str = pget("common_trajectory_duration", "40.0")
        controller_node_name = "clf_iris_trajectory_controller"

        for ns in self.all_drone_namespaces:
            start_param_name = "/{}/{}/trajectory_start_point".format(ns, controller_node_name)
            end_param_name = "/{}/{}/trajectory_end_point".format(ns, controller_node_name)
            duration_param_name = "/{}/{}/trajectory_duration".format(ns, controller_node_name)
            
            default_start_str = "[0.0, 0.0, 3.0]"
            default_end_str = "[10.0, 0.0, 3.0]"
            
            start_str = default_start_str
            end_str = default_end_str
            duration_val = 40.0 

            try:
                start_str = rospy.get_param(start_param_name)
            except KeyError:
                rospy.logwarn("Start point for {} not found at {}. Using default: {}".format(ns, start_param_name, default_start_str))

            try:
                end_str = rospy.get_param(end_param_name)
            except KeyError:
                rospy.logwarn("End point for {} not found at {}. Using default: {}".format(ns, end_param_name, default_end_str))
            
            try:
                duration_val = float(rospy.get_param(duration_param_name))
            except KeyError:
                try:
                    duration_val = float(common_duration_str)
                    rospy.logwarn("Duration for {} not found at {}, using common_trajectory_duration from plotter: {}".format(ns, duration_param_name, duration_val))
                except ValueError:
                    rospy.logerr("Invalid common_trajectory_duration: {}. Using default 40.0 for {}".format(common_duration_str, ns))
                    duration_val = 40.0
            except ValueError:
                 rospy.logwarn("Invalid duration value for {} at {}. Using common or default.".format(ns, duration_param_name))
                 try:
                    duration_val = float(common_duration_str)
                 except ValueError:
                    duration_val = 40.0
            
            try:
                line_start = np.array(ast.literal_eval(start_str), dtype=float)
                line_end = np.array(ast.literal_eval(end_str), dtype=float)
                if line_start.shape != (3,) or line_end.shape != (3,):
                    raise ValueError("Parsed line_start/line_end is not a 3-element list.")
                self.drone_trajectories[ns] = {
                    'start': line_start, 'end': line_end, 'duration': float(duration_val)
                }
                rospy.loginfo("Trajectory for {}: Start={}, End={}, Duration={}s".format(ns, line_start, line_end, float(duration_val)))
            except (ValueError, SyntaxError) as e:
                rospy.logerr("Error parsing trajectory for {} (start_str='{}', end_str='{}'): {}. Using defaults.".format(ns, start_str, end_str, e))
                self.drone_trajectories[ns] = {
                    'start': np.array(ast.literal_eval(default_start_str), dtype=float),
                    'end': np.array(ast.literal_eval(default_end_str), dtype=float),
                    'duration': 40.0
                }
        
        self.save_dir = pget("plot_save_dir", ".")
        self.w_lim = (pget("min_omega",0.), pget("max_omega",838.))
        self.fw_o, self.fp_o = pget("filter_window_odom",51), pget("filter_polyorder_odom",3)
        self.fw_w, self.fp_w = pget("filter_window_omega",31), pget("filter_polyorder_omega",3)
        self.fw_t, self.fp_t = pget("filter_window_thrust",31), pget("filter_polyorder_thrust",3)
        self.use_filt = pget("use_filtering",True)
        self.run_T = pget("run_duration_secs", 250.0)

        obstacles_str = pget("static_obstacles", "[]")
        self.obstacles = []
        try:
            parsed_obstacles = ast.literal_eval(obstacles_str)
            if isinstance(parsed_obstacles, list):
                for obs in parsed_obstacles:
                    if isinstance(obs, (list, tuple)) and len(obs) == 4 and all(isinstance(n, (int, float)) for n in obs):
                        self.obstacles.append(obs)
                    elif isinstance(obs, (list, tuple)) and len(obs) == 10 and all(isinstance(n, (int, float)) for n in obs[:4]): 
                        self.obstacles.append([obs[0], obs[1], obs[2], obs[9]])
                    else:
                        rospy.logwarn("Invalid obstacle format: %s. Expected [x,y,z,radius] or 10-elem CBF. Skipping.", str(obs))
            else:
                rospy.logwarn("Could not parse static_obstacles. Expected list. Got: %s", obstacles_str)
        except (ValueError, SyntaxError) as e:
            rospy.logwarn("Error parsing static_obstacles '%s': %s", obstacles_str, e)
        if self.obstacles:
            rospy.loginfo("Loaded %d static obstacles for plotting.", len(self.obstacles))

        self.t0_rec = None
        self.rec_started_by_drone = None
        self.rec = False
        self.done = False

        if len(self.all_drone_namespaces) > 0:
             cmap_name = 'gist_rainbow' 
             try: 
                 cmap = plt.colormaps.get_cmap(cmap_name, len(self.all_drone_namespaces))
             except AttributeError:
                 cmap = plt.cm.get_cmap(cmap_name, len(self.all_drone_namespaces))
             self.drone_colors = [cmap(i) for i in range(len(self.all_drone_namespaces))]
        else:
             self.drone_colors = []

        self.state_subs = {}
        self.omega_subs = {}
        self.thrust_subs = {}

        state_topic_template = pget("state_topic_template", "/{ns}/clf_iris_trajectory_controller/control/state")
        omega_topic_template = pget("omega_sq_topic_template", "/{ns}/clf_iris_trajectory_controller/control/omega_sq")
        thrust_topic_template = pget("thrust_topic_template", "/{ns}/clf_iris_trajectory_controller/control/U")
        model_states_topic = pget("model_states_topic", "/gazebo/model_states")

        for ns in self.all_drone_namespaces:
            self.state_subs[ns] = rospy.Subscriber(state_topic_template.format(ns=ns), String, self.cb_state, callback_args=ns, queue_size=2)
            self.omega_subs[ns] = rospy.Subscriber(omega_topic_template.format(ns=ns), Float64MultiArray, self.cb_omega, callback_args=ns, queue_size=200)
            self.thrust_subs[ns] = rospy.Subscriber(thrust_topic_template.format(ns=ns), Float64MultiArray, self.cb_thrust, callback_args=ns, queue_size=200)
        
        self.model_states_sub = rospy.Subscriber(model_states_topic, ModelStates, self.cb_model_states, queue_size=1, buff_size=2**24)

        rospy.on_shutdown(self.shutdown)
        rospy.loginfo("Multi-Drone Trajectory Plotter ready - waiting for TRAJECTORY phase.")

    def _init_drone_data_dict(self):
        return {'t':[],'x':[],'y':[],'z':[],'vx':[],'vy':[],'vz':[],
                'w_t':[],'w':[],'u_t':[],'u1':[], 'u2':[], 'u3':[], 'u4':[]}

    def cb_state(self, msg, drone_ns):
        if msg.data == "TRAJ" and not self.rec:
            self.rec = True
            self.t0_rec = rospy.Time.now()
            self.rec_started_by_drone = drone_ns
            rospy.loginfo("TRAJECTORY phase detected by {} - recording starts now (t0_rec={}).".format(drone_ns, self.t0_rec.to_sec()))

    def cb_model_states(self, msg):
        if not self.rec or self.done or self.t0_rec is None: return

        current_ros_time = rospy.Time.now()
        t_relative = (current_ros_time - self.t0_rec).to_sec()

        if t_relative < -0.01: 
            rospy.logwarn_throttle(5.0, "Negative relative time {:.3f}s. t0_rec={}, msg_stamp={}. Re-aligning t0_rec.".format(
                t_relative, self.t0_rec.to_sec(), current_ros_time.to_sec()))
            self.t0_rec = current_ros_time 
            t_relative = 0.0

        for i, model_name in enumerate(msg.name):
            if model_name in self.all_drone_namespaces:
                drone_ns = model_name
                data = self.drone_data[drone_ns]
                data['t'].append(t_relative)
                p = msg.pose[i].position
                v_world = msg.twist[i].linear
                data['x'].append(p.x); data['y'].append(p.y); data['z'].append(p.z)
                data['vx'].append(v_world.x); data['vy'].append(v_world.y); data['vz'].append(v_world.z)
        
        if t_relative >= self.run_T:
            rospy.loginfo("Run duration {}s reached. Shutting down plotter.".format(self.run_T))
            self.shutdown()

    def cb_omega(self, msg, drone_ns):
        if self.rec and not self.done and self.t0_rec is not None:
            t_relative = (rospy.Time.now() - self.t0_rec).to_sec()
            if t_relative < 0: return 
            self.drone_data[drone_ns]['w_t'].append(t_relative)
            self.drone_data[drone_ns]['w'].append(np.array(msg.data))

    def cb_thrust(self, msg, drone_ns):
        if self.rec and not self.done and self.t0_rec is not None:
            t_relative = (rospy.Time.now() - self.t0_rec).to_sec()
            if t_relative < 0: return
            self.drone_data[drone_ns]['u_t'].append(t_relative)
            if msg.data and len(msg.data) == 4:
                self.drone_data[drone_ns]['u1'].append(msg.data[0])
                self.drone_data[drone_ns]['u2'].append(msg.data[1])
                self.drone_data[drone_ns]['u3'].append(msg.data[2])
                self.drone_data[drone_ns]['u4'].append(msg.data[3])
            else:
                rospy.logwarn_throttle(10, "Thrust/Torque msg for {} with unexpected data: {}".format(drone_ns, msg.data))
                for key in ['u1','u2','u3','u4']: self.drone_data[drone_ns][key].append(0.0)
    
    def desired(self, t, drone_ns):
        traj_params = self.drone_trajectories[drone_ns]
        start, end, duration = traj_params['start'], traj_params['end'], traj_params['duration']
        if duration <= 1e-6:
            pos = start 
            vel = np.zeros(3)
        else:
            t_norm = np.clip(t / duration, 0.0, 1.0)
            pos = start + (end - start) * t_norm
            if 0.0 <= t < duration :
                vel = (end - start) / duration
            else:
                vel = np.zeros(3)
        return pos, vel

    def _get_eff_filter_window(self, w_nominal, p_order):
        w_eff = w_nominal if w_nominal % 2 != 0 else w_nominal + 1
        if w_eff <= p_order:
             w_eff = p_order + 1 + ( (p_order + 1) % 2) 
        return w_eff

    def shutdown(self):
        if self.done: return
        self.done = True

        if self.t0_rec is None or not any(self.drone_data[ns]['t'] for ns in self.all_drone_namespaces):
            rospy.logwarn("No data captured for any drone - nothing to plot.")
            return

        rospy.loginfo("Shutdown called. Processing and saving plots...")
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        base_save_path = os.path.join(self.save_dir, ts)
        if not os.path.isdir(self.save_dir): 
            try: os.makedirs(self.save_dir)
            except OSError as e: rospy.logerr("Could not create save directory {}: {}".format(self.save_dir, e)); return

        num_drones = len(self.all_drone_namespaces)
        if num_drones == 0: return

        for ns_idx, ns in enumerate(self.all_drone_namespaces):
            data_ns = self.drone_data[ns]
            if not data_ns['t']:
                rospy.logwarn("No odometry data for drone {}. Skipping its plots.".format(ns))
                continue

            t_arr = np.array(data_ns['t'])
            X, Y, Z = np.array(data_ns['x']), np.array(data_ns['y']), np.array(data_ns['z'])
            Vx, Vy, Vz = np.array(data_ns['vx']), np.array(data_ns['vy']), np.array(data_ns['vz'])

            if self.use_filt:
                fw_o_eff = self._get_eff_filter_window(self.fw_o, self.fp_o)
                X = FILT(X, fw_o_eff, self.fp_o); Y = FILT(Y, fw_o_eff, self.fp_o); Z = FILT(Z, fw_o_eff, self.fp_o)
                Vx = FILT(Vx, fw_o_eff, self.fp_o); Vy = FILT(Vy, fw_o_eff, self.fp_o); Vz = FILT(Vz, fw_o_eff, self.fp_o)
            
            data_ns['t_arr'] = t_arr
            data_ns['X_f'], data_ns['Y_f'], data_ns['Z_f'] = X, Y, Z
            data_ns['Vx_f'], data_ns['Vy_f'], data_ns['Vz_f'] = Vx, Vy, Vz

            Pd_list, Vd_list = [], []
            for ti in t_arr:
                p_d, v_d = self.desired(ti, ns)
                Pd_list.append(p_d); Vd_list.append(v_d)
            data_ns['Pd'] = np.array(Pd_list); data_ns['Vd'] = np.array(Vd_list)
            if len(X) > 0 and len(data_ns['Pd']) == len(X):
                 rms_val = np.sqrt(np.mean((X-data_ns['Pd'][:,0])**2 + (Y-data_ns['Pd'][:,1])**2 + (Z-data_ns['Pd'][:,2])**2))
                 rospy.loginfo("RMS 3D position error for {}: {:.3f} m".format(ns, rms_val))
            else:
                 rospy.logwarn("Could not compute RMS for {} due to data mismatch (X len: {}, Pd len: {}).".format(ns, len(X), len(data_ns['Pd'])))


        cols = int(math.ceil(math.sqrt(num_drones)))
        rows = int(math.ceil(num_drones / cols))

        fig3d = plt.figure(figsize=(12,10)); ax3d = fig3d.add_subplot(111, projection='3d')
        all_coords = []
        for ns_idx, ns in enumerate(self.all_drone_namespaces):
            data_ns = self.drone_data[ns]
            if 'X_f' not in data_ns: continue
            ax3d.plot(data_ns['X_f'], data_ns['Y_f'], data_ns['Z_f'], color=self.drone_colors[ns_idx], label='{} Actual'.format(ns))
            ax3d.plot(data_ns['Pd'][:,0], data_ns['Pd'][:,1], data_ns['Pd'][:,2], '--', color=self.drone_colors[ns_idx], alpha=0.7)
            start_pt = self.drone_trajectories[ns]['start']; end_pt = self.drone_trajectories[ns]['end']
            ax3d.scatter(start_pt[0], start_pt[1], start_pt[2], color='green', marker='o', s=100, depthshade=False, label='{} Start'.format(ns) if ns_idx==0 else None)
            ax3d.scatter(end_pt[0], end_pt[1], end_pt[2], color='lime', marker='x', s=100, depthshade=False, label='{} End'.format(ns) if ns_idx==0 else None) 
            all_coords.extend(np.column_stack((data_ns['X_f'], data_ns['Y_f'], data_ns['Z_f'])))
            all_coords.extend(data_ns['Pd'])
            all_coords.append(start_pt); all_coords.append(end_pt)
        if self.obstacles:
            u_s = np.linspace(0, 2*np.pi, 20); v_s = np.linspace(0, np.pi, 10)
            for obs_idx, obs in enumerate(self.obstacles):
                cx, cy, cz, r = obs
                x_surf = cx + r*np.outer(np.cos(u_s), np.sin(v_s)); y_surf = cy + r*np.outer(np.sin(u_s), np.sin(v_s)); z_surf = cz + r*np.outer(np.ones(np.size(u_s)), np.cos(v_s))
                ax3d.plot_surface(x_surf, y_surf, z_surf, color='grey', alpha=0.3, linewidth=0, antialiased=False, rstride=1, cstride=1, label='Obstacle' if obs_idx==0 else None)
                all_coords.extend([[cx-r,cy-r,cz-r], [cx+r,cy+r,cz+r]])
        if all_coords:
            all_coords_arr = np.array(all_coords); min_c, max_c = np.min(all_coords_arr, axis=0), np.max(all_coords_arr, axis=0)
            mid_c = (min_c+max_c)/2.0; max_range = np.max(max_c-min_c)*1.1; max_range = max(max_range, 1.0)
            ax3d.set_xlim(mid_c[0]-max_range/2, mid_c[0]+max_range/2); ax3d.set_ylim(mid_c[1]-max_range/2, mid_c[1]+max_range/2); ax3d.set_zlim(mid_c[2]-max_range/2, mid_c[2]+max_range/2)
        ax3d.set_xlabel('X (m)'); ax3d.set_ylabel('Y (m)'); ax3d.set_zlabel('Z (m)'); ax3d.set_title('3D Trajectories'); ax3d.legend(); ax3d.grid(True)
        plt.savefig(base_save_path + "_traj3d.png"); plt.close(fig3d)

        fig2d, ax2d = plt.subplots(figsize=(10,10)); all_coords_2d = []
        for ns_idx, ns in enumerate(self.all_drone_namespaces):
            data_ns = self.drone_data[ns]
            if 'X_f' not in data_ns: continue
            ax2d.plot(data_ns['X_f'], data_ns['Y_f'], color=self.drone_colors[ns_idx], label='{} Actual'.format(ns))
            ax2d.plot(data_ns['Pd'][:,0], data_ns['Pd'][:,1], '--', color=self.drone_colors[ns_idx], alpha=0.7)
            start_pt = self.drone_trajectories[ns]['start']; end_pt = self.drone_trajectories[ns]['end']
            ax2d.plot(start_pt[0], start_pt[1], color='green', marker='o', markersize=10, label='{} Start'.format(ns) if ns_idx==0 else None)
            ax2d.plot(end_pt[0], end_pt[1], color='lime', marker='x', markersize=10, label='{} End'.format(ns) if ns_idx==0 else None) 
            all_coords_2d.extend(np.column_stack((data_ns['X_f'], data_ns['Y_f'])))
            all_coords_2d.extend(data_ns['Pd'][:,:2])
            all_coords_2d.append(start_pt[:2]); all_coords_2d.append(end_pt[:2])
        if self.obstacles:
            for obs_idx, obs in enumerate(self.obstacles):
                cx, cy, _, r = obs
                circle = plt.Circle((cx, cy), r, color='grey', alpha=0.5, label='Obstacle' if obs_idx==0 else None); ax2d.add_patch(circle)
                all_coords_2d.extend([[cx-r,cy-r], [cx+r,cy+r]])
        if all_coords_2d:
            all_coords_2d_arr = np.array(all_coords_2d); min_c, max_c = np.min(all_coords_2d_arr, axis=0), np.max(all_coords_2d_arr, axis=0)
            mid_c = (min_c+max_c)/2.0; max_range = np.max(max_c-min_c)*1.1; max_range = max(max_range, 1.0)
            ax2d.set_xlim(mid_c[0]-max_range/2, mid_c[0]+max_range/2); ax2d.set_ylim(mid_c[1]-max_range/2, mid_c[1]+max_range/2)
        ax2d.set_xlabel('X (m)'); ax2d.set_ylabel('Y (m)'); ax2d.set_title('Top-Down Trajectories'); ax2d.legend(); ax2d.grid(True); ax2d.set_aspect('equal', adjustable='box')
        plt.savefig(base_save_path + "_traj2d_topdown.png"); plt.close(fig2d)

        fig_w, axs_w = plt.subplots(rows, cols, figsize=(6*cols, 4*rows), sharex=True, squeeze=False); axs_w_flat = axs_w.flatten()
        for ns_idx, ns in enumerate(self.all_drone_namespaces):
            ax = axs_w_flat[ns_idx]; data_ns = self.drone_data[ns]
            if not data_ns['w_t'] or not data_ns['w']: ax.set_title('{}-No Omega'.format(ns)); ax.grid(); continue
            Tw = np.array(data_ns['w_t']); W_sq_val = np.array(data_ns['w'])
            if W_sq_val.ndim == 1 and len(W_sq_val) > 0: W_sq_val = W_sq_val.reshape(-1,1)
            elif W_sq_val.size == 0 : ax.set_title('{}-Empty Omega'.format(ns)); ax.grid(); continue
            W_val = np.sqrt(np.maximum(W_sq_val, 0))
            fw_w_eff = self._get_eff_filter_window(self.fw_w, self.fp_w)
            if self.use_filt:
                for i in range(W_val.shape[1]): W_val[:,i] = FILT(W_val[:,i], fw_w_eff, self.fp_w)
            for i in range(W_val.shape[1]): ax.plot(Tw, W_val[:,i], label='M{}'.format(i+1))
            ax.set_ylim(*self.w_lim); ax.set_ylabel('ω (rad/s)'); ax.set_title('{} Motor Omegas'.format(ns)); ax.legend(); ax.grid()
        for i in range(num_drones, rows*cols): fig_w.delaxes(axs_w_flat[i])
        
        for r_idx in range(rows):
            for c_idx in range(cols):
                ax_idx = r_idx * cols + c_idx
                if ax_idx < num_drones: 
                    if r_idx == rows - 1 or (ax_idx + cols >= num_drones):
                         axs_w[r_idx,c_idx].set_xlabel('t (s)')
                
        fig_w.suptitle('Motor Angular Velocities (ω)', fontsize=16); fig_w.tight_layout(rect=[0,0.03,1,0.95])
        plt.savefig(base_save_path + "_omega_all_drones.png"); plt.close(fig_w)

        fig_u, axs_u = plt.subplots(rows, cols, figsize=(7*cols, 5*rows), sharex=True, squeeze=False); axs_u_flat = axs_u.flatten()
        u_labels = ['U1 (Thrust N)', 'U2 (Roll Nm)', 'U3 (Pitch Nm)', 'U4 (Yaw Nm)']
        for ns_idx, ns in enumerate(self.all_drone_namespaces):
            ax = axs_u_flat[ns_idx]; data_ns = self.drone_data[ns]
            if not data_ns['u_t']: ax.set_title('{}-No U Data'.format(ns)); ax.grid(); continue
            Tu = np.array(data_ns['u_t']); Us_val = [np.array(data_ns['u1']), np.array(data_ns['u2']), np.array(data_ns['u3']), np.array(data_ns['u4'])]
            fw_t_eff = self._get_eff_filter_window(self.fw_t, self.fp_t)
            if self.use_filt:
                for i in range(4): Us_val[i] = FILT(Us_val[i], fw_t_eff, self.fp_t)
            for i in range(4): ax.plot(Tu, Us_val[i], label=u_labels[i])
            ax.set_ylabel('Force/Torque'); ax.set_title('{} Control Inputs'.format(ns)); ax.legend(); ax.grid()
        for i in range(num_drones, rows*cols): fig_u.delaxes(axs_u_flat[i])
        
        for r_idx in range(rows):
            for c_idx in range(cols):
                ax_idx = r_idx * cols + c_idx
                if ax_idx < num_drones:
                    if r_idx == rows - 1 or (ax_idx + cols >= num_drones):
                        axs_u[r_idx,c_idx].set_xlabel('t (s)')

        fig_u.suptitle('Control Inputs (U)', fontsize=16); fig_u.tight_layout(rect=[0,0.03,1,0.95])
        plt.savefig(base_save_path + "_control_inputs_all_drones.png"); plt.close(fig_u)

        plot_types = [
            ("Position Error", "_pos_err_all_drones.png", ['X_f', 'Y_f', 'Z_f'], None, "Error (m)", lambda d, p: getattr(d,p.lower()) - d.Pd[:, ['X','Y','Z'].index(p[5:])]),
            ("Position Tracking", "_pos_track_all_drones.png", ['X', 'Y', 'Z'], None, "Position (m)", None),
            ("Velocity Tracking", "_vel_track_all_drones.png", ['Vx', 'Vy', 'Vz'], None, "Velocity (m/s)", None)
        ]
        for title_prefix, fname_suffix, comp_keys_base, legend_labels_err, ylabel, err_func in plot_types:
            fig, axs = plt.subplots(rows, cols, figsize=(8*cols, 6*rows), sharex=True, squeeze=False)
            axs_flat = axs.flatten()
            for ns_idx, ns in enumerate(self.all_drone_namespaces):
                ax = axs_flat[ns_idx]; data_ns = self.drone_data[ns]
                data_available = all('{}_f'.format(k) in data_ns for k in comp_keys_base) if "Tracking" in title_prefix else 'X_f' in data_ns 
                if not data_available or 'Pd' not in data_ns: ax.set_title('{}-No Data'.format(ns)); ax.grid(); continue

                if "Error" in title_prefix:
                    for i, comp_key_suffix in enumerate(['X', 'Y', 'Z']):
                        err_val = data_ns['{}_f'.format(comp_key_suffix)] - data_ns['Pd'][:,i]
                        ax.plot(data_ns['t_arr'], err_val, label='Error {}'.format(comp_key_suffix))
                else: 
                    for i, comp_key_suffix in enumerate(comp_keys_base):
                        actual_key = '{}_f'.format(comp_key_suffix) if comp_key_suffix in ['X','Y','Z'] else '{}{}_f'.format(comp_key_suffix[0].upper(), comp_key_suffix[1:]) 
                        desired_arr = data_ns['Pd'] if comp_key_suffix in ['X','Y','Z'] else data_ns['Vd']
                        
                        ax.plot(data_ns['t_arr'], data_ns[actual_key], '-', color=plt.cm.tab10(i*2), label='{} Actual'.format(comp_key_suffix))
                        ax.plot(data_ns['t_arr'], desired_arr[:,i], '--', color=plt.cm.tab10(i*2+1), label='{} Desired'.format(comp_key_suffix))
                
                ax.set_ylabel(ylabel); ax.set_title('{} {}'.format(ns, title_prefix)); ax.legend(fontsize='small'); ax.grid()
            for i in range(num_drones, rows*cols): fig.delaxes(axs_flat[i])
            
            for r_idx in range(rows):
                for c_idx in range(cols):
                    ax_idx = r_idx * cols + c_idx
                    if ax_idx < num_drones:
                        if r_idx == rows - 1 or (ax_idx + cols >= num_drones):
                            axs[r_idx,c_idx].set_xlabel('t (s)')
            
            fig.suptitle(title_prefix, fontsize=16); fig.tight_layout(rect=[0,0.03,1,0.95])
            plt.savefig(base_save_path + fname_suffix); plt.close(fig)
        rospy.loginfo("All plots saved to {}_*".format(base_save_path))

if __name__=="__main__":
    rospy.init_node("multi_drone_trajectory_plotter", anonymous=True)
    plotter_instance = None
    try:
        plotter_instance = MultiDronePlotter()
        if not rospy.is_shutdown():
             rospy.spin()
    except rospy.ROSInterruptException:
        rospy.loginfo("MultiDronePlotter interrupted.")
    except Exception as e:
        rospy.logerr("Unhandled exception in MultiDronePlotter: {}".format(e), exc_info=True)
    finally:
        if plotter_instance is not None and hasattr(plotter_instance, 'done') and not plotter_instance.done:
            rospy.loginfo("Ensuring plotter shutdown is called on script exit.")
            plotter_instance.shutdown()