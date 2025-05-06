#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import division
import rospy, math, numpy as np
from enum import Enum
from nav_msgs.msg import Odometry
from mav_msgs.msg import Actuators
from geometry_msgs.msg import Point, Vector3
from std_msgs.msg import Float64MultiArray, String
from tf.transformations import euler_from_quaternion
from cvxopt import matrix, solvers
import ast

solvers.options['show_progress'] = False

clip = np.clip
sign = np.sign
rt   = np.sqrt
def pget(n,d): return rospy.get_param("~"+n,d)

e3_world = np.array([0.0, 0.0, 1.0])
e3_body  = np.array([0.0, 0.0, 1.0])

class State(Enum):
    TAKEOFF=1; HOVER=2; TRAJ=3; LAND=4; IDLE=5
LOG_T = 1.0
DBG   = True

class ClfIrisController(object):
    def __init__(self):
        ns            = pget("namespace","iris")
        self.ns       = ns
        self.use_gz   = pget("use_model_states",False)
        # Offsets are no longer needed for the straight line trajectory starting at takeoff point
        # self.xy_offset= None
        # self.z_offset = None

        self.m        = pget("mass",1.5)
        self.g        = pget("gravity",9.81)
        self.Ix       = pget("I_x",0.0348)
        self.Iy       = pget("I_y",0.0459)
        self.Iz       = pget("I_z",0.0977)
        self.J_inv_diag = np.diag([1.0/self.Ix, 1.0/self.Iy, 1.0/self.Iz])
        self.J_mat    = np.diag([self.Ix, self.Iy, self.Iz])
        self.kf       = pget("motor_constant",8.54858e-06)
        self.km       = pget("moment_constant",1.3677728e-07)
        self.w_max    = pget("max_rot_velocity",838.0)
        self.min_f    = pget("min_thrust_factor",0.1)
        self.gc       = pget("gravity_comp_factor",1.022)
        self.max_tilt = math.radians(pget("max_tilt_angle_deg",30.0))
        self.r_drone  = pget("drone_radius", 0.5)
        #self.remote_topic = pget("remote_odom_topic", "")
        #self.remote_state_topic = pget("remote_state_topic", "")
        self.remote_topic       = pget("remote_odom_topic", "")
        self.remote_state_topic = pget("remote_state_topic", "")
        self.remote_r     = pget("remote_drone_radius", self.r_drone)
        self.remote_last  = None           # latest Odometry of other drone
        self.remote_state = None
        self.remote_prev_v= None           # for finite‑difference accel
        self.remote_prev_t= None
        if self.remote_topic:
            rospy.loginfo("Subscribing to remote UAV odom on %s", self.remote_topic)
            self.remote_odom_sub = rospy.Subscriber(
                    self.remote_topic,
                    Odometry, self.cb_remote_odom,
                    queue_size=5)
            
        if self.remote_state_topic:
            rospy.loginfo("Subscribing to remote UAV state on %s", self.remote_state_topic)
            self.remote_state_sub = rospy.Subscriber(
                    self.remote_state_topic,
                    String, self.cb_remote_state,
                    queue_size=5)

        # --- Straight Line Trajectory Parameters ---
        line_start_str = pget("line_start", "[30.0, 0.0, 3.0]")
        line_end_str   = pget("line_end", "[-30.0, 0.0, 3.0]")
        try:
            # Use ast.literal_eval to safely parse the string lists
            self.line_start = np.array(ast.literal_eval(line_start_str), dtype=float)
            self.line_end   = np.array(ast.literal_eval(line_end_str), dtype=float)
            # Ensure they are 3-element arrays
            if self.line_start.shape != (3,) or self.line_end.shape != (3,):
                 raise ValueError("Parsed line_start/line_end is not a 3-element list.")
            rospy.loginfo("[{self.ns}] Parsed line_start: {self.line_start}")
            rospy.loginfo("[{self.ns}] Parsed line_end: {self.line_end}")
        except (ValueError, SyntaxError) as e:
            rospy.logerr("[{self.ns}] Error parsing line_start/line_end parameters ('{line_start_str}', '{line_end_str}'): {e}. Using defaults.")
            # Fallback to default numerical arrays
            self.line_start = np.array([30.0, 0.0, 3.0], dtype=float)
            self.line_end   = np.array([-30.0, 0.0, 3.0], dtype=float)
        self.line_duration = pget("line_duration", 20.0) # Duration to complete the line in seconds
        self.yaw_fix    = math.radians(pget("fixed_yaw_deg",0.0))

        # Calculate derived trajectory properties
        self.line_vector = self.line_end - self.line_start
        self.line_length = np.linalg.norm(self.line_vector)
        if self.line_duration > 1e-6:
            self.line_velocity_const = self.line_vector / self.line_duration
        else:
            rospy.logwarn("Line duration is too small, setting velocity to zero.")
            self.line_velocity_const = np.zeros(3)
            self.line_duration = 0.0 # Avoid division by zero later

        # --- Takeoff Point ---
        # Set takeoff target to the start of the line trajectory
        tx = pget("takeoff_x", self.line_start[0])
        ty = pget("takeoff_y", self.line_start[1])
        th = pget("takeoff_height", self.line_start[2])
        self.x_to, self.y_to, self.z_to = tx, ty, th

        def gains(pref, k1, k2, a1, a2):
            return [pget(pref+k,i) for k,i in
                    zip(("pos1","pos2","att1","att2"),(k1,k2,a1,a2))]
        self.g_take = gains("k_take",0.22,0.8,2.05,4.1)
        self.g_traj = gains("k_traj",0.75,4.1,16.0,32.0)

        A = np.array([
            [self.kf]*4,
            [-0.22*self.kf,  0.20*self.kf,  0.22*self.kf, -0.20*self.kf],
            [-0.13*self.kf,  0.13*self.kf, -0.13*self.kf,  0.13*self.kf],
            [-self.km,       -self.km,       self.km,       self.km]
        ])
        self.invA = np.linalg.inv(A)

        self.beta   = pget("zcbf_beta", 1.5)
        self.a1     = pget("zcbf_a1", 0.5)
        self.a2     = pget("zcbf_a2", 1.0)
        self.gamma  = pget("zcbf_gamma", 5.0)
        self.kappa  = pget("zcbf_kappa", 18.0)
        self.a      = pget("zcbf_order_a", 0)
        obstacles_str = pget("dynamic_obstacles",
        "[[-8.96,-15.52,8.00, 0,0,0, 0,0,0, 1.00]]")  # x y z vx vy vz ax ay az r
        #obstacles_str = pget("static_obstacles", "[[-8.96, -15.52, 8.00, 1.00]]")
        #default_obs = [[-8.96, -15.52, 8.00, 1.00]]
        default_obs = [[-8.96,-15.52,8.00, 0,0,0, 0,0,0, 1.00]]
        try:
            obstacles_list = ast.literal_eval(obstacles_str)
            if not isinstance(obstacles_list, list):
                 rospy.logwarn("Parsed static_obstacles is not a list. Using default.")
                 obstacles_list = default_obs
            elif obstacles_list and not all(isinstance(o, (list, tuple)) and len(o) == 10 and all(isinstance(n, (int, float)) for n in o) for o in obstacles_list):
                 rospy.logwarn("Invalid format in static_obstacles list items. Expected [x,y,z,vx,vy,vz,ax,ay,az,r]. Using default.")
                 obstacles_list = default_obs

            self.obs = np.array(obstacles_list, dtype=float)
            if self.obs.ndim == 1 and self.obs.size == 0:
                 self.obs = np.empty((0, 10), dtype=float)
            elif self.obs.ndim == 1 and self.obs.shape[0] == 10:
                self.obs = self.obs.reshape(1, 10)
            elif self.obs.ndim != 2 or (self.obs.size > 0 and self.obs.shape[1] != 10):
                rospy.logwarn("Parsed static_obstacles does not have shape (N, 10). Using default.")
                self.obs = np.array(default_obs, dtype=float)

            if self.obs.size > 0:
                rospy.loginfo("Loaded %d dynamic obstacles.", self.obs.shape[0])
            else:
                rospy.loginfo("No dynamic obstacles loaded.")

        except (ValueError, SyntaxError) as e:
             rospy.logwarn("Error parsing static_obstacles parameter '%s': %s. Using default.", obstacles_str, e)
             self.obs = np.array(default_obs, dtype=float)
        except Exception as e:
             rospy.logwarn("Unexpected error processing static_obstacles '%s': %s. Using default.", obstacles_str, e)
             self.obs = np.array(default_obs, dtype=float)

        topic = lambda s: '~'+s
        self.cbf_pub = rospy.Publisher(topic("cbf/slack"),
                                       Float64MultiArray,
                                       queue_size=1)

        self.cmd_pub = rospy.Publisher('/{}/command/motor_speed'.format(ns),
                                       Actuators, queue_size=1)
        pubs = [
            ("control/state",String),("control/U",Float64MultiArray),
            ("control/omega_sq",Float64MultiArray),
            ("error/position",Point),("error/velocity",Vector3),
            ("error/attitude_deg",Point),("error/rates_deg_s",Vector3),
            ("control/desired_position",Point),
            ("control/desired_velocity",Vector3),
            ("control/desired_acceleration",Vector3),
            ("control/desired_attitude_deg",Point),
            ("control/virtual_inputs",Point),
        ]
        self.pubs = {n: rospy.Publisher(topic(n),t,queue_size=1)
                     for n,t in pubs}

        if self.use_gz:
            from gazebo_msgs.msg import ModelStates
            self.sub = rospy.Subscriber('/gazebo/model_states',
                                        ModelStates,self.cb_model,
                                        queue_size=5,buff_size=2**24)
        else:
            self.sub = rospy.Subscriber('/{}/ground_truth/odometry'.format(ns),
                                        Odometry,self.cb_odom,
                                        queue_size=10)

        self.state      = State.TAKEOFF
        self.last       = None
        rate            = pget("control_rate",100.0)
        self.timer      = rospy.Timer(rospy.Duration(1.0/rate),
                                      self.loop,reset=True)
        self.t0_traj    = None
        self.hover_ok_t = None
        rospy.on_shutdown(self.shutdown)


    def cb_remote_odom(self, msg):
        """Store the most recent odometry of the *other* Iris."""
        self.remote_last = msg

    def cb_odom(self,msg):
        self.last = msg
    def cb_remote_state(self, msg):
        self.remote_state = msg.data
    def cb_model(self,msg):
        try: idx = msg.name.index(self.ns)
        except ValueError:
            try: idx = msg.name.index(self.ns+'/')
            except ValueError: return
        o = Odometry()
        o.header.stamp          = rospy.Time.now()
        o.header.frame_id       = "world"
        o.child_frame_id        = self.ns+"/base_link"
        o.pose.pose, o.twist.twist = msg.pose[idx], msg.twist[idx]
        self.last = o

    @staticmethod
    def R(phi,th,psi):
        c,s=math.cos,math.sin
        return np.array([
            [c(th)*c(psi),
             s(phi)*s(th)*c(psi)-c(phi)*s(psi),
             c(phi)*s(th)*c(psi)+s(phi)*s(psi)],
            [c(th)*s(psi),
             s(phi)*s(th)*s(psi)+c(phi)*c(psi),
             c(phi)*s(th)*s(psi)-s(phi)*c(psi)],
            [   -s(th),
             s(phi)*c(th),
             c(phi)*c(th)]
        ])

    def traj_ref(self,t):
        """
        Calculates the reference position, velocity, and acceleration for a straight line.
        Args:
            t (float): Time elapsed since the start of the trajectory phase.
        Returns:
            tuple: (pos, vel, acc, yaw_d, yaw_rate_d)
                   pos (np.array): Desired position [x, y, z]
                   vel (np.array): Desired velocity [vx, vy, vz]
                   acc (np.array): Desired acceleration [ax, ay, az]
                   yaw_d (float): Desired yaw angle (radians)
                   yaw_rate_d (float): Desired yaw rate (radians/s)
        """
        if self.line_duration <= 1e-6: # Handle zero duration case
             pos = self.line_start
             vel = np.zeros(3)
             acc = np.zeros(3)
        else:
            # Calculate the normalized time, capped at 1.0
            t_norm = min(t / self.line_duration, 1.0)

            # Position: Linear interpolation from start to end
            pos = self.line_start + self.line_vector * t_norm

            # Velocity: Constant velocity along the line until the end, then zero
            if t < self.line_duration:
                vel = self.line_velocity_const
            else:
                vel = np.zeros(3) # Stop at the end point

            # Acceleration: Zero for constant velocity straight line
            acc = np.zeros(3)

        # Use fixed yaw
        yaw_d = self.yaw_fix
        yaw_rate_d = 0.0

        return pos, vel, acc, yaw_d, yaw_rate_d


    def loop(self,_):
        if self.last is None: return
        now = rospy.Time.now()

        p = self.last.pose.pose.position
        q_ros = self.last.pose.pose.orientation
        v = self.last.twist.twist.linear
        w = self.last.twist.twist.angular
        x,y,z = p.x, p.y, p.z
        phi,th,psi = euler_from_quaternion(
            [q_ros.x,q_ros.y,q_ros.z,q_ros.w])
        R_mat = self.R(phi,th,psi)
        v_w = np.array([v.x, v.y, v.z])
        w_body = np.array([w.x, w.y, w.z])
        p_vec = np.array([x, y, z])

        if self.state in (State.TAKEOFF,State.HOVER):
            tgt = np.array([self.x_to, self.y_to, self.z_to]) # Target is the takeoff point (line start)
            vd = ad_nom = np.zeros(3)
            yd,rd = self.yaw_fix, 0.0
            g1,g2,g3,g4 = self.g_take
            err_pos = np.linalg.norm(p_vec - tgt) # Use 3D error for hover check
            err_v = np.linalg.norm(v_w)

            current_pos_thresh = pget("hover_pos_threshold",.15)
            current_vel_thresh = pget("hover_vel_threshold",.1)

            if (self.state==State.TAKEOFF
                and err_pos < current_pos_thresh # Check position error
                and err_v < current_vel_thresh):
                rospy.loginfo("TRANSITIONING TO HOVER")
                self.state,self.hover_ok_t = State.HOVER,None
            if self.state==State.HOVER:
                remote_ok = (not self.remote_state_topic or
                             self.remote_state in (State.HOVER.name,
                                                   State.TRAJ.name))
                if (err_pos < current_pos_thresh
                    and err_v < current_vel_thresh
                    and remote_ok):
                    if self.hover_ok_t is None:
                        self.hover_ok_t = now
                    elif (now-self.hover_ok_t
                          >= rospy.Duration(
                              pget("hover_stabilization_secs",2.0))):
                        rospy.loginfo("TRANSITIONING TO TRAJ")
                        self.state,self.t0_traj = State.TRAJ, now
                        # No offsets needed as trajectory starts from the hover point
                        # self.xy_offset = np.array([
                        #     self.x_to - self.r0, # r0 is removed
                        #     self.y_to
                        # ])
                        # self.z_offset = self.z_to
                else:
                    self.hover_ok_t = None

        elif self.state==State.TRAJ:
            if self.t0_traj is None:
                rospy.logwarn("In TRAJ state but t0_traj is None, reverting to HOVER")
                self.state = State.HOVER
                return
            posd,vd,ad_nom,yd,rd = self.traj_ref((now-self.t0_traj).to_sec())
            tgt = posd
            g1,g2,g3,g4 = self.g_traj

        else: # LAND or IDLE (treat as hover at current pos for now)
            tgt = np.array([x,y,z])
            vd = ad_nom = np.zeros(3)
            yd,rd = psi,0.0 # Hold current yaw
            g1,g2,g3,g4 = self.g_take

        ex1 = p_vec - tgt
        ex2 = v_w - vd
        tof = math.cos(phi)*math.cos(th)
        if abs(tof)<self.min_f:
            tof = sign(tof or 1)*self.min_f

        U1_nom = (self.m/tof)*(-g1*ex1[2]
              +ad_nom[2]-g2*ex2[2]) \
             + self.m*self.g * self.gc/tof
        U1_nom = max(0.0,U1_nom)

        if U1_nom<1e-6:
            Uex=Uey=0.0
        else:
            Uex = (self.m/U1_nom)*(-g1*ex1[0]
                   +ad_nom[0]-g2*ex2[0])
            Uey = (self.m/U1_nom)*(-g1*ex1[1]
                   +ad_nom[1]-g2*ex2[1])

        sp,cp = math.sin(yd), math.cos(yd)
        try:
            phi_d = math.asin(clip(Uex*sp-Uey*cp,-1,1))
        except ValueError:
             phi_d = sign(Uex*sp-Uey*cp) * math.pi / 2.0

        cpd   = math.cos(phi_d)
        try:
            theta_d = (0.0 if abs(cpd)<self.min_f
                       else math.asin(
                           clip((Uex*cp+Uey*sp)/cpd,-1,1)))
        except ValueError:
             theta_d = sign((Uex*cp+Uey*sp)/cpd) * math.pi / 2.0 if abs(cpd)>=self.min_f else 0.0

        phi_d,theta_d = clip(phi_d,
                             -self.max_tilt,
                             self.max_tilt), \
                        clip(theta_d,
                             -self.max_tilt,
                             self.max_tilt)

        e_th = np.array([phi-phi_d,
                         th-theta_d,
                         (psi-yd+math.pi)%(2*math.pi)
                         -math.pi])
        e_w  = (w_body - np.array([0.0,0.0,rd]))

        U2_nom = (self.Ix*(-g3*e_th[0] - g4*e_w[0])
              - w_body[1]*w_body[2]*(self.Iy-self.Iz))
        U3_nom = (self.Iy*(-g3*e_th[1] - g4*e_w[1])
              - w_body[0]*w_body[2]*(self.Iz-self.Ix))
        U4_nom = (self.Iz*(-g3*e_th[2] - g4*e_w[2])
              - w_body[0]*w_body[1]*(self.Ix-self.Iy))

        U_nom = np.array([U1_nom, U2_nom, U3_nom, U4_nom])
        U = U_nom.copy()

        
        obs_current = self.obs
        if self.remote_last is not None:
            rp = self.remote_last.pose.pose.position
            rv = self.remote_last.twist.twist.linear

            # simple finite‑difference acceleration
            if self.remote_prev_t is not None:
                dt_r = (self.remote_last.header.stamp - self.remote_prev_t).to_sec()
                if dt_r > 1e-3:
                    ax = (rv.x - self.remote_prev_v.x) / dt_r
                    ay = (rv.y - self.remote_prev_v.y) / dt_r
                    az = (rv.z - self.remote_prev_v.z) / dt_r
                else:
                    ax = ay = az = 0.0
            else:
                ax = ay = az = 0.0

            remote_row = [rp.x, rp.y, rp.z,
                          rv.x, rv.y, rv.z,
                          ax,    ay,   az,
                          self.remote_r]

            if obs_current.size == 0:
                obs_current = np.array([remote_row], dtype=float)
            else:
                obs_current = np.vstack([obs_current, remote_row])

            self.remote_prev_v = rv
            self.remote_prev_t = self.remote_last.header.stamp
            
        if self.state == State.TRAJ and obs_current.size > 0:    
            G_cbf_list = []
            h_cbf_list = []

            for obs_row in obs_current:
                #xo = np.array([ox, oy, oz])
                ox,oy,oz,vx,vy,vz,ax,ay,az,r_o = obs_row
                xo   = np.array([ox,oy,oz])
                Vo   = np.array([vx,vy,vz])
                Ao   = np.array([ax,ay,az])
                r_safe = r_o + self.r_drone

                r      = p_vec - xo
                q      = R_mat[:,2]
                r_dot  = v_w - Vo      #   V - Vo
                s      = np.dot(r, q)
                sigma  = -self.a1*np.arctan(self.a2*s)
                sig_p  = -self.a1*self.a2/(1+(self.a2*s)**2)
                sig_pp  =  2.0 * self.a1 * (self.a2 ** 2) * s / (1.0 + (self.a2 * s) ** 2) ** 2

                g_hat  = np.dot(r,r) - self.beta*r_safe**2 - sigma
                R_Omxe3 = R_mat.dot(np.cross(w_body, e3_body))
                g_hat_d= 2.0*np.dot(r, r_dot) - sig_p*( np.dot(r_dot,q)
                         + np.dot(r, R_Omxe3) )
                #g_hat_d= 2.0*np.dot(r, v_w) - sig_p*( np.dot(v_w,q)
                 #         + np.dot(r, R_Omxe3) )
                h_val  = self.gamma*g_hat + g_hat_d

                Gamma1 = (2.0*s - sig_p)/self.m
                r_b = np.dot(R_mat.T, r)
                cross_rb_e3 = np.cross(r_b, e3_body)

                Gamma2_vec = sig_p * np.dot(cross_rb_e3, self.J_inv_diag)
                dot_s   = np.dot(r_dot, q) + np.dot(r, R_Omxe3)
                #term1 = self.gamma * g_hat_d
                #term2 = 2.0 * np.dot(v_w, v_w)
                #term3 = -2.0 * self.g * np.dot(r, e3_world)
                #term4 = -sig_pp * (dot_s ** 2)
                #term5 = sig_p * self.g * q[2]
                #term6 = -sig_p * (2.0 * np.dot(v_w, R_Omxe3))
                term1 = self.gamma * g_hat_d
                term2 = 2.0 * np.dot(r_dot, r_dot)
                term3 = -2.0 * self.g * np.dot(r, e3_world)
                term4 = -2.0 * np.dot(r, Ao)          # new acceleration term
                term5 = -sig_pp * (dot_s ** 2)
                term6 = sig_p * self.g * q[2]
                term7 = -sig_p * (2.0 * np.dot(r_dot, R_Omxe3))
                Omega_cross_e3 = np.cross(w_body, e3_body)
                Omega_cross_Omega_cross_e3 = np.cross(w_body, Omega_cross_e3)
                R_Omega_cross_Omega_cross_e3 = np.dot(R_mat, Omega_cross_Omega_cross_e3)
                #term7 = -sig_p * np.dot(r, R_Omega_cross_Omega_cross_e3)
                J_Omega = np.dot(self.J_mat, w_body)
                Omega_cross_JOmega = np.cross(w_body, J_Omega)
                xi = np.dot(self.J_inv_diag, Omega_cross_JOmega)
                xi_cross_e3 = np.cross(xi, e3_body)
                R_xi_cross_e3 = np.dot(R_mat, xi_cross_e3)
                term8 = -sig_p * np.dot(r, R_Omega_cross_Omega_cross_e3)
                term9 =  sig_p * np.dot(r, R_xi_cross_e3)
                #term8 = sig_p * np.dot(r, R_xi_cross_e3)

                Gamma3 = term1 + term2 + term3 + term4 + term5 + term6 + term7 + term8 + term9

                G_cbf_row  = np.hstack([Gamma1, Gamma2_vec])
                h_cbf_val  = + Gamma3 + self.kappa*(h_val**(2*self.a+1))

                G_cbf_list.append(-G_cbf_row)
                h_cbf_list.append(h_cbf_val)

            U1_max = 4 * self.kf * self.w_max**2
            # Define box constraints on U1 (thrust)
            # U1 >= 0 (already handled by clipping U1_nom and QP solver implicitly if needed)
            # U1 <= U1_max
            # G = [[-1, 0, 0, 0]], h = [-U1_min] -> U1 >= U1_min (e.g., 0)
            # G = [[ 1, 0, 0, 0]], h = [ U1_max] -> U1 <= U1_max
            box_G = np.array([[ 1., 0., 0., 0.]]) # Constraint: U1 <= U1_max
            box_h = np.array([[U1_max]])          # h value for U1 <= U1_max

            # Combine CBF and box constraints
            if G_cbf_list: # If there are CBF constraints
                cbf_G = np.vstack(G_cbf_list)
                cbf_h = np.array(h_cbf_list, dtype=float).reshape(-1,1)
                G_all = np.vstack([cbf_G, box_G])
                h_all = np.vstack([cbf_h, box_h])
            else: # Only box constraints
                G_all = box_G
                h_all = box_h

            # Setup QP: min || U - U_nom ||^2  s.t. G_all * U <= h_all
            P = matrix(np.eye(4))
            q_qp = matrix(-U_nom)
            G = matrix(G_all)
            h_qp = matrix(h_all)

            try:
                sol = solvers.qp(P, q_qp, G, h_qp)
                if sol['status'] == 'optimal':
                    U  = np.array(sol['x']).flatten()
                    # Calculate slack for CBF constraints only (first N rows of G/h)
                    if G_cbf_list:
                        slack_val = h_qp[:len(h_cbf_list)] - G[:len(h_cbf_list),:] * sol['x']
                        self.cbf_pub.publish(Float64MultiArray(data=list(slack_val)))
                    else:
                        self.cbf_pub.publish(Float64MultiArray(data=[])) # No CBF constraints
                else:
                     rospy.logwarn_throttle(1.0,"ZCBF-QP non-optimal (status: %s), using nominal U", sol['status'])
                     U = U_nom
                     self.cbf_pub.publish(Float64MultiArray(data=[0.0]*len(h_cbf_list))) # Publish zero slack
            except ValueError: # Often indicates infeasibility
                rospy.logwarn_throttle(1.0, "ZCBF-QP infeasible (ValueError), using nominal U")
                U = U_nom
                self.cbf_pub.publish(Float64MultiArray(data=[0.0]*len(h_cbf_list))) # Publish zero slack
            except Exception as e: # Catch other potential solver errors
                 rospy.logwarn_throttle(1.0, "ZCBF-QP solver error: %s, using nominal U", str(e))
                 U = U_nom
                 self.cbf_pub.publish(Float64MultiArray(data=[0.0]*len(h_cbf_list))) # Publish zero slack
        else: # Not in TRAJ state or no obstacles
             U = U_nom
             if self.state == State.TRAJ: # Publish empty slack if in TRAJ but no obstacles
                 self.cbf_pub.publish(Float64MultiArray(data=[]))

        # Ensure U1 (thrust) is non-negative after QP solution
        U[0] = max(0.0, U[0])

        w_sq = clip(np.dot(self.invA, U), 0, None) # Ensure non-negative squared speeds
        w_cmd= clip(rt(w_sq), 0, self.w_max)

        m = Actuators()
        m.header.stamp = now
        m.angular_velocities = w_cmd.tolist()
        self.cmd_pub.publish(m)

        # --- Publishing ---
        self.pubs["control/state"].publish(
            String(data=self.state.name))
        self.pubs["control/U"].publish(
            Float64MultiArray(data=U))
        self.pubs["control/omega_sq"].publish(
            Float64MultiArray(data=w_sq))
        self.pubs["error/position"].publish(
            Point(*ex1))
        self.pubs["error/velocity"].publish(
            Vector3(*ex2))
        self.pubs["error/attitude_deg"].publish(
            Point(*(math.degrees(i)
                    for i in e_th)))
        self.pubs["error/rates_deg_s"].publish(
            Vector3(*(math.degrees(i)
                      for i in e_w)))
        self.pubs["control/desired_position"].publish(
            Point(*tgt))
        self.pubs["control/desired_velocity"].publish(
            Vector3(*vd))
        self.pubs["control/desired_acceleration"].publish(
            Vector3(*ad_nom))
        self.pubs["control/desired_attitude_deg"].publish(
            Point(*(math.degrees(i)
                    for i in (phi_d,theta_d,yd))))
        self.pubs["control/virtual_inputs"].publish(
            Point(Uex,Uey,0.0))

        if DBG:
            rospy.loginfo_throttle(
                LOG_T,
                "[%s] U=[%.2f, %.2f, %.2f, %.2f] (Nominal U=[%.2f, %.2f, %.2f, %.2f])",
                self.state.name, U[0], U[1], U[2], U[3],
                U_nom[0], U_nom[1], U_nom[2], U_nom[3])

    def shutdown(self):
        rospy.loginfo("Shutting down controller, sending zero motor commands.")
        stop = Actuators()
        stop.angular_velocities = [0.0]*4
        # Send stop command multiple times to ensure it's received
        for _ in range(10):
            if rospy.is_shutdown():
                break
            self.cmd_pub.publish(stop)
            rospy.sleep(0.01)
        rospy.loginfo("Shutdown complete.")

if __name__=="__main__":
    rospy.init_node("zclf_iris_trajectory_controller",
                    anonymous=True)
    try:
        controller = ClfIrisController()
        rospy.loginfo("Straight Line Trajectory Controller Initialized.")
        rospy.spin()
    except rospy.ROSInterruptException:
        rospy.loginfo("ROS interrupt received.")
    except Exception as e:
        rospy.logerr("Unhandled exception in controller: %s", str(e))
    finally:
        # Ensure shutdown commands are sent even if there's an error during spin
        if 'controller' in locals() and hasattr(controller, 'shutdown'):
             controller.shutdown()
        rospy.loginfo("Controller node finished.")