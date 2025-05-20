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

clip = np.clip
sign = np.sign
rt   = np.sqrt
def pget(n,d): return rospy.get_param("~"+n,d)

class State(Enum):
    TAKEOFF=1; HOVER=2; TRAJ=3; LAND=4; IDLE=5
LOG_T = 1.0
DBG   = True

class ClfIrisController(object):
    def __init__(self):
        ns            = pget("namespace","iris")
        self.ns       = ns
        self.use_gz   = pget("use_model_states",False) 
        self.xy_offset= None
        self.z_offset = None

        # vehicle & motor
        self.m        = pget("mass",1.5)
        self.g        = pget("gravity",9.81)
        self.Ix       = pget("I_x",0.0348)
        self.Iy       = pget("I_y",0.0459)
        self.Iz       = pget("I_z",0.0977)
        self.kf       = pget("motor_constant",8.54858e-06)
        self.km       = pget("moment_constant",1.3677728e-07)
        self.w_max    = pget("max_rot_velocity",838.0)
        self.min_f    = pget("min_thrust_factor",0.1)
        self.gc       = pget("gravity_comp_factor",1.022)
        self.max_tilt = math.radians(pget("max_tilt_angle_deg",30.0))

        # helix parameters
        self.d_start    = pget("helix_start_diameter",40.0)
        self.d_end      = pget("helix_end_diameter",15.0) 
        self.height     = pget("helix_height",30.0)
        self.laps       = pget("helix_laps",4.0)
        self.omega_traj = pget("trajectory_omega",0.08) 
        self.yaw_fix    = math.radians(pget("fixed_yaw_deg",0.0)) 

        # initial radius & rates
        self.r0         = 0.5 * self.d_start
        theta_tot      = self.laps * 2.0 * math.pi
        self.k_r       = (self.r0 - 0.5*self.d_end) / theta_tot
        self.k_z       = self.height / theta_tot

        # take‑off/hover target
        tx = pget("takeoff_x", self.r0) 
        ty = pget("takeoff_y", 0.0)
        th = pget("takeoff_height", 3.0)
        self.x_to, self.y_to, self.z_to = tx, ty, th

        # gains
        def gains(pref, k1, k2, a1, a2):
            return [pget(pref+k,i) for k,i in
                    zip(("pos1","pos2","att1","att2"),(k1,k2,a1,a2))]
        self.g_take = gains("k_take",0.22,0.8,2.05,4.1) 
        self.g_traj = gains("k_traj",0.75,4.1,16.0,32.0) 
        

        # inverse mixer
        A = np.array([
            [self.kf]*4,
            [-0.22*self.kf,  0.20*self.kf,  0.22*self.kf, -0.20*self.kf],
            [-0.13*self.kf,  0.13*self.kf, -0.13*self.kf,  0.13*self.kf],
            [-self.km,       -self.km,       self.km,       self.km]
        ])
        self.invA = np.linalg.inv(A)

        # publishers & subscribers
        self.cmd_pub = rospy.Publisher(ns+'/command/motor_speed',Actuators,queue_size=1)
        topic=lambda s:'~'+s
        pubs=[
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
        self.pubs={n:rospy.Publisher(topic(n),t,queue_size=1) for n,t in pubs}

        if self.use_gz:
            from gazebo_msgs.msg import ModelStates
            self.sub=rospy.Subscriber('/gazebo/model_states',ModelStates,self.cb_model,
                                      queue_size=5,buff_size=2**24)
        else:
            self.sub=rospy.Subscriber(ns+'/ground_truth/odometry',Odometry,
                                      self.cb_odom,queue_size=10)

        self.state      = State.TAKEOFF
        self.last       = None
        rate            = pget("control_rate",100.0)
        self.timer      = rospy.Timer(rospy.Duration(1.0/rate),self.loop,reset=True)
        self.t0_traj    = None
        self.hover_ok_t = None
        rospy.on_shutdown(self.shutdown)

    def cb_odom(self,msg): self.last=msg # Used when use_gz is False

    def cb_model(self,msg): # Used when use_gz is True
        try: idx=msg.name.index(self.ns)
        except ValueError:
            try: idx=msg.name.index(self.ns+'/') 
            except ValueError: return
        o=Odometry(); o.header.stamp=rospy.Time.now()
        o.header.frame_id="world"; o.child_frame_id=self.ns+"/base_link" 
        o.pose.pose, o.twist.twist = msg.pose[idx], msg.twist[idx] 
        self.last=o

    @staticmethod
    def R(phi,th,psi): # Body-to-World from ZYX Euler angles
        c,s=math.cos,math.sin
        return np.array([
            [c(th)*c(psi),   s(phi)*s(th)*c(psi)-c(phi)*s(psi),  c(phi)*s(th)*c(psi)+s(phi)*s(psi)],
            [c(th)*s(psi),   s(phi)*s(th)*s(psi)+c(phi)*c(psi),  c(phi)*s(th)*s(psi)-s(phi)*c(psi)],
            [   -s(th),                 s(phi)*c(th),                   c(phi)*c(th)]
        ])

    def traj_ref(self,t):
        omt = self.omega_traj*t
        r  = self.r0 - self.k_r*omt
        z  = self.k_z*omt
        c,s=math.cos,math.sin
        x_rel = r*c(omt)
        y_rel = r*s(omt)
        pos = np.array([x_rel,y_rel,z])
        if self.xy_offset is not None:
            pos[0:2] += self.xy_offset
        if self.z_offset is not None:
            pos[2]   += self.z_offset
        dr = -self.k_r
        xp = dr*c(omt) - r*s(omt)
        yp = dr*s(omt) + r*c(omt)
        zp = self.k_z
        vel = np.array([xp,yp,zp]) * self.omega_traj
        a0 =  2*self.k_r*s(omt) - r*c(omt)
        a1 = -2*self.k_r*c(omt) - r*s(omt)
        acc= np.array([a0,a1,0.0]) * (self.omega_traj**2)
        psi_d = math.atan2(vel[1], vel[0])
        # compute desired yaw rate: ψ̇ = (vₓ·aᵧ − vᵧ·aₓ) / (vₓ² + vᵧ²)
        denom = vel[0]**2 + vel[1]**2
        rd = (vel[0]*acc[1] - vel[1]*acc[0]) / denom if denom > 1e-6 else 0.0
        return pos, vel, acc, psi_d, rd

    def loop(self,_):
        if self.last is None: return
        now=rospy.Time.now()
        
        p_msg = self.last.pose.pose.position
        q_msg = self.last.pose.pose.orientation
        v_raw_linear = self.last.twist.twist.linear    
        w_raw_angular = self.last.twist.twist.angular   

        x,y,z = p_msg.x, p_msg.y, p_msg.z
        phi,th,psi = euler_from_quaternion([q_msg.x,q_msg.y,q_msg.z,q_msg.w])
        
        current_R_body_to_world = self.R(phi, th, psi)
        

        if self.use_gz:

            v_w_world = np.array([v_raw_linear.x, v_raw_linear.y, v_raw_linear.z])
            

            omega_world_frame = np.array([w_raw_angular.x, w_raw_angular.y, w_raw_angular.z])
            omega_body_frame  = np.dot(current_R_body_to_world.T, omega_world_frame)
        else:

            v_in_body_frame = np.array([v_raw_linear.x, v_raw_linear.y, v_raw_linear.z])
            v_w_world    = np.dot(current_R_body_to_world, v_in_body_frame)
            
 
            omega_body_frame   = np.array([w_raw_angular.x, w_raw_angular.y, w_raw_angular.z])


        if self.state in (State.TAKEOFF,State.HOVER):
            tgt = np.array([self.x_to,self.y_to,self.z_to])
            vd=ad=np.zeros(3)
            yd,rd=self.yaw_fix,0.0 
            g1,g2,g3,g4=self.g_take
            
            err_p_vec = np.array([x-tgt[0],y-tgt[1],z-tgt[2]])
            err_p = np.linalg.norm(err_p_vec)
            err_v = np.linalg.norm(v_w_world - vd) # Check against desired velocity (0 for hover)

            if self.state==State.TAKEOFF and err_p<pget("hover_pos_threshold",.15) and err_v<pget("hover_vel_threshold",.1):
                self.state,self.hover_ok_t = State.HOVER,None
            if self.state==State.HOVER:
                if err_p<pget("hover_pos_threshold",.15) and err_v<pget("hover_vel_threshold",.1):
                    if self.hover_ok_t is None:
                        self.hover_ok_t=now
                    elif now-self.hover_ok_t>=rospy.Duration(pget("hover_stabilization_secs",2.0)):
                        self.state,self.t0_traj = State.TRAJ, now
                        self.xy_offset = np.array([self.x_to - self.r0, self.y_to])
                        self.z_offset  = self.z_to
                else:
                    self.hover_ok_t=None
        elif self.state==State.TRAJ:
            if self.t0_traj is None: 
                self.state=State.HOVER; rospy.logwarn_throttle(1.0, "TRAJ state entered with no t0_traj, reverting to HOVER"); return 
            posd,vd,ad,yd,rd = self.traj_ref((now-self.t0_traj).to_sec()) 
            tgt=posd; g1,g2,g3,g4=self.g_traj
        else: 
            tgt=np.array([x,y,z]) 
            vd=ad=np.zeros(3)
            yd,rd=psi,0.0 
            g1,g2,g3,g4=self.g_take

        ex1 = np.array([x,y,z]) - tgt
        ex2 = v_w_world - vd

        tof = math.cos(phi)*math.cos(th) 
        if abs(tof)<self.min_f: tof=sign(tof or 1.0)*self.min_f 
        
        Fz_des = self.m * (-g1*ex1[2] + ad[2] - g2*ex2[2] + self.g * self.gc)
        U1 = Fz_des / tof
        U1=max(0.0,U1) 

        if U1 < 1e-6: 
            Uex=Uey=0.0
        else:
            ax_des_world_component = -g1*ex1[0] + ad[0] - g2*ex2[0]
            ay_des_world_component = -g1*ex1[1] + ad[1] - g2*ex2[1]
            Uex = (self.m/U1) * ax_des_world_component
            Uey = (self.m/U1) * ay_des_world_component

        sp_yd, cp_yd = math.sin(yd), math.cos(yd)
        
        val_phi_d_arg = Uex*sp_yd - Uey*cp_yd
        phi_d = math.asin(clip(val_phi_d_arg, -1.0, 1.0))
        
        cos_phi_d = math.cos(phi_d)
        if abs(cos_phi_d) < self.min_f: 
            theta_d = 0.0 
            if abs(val_phi_d_arg) > 0.95 : # If desired roll is extreme
                 rospy.logwarn_throttle(1.0, "Desired roll is near +/-90 deg, desired pitch calculation might be unstable.")
        else:
            val_theta_d_arg = (Uex*cp_yd + Uey*sp_yd) / cos_phi_d
            theta_d = math.asin(clip(val_theta_d_arg, -1.0, 1.0))

        phi_d   = clip(phi_d, -self.max_tilt, self.max_tilt)
        theta_d = clip(theta_d, -self.max_tilt, self.max_tilt)

        e_phi = phi - phi_d
        e_theta = th - theta_d
        e_psi = (psi - yd + math.pi) % (2*math.pi) - math.pi
        attitude_error = np.array([e_phi, e_theta, e_psi])

        angular_rate_error  = omega_body_frame - np.array([0.0, 0.0, rd]) 

        p_b, q_b, r_b = omega_body_frame[0], omega_body_frame[1], omega_body_frame[2]
        U2 = self.Ix*(-g3*attitude_error[0] - g4*angular_rate_error[0]) - q_b*r_b*(self.Iy-self.Iz) 
        U3 = self.Iy*(-g3*attitude_error[1] - g4*angular_rate_error[1]) - p_b*r_b*(self.Iz-self.Ix) 
        U4 = self.Iz*(-g3*attitude_error[2] - g4*angular_rate_error[2]) - p_b*q_b*(self.Ix-self.Iy) 

        U_control_inputs = np.array([U1,U2,U3,U4])
        w_sq = clip(np.dot(self.invA, U_control_inputs), 0, None) 
        w_cmd= clip(rt(w_sq), 0, self.w_max)

        m_actuators=Actuators(); m_actuators.header.stamp=now; m_actuators.angular_velocities=w_cmd.tolist()
        self.cmd_pub.publish(m_actuators)

        self.pubs["control/state"].publish(String(data=self.state.name))
        self.pubs["control/U"].publish(Float64MultiArray(data=U_control_inputs))
        self.pubs["control/omega_sq"].publish(Float64MultiArray(data=w_sq))
        self.pubs["error/position"].publish(Point(*ex1))
        self.pubs["error/velocity"].publish(Vector3(*ex2))
        self.pubs["error/attitude_deg"].publish(Point(*(math.degrees(i) for i in attitude_error)))
        self.pubs["error/rates_deg_s"].publish(Vector3(*(math.degrees(i) for i in angular_rate_error)))
        self.pubs["control/desired_position"].publish(Point(*tgt))
        self.pubs["control/desired_velocity"].publish(Vector3(*vd))
        self.pubs["control/desired_acceleration"].publish(Vector3(*ad))
        self.pubs["control/desired_attitude_deg"].publish(Point(*(math.degrees(i) for i in (phi_d,theta_d,yd))))
        self.pubs["control/virtual_inputs"].publish(Point(Uex,Uey,0.0)) 

        if DBG:
             rospy.loginfo_throttle(LOG_T,"[%s] U1=%.2f U2=%.2f U3=%.2f U4=%.2f. PosErr: (%.2f,%.2f,%.2f). VelErr: (%.2f,%.2f,%.2f)",
                                   self.state.name,U1,U2,U3,U4, ex1[0],ex1[1],ex1[2], ex2[0],ex2[1],ex2[2])


    def shutdown(self):
        rospy.loginfo_once("CLF Iris Controller shutting down, sending zero motor commands.")
        stop_msg=Actuators(); stop_msg.angular_velocities=[0.0]*4
        try:
            control_rate = pget("control_rate", 100.0)
            sleep_duration = rospy.Duration(1.0 / control_rate)
            for _ in range(20): # Send for ~0.2s if rate is 100Hz
                if rospy.is_shutdown(): break
                self.cmd_pub.publish(stop_msg)
                rospy.sleep(sleep_duration)
        except Exception as e:
            rospy.logerr("Error during shutdown motor stop: %s", e)


if __name__=="__main__":
    rospy.init_node("clf_iris_trajectory_controller",anonymous=True)
    controller_instance = None
    try:
        controller_instance = ClfIrisController()
        rospy.spin()
    except rospy.ROSInterruptException:
        rospy.loginfo("CLF Iris Controller interrupted.")
    finally:
        # The rospy.on_shutdown hook is generally preferred,
        # but an explicit call here ensures it happens if shutdown is called on the instance.
        if controller_instance and hasattr(controller_instance, 'shutdown') and callable(controller_instance.shutdown):
             rospy.loginfo("Calling controller shutdown method in finally block.")
             controller_instance.shutdown() # Ensure motors are commanded to stop