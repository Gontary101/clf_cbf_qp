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

# ---------- helpers ----------
clip  = np.clip
sign  = np.sign
sq, rt = np.square, np.sqrt
def pget(name, default): return rospy.get_param("~"+name, default)

class State(Enum): TAKEOFF=1; HOVER=2; TRAJ=3; LAND=4; IDLE=5
LOG_T   = 1.0
DBG     = True

# ---------- main class ----------
class ClfIrisController(object):
    def __init__(self):
        ns               = pget("namespace","iris")
        self.ns          = ns
        self.use_gz      = pget("use_model_states",False)

        # vehicle & motor constants
        self.m, g        = pget("mass",1.5), pget("gravity",9.81)
        self.Ix, self.Iy,self.Iz = pget("I_x",.0348), pget("I_y",.0459), pget("I_z",.0977)
        self.kf, self.km = pget("motor_constant",8.54858e-06), pget("moment_constant",1.3677728e-07)
        self.w_max       = pget("max_rot_velocity",838.0)
        self.min_f       = pget("min_thrust_factor",0.1)
        self.gc          = pget("gravity_comp_factor",1.022)
        self.max_tilt    = math.radians(pget("max_tilt_angle_deg",30.0))

        # trajectory + take‑off targets
        self.a, self.b   = pget("ellipse_a",31.0), pget("ellipse_b",21.0)
        self.z_traj      = pget("trajectory_z",2.0)
        self.omega_traj  = pget("trajectory_omega",0.1)
        self.yaw_fix     = math.radians(pget("fixed_yaw_deg",0.0))

        self.x_to, self.y_to, self.z_to = self.a,0.0,self.z_traj   # take‑off target

        # gains
        def gains(prefix, kp1, kp2, ka1, ka2):
            return [pget(prefix+k,i) for k,i in
                    zip(("pos1","pos2","att1","att2"),(kp1,kp2,ka1,ka2))]
        self.g_take = gains("k_take",0.18,0.65,2.5,5)
        self.g_traj = gains("k_traj",0.75,4.1,16.0,32.0)

        # inverse mixer (fixed geometry)
        A=np.array([[self.kf]*4,
                    [-0.22*self.kf, 0.20*self.kf, 0.22*self.kf,-0.20*self.kf],
                    [-0.13*self.kf, 0.13*self.kf,-0.13*self.kf, 0.13*self.kf],
                    [-self.km,-self.km,self.km,self.km]])
        self.invA=np.linalg.inv(A)

        # pubs / subs
        self.cmd_pub=rospy.Publisher(ns+'/command/motor_speed',Actuators,queue_size=1)
        topic=lambda s:'~'+s
        self.pubs={n:rospy.Publisher(topic(n),t,queue_size=1) for n,t in
                   [("control/state",String),("control/U",Float64MultiArray),
                    ("error/position",Point),("error/velocity",Vector3),
                    ("error/attitude_deg",Point),("error/rates_deg_s",Vector3),
                    ("control/desired_position",Point),
                    ("control/desired_velocity",Vector3),
                    ("control/desired_acceleration",Vector3),
                    ("control/desired_attitude_deg",Point),
                    ("control/virtual_inputs",Point),
                    ("control/omega_sq",Float64MultiArray)]}

        if self.use_gz:
            from gazebo_msgs.msg import ModelStates
            self.sub=rospy.Subscriber('/gazebo/model_states',ModelStates,
                                       self.cb_model,queue_size=5,buff_size=2**24)
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

    # ---------- callbacks ----------
    def cb_odom (self,msg): self.last=msg
    def cb_model(self,msg):
        try:
            idx=msg.name.index(self.ns)
        except ValueError:
            try: idx=msg.name.index(self.ns+'/')
            except ValueError: return
        o=Odometry(); o.header.stamp=rospy.Time.now()
        o.header.frame_id="world"; o.child_frame_id=self.ns+"/base_link"
        o.pose.pose, o.twist.twist = msg.pose[idx], msg.twist[idx]
        self.last=o

    # ---------- math ----------
    @staticmethod
    def R(phi,th,psi):
        c,s=math.cos,math.sin
        return np.array([[ c(th)*c(psi), s(phi)*s(th)*c(psi)-c(phi)*s(psi),
                           c(phi)*s(th)*c(psi)+s(phi)*s(psi)],
                         [ c(th)*s(psi), s(phi)*s(th)*s(psi)+c(phi)*c(psi),
                           c(phi)*s(th)*s(psi)-s(phi)*c(psi)],
                         [-s(th),        s(phi)*c(th),                   c(phi)*c(th)]])

    def traj_ref(self,t):
        th=self.omega_traj*t
        c,s=math.cos,math.sin
        pos=np.array([ self.a*c(th), self.b*s(th), self.z_traj])
        vel=np.array([-self.a*s(th), self.b*c(th),0])*self.omega_traj
        acc=-np.array([ self.a*c(th), self.b*s(th),0])*self.omega_traj**2
        return pos,vel,acc,self.yaw_fix,0.0

    # ---------- control loop ----------
    def loop(self,_):
        if self.last is None: return
        now=rospy.Time.now()

        # pose / twist
        p=self.last.pose.pose.position
        q=self.last.pose.pose.orientation
        v=self.last.twist.twist.linear
        w=self.last.twist.twist.angular
        x,y,z, vx,vy,vz = p.x,p.y,p.z, v.x,v.y,v.z
        phi,th,psi = euler_from_quaternion([q.x,q.y,q.z,q.w])
        p_body=np.array([vx,vy,vz])
        v_w = np.dot(self.R(phi,th,psi),p_body)

        # choose targets & gains
        if self.state in (State.TAKEOFF,State.HOVER):
            tgt=np.array([self.x_to,self.y_to,self.z_to])
            vd = ad = np.zeros(3)
            yd,rd = self.yaw_fix,0.0
            g1,g2,g3,g4 = self.g_take
            err_pos=np.linalg.norm([x-tgt[0],y-tgt[1],z-tgt[2]])
            err_v  = np.linalg.norm(v_w)
            if self.state==State.TAKEOFF and err_pos< pget("hover_pos_threshold",.15) \
               and err_v   < pget("hover_vel_threshold",.1):
                self.state=State.HOVER; self.hover_ok_t=None
            if self.state==State.HOVER:
                if err_pos<pget("hover_pos_threshold",.15) and err_v<pget("hover_vel_threshold",.1):
                    if self.hover_ok_t is None: self.hover_ok_t=now
                    elif now-self.hover_ok_t>=rospy.Duration(pget("hover_stabilization_secs",2.0)):
                        self.state=State.TRAJ; self.t0_traj=now
                else: self.hover_ok_t=None
        elif self.state==State.TRAJ:
            if self.t0_traj is None: self.state=State.HOVER; return
            t=(now-self.t0_traj).to_sec()
            posd,vd,ad,yd,rd = self.traj_ref(t)
            tgt=posd; g1,g2,g3,g4 = self.g_traj
        else: # fallback safe hold
            tgt=np.array([x,y,z]); vd=ad=np.zeros(3); yd,rd=psi,0.0
            g1,g2,g3,g4=self.g_take

        ex1=np.array([x,y,z])-tgt
        ex2=v_w-vd
        cphi,cth=math.cos(phi),math.cos(th)
        tof=cphi*cth
        if abs(tof)<self.min_f: tof=sign(tof or 1)*self.min_f
        U1=(self.m/tof)*(-g1*ex1[2]+ad[2]-g2*ex2[2])+self.m*pget("gravity",9.81)*self.gc/tof
        U1=max(0.0,U1)
        if U1<1e-6: Uex=Uey=0.0
        else:
            Uex=(self.m/U1)*(-g1*ex1[0]+ad[0]-g2*ex2[0])
            Uey=(self.m/U1)*(-g1*ex1[1]+ad[1]-g2*ex2[1])

        sp,cp=math.sin(yd),math.cos(yd)
        phi_d=math.asin(clip(Uex*sp-Uey*cp,-1,1))
        cpd  =math.cos(phi_d)
        theta_d=0.0 if abs(cpd)<self.min_f else \
                math.asin(clip((Uex*cp+Uey*sp)/cpd,-1,1))
        phi_d,theta_d = clip(phi_d,-self.max_tilt,self.max_tilt),clip(theta_d,-self.max_tilt,self.max_tilt)

        e_th = np.array([phi-phi_d, th-theta_d, (psi-yd+math.pi)%(2*math.pi)-math.pi])
        e_w  = np.array([w.x, w.y, w.z]) - np.array([0.0,0.0,rd])
        U2 = self.Ix*(-g3*e_th[0] - g4*e_w[0]) - w.y*w.z*(self.Iy-self.Iz)
        U3 = self.Iy*(-g3*e_th[1] - g4*e_w[1]) - w.x*w.z*(self.Iz-self.Ix)
        U4 = self.Iz*(-g3*e_th[2] - g4*e_w[2]) - w.x*w.y*(self.Ix-self.Iy)

        U=np.array([U1,U2,U3,U4])
        w_sq= np.dot(self.invA,U); w_sq=clip(w_sq,0,None); w_cmd=rt(w_sq); w_cmd=clip(w_cmd,0,self.w_max)

        # motor cmd
        msg=Actuators(); msg.header.stamp=now; msg.angular_velocities=w_cmd.tolist()
        self.cmd_pub.publish(msg)

        # debug pubs
        self.pubs["control/state"].publish(String(data=self.state.name))
        self.pubs["control/U"].publish(Float64MultiArray(data=U))
        self.pubs["control/omega_sq"].publish(Float64MultiArray(data=w_sq))
        self.pubs["error/position"].publish(Point(*ex1))
        self.pubs["error/velocity"].publish(Vector3(*ex2))
        self.pubs["error/attitude_deg"].publish(Point(*(math.degrees(i) for i in e_th)))
        self.pubs["error/rates_deg_s"].publish(Vector3(*(math.degrees(i) for i in e_w)))
        self.pubs["control/desired_position"].publish(Point(*tgt))
        self.pubs["control/desired_velocity"].publish(Vector3(*vd))
        self.pubs["control/desired_acceleration"].publish(Vector3(*ad))
        self.pubs["control/desired_attitude_deg"].publish(Point(*(math.degrees(i) for i in (phi_d,theta_d,yd))))
        self.pubs["control/virtual_inputs"].publish(Point(Uex,Uey,0.0))

        # log
        if DBG:
            rospy.loginfo_throttle(LOG_T,
                "[%s] x=%.2f y=%.2f z=%.2f | U1=%.2f U2=%.2f U3=%.2f U4=%.2f | ω %s",
                self.state.name,x,y,z,U1,U2,U3,U4,", ".join("%.0f"%w for w in w_cmd))

    # ---------- shutdown ----------
    def shutdown(self):
        msg=Actuators(); msg.angular_velocities=[0.0]*4
        for _ in range(10):
            self.cmd_pub.publish(msg); rospy.sleep(0.01)

# ---------- main ----------
if __name__=="__main__":
    rospy.init_node("clf_iris_trajectory_controller",anonymous=True)
    try: ClfIrisController(); rospy.spin()
    except rospy.ROSInterruptException: pass
