import numpy as np
import rospy
from components.state_enum import State
from utils.dynamics_utils2 import pget

class FlightStateManager(object):
    def __init__(self, initial_takeoff_x, initial_takeoff_y, initial_takeoff_z,
                 gains_takeoff, gains_traj, trajectory_module):
        self.current_state = State.TAKEOFF
        self.t0_traj = None
        self.hover_ok_t = None

        self.x_to = initial_takeoff_x 
        self.y_to = initial_takeoff_y
        self.z_to = initial_takeoff_z
        
        self.g_take = gains_takeoff
        self.g_traj = gains_traj

        self.trajectory_module = trajectory_module
        self.psi_traj0 = self.trajectory_module.get_psi_traj0()
        self.yaw_fix = self.trajectory_module.get_yaw_fix()
        
        self.pos_thr = pget("hover_pos_threshold", 0.15)
        self.vel_thr = pget("hover_vel_threshold", 0.10)    
        self.hover_stabilization_duration = rospy.Duration(pget("hover_stabilization_secs", 2.0))

    def update_state(self, current_time, current_kinematics):
        p_vec = current_kinematics["p_vec"]
        v_world = current_kinematics["v_world"]
        psi = current_kinematics["psi"]
        
        t_traj_sec = 0.0
        if self.current_state == State.TRAJ and self.t0_traj is not None:
            t_traj_sec = (current_time - self.t0_traj).to_sec()

        gains = self.g_take
        tgt = np.zeros(3)
        vd = np.zeros(3)
        ad_nom = np.zeros(3)
        yd = 0.0
        rd = 0.0
        
        if self.current_state in (State.TAKEOFF, State.HOVER):
            tgt = np.array([self.x_to, self.y_to, self.z_to])
            vd = ad_nom = np.zeros(3)
            yd, rd = (self.psi_traj0, 0.0) if self.current_state == State.HOVER else (self.yaw_fix, 0.0)
            gains = self.g_take

            err_z = abs(p_vec[2] - tgt[2])
            err_v = np.linalg.norm(v_world - vd)

            if self.current_state == State.TAKEOFF and err_z < self.pos_thr and err_v < self.vel_thr:
                rospy.loginfo("TRANSITION  ->  HOVER")
                self.current_state, self.hover_ok_t = State.HOVER, None
                yd, rd = self.psi_traj0, 0.0 

            if self.current_state == State.HOVER: # Check again in case of transition
                if err_z < self.pos_thr and err_v < self.vel_thr:
                    if self.hover_ok_t is None:
                        self.hover_ok_t = current_time
                    elif (current_time - self.hover_ok_t) >= self.hover_stabilization_duration:
                        rospy.loginfo("TRANSITION ->  TRAJ")
                        self.current_state, self.t0_traj = State.TRAJ, current_time
                        self.trajectory_module.set_offsets(self.x_to, self.y_to, self.z_to)
                        t_traj_sec = (current_time - self.t0_traj).to_sec()
                else:
                    self.hover_ok_t = None
        
        if self.current_state == State.TRAJ:
            if self.t0_traj is None:
                rospy.logwarn_throttle(5.0, "In TRAJ state but t0_traj is None. Reverting to HOVER.")
                self.current_state = State.HOVER
                tgt = np.array([self.x_to, self.y_to, self.z_to])
                vd = ad_nom = np.zeros(3)
                yd, rd = self.psi_traj0, 0.0
                gains = self.g_take
            else:
                posd, vd, ad_nom, yd, rd = self.trajectory_module.get_reference(t_traj_sec)
                tgt = posd
                gains = self.g_traj
        
        elif self.current_state not in (State.TAKEOFF, State.HOVER, State.TRAJ):
            tgt = p_vec 
            vd = ad_nom = np.zeros(3)
            yd, rd = psi, 0.0 
            gains = self.g_take

        references = {
            "tgt": tgt,
            "vd": vd,
            "ad": ad_nom,
            "yd": yd,
            "rd": rd,
            "t_traj_secs": t_traj_sec,
            "state_name": self.current_state.name
        }
        
        return references, gains

    def get_current_state_name(self):
        return self.current_state.name

    def get_current_state_enum(self):
        return self.current_state