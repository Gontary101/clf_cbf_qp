import numpy as np
from trajectory.straight_line import StraightLineTrajectory

class TrajectoryModuleStraight(object):
    def __init__(self):
        self.trajectory = StraightLineTrajectory()
        self.yaw_fix = self.trajectory.yaw_fix
        self.p0 = self.trajectory.p0
        
        self.xy_offset = None 
        self.z_offset = None
        
        self.psi_traj0 = self.trajectory.psi_d

    def set_offsets(self, x_offset_val, y_offset_val, z_offset_val):
        self.xy_offset = np.array([x_offset_val, y_offset_val])
        self.z_offset = z_offset_val
        
        self.trajectory.xy_offset = self.xy_offset
        self.trajectory.z_offset = self.z_offset

    def get_reference(self, t_traj):
        return self.trajectory.ref(t_traj)

    def get_psi_traj0(self):
        return self.psi_traj0

    def get_yaw_fix(self):
        return self.yaw_fix

    def get_p0(self):
        return self.p0
