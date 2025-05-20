import numpy as np
from trajectory.helix import HelixTrajectory

class TrajectoryModule(object):
    def __init__(self):
        self.trajectory = HelixTrajectory()
        self.omega_traj = self.trajectory.omega
        self.yaw_fix = self.trajectory.yaw_fix
        self.r0 = self.trajectory.r0
        
        self.xy_offset = None
        self.z_offset = None
        self.psi_traj0 = self.trajectory.ref(0.0)[3]

    def set_offsets(self, x_to, y_to, z_to):
        if self.r0 is not None:
            self.xy_offset = np.array([x_to - self.r0, y_to])
        else:
            self.xy_offset = np.array([x_to, y_to])
        self.z_offset = z_to
        
        self.trajectory.xy_offset = self.xy_offset
        self.trajectory.z_offset = self.z_offset

    def get_reference(self, t_traj):
        return self.trajectory.ref(t_traj)

    def get_psi_traj0(self):
        return self.psi_traj0

    def get_yaw_fix(self):
        return self.yaw_fix