import numpy as np
from trajectory.helix import HelixTrajectory
import rospy

class TrajectoryModule(object):
    def __init__(self):
        self.trajectory = HelixTrajectory()
        self.omega_traj = self.trajectory.omega
        self.yaw_fix = self.trajectory.yaw_fix
        self.r0 = self.trajectory.r0 # Helix radius
        
        self.xy_offset = None
        self.z_offset = None
        self.psi_traj0 = self.trajectory.ref(0.0)[3]

    def set_offsets(self, x_offset_val, y_offset_val, z_offset_val):
        self.xy_offset = np.array([x_offset_val, y_offset_val])
        self.z_offset = z_offset_val

        self.trajectory.xy_offset = self.xy_offset
        self.trajectory.z_offset = self.z_offset

    def get_reference(self, t_traj):
        return self.trajectory.ref(t_traj)

    def get_p0(self):
        if self.r0 is not None:
            return np.array([self.r0, 0.0, 0.0])
        else:
            rospy.logwarn_throttle(5.0, "Helix TrajectoryModule: r0 is None, returning [0,0,0] for get_p0(). This might be incorrect.")
            return np.array([0.0, 0.0, 0.0])

    def get_psi_traj0(self):
        return self.psi_traj0

    def get_yaw_fix(self):
        return self.yaw_fix