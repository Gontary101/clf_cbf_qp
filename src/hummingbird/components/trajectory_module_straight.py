#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
from trajectory.straight_line import StraightLineTrajectory

class TrajectoryModuleStraight(object):
    def __init__(self):
        """
        Initializes the straight line trajectory module.
        """
        self.trajectory = StraightLineTrajectory()
        self.yaw_fix = self.trajectory.yaw_fix
        self.p0 = self.trajectory.p0  # Initial point of the straight line
        
        # Offsets will be set by FlightStateManager or similar
        self.xy_offset = None 
        self.z_offset = None
        
        # Initial desired yaw from the trajectory definition
        self.psi_traj0 = self.trajectory.psi_d

    def set_offsets(self, x_offset_val, y_offset_val, z_offset_val):
        """
        Sets the trajectory offsets. These are typically determined by the
        stabilization point after TAKEOFF or HOVER.
        """
        self.xy_offset = np.array([x_offset_val, y_offset_val])
        self.z_offset = z_offset_val
        
        self.trajectory.xy_offset = self.xy_offset
        self.trajectory.z_offset = self.z_offset

    def get_reference(self, t_traj):
        """
        Gets the reference signals (position, velocity, acceleration, yaw, yaw_rate)
        from the straight line trajectory at a given trajectory time t_traj.

        Args:
            t_traj (float): The current trajectory time.

        Returns:
            tuple: A tuple containing (pos_d, vel_d, acc_d, yaw_d, yaw_rate_d)
                   where:
                     pos_d (np.ndarray): Desired position [x, y, z].
                     vel_d (np.ndarray): Desired velocity [vx, vy, vz].
                     acc_d (np.ndarray): Desired acceleration [ax, ay, az].
                     yaw_d (float): Desired yaw angle.
                     yaw_rate_d (float): Desired yaw rate.
        """
        return self.trajectory.ref(t_traj)

    def get_psi_traj0(self):
        """
        Returns the initial desired yaw angle of the trajectory.
        """
        return self.psi_traj0

    def get_yaw_fix(self):
        """
        Returns whether the yaw is fixed for this trajectory.
        """
        return self.yaw_fix

    def get_p0(self):
        """
        Returns the initial point (p0) of the straight line trajectory.
        This can be used to determine hover targets or reference points.
        """
        return self.p0
