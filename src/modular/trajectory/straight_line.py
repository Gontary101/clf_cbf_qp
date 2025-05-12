#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import division
import math
import numpy as np
import ast
from dynamics_utils import pget

class StraightLineTrajectory(object):
    def __init__(self):
        self.omega = 0.5
        self.yaw_fix = math.radians(pget("fixed_yaw_deg", 0.0))

        default_xyz = [-20.0, -10.0, 7.0]
        default_target = np.array(default_xyz, dtype=float)

        target_param_val = pget("straight_line_target_xyz", default_xyz)
        if isinstance(target_param_val, str):
            try:
                self.target_pos_world_final = np.array(
                    ast.literal_eval(target_param_val), dtype=float
                )
            except (ValueError, SyntaxError):
                self.target_pos_world_final = default_target
        elif (
            isinstance(target_param_val, list)
            and len(target_param_val) == 3
            and all(isinstance(n, (int, float)) for n in target_param_val)
        ):
            self.target_pos_world_final = np.array(target_param_val, dtype=float)
        else:
            self.target_pos_world_final = default_target

        if self.target_pos_world_final.shape != (3,):
            self.target_pos_world_final = default_target

        # ------------------- start position config ----------------------
        # allow overriding the start in world‐coords, with a sane default
        default_start_xyz = [20.0, 10.0, 5.0]
        default_start     = np.array(default_start_xyz, dtype=float)
        # if no param given, fall back to our default_start_xyz
        start_param_val = pget("straight_line_start_xyz", default_start_xyz)
        if isinstance(start_param_val, str):
            try:
                self.start_pos_world_initial = np.array(
                    ast.literal_eval(start_param_val), dtype=float
                )
            except (ValueError, SyntaxError):
                self.start_pos_world_initial = default_start
        elif (
            isinstance(start_param_val, list)
            and len(start_param_val) == 3
            and all(isinstance(n, (int, float)) for n in start_param_val)
        ):
            self.start_pos_world_initial = np.array(start_param_val, dtype=float)
        else:
            # no user override → use the built-in default_start
            self.start_pos_world_initial = default_start

        # Helix parameters
        self.r0_for_offset_calculation = 0.5 * pget("helix_start_diameter", 40.0)

        # Placeholders for computed offsets and actuals
        self.xy_offset = None
        self.z_offset = None
        self.actual_start_pos_world = None
        self.actual_direction_vector_world = None
        self.actual_total_distance_world = None
        self.actual_unit_direction_vector_world = None
        self.actual_total_duration = None
        self._initialized_internals = False

    def _initialize_internals_if_needed(self):
        # if no explicit start override and offsets aren't set yet, wait
        if self.start_pos_world_initial is None and (self.xy_offset is None or self.z_offset is None):
            return 

        if not self._initialized_internals:
            if self.start_pos_world_initial is not None:
                # use the user‐provided start in world frame
                self.actual_start_pos_world = self.start_pos_world_initial
            else:
                # compute start from local frame + offsets
                local_start_pos_traj_frame = np.array([self.r0_for_offset_calculation, 0.0, 0.0])
                self.actual_start_pos_world = np.array([
                    local_start_pos_traj_frame[0] + self.xy_offset[0],
                    local_start_pos_traj_frame[1] + self.xy_offset[1],
                    local_start_pos_traj_frame[2] + self.z_offset
                ])
            
            self.actual_direction_vector_world = self.target_pos_world_final - self.actual_start_pos_world
            self.actual_total_distance_world = np.linalg.norm(self.actual_direction_vector_world)

            if self.actual_total_distance_world < 1e-6:
                self.actual_unit_direction_vector_world = np.zeros(3)
                self.actual_total_duration = 0.0
            else:
                self.actual_unit_direction_vector_world = self.actual_direction_vector_world / self.actual_total_distance_world
                if self.omega <= 1e-6: 
                    self.actual_total_duration = float('inf') 
                else:
                    self.actual_total_duration = self.actual_total_distance_world / self.omega
            
            self._initialized_internals = True

    def ref(self, t):
        if not self._initialized_internals:
            self._initialize_internals_if_needed()

        if not self._initialized_internals:
            pos = np.array([self.r0_for_offset_calculation, 0.0, 0.0]) 
            vel = np.zeros(3)
            acc = np.zeros(3)
        else:
            if t >= self.actual_total_duration or self.actual_total_distance_world < 1e-6:
                pos = self.target_pos_world_final if self.actual_total_distance_world >= 1e-6 else self.actual_start_pos_world
                vel = np.zeros(3)
                acc = np.zeros(3)
            elif self.omega <= 1e-6:
                pos = self.actual_start_pos_world
                vel = np.zeros(3)
                acc = np.zeros(3)
            else:
                distance_covered = self.omega * t
                pos = self.actual_start_pos_world + self.actual_unit_direction_vector_world * distance_covered
                vel = self.actual_unit_direction_vector_world * self.omega
                acc = np.zeros(3) 

        psi_d = self.yaw_fix
        rd = 0.0

        return pos, vel, acc, psi_d, rd