# Copyright 2025 Enactic, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Custom command terms for the OpenArm reach tasks."""

from __future__ import annotations

import math
from typing import Sequence

import torch
import torch.nn.functional as F

from isaaclab.envs.mdp.commands.commands_cfg import UniformPoseCommandCfg
from isaaclab.envs.mdp.commands.pose_command import UniformPoseCommand
from isaaclab.utils import configclass


def _smooth_interp(progress: torch.Tensor, mode: str) -> torch.Tensor:
    """Apply a scalar easing function to the interpolation fraction."""

    if mode == "linear":
        return progress
    if mode == "smoothstep":
        return progress * progress * (3.0 - 2.0 * progress)
    if mode == "cosine":
        return 0.5 * (1.0 - torch.cos(math.pi * progress))
    raise ValueError(f"Unsupported interpolation mode: {mode}")


def _quat_slerp(q0: torch.Tensor, q1: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """Spherical linear interpolation for unit quaternions.

    Args:
        q0: Starting quaternion tensor with shape (..., 4) ordered as (w, x, y, z).
        q1: Target quaternion tensor with shape (..., 4).
        t: Interpolation parameter in [0, 1] with shape (...,).
    """

    # ensure both inputs point to the same hemisphere
    dot = torch.sum(q0 * q1, dim=-1)
    negate_mask = dot < 0.0
    q1 = torch.where(negate_mask.unsqueeze(-1), -q1, q1)
    dot = torch.where(negate_mask, -dot, dot)

    # handle nearly identical quaternions with lerp to avoid numerical issues
    threshold = 0.9995
    result = torch.empty_like(q0)
    close_mask = dot > threshold
    far_mask = ~close_mask

    if torch.any(close_mask):
        t_close = t[close_mask]
        q_close = F.normalize(
            q0[close_mask] + (q1[close_mask] - q0[close_mask]) * t_close.unsqueeze(-1), dim=-1
        )
        result[close_mask] = q_close

    if torch.any(far_mask):
        dot_far = dot[far_mask]
        q0_far = q0[far_mask]
        q1_far = q1[far_mask]
        t_far = t[far_mask]
        omega = torch.acos(torch.clamp(dot_far, -1.0, 1.0))
        sin_omega = torch.sin(omega)
        sin_omega = torch.where(sin_omega.abs() < 1e-6, torch.ones_like(sin_omega), sin_omega)

        coeff_0 = torch.sin((1.0 - t_far) * omega) / sin_omega
        coeff_1 = torch.sin(t_far * omega) / sin_omega

        result[far_mask] = coeff_0.unsqueeze(-1) * q0_far + coeff_1.unsqueeze(-1) * q1_far

    return F.normalize(result, dim=-1)


class SmoothPoseCommand(UniformPoseCommand):
    """Pose command that interpolates smoothly between randomly sampled targets."""

    cfg: "SmoothPoseCommandCfg"

    def __init__(self, cfg: "SmoothPoseCommandCfg", env):
        super().__init__(cfg, env)

        # buffers to track the segment start and end pose in the robot base frame
        self._segment_start_b = self.pose_command_b.clone()
        self._segment_goal_b = self.pose_command_b.clone()
        # duration buffer stores how long the current segment should take
        self._segment_duration = torch.ones(self.num_envs, device=self.device)

    def _resample_command(self, env_ids: Sequence[int]):
        if len(env_ids) == 0:
            return

        # preserve the current command as the starting pose for the new segment
        start_pose = self.pose_command_b[env_ids].clone()

        # sample a new goal pose using the base implementation
        super()._resample_command(env_ids)
        goal_pose = self.pose_command_b[env_ids].clone()

        # detect the first resample inside the episode (command_counter is updated afterwards)
        is_first_command = self.command_counter[env_ids] == 0

        # update cached segment data
        start_pose = torch.where(is_first_command.unsqueeze(-1), goal_pose, start_pose)
        self._segment_start_b[env_ids] = start_pose
        self._segment_goal_b[env_ids] = goal_pose
        self._segment_duration[env_ids] = torch.clamp(self.time_left[env_ids], min=1e-6)

        # normalize orientations to avoid numerical drift
        self._segment_start_b[env_ids, 3:] = F.normalize(self._segment_start_b[env_ids, 3:], dim=-1)
        self._segment_goal_b[env_ids, 3:] = F.normalize(self._segment_goal_b[env_ids, 3:], dim=-1)

        # reset the live command to the start pose so it can progress smoothly
        self.pose_command_b[env_ids] = self._segment_start_b[env_ids]

    def _update_command(self):
        # compute interpolation progress for each environment
        duration = torch.clamp(self._segment_duration, min=1e-6)
        remaining = torch.clamp(self.time_left, min=0.0)
        progress = torch.clamp((duration - remaining) / duration, 0.0, 1.0)
        eased_progress = _smooth_interp(progress, self.cfg.interpolation)

        # interpolate positions with the eased fraction
        eased_progress_expanded = eased_progress.unsqueeze(-1)
        self.pose_command_b[:, :3] = (
            self._segment_start_b[:, :3]
            + (self._segment_goal_b[:, :3] - self._segment_start_b[:, :3]) * eased_progress_expanded
        )

        # interpolate orientations either via slerp or by holding the goal orientation
        if self.cfg.orientation_slerp:
            self.pose_command_b[:, 3:] = _quat_slerp(
                self._segment_start_b[:, 3:], self._segment_goal_b[:, 3:], eased_progress
            )
        else:
            self.pose_command_b[:, 3:] = self._segment_goal_b[:, 3:]


@configclass
class SmoothPoseCommandCfg(UniformPoseCommandCfg):
    """Configuration for the smooth pose command generator."""

    class_type: type = SmoothPoseCommand

    interpolation: str = "smoothstep"
    """Interpolation mode used for the position timeline (linear, smoothstep, cosine)."""

    orientation_slerp: bool = True
    """Whether to interpolate the command orientation with SLERP."""
