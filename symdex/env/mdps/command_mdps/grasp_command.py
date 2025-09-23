from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

from isaaclab.assets import RigidObject, Articulation
from isaaclab.managers import CommandTerm
import isaaclab.utils.math as math_utils
from isaaclab.markers.visualization_markers import VisualizationMarkers

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

    from .grasp_command_cfg import TargetPositionCommandCfg

class TargetPositionCommand(CommandTerm):
    cfg: TargetPositionCommandCfg
    """Configuration for the command term."""

    def __init__(self, cfg: TargetPositionCommandCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.env = env
        # the target pick object is the first object spawned
        if isinstance(cfg.object_id, str):
            self.object: Articulation = self._env.scene[cfg.object_id]
        else:
            self.object: RigidObject = self._env.scene[f"object_{cfg.object_id}"]

        # Check that exactly one of pose_range and z_height is specified
        assert (cfg.pose_range is None) != (cfg.z_height is None), "Exactly one of pose_range and z_height must be specified"
        # goal pose range
        if cfg.pose_range is not None:
            range_list = [cfg.pose_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
            self.ranges = torch.tensor(range_list, device=self.device)
        else:
            self.ranges = None

        # goal height
        if cfg.z_height is not None:
            self.z_height = cfg.z_height
        else:
            self.z_height = None

        self.return_type = cfg.return_type

        # create buffers to store the command
        # -- command: (x, y, z)
        self.pos_command_e = torch.zeros(self.num_envs, 3, device=self.device)
        self.pos_command_w = self.pos_command_e + self._env.scene.env_origins
        # -- orientation: (w, x, y, z)
        self.quat_command_w = torch.zeros(self.num_envs, 4, device=self.device)
        self.quat_command_w[:, 0] = 1.0  # set the scalar component to 1.0

        # -- metrics
        self.metrics["orientation_error"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["position_error"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["consecutive_success"] = torch.zeros(self.num_envs, device=self.device)

    def __str__(self) -> str:
        msg = "GraspCommandGenerator:\n"
        msg += f"\tCommand dimension: {tuple(self.command.shape[1:])}\n"
        return msg

    """
    Properties
    """

    @property
    def command(self) -> torch.Tensor:
        """The desired goal position in the environment frame. Shape is (num_envs, 7)."""
        # return self.pos_command_e
        if self.ranges is not None:
            if self.return_type == "pos":
                return self.pos_command_e
            elif self.return_type == "quat":
                return self.quat_command_w
            else:
                return torch.cat((self.pos_command_e, self.quat_command_w), dim=-1)
        else:
            return self.pos_command_e

    """
    Implementation specific functions.
    """
    def _update_metrics(self):
        # logs data
        # -- compute the orientation error
        from symdex.utils.isaac_utils import get_angle_from_quat
        target_axis = get_angle_from_quat(self.quat_command_w, axis="z", normalize=True)
        cur_axis = get_angle_from_quat(self.object.data.root_quat_w, axis="z", normalize=True)
        self.metrics["orientation_error"] = torch.sum(target_axis * cur_axis, dim=-1)
        # -- compute the position error
        self.metrics["position_error"] = torch.norm(self.object.data.root_pos_w - self.pos_command_w, dim=1)
        # -- compute the number of consecutive successes
        if self.ranges is not None and self.return_type != "pos":
            successes = (torch.logical_and(self.metrics["position_error"] < self.cfg.success_threshold, self.metrics["orientation_error"] > self.cfg.success_threshold_orient)).float()
        else:
            successes = (self.metrics["position_error"] < self.cfg.success_threshold).float()
        unsuccesses = torch.where(successes == 0.0)[0]
        self.metrics["consecutive_success"] += successes.float()
        self.metrics["consecutive_success"][unsuccesses] = 0.0

    def _resample_command(self, env_ids: Sequence[int]):
        # sample the goal position       
        if self.ranges is not None:
            rand_samples = math_utils.sample_uniform(self.ranges[:, 0], self.ranges[:, 1], (len(env_ids), 6), device=self.device)
            if self.cfg.offset:
                if isinstance(self.cfg.object_id, str) or self.cfg.use_initial_pose:
                    rand_samples[:, 0:3] += self.object.data.root_pos_w[env_ids] - self._env.scene.env_origins[env_ids]
                else:
                    rand_samples[:, 0:3] += self.env.object_init_pos[self.cfg.object_id, env_ids] - self._env.scene.env_origins[env_ids]
            self.pos_command_e[env_ids] = rand_samples[:, 0:3]
            self.pos_command_w[env_ids] = self.pos_command_e[env_ids] + self._env.scene.env_origins[env_ids]
            orientations_delta = math_utils.quat_from_euler_xyz(rand_samples[:, 3], rand_samples[:, 4], rand_samples[:, 5])
            self.quat_command_w[env_ids] = math_utils.quat_mul(self.object.data.default_root_state[env_ids, 3:7], orientations_delta)
        elif self.z_height is not None:
            if self.cfg.use_initial_pose:
                self.pos_command_w[env_ids] = self.object.data.root_pos_w[env_ids]
            else:
                self.pos_command_w[env_ids] = self.env.object_init_pos[self.cfg.object_id, env_ids]
            self.pos_command_w[env_ids, 2] += self.z_height
            self.pos_command_e[env_ids] = self.pos_command_w[env_ids] - self._env.scene.env_origins[env_ids]

    def _update_command(self):
        # update the command if goal is reached
        if self.cfg.update_goal_on_success:
            # compute the goal resets
            if self.ranges is not None:
                goal_resets = torch.logical_and(self.metrics["position_error"] < self.cfg.success_threshold, self.metrics["orientation_error"] > self.cfg.success_threshold_orient)
            elif self.z_height is not None:
                goal_resets = self.metrics["position_error"] < self.cfg.success_threshold
            else:
                raise ValueError("No goal range or height specified")
            goal_reset_ids = goal_resets.nonzero(as_tuple=False).squeeze(-1)
            # resample the goals
            self._resample(goal_reset_ids)

    def _set_debug_vis_impl(self, debug_vis: TYPE_CHECKING):
        # set visibility of markers
        # note: parent only deals with callbacks. not their visibility
        if debug_vis:
            # create markers if necessary for the first time
            if not hasattr(self, "goal_marker_visualizer"):
                self.goal_marker_visualizer = VisualizationMarkers(self.cfg.visualizer_cfg)
            # set visibility
            self.goal_marker_visualizer.set_visibility(True)
        else:
            if hasattr(self, "goal_marker_visualizer"):
                self.goal_marker_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        # add an offset to the marker position to visualize the goal
        marker_pos = self.pos_command_w
        marker_quat = self.quat_command_w
        # visualize the goal marker
        self.goal_marker_visualizer.visualize(translations=marker_pos, orientations=marker_quat)