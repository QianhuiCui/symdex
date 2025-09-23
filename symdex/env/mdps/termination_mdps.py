from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def max_consecutive_success(env: ManagerBasedRLEnv, num_success: int, command_names: str | list[str]) -> torch.Tensor:
    """Check if the task has been completed consecutively for a certain number of times.

    Args:
        env: The environment object.
        num_success: Threshold for the number of consecutive successes required.
        command_name: The command term to be used for extracting the goal.
    """
    if isinstance(command_names, str):
        command_term = env.command_manager.get_term(command_names)
        success = command_term.metrics["consecutive_success"] >= num_success
    else:
        success = torch.ones(env.num_envs, device=env.device)
        for command_name in command_names:
            command_term = env.command_manager.get_term(command_name)
            success = torch.logical_and(success, command_term.metrics["consecutive_success"] >= num_success)
    env.success_tracker = success.float()
    return success

def object_away_from_robot(
    env: ManagerBasedRLEnv,
    threshold: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    link_name = None,
    object_id: int = 0,
) -> torch.Tensor:
    """Check if object has gone far from the robot.

    The object is considered to be out-of-reach if the distance between the robot and the object is greater
    than the threshold.

    Args:
        env: The environment object.
        threshold: The threshold for the distance between the robot and the object.
        asset_cfg: The configuration for the robot entity. Default is "robot".
        object_id: The id for the object entity. Default is 0.
    """
    # extract useful elements
    robot = env.scene[asset_cfg.name]
    object: RigidObject = env.scene[f"object_{object_id}"]

    if link_name is not None:
        link_idx = robot.find_bodies(link_name)[0]
        dist = torch.norm(robot.data.body_state_w[:, link_idx, :3].reshape(-1, 3) - object.data.root_pos_w, dim=1)
    else:
        dist = torch.norm(robot.data.root_pos_w - object.data.root_pos_w, dim=1)

    return dist > threshold