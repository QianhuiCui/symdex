from __future__ import annotations

import torch
from typing import TYPE_CHECKING
from functools import reduce
from isaaclab.assets import RigidObject

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

from symdex.env.mdps.reward_mdps import get_allegro_contact

def cmd_success_bonus(
    env: ManagerBasedRLEnv, command_names: str | list[str], num_success: int = 0,
) -> torch.Tensor:
    if isinstance(command_names, str):
        command_term = env.command_manager.get_term(command_names)
        success = command_term.metrics["consecutive_success"] >= num_success
    else:
        success = torch.ones(env.num_envs, device=env.device)
        for command_name in command_names:
            command_term = env.command_manager.get_term(command_name)
            success = torch.logical_and(success, command_term.metrics["consecutive_success"] >= num_success)
    rew = success.float()
    return rew

def if_egg_beater_success(env: ManagerBasedRLEnv, sensor_names: list):
    egg_beater_command_term = env.command_manager.get_term("target_pos")
    within_range = torch.ones(env.num_envs, device=env.device)
    within_range = reduce(torch.logical_and, 
                          [within_range, 
                           egg_beater_command_term.metrics["position_error"] < egg_beater_command_term.cfg.success_threshold, 
                           egg_beater_command_term.metrics["orientation_error"] > 0.8])
    if sensor_names is not None:
        within_range = within_range * get_allegro_contact(env, sensor_names)
    return within_range.float()

def success_bonus(
    env: ManagerBasedRLEnv, num_success: int = 0,   
) -> torch.Tensor:
    bowl_command_term = env.command_manager.get_term("bowl_target_pos")
    bowl_within_range = bowl_command_term.metrics["position_error"] < 0.1
    egg_beater_command_term = env.command_manager.get_term("target_pos")
    egg_beater_within_range = egg_beater_command_term.metrics["position_error"] < 0.1
    egg_beater_orient_within_range = egg_beater_command_term.metrics["orientation_error"] > 0.7
    rew = bowl_within_range * egg_beater_within_range * egg_beater_orient_within_range
    env.success_tracker = rew.float()
    return rew

def object_vel(
    env: ManagerBasedRLEnv,
    object_id: int | list[int] = 0,
    sensor_names: list = ["contact_sensors_0", "contact_sensors_1", "contact_sensors_2", "contact_sensors_3"],
) -> torch.Tensor:
    """Reward for encourage the velocity"""
    # extract the used quantities (to enable type-hinting)
    if isinstance(object_id, int):
        object: RigidObject = env.scene[f"object_{object_id}"]
        vel = torch.abs(object.data.root_lin_vel_w[:, :2])
        # vel += torch.abs(object.data.root_ang_vel_w[:, :2])
    else:
        vel = None
        for id in object_id:
            object = env.scene[f"object_{id}"]
            if vel is None:
                vel = torch.abs(object.data.root_lin_vel_w[:, :2])
            else:
                vel += torch.abs(object.data.root_lin_vel_w[:, :2])
    bowl_command_term = env.command_manager.get_term("bowl_target_pos")
    success = bowl_command_term.metrics["consecutive_success"] >= 5
    rew = torch.norm(vel, dim=1) * if_egg_beater_success(env, sensor_names) * success.float()
    return rew

def object_goal_distance(
    env: ManagerBasedRLEnv,
    command_name: str,
    object_id: int = 0,
) -> torch.Tensor:
    """Reward the agent for tracking the goal pose."""
    object: RigidObject = env.scene[f"object_{object_id}"]
    command = env.command_manager.get_command(command_name)
    des_pos_w = command[:, :3] + env.scene.env_origins
    # distance of the object to the goal: (num_envs,)
    distance = torch.norm(des_pos_w - object.data.root_pos_w[:, :3], dim=1)
    # initial distance
    max_distance = torch.norm(des_pos_w - env.object_init_pos[object_id], dim=1)
    distance = torch.clamp(max_distance - distance, min=0.0)
    within_range = torch.ones(env.num_envs, device=env.device)
    command_term = env.command_manager.get_term(command_name)
    within_range = torch.logical_and(within_range, command_term.metrics["orientation_error"] > command_term.cfg.success_threshold_orient)
    rew = distance * within_range.float()
    return rew