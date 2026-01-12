from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from functools import reduce
from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from symdex.env.mdps.reward_mdps import get_allegro_contact, check_release
if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

def object_goal_distance(
    env: ManagerBasedRLEnv,
    object_id: int = 0,
    command_name: str = "target_pos",
    sensor_names: list = ["contact_sensors_0", "contact_sensors_1", "contact_sensors_2", "contact_sensors_3"],
    delay: bool = False,
    switch: bool = False,
    distance_threshold: float = 0.3,
) -> torch.Tensor:
    """Reward the agent for tracking the goal pose."""
    object: RigidObject = env.scene[f"object_{object_id}"]
    if switch:
        assert object_id == 2
        des_pos_w = generated_commands(env, object_id) + env.scene.env_origins
    else:
        command = env.command_manager.get_command(command_name)
        des_pos_w = command[:, :3] + env.scene.env_origins
    distance = torch.norm(des_pos_w - object.data.root_pos_w[:, :3], dim=1)
    # initial distance
    distance = torch.clamp(distance_threshold - distance, min=0.0)
    # only enable when the gripper is in contact with the object
    rew = distance * get_allegro_contact(env, sensor_names) * (object.data.root_pos_w[:, 2] > (des_pos_w[:, 2] - 0.05))
    if delay:
        assert object_id == 2
        if object_id == 1:
            rew = rew * (env.object_in_tote_tracker[2] >= 3)
        elif object_id == 2:
            rew = rew * (env.object_in_tote_tracker[1] >= 3)
    return rew

def if_in_tote(
    env: ManagerBasedRLEnv,
    object_id: int = 0,
    distance_threshold: float = 0.2,
    delay: bool = False,
    symmetry: bool = False,
    sensor_names: list = ["contact_sensors_0", "contact_sensors_1", "contact_sensors_2", "contact_sensors_3"],
) -> torch.Tensor:
    object: RigidObject = env.scene[f"object_{object_id}"]
    distance = torch.norm(object.data.root_pos_w[:, :3] - env.scene["object_0"].data.root_pos_w[:, :3], dim=-1)
    in_tote = check_release(env, sensor_names) * (distance < distance_threshold) * (object.data.root_pos_w[:, 2] < 0.2) * (env.object_on_tote_tracker[object_id] > 0)
    
    # only the correct rew term can update the tracker
    if symmetry:
        in_tote = in_tote * (env.symmetry_tracker == 1)
    else:
        in_tote = in_tote * (env.symmetry_tracker == 0)

    if delay:
        assert object_id == 2
    # object 1 should be in tote first, then object 2 can be in tote
    if object_id == 1:
        env.object_in_tote_tracker[object_id] += in_tote
        rew = env.object_in_tote_tracker[object_id] == 3
    elif object_id == 2:
        env.object_in_tote_tracker[object_id] += in_tote * (env.object_in_tote_tracker[1] >= 3)
        rew = (env.object_in_tote_tracker[object_id] == 3) * (env.object_in_tote_tracker[1] >= 3)
    return rew.float()

def robot_goal_distance(
    env: ManagerBasedRLEnv,
    target_pos: list,
    target_link: str,
    object_id: int = 0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward the agent for tracking the goal pose."""
    des_pos_w = torch.tensor(target_pos, device=env.device) + env.scene.env_origins
    target_link_idx = env.scene[asset_cfg.name].find_bodies([target_link])[0]
    distance = torch.norm(des_pos_w - env.scene[asset_cfg.name].data.body_state_w[:, target_link_idx, 0:3].squeeze(1), dim=1)
    distance = torch.clamp(0.25 - distance, min=0.0)
    rew = distance * (env.object_in_tote_tracker[object_id] >= 3)
    return rew

def punish_collision(
    env: ManagerBasedRLEnv,
    sensor: str | list,
    filtered_sensor: str | list = None,
) -> torch.Tensor:
    force = env.scene[sensor].data.net_forces_w
    is_contact = (torch.norm(force, dim=-1) > 1.0)
    is_contact = torch.any(is_contact, dim=-1)
    return is_contact.float()

def success_bonus(
    env: ManagerBasedRLEnv,
) -> torch.Tensor:
    rew = env.success_tracker.float()
    return rew

def generated_commands(env: ManagerBasedRLEnv, object_id: int = 0):
    waiting_pos = env.command_manager.get_command("waiting_pos").clone()
    target_pos = env.command_manager.get_command("target_pos").clone()
    if not hasattr(env, "object_in_tote_tracker"):
        return torch.zeros((env.num_envs, 3), device=env.device)
    if object_id == 1:
        tote_not_in = torch.where(env.object_in_tote_tracker[2] < 3)[0]
    elif object_id == 2:
        tote_not_in = torch.where(env.object_in_tote_tracker[1] < 3)[0]
    elif object_id == 0:
        return target_pos
    target_pos[tote_not_in] = waiting_pos[tote_not_in]
    return target_pos

def max_consecutive_success(env: ManagerBasedRLEnv, num_success: int) -> torch.Tensor:
    object_1: RigidObject = env.scene["object_1"]
    object_2: RigidObject = env.scene["object_2"]
    tote = env.scene["object_0"]
    sensor_names = ["contact_sensors_0", "contact_sensors_1", "contact_sensors_2", "contact_sensors_3"]
    distance_threshold = 0.2

    dist_1 = torch.norm(object_1.data.root_pos_w[:, :3] - tote.data.root_pos_w[:, :3], dim=-1)
    in_tote_1 = check_release(env, sensor_names) * (dist_1 < distance_threshold) * (object_1.data.root_pos_w[:, 2] < 0.2) * (env.object_on_tote_tracker[1] > 0)
    dist_2 = torch.norm(object_2.data.root_pos_w[:, :3] - tote.data.root_pos_w[:, :3], dim=-1)
    in_tote_2 = check_release(env, sensor_names) * (dist_2 < distance_threshold) * (object_2.data.root_pos_w[:, 2] < 0.2) * (env.object_on_tote_tracker[2] > 0)

    env.object_in_tote_tracker[1] += in_tote_1.float()
    env.object_in_tote_tracker[2] += (in_tote_2 & (env.object_in_tote_tracker[1] >= 3)).float()
    success = reduce(torch.logical_and, [env.object_in_tote_tracker[1] >= 3, env.object_in_tote_tracker[2] >= 3]).bool()

    env.success_tracker_step[success] += 1
    env.success_tracker_step[~success] = 0
    success_consecutive = env.success_tracker_step >= num_success
    env.success_tracker = success_consecutive.float()
    return success_consecutive

def obj_out_space(
        env: ManagerBasedRLEnv, 
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
        object_id: int = 0,
        workspace_radius: float = 0.7,
        workspace_height_range: tuple = (0.0, 1.5),
        ) -> torch.Tensor:
    """Terminate if the object is out of the workspace."""
    robot = env.scene[asset_cfg.name]
    object: RigidObject = env.scene[f"object_{object_id}"]

    robot_base = robot.data.root_link_pos_w[:, :3]
    object_pos = object.data.root_pos_w[:, :3]

    offset = object_pos - robot_base
    dist_xy = torch.norm(offset[:, :2], dim=-1)
    dist_z = offset[:, 2]

    out_of_radius = (dist_xy > workspace_radius) | (dist_xy < 0.1)
    out_of_height = (dist_z < workspace_height_range[0]) | (dist_z > workspace_height_range[1])
    out_of_space = out_of_radius | out_of_height

    return out_of_space