from __future__ import annotations

import torch
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from symdex.env.mdps.reward_mdps import get_allegro_contact

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

def object_goal_orient_distance(
    env: ManagerBasedRLEnv,
    command_name: str,
    object_id: int = 0,
    axis: None | str = "z", # roll
    sensor_names: list = ["contact_sensors_0", "contact_sensors_1", "contact_sensors_2", "contact_sensors_3"],
) -> torch.Tensor:
    '''
    If axis is not None, measure if the object's specific axis is aligned with the goal's axis.
    If axis is None, measure the orientation error for all three axes.
    '''
    object: RigidObject = env.scene[f"object_{object_id}"]
    command_term = env.command_manager.get_term(command_name)
    des_orient_w = command_term.quat_command_w
    if axis is not None:
        from symdex.utils.isaac_utils import get_angle_from_quat
        target_axis = get_angle_from_quat(des_orient_w, axis=axis, normalize=True)
        cur_axis = get_angle_from_quat(object.data.root_quat_w, axis=axis, normalize=True)
        distance = torch.sum(target_axis * cur_axis, dim=-1) * (2**0.5)  # linearize the reward (times sqrt(2))
        distance = torch.clamp(distance, min=0.0)
    else:
        distance = math_utils.quat_error_magnitude(object.data.root_quat_w, des_orient_w)
        # initial distance
        max_distance = math_utils.quat_error_magnitude(env.object_init_orient[object_id], des_orient_w)
        distance = torch.clamp(max_distance - distance, min=0.0)
    # only enable when the gripper is in contact with the object
    rew = distance * get_allegro_contact(env, sensor_names) * (object.data.root_pos_w[:, 2] > command_term.command[:, 2] - 0.05)
    return rew

def success_bonus(
    env: ManagerBasedRLEnv, object_id: int, frame_name: str, num_success: int = 0,
) -> torch.Tensor:
    object: RigidObject = env.scene[f"object_{object_id}"]
    distance = torch.norm(env.scene[frame_name].data.target_pos_w.reshape(-1, 3) - object.data.root_pos_w[:, :3], dim=-1)
    success = distance < 0.03 * (object.data.root_pos_w[:, 2] > 0.2)
    env.success_tracker_step[success] += 1
    env.success_tracker_step[~success] = 0
    rew = env.success_tracker_step >= num_success
    return rew.float()

def max_consecutive_success(env: ManagerBasedRLEnv, num_success: int) -> torch.Tensor:
    success = env.success_tracker_step >= num_success
    env.success_tracker = success.float()
    return success

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

def object_goal_distance(
    env: ManagerBasedRLEnv,
    command_name: str,
    object_id: int = 0,
    sensor_names: list = ["contact_sensors_0", "contact_sensors_1", "contact_sensors_2", "contact_sensors_3"],
) -> torch.Tensor:
    """Reward the agent for tracking the goal pose."""
    object: RigidObject = env.scene[f"object_{object_id}"]
    command = env.command_manager.get_command(command_name)
    command_term = env.command_manager.get_term(command_name)
    des_pos_w = command[:, :3] + env.scene.env_origins
    # distance of the object to the goal: (num_envs,)
    distance = torch.norm(des_pos_w - object.data.root_pos_w[:, :3], dim=1)
    # initial distance
    max_distance = torch.norm(des_pos_w - env.object_init_pos[object_id], dim=1)
    distance = torch.clamp(max_distance - distance, min=0.0)
    if object_id == 0:
        rew = distance * get_allegro_contact(env, sensor_names) * (object.data.root_pos_w[:, 2] > (des_pos_w[:, 2] - 0.1))
    else:
        rew = distance * (command_term.metrics["orientation_error"] > 0.75)
    return rew

def drill_goal_orient_distance(
    env: ManagerBasedRLEnv,
    command_name: str,
    object_id: int = 0,
) -> torch.Tensor:
    '''
    If axis is not None, measure if the object's specific axis is aligned with the goal's axis.
    If axis is None, measure the orientation error for all three axes.
    '''
    object: RigidObject = env.scene[f"object_{object_id}"]
    command_term = env.command_manager.get_term(command_name)
    des_orient_w = command_term.quat_command_w
    distance = math_utils.quat_error_magnitude(object.data.root_quat_w, des_orient_w)
    # only enable when the gripper is in contact with the object
    rew = 1 / distance * (object.data.root_pos_w[:, 2] > command_term.command[:, 2] - 0.1)
    return rew

def drill_cube_distance(
    env: ManagerBasedRLEnv,
    frame_name: str,
    cube_id: int = 0,
    drill_id: int = 1,
) -> torch.Tensor:
    cube: RigidObject = env.scene[f"object_{cube_id}"]
    drill: RigidObject = env.scene[f"object_{drill_id}"]
    distance = torch.norm(env.scene[frame_name].data.target_pos_w.reshape(-1, 3) - cube.data.root_pos_w[:, :3], dim=-1)
    distance = torch.clamp(0.1 - distance, min=0.0) * (cube.data.root_pos_w[:, 2] > 0.25) * (drill.data.root_pos_w[:, 2] > 0.2)
    return distance 