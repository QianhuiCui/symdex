from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from functools import reduce
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from symdex.env.mdps.reward_mdps import get_force, check_release

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def drawer_handle_robot_distance(
    env: ManagerBasedRLEnv,
    weight: list,
    link_name: list,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward the agent for reaching the drawer handle using tanh-kernel."""
    weight = torch.tensor(weight, device=env.device)
    # extract the used quantities (to enable type-hinting)
    drawer: Articulation = env.scene["drawer"]
    # Target object position: (num_envs, 3)
    drawer_idx = drawer.find_bodies("handle_grip")[0]
    drawer_handle_pos_w = drawer.data.body_state_w[:, drawer_idx, :3]
    # Fingertip position: (num_envs, num_fingertip, 3)
    robot: Articulation = env.scene[asset_cfg.name]
    link_idx = robot.find_bodies(link_name)[0]
    link_w = robot.data.body_pos_w[:, link_idx, :3]
    # Distance of the fingertip to the object: (num_envs,)
    drawer_handle_link_distance = torch.norm(drawer_handle_pos_w - link_w, dim=-1) * weight
    drawer_handle_link_distance = torch.mean(drawer_handle_link_distance, dim=1)
    rew = 1 / drawer_handle_link_distance
    return rew

def drawer_move(
    env: ManagerBasedRLEnv,
    sensor_names: str = "contact_sensors_0_left",
    joints = None,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("drawer"),
) -> torch.Tensor:
    drawer: Articulation = env.scene[asset_cfg.name]
    if joints is None:
        joint_ids = asset_cfg.joint_ids
    else:
        joint_ids = drawer.find_joints(joints)[0]
    joint_pos = drawer.data.joint_pos[:, joint_ids].reshape(-1)
    # if object is inside the drawer, encourage the drawer to move inside
    max_joint_pos = drawer.data.default_joint_limits[:, joint_ids, 1].reshape(-1)
    is_inside = if_in_drawer(env, object_id=0)
    is_inside_idx = torch.where(is_inside)[0]
    joint_pos[is_inside_idx] = max_joint_pos[is_inside_idx]

    force = get_force(env, sensor_names, if_filter=True)
    is_contact = (torch.norm(force, dim=-1) > 1.0).reshape(-1)
    rew = joint_pos * is_contact
    return rew

def drawer_move_inside(
    env: ManagerBasedRLEnv,
    sensor_names: str = "contact_sensors_0_left",
    joints = None,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("drawer"),
) -> torch.Tensor:
    drawer: Articulation = env.scene[asset_cfg.name]
    if joints is None:
        joint_ids = asset_cfg.joint_ids
    else:
        joint_ids = drawer.find_joints(joints)[0]
    joint_pos = drawer.data.joint_pos[:, joint_ids].reshape(-1)
    rew = torch.zeros_like(joint_pos)
    # if object is inside the drawer, encourage the drawer to move inside
    max_joint_pos = drawer.data.default_joint_limits[:, joint_ids, 1].reshape(-1)
    is_inside = if_in_drawer(env, object_id=0)
    is_inside_idx = torch.where(is_inside)[0]
    rew[is_inside_idx] = 10 * (max_joint_pos[is_inside_idx] - joint_pos[is_inside_idx])

    force = get_force(env, sensor_names, if_filter=True)
    is_contact = (torch.norm(force, dim=-1) > 1.0).reshape(-1)
    rew = rew * is_contact
    return rew

def if_in_drawer(
    env: ManagerBasedRLEnv,
    object_id: int = 0,
    sensor_names: list = ["contact_sensors_0", "contact_sensors_1", "contact_sensors_2", "contact_sensors_3"],
) -> torch.Tensor:
    drawer: Articulation = env.scene["drawer"]
    drawer_idx = drawer.find_bodies("drawer")[0]
    drawer_pos_w = drawer.data.body_state_w[:, drawer_idx, :3].reshape(-1, 3)
    object: RigidObject = env.scene[f"object_{object_id}"]
    distance = torch.norm(object.data.root_pos_w[:, :3] - drawer_pos_w, dim=-1)
    rew = check_release(env, sensor_names) * (distance < 0.2) * (object.data.root_pos_w[:, 2] < 1.0)
    return rew

def reset_robot_joint_pos(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    robot: Articulation = env.scene[asset_cfg.name]
    robot_default_joint_pos = robot.data.default_joint_pos
    cur_joint_pos = robot.data.joint_pos
    distance = torch.norm(robot_default_joint_pos - cur_joint_pos, dim=-1)
    rew = 1 / (distance + 1e-6) * if_in_drawer(env, object_id=0).float()
    return rew

def robot_goal_distance(
    env: ManagerBasedRLEnv,
    target_pos: list,
    target_link: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward the agent for tracking the goal pose."""
    des_pos_w = torch.tensor(target_pos, device=env.device) + env.scene.env_origins
    target_link_idx = env.scene[asset_cfg.name].find_bodies([target_link])[0]
    distance = torch.norm(des_pos_w - env.scene[asset_cfg.name].data.body_state_w[:, target_link_idx, 0:3].squeeze(1), dim=1)
    rew = 1 / (distance + 1e-6) * if_in_drawer(env, object_id=0).float()
    return rew

def success_bonus(
    env: ManagerBasedRLEnv,
) -> torch.Tensor:
    rew = env.success_tracker
    return rew.float()

def max_consecutive_success(env: ManagerBasedRLEnv, num_success: int) -> torch.Tensor:
    drawer: Articulation = env.scene["drawer"]
    joint_ids = drawer.find_joints("base_drawer_joint")[0]
    joint_pos = drawer.data.joint_pos[:, joint_ids].reshape(-1)
    success = (joint_pos < 0.1) * if_in_drawer(env)
    env.success_tracker_step[success] += 1
    env.success_tracker_step[~success] = 0
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