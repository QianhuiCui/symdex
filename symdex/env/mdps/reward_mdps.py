from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from functools import reduce
import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv
    
def object_robot_distance(
    env: ManagerBasedRLEnv,
    weight: list,
    link_name: list,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_id: int = 0,
) -> torch.Tensor:
    """Reward the agent for reaching the object using tanh-kernel."""
    weight = torch.tensor(weight, device=env.device)
    # extract the used quantities (to enable type-hinting)
    object: RigidObject = env.scene[f"object_{object_id}"]
    # Target object position: (num_envs, 3)
    object_pos_w = object.data.root_pos_w[:, None, :] 
    # Fingertip position: (num_envs, num_fingertip, 3)
    robot: Articulation = env.scene[asset_cfg.name]
    link_idx = robot.find_bodies(link_name)[0]
    link_w = robot.data.body_pos_w[:, link_idx, :3]
    # Distance of the fingertip to the object: (num_envs,)
    object_link_distance = torch.norm(object_pos_w - link_w, dim=-1) * weight
    object_link_distance = torch.mean(object_link_distance, dim=1)
    rew = 1 / object_link_distance
    return rew

def get_force(env: ManagerBasedRLEnv, sensor_names: list, if_filter: bool = False):
    if if_filter:
        forces = []
        for sensor_name in sensor_names:
            force = env.scene[sensor_name].data.force_matrix_w.squeeze(2)
            forces.append(force)
        return torch.cat(forces, dim=1)
    else:
        forces = []
        for sensor_name in sensor_names:
            force = env.scene[sensor_name].data.net_forces_w.squeeze(2)
            forces.append(force)
        return torch.cat(forces, dim=1)

def get_allegro_contact(env: ManagerBasedRLEnv, sensor_names: list):
    force = get_force(env, sensor_names, if_filter=True)
    is_contact = (torch.norm(force, dim=-1) > 1.0)
    is_contact_index_or_middle_or_ring = reduce(torch.logical_or, [is_contact[:, 0], is_contact[:, 1], is_contact[:, 2]])
    is_contact = reduce(torch.logical_and, [is_contact_index_or_middle_or_ring, is_contact[:, 3]])
    return is_contact

def check_release(env: ManagerBasedRLEnv, sensor_names: list):
    force = get_force(env, sensor_names, if_filter=True)
    is_contact = (torch.norm(force, dim=-1) < 1.0)
    is_release = reduce(torch.logical_and, [is_contact[:, 0], is_contact[:, 1], is_contact[:, 2], is_contact[:, 3]])
    return is_release

def get_energy_consumption(env: ManagerBasedEnv, robot_name: str):
    # it only works for **explicit** actuator
    robot: Articulation = env.scene[robot_name]
    jnt_vel = robot.data.joint_vel
    jnt_effort = robot.data.applied_torque
    energy = torch.abs(jnt_vel * jnt_effort).sum(dim=-1)
    return energy

def get_actuator_energy_consumption(
    env: ManagerBasedEnv, robot_name: str, actuator_name: str | list[str]
):
    robot: Articulation = env.scene[robot_name]
    jnt_vel = robot.data.joint_vel
    jnt_effort = robot.data.applied_torque
    if isinstance(actuator_name, str):
        joint_ids = robot.actuators[actuator_name].joint_indices
    else:
        joint_ids = []
        for name in actuator_name:
            joint_ids.extend(robot.actuators[name].joint_indices)
    jnt_vel = jnt_vel[:, joint_ids]
    jnt_effort = jnt_effort[:, joint_ids]
    energy = torch.abs(jnt_vel * jnt_effort).sum(dim=-1)
    return energy

def energy_punishment(
    env: ManagerBasedRLEnv,
    actuator_name = None,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    if actuator_name is None:
        energy = get_energy_consumption(env=env, robot_name=asset_cfg.name)
    else:
        energy = get_actuator_energy_consumption(env=env, robot_name=asset_cfg.name, actuator_name=actuator_name)
    rew = -energy
    return rew

def collision_penalty(
    env: ManagerBasedRLEnv,
    sensor_names: list = ["contact_sensors_0", "contact_sensors_1", "contact_sensors_2", "contact_sensors_3"],
) -> torch.Tensor:
    filtered_is_contact = []
    for s in sensor_names:
        # collide with objects that are not the target object
        filtered_force = env.scene[s].data.force_matrix_w.mean(dim=tuple(range(1, env.scene[s].data.force_matrix_w.ndim))) == 0.0
        normal_force = env.scene[s].data.net_forces_w.mean(dim=tuple(range(1, env.scene[s].data.net_forces_w.ndim))) != 0.0
        filtered_is_contact.append(torch.logical_and(filtered_force, normal_force))
    
    is_contact = reduce(torch.logical_or, filtered_is_contact)
    return is_contact.float()

def lift_distance(
    env: ManagerBasedRLEnv,
    command_name: str,
    minimal_height = None,
    object_id: int = 0,
    sensor_names: list = ["contact_sensors_0", "contact_sensors_1", "contact_sensors_2", "contact_sensors_3"],
) -> torch.Tensor:
    command = env.command_manager.get_command(command_name)
    des_pos_w = command[:, :3] + env.scene.env_origins
    if minimal_height is None:
        minimal_height = des_pos_w[:, 2] + 0.05
    object: RigidObject = env.scene[f"object_{object_id}"]
    # scale to [0, )
    z_distance = (object.data.root_pos_w[:, 2] - env.object_init_pos[object_id][:, 2]) / (minimal_height - env.object_init_pos[object_id][:, 2]) # linear
    z_distance = torch.clamp(z_distance, min=0.0)
    rew = (object.data.root_pos_w[:, 2] < minimal_height) * z_distance * get_allegro_contact(env, sensor_names)
    return rew

def object_goal_distance(
    env: ManagerBasedRLEnv,
    command_name: str,
    object_id: int = 0,
    sensor_names: list = ["contact_sensors_0", "contact_sensors_1", "contact_sensors_2", "contact_sensors_3"],
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
    # only enable when the gripper is in contact with the object
    rew = distance * get_allegro_contact(env, sensor_names) * torch.where(object.data.root_pos_w[:, 2] > (des_pos_w[:, 2] - 0.05), 1.0, 0.0)
    return rew

def object_goal_distance_orient(
    env: ManagerBasedRLEnv,
    command_name: str,
    object_id: int = 0,
    axis: None | str = "z", # roll
    pos_success_threshold: None | float = None,
    sensor_names: list = ["contact_sensors_0", "contact_sensors_1", "contact_sensors_2", "contact_sensors_3"],
) -> torch.Tensor:
    '''
    If axis is not None, measure if the object's specific axis is aligned with the goal's axis.
    If axis is None, measure the orientation error for all three axes.
    '''
    object: RigidObject = env.scene[f"object_{object_id}"]
    command = env.command_manager.get_command(command_name)
    des_orient_w = command[:, 3:]
    if axis is not None:
        from symdex.utils.isaac_utils import get_angle_from_quat
        target_axis = get_angle_from_quat(des_orient_w, axis=axis, normalize=True)
        init_axis = get_angle_from_quat(env.object_init_orient[object_id], axis=axis, normalize=True)
        cur_axis = get_angle_from_quat(object.data.root_quat_w, axis=axis, normalize=True)
        distance = torch.sum(target_axis * cur_axis, dim=-1) * (2**0.5)  # linearize the reward (times sqrt(2))
        distance = torch.clamp(distance, min=0.0)
    else:
        distance = math_utils.quat_error_magnitude(object.data.root_quat_w, des_orient_w)
        # initial distance
        max_distance = math_utils.quat_error_magnitude(env.object_init_orient[object_id], des_orient_w)
        distance = torch.clamp(max_distance - distance, min=0.0)
    # only enable when the gripper is in contact with the object
    within_range = torch.ones(env.num_envs, device=env.device)
    command_term = env.command_manager.get_term(command_name)
    if pos_success_threshold is not None:
        within_range = torch.logical_and(within_range, command_term.metrics["position_error"] < pos_success_threshold)
    else:
        within_range = torch.logical_and(within_range, command_term.metrics["position_error"] < command_term.cfg.success_threshold)
    rew = distance * within_range.float() * get_allegro_contact(env, sensor_names)
    return rew

def success_bonus(
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
    env.success_tracker = success.float()
    return rew

def align_palm_to_quat(
    env: ManagerBasedRLEnv,
    link_name: list,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    frame_name: str = "approach_frame",
    ):
    robot: Articulation = env.scene[asset_cfg.name]
    link_idx = robot.find_bodies(link_name)[0]
    robot_state_w = robot.data.body_state_w[:, link_idx, :7].reshape(-1, 7)
    # orientation distance
    ori_distance = math_utils.quat_error_magnitude(
        robot_state_w[:, 3:7], env.scene[frame_name].data.target_quat_w.reshape(-1, 4)
    )
    return -ori_distance

def align_palm_to_pos(
    env: ManagerBasedRLEnv,
    link_name: list,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    frame_name: str = "approach_frame",
    ):
    robot: Articulation = env.scene[asset_cfg.name]
    link_idx = robot.find_bodies(link_name)[0]
    robot_state_w = robot.data.body_state_w[:, link_idx, :7].reshape(-1, 7)
    # position distance
    distance = torch.norm(robot_state_w[:, :3] - env.scene[frame_name].data.target_pos_w.reshape(-1, 3), dim=-1)
    return -distance