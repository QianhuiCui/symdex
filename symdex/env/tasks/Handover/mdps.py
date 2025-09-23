from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from functools import reduce
import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from symdex.env.mdps.reward_mdps import get_force
if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def get_allegro_contact(env: ManagerBasedRLEnv, sensor_names: list):
    force = get_force(env, sensor_names, if_filter=True)
    is_contact = (torch.norm(force, dim=-1) > 1.0)
    if len(sensor_names) == 4:
        is_contact_index_or_middle_or_ring = reduce(torch.logical_or, [is_contact[:, 0], is_contact[:, 1], is_contact[:, 2]])
        is_contact = reduce(torch.logical_and, [is_contact_index_or_middle_or_ring, is_contact[:, 3]])
    elif len(sensor_names) == 8:
        is_contact_index_or_middle_or_ring = reduce(torch.logical_or, [is_contact[:, 0], is_contact[:, 1], is_contact[:, 2], is_contact[:, 4], is_contact[:, 5], is_contact[:, 6]])
        is_contact_thumb = reduce(torch.logical_or, [is_contact[:, 3], is_contact[:, 7]])
        is_contact = reduce(torch.logical_and, [is_contact_index_or_middle_or_ring, is_contact_thumb])
    return is_contact

def get_allegro_contact_with_palm(env: ManagerBasedRLEnv, sensor_names: list):
    force = get_force(env, sensor_names, if_filter=True)
    is_contact = (torch.norm(force, dim=-1) > 1.0)
    is_contact_index_or_middle_or_ring = reduce(torch.logical_or, [is_contact[:, 0], is_contact[:, 1], is_contact[:, 2]])
    is_contact = reduce(torch.logical_and, [is_contact_index_or_middle_or_ring, is_contact[:, 3], is_contact[:, 4]])
    return is_contact

def check_release_object(env: ManagerBasedRLEnv, sensor_names: list):
    force = get_force(env, sensor_names, if_filter=True)
    is_not_contact = (torch.norm(force, dim=-1) < 1.0)
    is_not_contact = reduce(torch.logical_and, [is_not_contact[:, i] for i in range(len(sensor_names))])
    return is_not_contact, force

def contact_bottle_punish(env: ManagerBasedRLEnv, sensor_names: list):
    is_not_contact, force = check_release_object(env, sensor_names)
    rew = -torch.mean(torch.abs(force), dim=(-1, -2))
    rew[is_not_contact] = 20.0
    return rew * (env.reach_middle > 10)

def robot_goal_distance(
    env: ManagerBasedRLEnv,
    target_pos: list,
    target_link: str,
    sensor_names: list,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward the agent for tracking the goal pose."""
    des_pos_w = torch.tensor(target_pos, device=env.device) + env.scene.env_origins
    target_link_idx = env.scene[asset_cfg.name].find_bodies([target_link])[0]
    distance = torch.norm(des_pos_w - env.scene[asset_cfg.name].data.body_state_w[:, target_link_idx, 0:3].squeeze(1), dim=1)
    # print("robot_goal_distance", distance)
    is_not_contact, force = check_release_object(env, sensor_names)
    rew = torch.clamp(0.5 - distance, min=0.0) * (is_not_contact).float() * (env.reach_middle > 10).float()
    return rew

def frame_marker_robot_distance(
    env: ManagerBasedRLEnv,
    weight: list,
    link_name: list,
    if_left: bool = False,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    frame_name: str = "bottle_top",
) -> torch.Tensor:
    """Reward the agent for reaching the object using tanh-kernel."""
    weight = torch.tensor(weight, device=env.device)
    # extract the used quantities (to enable type-hinting)
    # Target object position: (num_envs, 3)
    object_pos_w = env.scene[frame_name].data.target_pos_w.reshape(-1, 1, 3)
    # Fingertip position: (num_envs, num_fingertip, 3)
    robot: Articulation = env.scene[asset_cfg.name]
    link_idx = robot.find_bodies(link_name)[0]
    link_w = robot.data.body_pos_w[:, link_idx, :3]
    # Distance of the fingertip to the object: (num_envs,)
    object_link_distance = torch.norm(object_pos_w - link_w, dim=-1) * weight
    if if_left:
        weighted_object_link_distance = torch.sum(object_link_distance, dim=1)
        # print(weighted_object_link_distance)
        rew = torch.clamp(0.6 - weighted_object_link_distance, min=0.0) * (env.reach_middle > 0).float()
    else:
        weighted_object_link_distance = torch.mean(object_link_distance, dim=1)
        rew = 1 / weighted_object_link_distance
    return rew

def align_hand_pose(
    env: ManagerBasedRLEnv,
    link_name: str,
    command_name: str = "target_pos",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    command_term = env.command_manager.get_term(command_name)
    robot: Articulation = env.scene[asset_cfg.name]
    link_idx = robot.find_bodies(link_name)[0]
    robot_state_w = robot.data.body_state_w[:, link_idx, :7].reshape(-1, 7)
    # orientation distance
    ori_distance = math_utils.quat_error_magnitude(
        robot_state_w[:, 3:7], command_term.quat_command_w
    )
     # position distance
    distance = torch.norm(robot_state_w[:, :3] - command_term.pos_command_w, dim=-1)
    rew = -ori_distance - 2 * distance * (env.reach_middle == 0).float()
    return rew

def align_finger_joint(
    env: ManagerBasedRLEnv,
    link_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    joint_idx = env.scene[asset_cfg.name].find_joints(link_name)[0]
    joint_pos = env.scene[asset_cfg.name].data.joint_pos[:, joint_idx]
    default_joint_pos = env.scene[asset_cfg.name].data.default_joint_pos[:, joint_idx]
    distance = torch.norm(joint_pos - default_joint_pos, dim=-1)
    distance = -distance * (env.reach_middle == 0).float()
    return distance

def object_goal_distance(
    env: ManagerBasedRLEnv,
    command_name: str,
    object_id: int = 0,
    if_left: bool = False,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    sensor_names: list = ["contact_sensors_0", "contact_sensors_1", "contact_sensors_2", "contact_sensors_3"],
) -> torch.Tensor:
    """Reward the agent for tracking the goal pose."""
    object: RigidObject = env.scene[f"object_{object_id}"]
    command = env.command_manager.get_command(command_name)
    des_pos_w = command[:, :3] + env.scene.env_origins
    # distance of the object to the goal: (num_envs,)
    distance = torch.norm(des_pos_w - object.data.root_pos_w[:, :3], dim=1)
    if if_left:
        distance = torch.clamp(0.33 - distance, min=0.0)
        # only enable when the gripper is in contact with the object
        rew = distance * get_allegro_contact(env, sensor_names)
        rew = rew * (env.reach_middle > 0).float()
    else:
            # initial distance
        max_distance = torch.norm(des_pos_w - env.object_init_pos[object_id], dim=1)
        distance = torch.clamp(max_distance - distance, min=0.0)
        # only enable when the gripper is in contact with the object
        rew = distance * get_allegro_contact(env, sensor_names)
        object_pos_w = env.scene["bottle_bottom"].data.target_pos_w.reshape(-1, 1, 3)
        # Fingertip position: (num_envs, num_fingertip, 3)
        robot: Articulation = env.scene[asset_cfg.name]
        link_idx = robot.find_bodies("palm_link")[0]
        link_w = robot.data.body_pos_w[:, link_idx, :3]
        # Distance of the fingertip to the object: (num_envs,)
        object_link_distance = torch.norm(object_pos_w - link_w, dim=-1)
        rew = rew * (object_link_distance[:, -1] < 0.15).float()
    return rew

def object_goal_orient_distance(
    env: ManagerBasedRLEnv,
    command_name: str,
    object_id: int = 0,
    if_left: bool = False,
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
    if if_left:
        rew = distance * get_allegro_contact(env, sensor_names) * (env.reach_middle > 0).float()
    else:
        rew = distance * get_allegro_contact(env, sensor_names) * (object.data.root_pos_w[:, 2] > command_term.command[:, 2] - 0.08)
    return rew

def cmd_success_bonus(
    env: ManagerBasedRLEnv, 
    command_names: str, 
    num_success: int = 0, 
    if_right: bool = False,
    if_left: bool = False,
    sensor_names: list = ["contact_sensors_0", "contact_sensors_1", "contact_sensors_2", "contact_sensors_3"],
) -> torch.Tensor:
    command_term = env.command_manager.get_term(command_names)
    rew = command_term.metrics["consecutive_success"] >= num_success
    if if_right:
        num_success = env.command_manager.get_term(command_names).metrics["consecutive_success"]
        rew[num_success > 15] = 0.0
    elif if_left:
        rew = rew * get_allegro_contact(env, sensor_names)
    return rew

def success_bonus(
    env: ManagerBasedRLEnv, 
    command_names: str, 
    num_success: int = 0,
    symmetry: bool = False,
    not_contact_sensor_names: list = None,
    is_contact_sensor_names: list = None,
) -> torch.Tensor:
    is_not_contact, _ = check_release_object(env, not_contact_sensor_names)
    cmd_term = env.command_manager.get_term(command_names)
    within_range = torch.logical_and(cmd_term.metrics["position_error"] < 0.08, cmd_term.metrics["orientation_error"] > 0.95)
    success = within_range * is_not_contact * get_allegro_contact(env, is_contact_sensor_names)
    if symmetry:
        valid_envs = torch.where(env.symmetry_tracker == 1)[0]
    else:
        valid_envs = torch.where(env.symmetry_tracker == 0)[0]
    if len(valid_envs) > 0:
        success_idx = valid_envs[success[valid_envs].nonzero(as_tuple=True)[0] ]
        # get the indices of those envs that failed
        failure_idx = valid_envs[(~success[valid_envs]).nonzero(as_tuple=True)[0] ]

        # now update the original tensor in-place
        old_success_tracker_step = env.success_tracker_step.clone()
        env.success_tracker_step[success_idx] += 1   # increment successes
        env.success_tracker_step[failure_idx] = 0
        rew = (old_success_tracker_step == num_success - 1) * (env.success_tracker_step == num_success)
        return rew.float()
    else:
        return torch.zeros_like(success).float()

def max_consecutive_success(env: ManagerBasedRLEnv, num_success: int) -> torch.Tensor:
    success = env.success_tracker_step >= num_success
    env.success_tracker = success.float()
    return success

