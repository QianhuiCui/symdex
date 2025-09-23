from __future__ import annotations

import torch
from typing import TYPE_CHECKING, Literal

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
import isaacsim.core.utils.bounds as bounds_utils
if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv


def align_palm_to_pos(
    env: ManagerBasedRLEnv,
    link_name: list,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    side: Literal["left", "right"] = "right",
    ):
    robot: Articulation = env.scene[asset_cfg.name]
    link_idx = robot.find_bodies(link_name)[0]
    robot_state_w = robot.data.body_state_w[:, link_idx, :7].reshape(-1, 7)
    # position distance
    side_pos = compute_side_points(env.scene["object_0"].data.root_state_w, env.side_points, side)
    distance = torch.norm(robot_state_w[:, :3] - side_pos, dim=-1)
    return -distance

def if_aligned_quat(
    env: ManagerBasedRLEnv,
    link_names: list = ["palm_link", "palm_link"],
    frame_names: list = ["tote_right", "tote_left"],
    asset_cfgs: list = [SceneEntityCfg("robot"), SceneEntityCfg("robot_left")],
    error_threshold: float = 0.5,
    ):
    aligned = []
    # check if palms are aligned with the object
    for i in range(len(link_names)):
        link_name = link_names[i]
        frame_name = frame_names[i]
        asset_cfg = asset_cfgs[i]
        robot: Articulation = env.scene[asset_cfg.name]
        link_idx = robot.find_bodies(link_name)[0]
        robot_state_w = robot.data.body_state_w[:, link_idx, :7].reshape(-1, 7)
        # orientation distance
        ori_distance = math_utils.quat_error_magnitude(
            robot_state_w[:, 3:7], env.scene[frame_name].data.target_quat_w.reshape(-1, 4)
        )
        aligned.append(torch.where(ori_distance < error_threshold, 1.0, 0.0))
    return torch.logical_and(aligned[0], aligned[1])

def if_aligned_pos(
    env: ManagerBasedRLEnv,
    link_names: list = ["palm_link", "palm_link"],
    frame_names: list = ["right", "left"],
    asset_cfgs: list = [SceneEntityCfg("robot"), SceneEntityCfg("robot_left")],
    error_threshold: float = 0.1,
):
    aligned = []
    # check if palms are aligned with the object
    for i in range(len(link_names)):
        link_name = link_names[i]
        frame_name = frame_names[i]
        asset_cfg = asset_cfgs[i]
        robot: Articulation = env.scene[asset_cfg.name]
        link_idx = robot.find_bodies(link_name)[0]
        robot_state_w = robot.data.body_state_w[:, link_idx, :7].reshape(-1, 7)
        # position distance
        side_pos = compute_side_points(env.scene["object_0"].data.root_state_w, env.side_points, frame_name)
        distance = torch.norm(robot_state_w[:, :3] - side_pos, dim=-1)
        aligned.append(torch.where(distance < error_threshold, 1.0, 0.0))
    return torch.logical_and(aligned[0], aligned[1])

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
    rew = distance * if_aligned_quat(env) * if_aligned_pos(env)
    return rew

def object_goal_orient_distance(
    env: ManagerBasedRLEnv,
    command_name: str,
    object_id: int = 0,
) -> torch.Tensor:
    """Reward the agent for tracking the goal pose."""
    object: RigidObject = env.scene[f"object_{object_id}"]
    command_term = env.command_manager.get_term(command_name)
    distance = math_utils.quat_error_magnitude(
        object.data.root_quat_w, command_term.quat_command_w
    )
    rew = (2.5 - distance) * if_aligned_quat(env) * if_aligned_pos(env)
    return rew

def punish_collision(
    env: ManagerBasedRLEnv,
    sensor: str,
    filter_force: bool = False,
) -> torch.Tensor:
    if filter_force:
        force = env.scene[sensor].data.force_matrix_w.squeeze(1)
    else:
        force = env.scene[sensor].data.net_forces_w
    is_contact = (torch.norm(force, dim=-1) > 1.0)
    is_contact = torch.any(is_contact, dim=-1)
    return is_contact.float()

def compute_box_side_transform(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    object_id: int = 0,
    axis: Literal["x", "y", "z"] = "x",
):
    """Compute the side position of the box."""
    object: RigidObject = env.scene[f"object_{object_id}"]
    cache = bounds_utils.create_bbox_cache()
    side_lengths = []
    side_points = []
    for i in range(len(object.root_physx_view.prim_paths)):
        min_x, min_y, min_z, max_x, max_y, max_z = bounds_utils.compute_combined_aabb(cache, prim_paths=[object.root_physx_view.prim_paths[i]])
        side_point = torch.tensor([[(min_x+max_x)/2, min_y, (min_z+max_z)/2], [(min_x+max_x)/2, max_y, (min_z+max_z)/2]], device=env.device, dtype=torch.float32)
        side_point -= env.scene.env_origins[i]
        side_points.append(side_point.unsqueeze(0))
        if axis == "x":
            side_lengths.append((max_x - min_x)/2)
        elif axis == "y":
            side_lengths.append((max_y - min_y)/2)
        elif axis == "z":
            side_lengths.append((max_z - min_z)/2)
    env.side_points = torch.cat(side_points, dim=0)
    env.side_lengths = torch.tensor(side_lengths, device=env.device, dtype=torch.float32).reshape(-1, 1)

def box_length(
    env: ManagerBasedRLEnv,
):
    if not hasattr(env, "side_lengths"):
        return torch.zeros((env.num_envs, 1), device=env.device, dtype=torch.float32)
    return env.side_lengths

def box_side(env: ManagerBasedRLEnv, side: Literal["left", "right"]) -> torch.Tensor:
    """Frame position in the environment frame."""
    # extract the used quantities (to enable type-hinting)
    if not hasattr(env, "side_points"):
        return torch.zeros((env.num_envs, 3), device=env.device, dtype=torch.float32)
    return compute_side_points(env.scene["object_0"].data.root_state_w, env.side_points, side) - env.scene.env_origins

def compute_side_points(object_state_w, side_points, side: Literal["left", "right"] = "right"):
    """Compute the side points of the object."""
    object_rot_mat = math_utils.matrix_from_quat(object_state_w[:, 3:7])
    object_pos_w = object_state_w[:, :3]
    side_points_rotated = torch.bmm(object_rot_mat, side_points.transpose(1, 2)).transpose(1, 2)
    # Apply the translation (box center in world coordinates) to the rotated points
    side_points_w = side_points_rotated + object_pos_w.unsqueeze(1)
    if side == "right":
        return side_points_w[:, 0, :]
    elif side == "left":
        return side_points_w[:, 1, :]
    else:
        raise ValueError(f"Invalid side: {side}")