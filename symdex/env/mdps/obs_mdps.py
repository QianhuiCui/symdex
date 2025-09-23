
from __future__ import annotations

import torch
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv
    

def joint_pos_limit_normalized(
    env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), joints = None, joint_lower_limit = None, joint_upper_limit = None
) -> torch.Tensor:
    """The joint positions of the asset normalized with the asset's joint limits.

    Note: Only the joints configured in :attr:`asset_cfg.joint_ids` will have their normalized positions returned.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    if joints is None:
        joint_ids = asset_cfg.joint_ids
    else:
        joint_ids = asset.find_joints(joints)[0]

    if joint_lower_limit is None:
        joint_lower_limit = asset.data.soft_joint_pos_limits[:, joint_ids, 0]
    else:
        joint_lower_limit = torch.tensor(joint_lower_limit, device=env.device)
    if joint_upper_limit is None:
        joint_upper_limit = asset.data.soft_joint_pos_limits[:, joint_ids, 1]
    else:
        joint_upper_limit = torch.tensor(joint_upper_limit, device=env.device)

    assert len(joint_lower_limit) == len(joint_upper_limit)

    return math_utils.scale_transform(
        asset.data.joint_pos[:, joint_ids],
        joint_lower_limit,
        joint_upper_limit,
    )

def joint_vel(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), joints = None):
    """The joint velocities of the asset.

    Note: Only the joints configured in :attr:`asset_cfg.joint_ids` will have their velocities returned.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    if joints is None:
        joint_ids = asset_cfg.joint_ids
    else:
        joint_ids = asset.find_joints(joints)[0]
    return asset.data.joint_vel[:, joint_ids]

def ee_pose(env: ManagerBasedEnv, ee_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), symmetry: bool = True) -> torch.Tensor:
    """EE pose in the environment frame."""
    # extract the used quantities (to enable type-hinting)
    robot: Articulation = env.scene[asset_cfg.name]
    ee_idx = robot.find_bodies(ee_name)[0]
    ee_w = robot.data.body_state_w[:, ee_idx, :7].clone().reshape(-1, 7)
    ee_w[:, :3] = ee_w[:, :3] - env.scene.env_origins
    if symmetry:
        ee_R = math_utils.matrix_from_quat(ee_w[:, 3:7])
        ee_R_flat = ee_R.transpose(1, 2).reshape(-1, 9)
        ee_w = torch.cat([ee_w[:, :3], ee_R_flat], dim=-1)
    return ee_w

def ee_pos(env: ManagerBasedEnv, ee_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), symmetry: bool = True) -> torch.Tensor:
    """EE position in the environment frame."""
    # extract the used quantities (to enable type-hinting)
    robot: Articulation = env.scene[asset_cfg.name]
    ee_idx = robot.find_bodies(ee_name)[0]
    ee_w = robot.data.body_state_w[:, ee_idx, :3].clone().reshape(-1, 3)
    ee_w[:, :3] = ee_w[:, :3] - env.scene.env_origins
    return ee_w

def ee_quat(env: ManagerBasedEnv, ee_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), symmetry: bool = True) -> torch.Tensor:
    """EE orientation in the environment frame."""
    # extract the used quantities (to enable type-hinting)
    robot: Articulation = env.scene[asset_cfg.name]
    ee_idx = robot.find_bodies(ee_name)[0]
    ee_quat = robot.data.body_state_w[:, ee_idx, 3:7].clone().reshape(-1, 4)
    if symmetry:
        ee_R = math_utils.matrix_from_quat(ee_quat)
        ee_R_flat = ee_R.transpose(1, 2).reshape(-1, 9)
        return ee_R_flat
    else:
        return ee_quat

def object_pos(env: ManagerBasedEnv, object_id: int = 0) -> torch.Tensor:
    """Object root position in the environment frame."""
    # extract the used quantities (to enable type-hinting)
    object: RigidObject = env.scene[f"object_{object_id}"]
    return object.data.root_pos_w - env.scene.env_origins

def frame_pos(env: ManagerBasedEnv, frame_name: str) -> torch.Tensor:
    """Frame position in the environment frame."""
    # extract the used quantities (to enable type-hinting)
    return env.scene[frame_name].data.target_pos_w.reshape(-1, 3) - env.scene.env_origins

def object_quat(env: ManagerBasedEnv, make_quat_unique: bool = False, object_id: int = 0, symmetry: bool = True) -> torch.Tensor:
    """Object root orientation (w, x, y, z) in the environment frame.

    If :attr:`make_quat_unique` is True, then returned quaternion is made unique by ensuring
    the quaternion has non-negative real component. This is because both ``q`` and ``-q`` represent
    the same orientation.
    """
    # extract the used quantities (to enable type-hinting)
    object: RigidObject = env.scene[f"object_{object_id}"]
    if symmetry:
        object_R = math_utils.matrix_from_quat(object.data.root_quat_w)
        object_R_flat = object_R.transpose(1, 2).reshape(-1, 9)
        return object_R_flat
    else:
        return math_utils.quat_unique(object.data.root_quat_w) if make_quat_unique else object.data.root_quat_w
    
def frame_quat(env: ManagerBasedEnv, frame_name: str, make_quat_unique: bool = False, symmetry: bool = True) -> torch.Tensor:
    if symmetry:
        frame_R = math_utils.matrix_from_quat(env.scene[frame_name].data.target_quat_w.reshape(-1, 4))
        frame_R_flat = frame_R.transpose(1, 2).reshape(-1, 9)
        return frame_R_flat
    else:
        return math_utils.quat_unique(env.scene[frame_name].data.target_quat_w.reshape(-1, 4) ) if make_quat_unique else env.scene[frame_name].data.target_quat_w.reshape(-1, 4)

def object_lin_vel(env: ManagerBasedEnv, object_id: int = 0) -> torch.Tensor:
    """Object root linear velocity in the environment frame."""
    object: RigidObject = env.scene[f"object_{object_id}"]
    return object.data.root_lin_vel_w

def last_action(env: ManagerBasedEnv) -> torch.Tensor:
    """The last input action to the environment."""
    if hasattr(env, "last_action"):
        return env.last_action
    else:
        return torch.zeros((env.num_envs, env.action_dim), device=env.device)
    
def symmetry_tracker(env: ManagerBasedEnv) -> torch.Tensor:
    """The last input action to the environment."""
    if hasattr(env, "symmetry_tracker"):
        if env.symmetry_tracker is not None:
            symmetry_tracker = env.symmetry_tracker.clone()
            symmetry_tracker[symmetry_tracker == 0] = -1
            return symmetry_tracker.reshape(-1, 1)
        else:
            return torch.ones((env.num_envs, 1), device=env.device) * -1
    else:
        return torch.ones((env.num_envs, 1), device=env.device) * -1

def generated_commands(env: ManagerBasedRLEnv, command_name: str, symmetry: bool = True) -> torch.Tensor:
    """The generated command from command term in the command manager with the given name."""
    command = env.command_manager.get_command(command_name)
    if symmetry:
        command_R = math_utils.matrix_from_quat(command[:, 3:])
        command_R = command_R.transpose(1, 2).reshape(-1, 9)
    else:
        command_R = command[:, :4]
    return torch.cat([command[:, :3], command_R], dim=-1)