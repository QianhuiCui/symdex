
from __future__ import annotations

import torch
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg

from symdex.utils.point_cloud_utils import create_pointcloud_from_depth

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

def last_action_side(env: ManagerBasedEnv, side=None) -> torch.Tensor:
    """The last input action to the environment."""
    if hasattr(env, "last_action"):
        last_action = env.last_action
    else:
        last_action = torch.zeros((env.num_envs, env.action_dim), device=env.device)

    if side is None:
        return last_action
    else:
        if side == "right":
            return last_action[:, :22]
        elif side == "left":
            return last_action[:, 22:]
        else:
            raise ValueError(f"Invalid side: {side}")
    
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

def rgb_image(env: ManagerBasedEnv, camera_name: list[str]) -> torch.Tensor:    
    # check which envs' the frame stack is all zeros
    reset_env_ids = torch.where(env.frame_stacks.sum(dim=(1,2,3,4,5)) == 0)[0]
    # check which envs' the frame stack is not all zeros
    not_reset_env_ids = torch.where(env.frame_stacks.sum(dim=(1,2,3,4,5)) != 0)[0]

    for i in range(len(camera_name)):
        camera = env.scene[camera_name[i]]
        if hasattr(env, "camera_delay") and camera_name[i] in env.camera_delay.keys():
            img = env.camera_delay[camera_name[i]]["rgb"].compute(camera.data.output['rgb'].clone())
        else:
            img = camera.data.output['rgb']
        # fill all frames with the current frame
        env.frame_stacks[reset_env_ids, i, :, :, :, :] = img[reset_env_ids].unsqueeze(1).repeat(1, env.cfg.hydra_cfg.task.cam.frame_stack, 1, 1, 1)
        # fill the frame stack with the current frame
        env.frame_stacks[not_reset_env_ids, i, :, :, :, :] = torch.cat([env.frame_stacks[not_reset_env_ids, i, 1:, :, :, :], img[not_reset_env_ids].unsqueeze(1)], dim=1)
    
    return env.frame_stacks.reshape(env.num_envs, env.num_cams, env.cfg.hydra_cfg.task.cam.resolution, env.cfg.hydra_cfg.task.cam.resolution, env.cfg.hydra_cfg.task.cam.frame_stack * 3)

def depth_image(env: ManagerBasedEnv, camera_name: list[str]) -> torch.Tensor:    
    depth_stacks = []
    for i in range(len(camera_name)):
        camera = env.scene[camera_name[i]]
        img = camera.data.output['depth']
        depth_stacks.append(img)
    depth_stack = torch.stack(depth_stacks, dim=1)
    return depth_stack

def point_cloud(env: ManagerBasedEnv, 
                camera_name: list[str], 
                wrist_cam_name: list[str] = [],
                crop_range: list[list[float]] = [[-0.5, 0.5], [-0.5, 0.0], [-0.8, 0.0]], 
                max_points: int = 1024, 
                downsample: str = "random",
                add_noise: bool = False,
                ) -> torch.Tensor:
    """
    Args:
        camera_name: list of camera names
        crop_range: list of crop ranges for each camera
        max_points: maximum number of points to sample
        downsample: method to downsample the point cloud, "random" or "FSP"
    """
    if not hasattr(env, "camera_offset"):
        return torch.zeros((env.num_envs, max_points, 3), device=env.device)

    intrinsic_matrix = []
    depth = []
    position_offset = []
    orientation_offset = []
    for cam_name in camera_name:
        intrinsic_matrix.append(env.scene[cam_name].data.intrinsic_matrices.clone())

        out = env.scene[cam_name].data.output
        print(cam_name, list(out.keys()))
        
        depth.append(env.scene[cam_name].data.output['depth'].clone())
        position_offset.append(env.camera_offset[cam_name]["pos"])
        orientation_offset.append(env.camera_offset[cam_name]["orientation"])
    for cam_name in wrist_cam_name:
        intrinsic_matrix.append(env.scene[cam_name].data.intrinsic_matrices.clone())
        position_offset.append(env.scene[f"{cam_name}_marker"].data.target_pos_w.reshape(-1, 3).clone() - env.scene.env_origins)
        orientation_offset.append(env.scene[f"{cam_name}_marker"].data.target_quat_w.reshape(-1, 4).clone())
        if hasattr(env, "camera_delay") and cam_name in env.camera_delay.keys():
            depth.append(env.camera_delay[cam_name]["depth"].compute(env.scene[cam_name].data.output['depth'].clone()))
        else:
            depth.append(env.scene[cam_name].data.output['depth'].clone())
        
    intrinsic_matrix = torch.cat(intrinsic_matrix, dim=0)
    depth = torch.cat(depth, dim=0)
    position_offset = torch.cat(position_offset, dim=0)
    orientation_offset = torch.cat(orientation_offset, dim=0)
    pc = create_pointcloud_from_depth(intrinsic_matrix=intrinsic_matrix,
                                                depth=depth,
                                                position=position_offset,
                                                orientation=orientation_offset,
                                                crop_range=crop_range,
                                                max_points=max_points,
                                                num_cams=len(camera_name) + len(wrist_cam_name),
                                                downsample=downsample,
                                                add_noise=add_noise)
    return pc