from __future__ import annotations

import torch
import isaacsim.core.utils.bounds as bounds_utils
import isaaclab.utils.math as math_utils


def compute_bbox(env):
    object = env.unwrapped.scene["table"]
    cache = bounds_utils.create_bbox_cache()
    for i in range(len(object.root_physx_view.prim_paths)):
        min_x, min_y, min_z, max_x, max_y, max_z = bounds_utils.compute_combined_aabb(cache, prim_paths=[object.root_physx_view.prim_paths[i]])
        print(f"min: {min_x, min_y, min_z}")
        print(f"max: {max_x, max_y, max_z}")
        print(f"center: {(min_x + max_x) / 2, (min_y + max_y) / 2, (min_z + max_z) / 2}")


def angle_distance_tensor(theta1, theta2):
    """
    Compute the shortest angular distance between two tensors of angles in radians.
    
    Args:
        theta1 (torch.Tensor): Tensor of angles (shape: num_env).
        theta2 (torch.Tensor): Tensor of angles (shape: num_env).
    
    Returns:
        torch.Tensor: Tensor of angular distances (shape: num_env).
    """
    delta = (theta2 - theta1 + torch.pi) % (2 * torch.pi) - torch.pi
    return delta


def get_angle_from_quat(quat_w, 
                   axis: str = "x", 
                   normalize: bool = False,
                   ):
    """Get the angle from a quaternion."""
    rot_mat = math_utils.matrix_from_quat(quat_w)

    if axis == "x":
        angle = rot_mat[:, :, 0]
    elif axis == "y":
        angle = rot_mat[:, :, 1]
    elif axis == "z":
        angle = rot_mat[:, :, 2]

    if normalize:
        angle = angle / torch.linalg.norm(angle, dim=-1, keepdim=True)
    return angle