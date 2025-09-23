from __future__ import annotations

import torch
import numpy as np
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv


def randomize_mass(env: ManagerBasedEnv, mass: torch.Tensor):
    # resolve environment ids
    env_ids = torch.arange(env.scene.num_envs, device="cpu")

    for object_id in range(env.num_object):
        asset = env.scene[f"object_{object_id}"]
        # resolve body indices
        assert asset.num_bodies == 1
        mass = mass.reshape(env.num_envs, 1)
        # set the mass into the physics simulation
        asset.root_physx_view.set_masses(mass, env_ids)

def randomize_material(
    env: ManagerBasedEnv,
    static_friction: torch.Tensor,
    static_friction_range: list[float],
    dynamic_friction: torch.Tensor,
    dynamic_friction_range: list[float],
    restitution: torch.Tensor,
    restitution_range: list[float],
    num_buckets: int = 250,
):
   # resolve environment ids
    env_ids = torch.arange(env.scene.num_envs, device="cpu")
    assets = [env.scene[f"object_{object_id}"] for object_id in range(env.num_object)]
    assets.append(env.scene["robot"])
    if 'robot_left' in env.scene.keys():
        assets.append(env.scene["robot_left"])

    static_friction = static_friction.cpu().numpy().reshape(-1, 1)
    dynamic_friction = dynamic_friction.cpu().numpy().reshape(-1, 1)
    restitution = restitution.cpu().numpy().reshape(-1, 1)

    for asset in assets:
        # retrieve material buffer
        materials = asset.root_physx_view.get_material_properties()

        # sample material properties from the given ranges
        material_samples = np.zeros(materials[env_ids].shape)
        shape = material_samples[..., 0].shape[-1]
        material_samples[..., 0] = np.tile(static_friction, (1, shape))
        shape = material_samples[..., 1].shape[-1]
        material_samples[..., 1] = np.tile(dynamic_friction, (1, shape))
        shape = material_samples[..., 2].shape[-1]
        material_samples[..., 2] = np.tile(restitution, (1, shape))

        # create uniform range tensor for bucketing
        lo = np.array([static_friction_range[0], dynamic_friction_range[0], restitution_range[0]])
        hi = np.array([static_friction_range[1], dynamic_friction_range[1], restitution_range[1]])

        # to avoid 64k material limit in physx, we bucket materials by binning randomized material properties
        # into buckets based on the number of buckets specified
        for d in range(3):
            buckets = np.array([(hi[d] - lo[d]) * i / num_buckets + lo[d] for i in range(num_buckets)])
            material_samples[..., d] = buckets[np.searchsorted(buckets, material_samples[..., d]) - 1]

        materials[env_ids] = torch.from_numpy(material_samples).to(dtype=torch.float)
        # apply to simulation
        asset.root_physx_view.set_material_properties(materials, env_ids)

def randomize_rew_weight(env: ManagerBasedEnv, rew_name: str, rew_weight: float):
    term_cfg = env.reward_manager.get_term_cfg(rew_name)
    # update term settings
    term_cfg.weight = rew_weight

def randomize_external_force_torque(
    env: ManagerBasedEnv,
    force: float,
    torque: float,
):
    # resolve environment ids
    env_ids = torch.arange(env.scene.num_envs, device="cpu")

    for object_id in range(env.num_object):
        # extract the used quantities (to enable type-hinting)
        asset = env.scene[f"object_{object_id}"]
        # resolve number of bodies
        num_bodies = asset.num_bodies

        # sample random forces and torques
        size = (len(env_ids), num_bodies, 3)
        forces = math_utils.sample_uniform(0.0, force, size, asset.device)
        torques = math_utils.sample_uniform(0.0, torque, size, asset.device)
        # set the forces and torques into the buffers
        # note: these are only applied when you call: `asset.write_data_to_sim()`
        asset.set_external_force_and_torque(forces, torques, env_ids=env_ids)

def randomize_reset_pose(env: ManagerBasedEnv, event_name: str, values: list[list[float]]):
    term_cfg = env.event_manager.get_term_cfg(event_name)
    # update term settings
    term_cfg.params["pose_range"]["x"] = values[0]
    term_cfg.params["pose_range"]["y"] = values[1]
    term_cfg.params["pose_range"]["z"] = values[2]
    if len(values) > 3:
        term_cfg.params["pose_range"]["roll"] = values[3]
        term_cfg.params["pose_range"]["pitch"] = values[4]
        term_cfg.params["pose_range"]["yaw"] = values[5]