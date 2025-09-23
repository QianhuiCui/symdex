from __future__ import annotations

import torch
import numpy as np
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
from isaaclab.actuators import ImplicitActuator
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.envs.mdp.events import _randomize_prop_by_op

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def reset_joints_by_symmetry(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    position_range: tuple[float, float],
    velocity_range: tuple[float, float],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Reset the robot joints by scaling the default position and velocity by the given ranges.

    This function samples random values from the given ranges and scales the default joint positions and velocities
    by these values. The scaled values are then set into the physics simulation.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # get default joint state
    joint_pos = asset.data.default_joint_pos[env_ids].clone()
    joint_vel = asset.data.default_joint_vel[env_ids].clone()
    # symmetry
    symmetric_idx = torch.where(env.symmetry_tracker[env_ids] == 1)[0]
    env_symmetric_idx = env_ids[symmetric_idx]
    if asset_cfg.name == "robot":
        symmetric_joint_pos = env.scene["robot_left"].data.default_joint_pos[env_symmetric_idx].clone() @ env.rep_Q_js
        symmetric_joint_vel = env.scene["robot_left"].data.default_joint_vel[env_symmetric_idx].clone() @ env.rep_Q_js
    else:
        symmetric_joint_pos = env.scene["robot"].data.default_joint_pos[env_symmetric_idx].clone() @ env.rep_Q_js
        symmetric_joint_vel = env.scene["robot"].data.default_joint_vel[env_symmetric_idx].clone() @ env.rep_Q_js
    joint_pos[symmetric_idx] = symmetric_joint_pos
    joint_vel[symmetric_idx] = symmetric_joint_vel

    # scale these values randomly
    joint_pos *= math_utils.sample_uniform(*position_range, joint_pos.shape, joint_pos.device)
    joint_vel *= math_utils.sample_uniform(*velocity_range, joint_vel.shape, joint_vel.device)

    # clamp joint pos to limits
    joint_pos_limits = asset.data.soft_joint_pos_limits[env_ids]
    joint_pos = joint_pos.clamp_(joint_pos_limits[..., 0], joint_pos_limits[..., 1])
    # clamp joint vel to limits
    joint_vel_limits = asset.data.soft_joint_vel_limits[env_ids]
    joint_vel = joint_vel.clamp_(-joint_vel_limits, joint_vel_limits)
    # set into the physics simulation
    asset.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)


def reset_object(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    pose_range: dict[str, tuple[float, float]],
    velocity_range: dict[str, tuple[float, float]],
    object_id: int,
):
    object = env.scene[f"object_{object_id}"]
    root_states = object.data.default_root_state[env_ids].clone()
    # poses
    range_list = [pose_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
    ranges = torch.tensor(range_list, device=object.device)
    rand_samples = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=object.device)
    positions = root_states[:, 0:3] + env.scene.env_origins[env_ids] + rand_samples[:, 0:3]
    orientations_delta = math_utils.quat_from_euler_xyz(rand_samples[:, 3], rand_samples[:, 4], rand_samples[:, 5])
    orientations = math_utils.quat_mul(root_states[:, 3:7], orientations_delta)
    # velocities
    range_list = [velocity_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
    ranges = torch.tensor(range_list, device=object.device)
    rand_samples = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=object.device)
    velocities = root_states[:, 7:13] + rand_samples

        # set into the physics simulation
    object.write_root_pose_to_sim(torch.cat([positions, orientations], dim=-1), env_ids=env_ids)
    object.write_root_velocity_to_sim(velocities, env_ids=env_ids)

def reset_articulation(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    pose_range: dict[str, tuple[float, float]],
    velocity_range: dict[str, tuple[float, float]],
    asset_cfg: SceneEntityCfg,
):
    asset: Articulation = env.scene[asset_cfg.name]
    root_states = asset.data.default_root_state[env_ids].clone()
    # poses
    range_list = [pose_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
    ranges = torch.tensor(range_list, device=asset.device)
    rand_samples = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=asset.device)
    positions = root_states[:, 0:3] + env.scene.env_origins[env_ids] + rand_samples[:, 0:3]
    orientations_delta = math_utils.quat_from_euler_xyz(rand_samples[:, 3], rand_samples[:, 4], rand_samples[:, 5])
    orientations = math_utils.quat_mul(root_states[:, 3:7], orientations_delta)
    # velocities
    range_list = [velocity_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
    ranges = torch.tensor(range_list, device=asset.device)
    rand_samples = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=asset.device)
    velocities = root_states[:, 7:13] + rand_samples

    # set into the physics simulation
    asset.write_root_pose_to_sim(torch.cat([positions, orientations], dim=-1), env_ids=env_ids)
    asset.write_root_velocity_to_sim(velocities, env_ids=env_ids)

def reset_tote(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    object_id: int,
):
    object: RigidObject = env.scene[f"object_{object_id}"]
    object_pos_w = object.data.root_pos_w[env_ids]
    object_pos_w[:, 2] = 0.0
    tote_quat_w = env.scene["tote"].data.default_root_state[env_ids, 3:7].clone()    
    # set into the physics simulation
    env.scene["tote"].write_root_pose_to_sim(torch.cat([object_pos_w, tote_quat_w], dim=-1), env_ids=env_ids)

def reset_joints_random_pose(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    velocity_range: tuple[float, float],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Reset the robot joints by scaling the default position and velocity by the given ranges.

    This function samples random values from the given ranges and scales the default joint positions and velocities
    by these values. The scaled values are then set into the physics simulation.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # get default joint state
    joint_pos = asset.data.default_joint_pos[env_ids].clone()
    joint_vel = asset.data.default_joint_vel[env_ids].clone()

    # scale these values randomly
    indices = torch.randint(0, env.arm_pose.shape[0], (env_ids.shape[0],)).to(env.device)
    joint_pos[:, env.right_arm_joints_idx] = env.arm_pose[indices].clone()
    joint_vel *= math_utils.sample_uniform(*velocity_range, joint_vel.shape, joint_vel.device)

    # clamp joint pos to limits
    joint_pos_limits = asset.data.soft_joint_pos_limits[env_ids]
    joint_pos = joint_pos.clamp_(joint_pos_limits[..., 0], joint_pos_limits[..., 1])
    # clamp joint vel to limits
    joint_vel_limits = asset.data.soft_joint_vel_limits[env_ids]
    joint_vel = joint_vel.clamp_(-joint_vel_limits, joint_vel_limits)

    # set into the physics simulation
    asset.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)

def reset_dual_joints_random_pose(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    velocity_range: tuple[float, float],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Reset the robot joints by scaling the default position and velocity by the given ranges.

    This function samples random values from the given ranges and scales the default joint positions and velocities
    by these values. The scaled values are then set into the physics simulation.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # get default joint state
    joint_pos = asset.data.default_joint_pos[env_ids].clone()
    joint_vel = asset.data.default_joint_vel[env_ids].clone()

    # scale these values randomly
    indices = torch.randint(0, env.arm_pose.shape[0], (env_ids.shape[0],)).to(env.device)
    joint_pos[:, env.right_arm_joints_idx] = env.arm_pose[indices].clone()
    indices = torch.randint(0, env.arm_pose.shape[0], (env_ids.shape[0],)).to(env.device)
    joint_pos[:, env.left_arm_joints_idx] = env.arm_pose[indices].clone()
    joint_pos[:, env.left_arm_joints_idx] *= torch.tensor([1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0], device=env.device)
    joint_vel *= math_utils.sample_uniform(*velocity_range, joint_vel.shape, joint_vel.device)

    # clamp joint pos to limits
    joint_pos_limits = asset.data.soft_joint_pos_limits[env_ids]
    joint_pos = joint_pos.clamp_(joint_pos_limits[..., 0], joint_pos_limits[..., 1])
    # clamp joint vel to limits
    joint_vel_limits = asset.data.soft_joint_vel_limits[env_ids]
    joint_vel = joint_vel.clamp_(-joint_vel_limits, joint_vel_limits)

    # set into the physics simulation
    asset.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
    
def randomize_rigid_body_mass(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    mass_distribution_params: tuple[float, float],
    operation: Literal["add", "scale", "abs"],
    distribution: Literal["uniform", "log_uniform", "gaussian"] = "uniform",
    object_id: list[int] = None,
):
    """Randomize the mass of the bodies by adding, scaling, or setting random values.

    This function allows randomizing the mass of the bodies of the asset. The function samples random values from the
    given distribution parameters and adds, scales, or sets the values into the physics simulation based on the operation.

    .. tip::
        This function uses CPU tensors to assign the body masses. It is recommended to use this function
        only during the initialization of the environment.
    """
    # resolve environment ids
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device="cpu")
    else:
        env_ids = env_ids.cpu()

    if object_id is None:
        object_id = range(env.num_object)

    for object_id in object_id:
        asset = env.scene[f"object_{object_id}"]
        # resolve body indices
        body_ids = torch.arange(asset.num_bodies, dtype=torch.int, device="cpu")
        # get the current masses of the bodies (num_assets, num_bodies)
        masses = asset.root_physx_view.get_masses()
        # apply randomization on default values
        masses[env_ids[:, None], body_ids] = asset.data.default_mass[env_ids[:, None], body_ids].clone()
        # sample from the given range
        # note: we modify the masses in-place for all environments
        #   however, the setter takes care that only the masses of the specified environments are modified
        masses = _randomize_prop_by_op(
            masses, mass_distribution_params, env_ids, body_ids, operation=operation, distribution=distribution
        )
        # set the mass into the physics simulation
        asset.root_physx_view.set_masses(masses, env_ids)

def randomize_rigid_body_material(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    static_friction_range: tuple[float, float],
    dynamic_friction_range: tuple[float, float],
    restitution_range: tuple[float, float],
    num_buckets: int,
):
    """Randomize the physics materials on all geometries of the asset.

    This function creates a set of physics materials with random static friction, dynamic friction, and restitution
    values. The number of materials is specified by ``num_buckets``. The materials are generated by sampling
    uniform random values from the given ranges.

    The material properties are then assigned to the geometries of the asset. The assignment is done by
    creating a random integer tensor of shape  (num_instances, max_num_shapes) where ``num_instances``
    is the number of assets spawned and ``max_num_shapes`` is the maximum number of shapes in the asset (over
    all bodies). The integer values are used as indices to select the material properties from the
    material buckets.

    .. attention::
        This function uses CPU tensors to assign the material properties. It is recommended to use this function
        only during the initialization of the environment. Otherwise, it may lead to a significant performance
        overhead.

    .. note::
        PhysX only allows 64000 unique physics materials in the scene. If the number of materials exceeds this
        limit, the simulation will crash.
    """
    # resolve environment ids
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device="cpu")
    else:
        env_ids = env_ids.cpu()

    for object_id in range(env.num_object):
        asset = env.scene[f"object_{object_id}"]
        # retrieve material buffer
        materials = asset.root_physx_view.get_material_properties()

        # sample material properties from the given ranges
        material_samples = np.zeros(materials[env_ids].shape)
        material_samples[..., 0] = np.random.uniform(*static_friction_range)
        material_samples[..., 1] = np.random.uniform(*dynamic_friction_range)
        material_samples[..., 2] = np.random.uniform(*restitution_range)

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

def randomize_actuator_gains(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
    stiffness_distribution_params: tuple[float, float] | None = None,
    damping_distribution_params: tuple[float, float] | None = None,
    operation: Literal["add", "scale", "abs"] = "abs",
    distribution: Literal["uniform", "log_uniform", "gaussian"] = "uniform",
):
    """Randomize the actuator gains in an articulation by adding, scaling, or setting random values.

    This function allows randomizing the actuator stiffness and damping gains.

    The function samples random values from the given distribution parameters and applies the operation to the joint properties.
    It then sets the values into the actuator models. If the distribution parameters are not provided for a particular property,
    the function does not modify the property.

    .. tip::
        For implicit actuators, this function uses CPU tensors to assign the actuator gains into the simulation.
        In such cases, it is recommended to use this function only during the initialization of the environment.

    Raises:
        NotImplementedError: If the joint indices are in explicit motor mode. This operation is currently
            not supported for explicit actuator models.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]

    # resolve environment ids
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device=asset.device)

    # resolve joint indices
    if asset_cfg.joint_ids == slice(None):
        joint_ids_list = range(asset.num_joints)
        joint_ids = slice(None)  # for optimization purposes
    else:
        joint_ids_list = asset_cfg.joint_ids
        joint_ids = torch.tensor(asset_cfg.joint_ids, dtype=torch.int, device=asset.device)

    # check if none of the joint indices are in explicit motor mode
    for joint_index in joint_ids_list:
        for act_name, actuator in asset.actuators.items():
            # if joint indices are a slice (i.e., all joints are captured) or the joint index is in the actuator
            if actuator.joint_indices == slice(None) or joint_index in actuator.joint_indices:
                if not isinstance(actuator, ImplicitActuator):
                    raise NotImplementedError(
                        "Event term 'randomize_actuator_stiffness_and_damping' is performed on asset"
                        f" '{asset_cfg.name}' on the joint '{asset.joint_names[joint_index]}' ('{joint_index}') which"
                        f" uses an explicit actuator model '{act_name}<{actuator.__class__.__name__}>'. This operation"
                        " is currently not supported for explicit actuator models."
                    )

    # sample joint properties from the given ranges and set into the physics simulation
    # Create a grid of stiffness and damping values
    num_grids = 10000
    grid_size = int(num_grids**0.5)  # Square root of 10000 for a square grid
    
    stiffness_range = torch.linspace(1, 2000, steps=grid_size, device=asset.device)
    damping_range = torch.linspace(1, 500, steps=grid_size, device=asset.device)
    
    # Create meshgrid
    stiffness_grid, damping_grid = torch.meshgrid(stiffness_range, damping_range, indexing='ij')
    
    # Flatten the grids
    stiffness_values = stiffness_grid.reshape(-1)
    damping_values = damping_grid.reshape(-1)
    
    # Ensure we have exactly 10000 pairs
    stiffness_values = stiffness_values[:num_grids]
    damping_values = damping_values[:num_grids]
    
    # Shuffle the pairs to randomize the order
    shuffle_indices = torch.randperm(num_grids)
    stiffness_values = stiffness_values[shuffle_indices]
    damping_values = damping_values[shuffle_indices]
    
    # Assign values to environments
    num_envs = env_ids.shape[0]
    num_joints = asset.data.default_joint_stiffness.shape[1]
    
    stiffness = asset.data.default_joint_stiffness.to(asset.device).clone()
    damping = asset.data.default_joint_damping.to(asset.device).clone()
    
    for i in range(num_envs):
        env_stiffness = stiffness_values[i % num_grids]
        env_damping = damping_values[i % num_grids]
        
        stiffness[env_ids[i], :] = env_stiffness.expand(num_joints)
        damping[env_ids[i], :] = env_damping.expand(num_joints)
    
    env.stiffness = stiffness
    env.damping = damping
    asset.write_joint_stiffness_to_sim(stiffness, joint_ids=joint_ids, env_ids=env_ids)
    asset.write_joint_damping_to_sim(damping, joint_ids=joint_ids, env_ids=env_ids)