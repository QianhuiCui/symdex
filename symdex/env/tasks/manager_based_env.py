from __future__ import annotations

import gymnasium as gym
import math
import os
import numpy as np
import torch
from collections.abc import Sequence
from typing import Any, ClassVar
import builtins

import carb
from isaacsim.core.version import get_version
import omni.log
import isaacsim.core.utils.stage as stage_utils
from isaaclab.managers import CommandManager, CurriculumManager, RewardManager, TerminationManager
from isaaclab.envs.manager_based_rl_env import ManagerBasedRLEnv
from isaaclab.envs.common import VecEnvStepReturn
from isaaclab.envs.ui import ViewportCameraController
from isaaclab.scene import InteractiveScene
from isaaclab.sim import SimulationContext
import isaaclab.sim as sim_utils
import isaacsim.core.utils.torch as torch_utils
from isaaclab.utils.timer import Timer
from isaaclab.utils import DelayBuffer
from isaaclab.assets import Articulation, ArticulationCfg, RigidObject, RigidObjectCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.sensors.contact_sensor import ContactSensor

from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.markers.visualization_markers import VisualizationMarkers
import isaaclab.utils.math as math_utils
from isaaclab.utils.math import (
    convert_camera_frame_orientation_convention,
    create_rotation_matrix_from_view,
    quat_from_matrix,
    quat_inv,
)

from symdex.env.tasks.manager_based_env_cfg import BaseEnvCfg
from symdex.utils.domain_random import DomainRandomizer
from symdex.env.mdps.randomization_mdps import *
from symdex.utils.symmetry import load_symmetric_system
from symdex.utils.random_utils import *

# OpenCV to OpenGL conversion
CV_TO_GL = np.array([
    [1,  0,  0],
    [0, -1,  0],
    [0,  0, -1]
])

GL_TO_CV = np.array([
    [1,  0,  0],
    [0, -1,  0],
    [0,  0, -1]
])


class BaseEnv(ManagerBasedRLEnv):
    is_vector_env: ClassVar[bool] = True
    """Whether the environment is a vectorized environment."""
    metadata: ClassVar[dict[str, Any]] = {
        "render_modes": [None, "human", "rgb_array"],
        "isaac_sim_version": get_version(),
    }
    """Metadata for the environment."""

    cfg: BaseEnvCfg
    """Configuration for the environment."""

    def __init__(self, cfg, render_mode: str | None = None, **kwargs):
        # store inputs to class
        self.cfg = cfg
        # initialize internal variables
        self._is_closed = False

        # set the seed for the environment
        if self.cfg.seed is not None:
            self.cfg.seed = self.seed(self.cfg.seed)
        else:
            omni.log.warn("Seed not set for the environment. The environment creation may not be deterministic.")

        # create a simulation context to control the simulator
        if SimulationContext.instance() is None:
            # the type-annotation is required to avoid a type-checking error
            # since it gets confused with Isaac Sim's SimulationContext class
            self.sim: SimulationContext = SimulationContext(self.cfg.sim)
        else:
            # simulation context should only be created before the environment
            # when in extension mode
            if not builtins.ISAAC_LAUNCHED_FROM_TERMINAL:
                raise RuntimeError("Simulation context already exists. Cannot create a new one.")
            self.sim: SimulationContext = SimulationContext.instance()

        # print useful information
        print("[INFO]: Base environment:")
        print(f"\tEnvironment device    : {self.device}")
        print(f"\tEnvironment seed      : {self.cfg.seed}")
        print(f"\tPhysics step-size     : {self.physics_dt}")
        print(f"\tRendering step-size   : {self.physics_dt * self.cfg.sim.render_interval}")
        print(f"\tEnvironment step-size : {self.step_dt}")

        if self.cfg.sim.render_interval < self.cfg.decimation:
            msg = (
                f"The render interval ({self.cfg.sim.render_interval}) is smaller than the decimation "
                f"({self.cfg.decimation}). Multiple render calls will happen for each environment step. "
                "If this is not intended, set the render interval to be equal to the decimation."
            )
            omni.log.warn(msg)

        # counter for simulation steps
        self._sim_step_counter = 0

        # generate scene
        with Timer("[INFO]: Time taken for scene creation", "scene_creation"):
            self.scene = InteractiveScene(self.cfg.scene)
        print("[INFO]: Scene manager: ", self.scene)
        # camera frame stack
        self.num_cams = self.cfg.hydra_cfg.task.cam.num_cams
        self.frame_stacks = torch.zeros((self.num_envs, self.num_cams, self.cfg.hydra_cfg.task.cam.frame_stack, self.cfg.hydra_cfg.task.cam.resolution, self.cfg.hydra_cfg.task.cam.resolution, 3), device=self.device, dtype=torch.uint8)

        # environment specific initialization
        self._pre_init_process()

        # set up camera viewport controller
        # viewport is not available in other rendering modes so the function will throw a warning
        # FIXME: This needs to be fixed in the future when we unify the UI functionalities even for
        # non-rendering modes.
        if self.sim.render_mode >= self.sim.RenderMode.PARTIAL_RENDERING:
            self.viewport_camera_controller = ViewportCameraController(self, self.cfg.viewer)
        else:
            self.viewport_camera_controller = None

        # play the simulator to activate physics handles
        # note: this activates the physics simulation view that exposes TensorAPIs
        # note: when started in extension mode, first call sim.reset_async() and then initialize the managers
        if builtins.ISAAC_LAUNCHED_FROM_TERMINAL is False:
            print("[INFO]: Starting the simulation. This may take a few seconds. Please wait...")
            with Timer("[INFO]: Time taken for simulation start", "simulation_start"):
                self.sim.reset()
                # update scene to pre populate data buffers for assets and sensors.
                # this is needed for the observation manager to get valid tensors for initialization.
                # this shouldn't cause an issue since later on, users do a reset over all the environments so the lazy buffers would be reset.
                self.scene.update(dt=self.physics_dt)
            # add timeline event to load managers
            self.load_managers()

        # make sure torch is running on the correct device
        if "cuda" in self.device:
            torch.cuda.set_device(self.device)

        # extend UI elements
        # we need to do this here after all the managers are initialized
        # this is because they dictate the sensors and commands right now
        if self.sim.has_gui() and self.cfg.ui_window_class_type is not None:
            # setup live visualizers
            self.setup_manager_visualizers()
            self._window = self.cfg.ui_window_class_type(self, window_name="IsaacLab")
        else:
            # if no window, then we don't need to store the window
            self._window = None

        # allocate dictionary to store metrics
        self.extras = {}

        # initialize observation buffers
        self.obs_buf = {}

        # initialize language instruction buffer
        self.language_instruction_buf = [""] * self.num_envs

        # store the render mode
        self.render_mode = render_mode

        # initialize data and constants
        # -- counter for curriculum
        self.common_step_counter = 0
        # -- init buffers
        self.episode_length_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        # -- set the framerate of the gym video recorder wrapper so that the playback speed of the produced video matches the simulation
        self.metadata["render_fps"] = 1 / self.step_dt

        # post init
        self._post_init_process()
    
    def step(self, action: torch.Tensor) -> VecEnvStepReturn:
        # update last action inferred from the policy. this is different from the last action used in IsaacLab action manager
        self.last_action = action.clone()
        # scale the action
        action = action * self._scale
        super().step(action)

        self.extras['success'] = self.success_tracker
        self.extras['detailed_reward'] = self.detailed_reward_buf
        self.extras["language_instruction"] = list(self.language_instruction_buf)
        # post reset process
        reset_indices = torch.where(self.episode_length_buf == 1)[0]
        if len(reset_indices) > 0:
            self._post_reset_process(reset_indices)
        
        # return observations, rewards, resets and extras
        return self.obs_buf, self.reward_buf, self.reset_terminated, self.reset_time_outs, self.extras
    
    def update_randomization(self, success_rate):
        self.domain_randomizer.update(success_rate)
        randomized_values, randomization_state, curriculum_state = self.domain_randomizer.sample()
        if "object_mass" in randomized_values:
            randomize_mass(self, randomized_values["object_mass"])
        if "static_friction" in randomized_values:
            randomize_material(self, static_friction=randomized_values["static_friction"],
                               static_friction_range=list(self.cfg.hydra_cfg.task.randomize.randomization.static_friction),
                               dynamic_friction=randomized_values["dynamic_friction"], 
                               dynamic_friction_range=list(self.cfg.hydra_cfg.task.randomize.randomization.dynamic_friction),
                               restitution=randomized_values["restitution"], 
                               restitution_range=list(self.cfg.hydra_cfg.task.randomize.randomization.restitution), 
                               num_buckets=250)
        if "action_scale" in randomized_values:
            self._scale[:6] = randomized_values["action_scale"]
        if "energy_penalty" in randomized_values:
            for rew_name in curriculum_state["energy_penalty"]["names"]:
                randomize_rew_weight(self, rew_name, randomized_values["energy_penalty"])
        if "collision_penalty" in randomized_values:
            for rew_name in curriculum_state["collision_penalty"]["names"]:
                randomize_rew_weight(self, rew_name, randomized_values["collision_penalty"])
        if "external_force_torque" in randomized_values:
            randomize_external_force_torque(self, force=randomized_values["external_force_torque"], torque=randomized_values["external_force_torque"])
        # reset pose
        for rand_name in randomized_values.keys():
            if "reset_pose" in rand_name:
                randomize_reset_pose(self, curriculum_state[rand_name]["names"], randomized_values[rand_name])
        return randomization_state, curriculum_state, self.domain_randomizer.best_so_far
    
    """
    Helper functions.
    """
    def _pre_init_process(self):
        self.action_dim = self.cfg.action_dim
        self.num_object = self.cfg.num_object

    def _post_init_process(self):
        # initialize IK visualizer
        if self.cfg.visualize_marker:
            self.markers = dict()
            # initialize marker for each arm
            frame_marker_cfg = FRAME_MARKER_CFG.copy()
            frame_marker_cfg.markers["frame"].scale = self.cfg.visualizer_scale
            self.markers['arm_l'] = dict()
            self.markers['arm_l']['ee_marker'] = VisualizationMarkers(frame_marker_cfg.replace(prim_path="/Visuals/left_ee_current"))
            self.markers['arm_l']['goal_marker'] = VisualizationMarkers(frame_marker_cfg.replace(prim_path="/Visuals/left_ee_goal"))
            self.markers['arm_r'] = dict()
            self.markers['arm_r']['ee_marker'] = VisualizationMarkers(frame_marker_cfg.replace(prim_path="/Visuals/right_ee_current"))
            self.markers['arm_r']['goal_marker'] = VisualizationMarkers(frame_marker_cfg.replace(prim_path="/Visuals/right_ee_goal"))

        # save the scale as tensors
        if self.cfg.action_scale is not None:
            self._scale = torch.tensor(self.cfg.action_scale, device=self.device)
        else:
            self._scale = torch.ones(self.action_space.shape, device=self.device)

        self.last_action = torch.zeros(self.action_space.shape, device=self.device)
        self.object_init_pos = torch.zeros((self.num_object, self.num_envs, 3), device=self.device)
        self.object_init_orient = torch.zeros((self.num_object, self.num_envs, 4), device=self.device)
        self.success_tracker = None
        self.symmetry_tracker = None
        self.domain_randomizer = DomainRandomizer(self.cfg.hydra_cfg.task.randomize, num_envs=self.num_envs)

        # symmetry
        self.G = load_symmetric_system(cfg=self.cfg.hydra_cfg.task.symmetry)
        assert self.cfg.hydra_cfg.task.symmetry.group_label == 'C2'
        symmetric_group = self.G.elements[-1]
        self.rep_Rd = self.G.representations['R3'](symmetric_group)
        self.rep_Rd = torch.tensor(self.rep_Rd, device=self.device, dtype=torch.float32)
        self.rep_SO3_flat = self.G.representations['SO3_flat'](symmetric_group)
        self.rep_SO3_flat = torch.tensor(self.rep_SO3_flat, device=self.device, dtype=torch.float32)
        self.rep_Q_js = self.G.representations['Q_js'](symmetric_group)
        self.rep_Q_js = torch.tensor(self.rep_Q_js, device=self.device, dtype=torch.float32)

        self.all_time_first = True

        # initialize camera
        if self.cfg.hydra_cfg.task.cam.enable:
            self.camera_offset = {}
            self._process_camera_cfg(self.scene["robot"]._ALL_INDICES)
            # set camera delay
            self.camera_delay = {}
            for cam_name, cam_cfg in self.cfg.hydra_cfg.task.cam.cam_params.items():
                if not cam_cfg.enable or "wrist" not in cam_name or cam_cfg.delay == 0:
                    continue
                self.camera_delay[cam_name] = {
                    "rgb": DelayBuffer(cam_cfg.delay, self.num_envs, device=self.device),
                    "depth": DelayBuffer(cam_cfg.delay, self.num_envs, device=self.device),
                    "position_offset": DelayBuffer(cam_cfg.delay, self.num_envs, device=self.device),
                    "orientation_offset": DelayBuffer(cam_cfg.delay, self.num_envs, device=self.device),
                }
            # Set each delay buffer .set_time_lag(cam_cfg.delay)
            for cam_name in self.camera_delay.keys():
                self.camera_delay[cam_name]["rgb"].set_time_lag(self.cfg.hydra_cfg.task.cam.cam_params[cam_name].delay)
                self.camera_delay[cam_name]["depth"].set_time_lag(self.cfg.hydra_cfg.task.cam.cam_params[cam_name].delay)

        print("[INFO]: Completed setting up the environment...")

    def _configure_gym_env_spaces(self):
        """Configure the action and observation spaces for the Gym environment."""
        # observation space (unbounded since we don't impose any limits)
        self.single_observation_space = gym.spaces.Dict()
        for group_name, group_term_names in self.observation_manager.active_terms.items():
            # extract quantities about the group
            has_concatenated_obs = self.observation_manager.group_obs_concatenate[group_name]
            group_dim = self.observation_manager.group_obs_dim[group_name]
            group_term_dim = self.observation_manager.group_obs_term_dim[group_name]
            # check if group is concatenated or not
            # if not concatenated, then we need to add each term separately as a dictionary
            if has_concatenated_obs:
                self.single_observation_space[group_name] = gym.spaces.Box(low=-np.inf, high=np.inf, shape=group_dim)
            else:
                self.single_observation_space[group_name] = gym.spaces.Dict({
                    term_name: gym.spaces.Box(low=-np.inf, high=np.inf, shape=term_dim)
                    for term_name, term_dim in zip(group_term_names, group_term_dim)
                })
        # action space (unbounded since we don't impose any limits)
        self.single_action_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.action_dim,))

        # batch the spaces for vectorized environments
        self.observation_space = gym.vector.utils.batch_space(self.single_observation_space, self.num_envs)
        self.action_space = gym.vector.utils.batch_space(self.single_action_space, self.num_envs)
    
    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self.scene["robot"]._ALL_INDICES
            env_id_regex = r"\d+"                       
        else:
            env_id_regex = "|".join(map(str, env_ids.tolist()))

        self.scene["robot"].reset(env_ids)
        if 'robot_left' in self.scene.keys():
            self.scene["robot_left"].reset(env_ids)
        
        if self.symmetry_tracker is None:
            self.symmetry_tracker = torch.zeros(self.num_envs, device=self.device)

        if self.cfg.hydra_cfg.task.symmetry.symmetric_envs:
            if self.all_time_first:
                self.all_time_first = False
                symmetric_env_ids = None
            else:
                mask = torch.rand(env_ids.shape) < 0.5
                symmetric_env_ids = env_ids[mask]
                other_env_ids = env_ids[~mask]
                self.symmetry_tracker[symmetric_env_ids] = 1
                self.symmetry_tracker[other_env_ids] = 0

        super()._reset_idx(env_ids)
        if self.cfg.hydra_cfg.task.symmetry.symmetric_envs and symmetric_env_ids is not None:
            if self.cfg.hydra_cfg.task.symmetry.group_label == 'C4':
                self._reset_symmetry_c4(env_ids)
            else:
                self._reset_symmetry(symmetric_env_ids)

        # camera frame stack
        if self.cfg.hydra_cfg.task.cam.enable:
            self.frame_stacks[env_ids] = torch.zeros((env_ids.shape[0], self.num_cams, self.cfg.hydra_cfg.task.cam.frame_stack, self.cfg.hydra_cfg.task.cam.resolution, self.cfg.hydra_cfg.task.cam.resolution, 3), device=self.device, dtype=torch.uint8)
            # Set each delay buffer .set_time_lag(cam_cfg.delay)
            for cam_name in self.camera_delay.keys():
                self.camera_delay[cam_name]["rgb"].reset(env_ids)
                self.camera_delay[cam_name]["depth"].reset(env_ids)
            self._process_camera_cfg(env_ids)
        
        stage = omni.usd.get_context().get_stage()
        # randomize light
        random_light_flag = getattr(self.cfg.scene.light, "random_light_reset", False)
        random_freq = getattr(self.cfg.scene.light, "random_light_freq", 100)
        if random_light_flag and self.global_step_tracker % random_freq == 0:
            light_prim_path = "/World/light"
            light_prim = stage.GetPrimAtPath(light_prim_path)
            if light_prim.IsValid():
                stage.RemovePrim(light_prim_path)
            light_cfg = get_light(random_light=random_light_flag)
            create_light(stage, light_prim_path, light_cfg, translation=(0.0, 0.0, 1.5))
        
        for obj_id in range(self.num_object):
            spawn_cfg = getattr(self.cfg.scene, f"object_{obj_id}").spawn
            prims = find_prims_by_regex(stage, rf"^/World/envs/env_({env_id_regex})/Object_{obj_id}$")
            # get object category
            if getattr(spawn_cfg, "obj_label", False):
                object_label = get_object_label(spawn_cfg.usd_path, prims)
            else:
                object_label = [spawn_cfg.usd_path] * len(prims)
            # randomize surface
            # if getattr(spawn_cfg, "random_texture", False):
            #     randomise_textures(prims, spawn_cfg.texture_path)
            surface_cfg = getattr(spawn_cfg, "preview_surface", None)
            if getattr(spawn_cfg, "preview_surface", None) is not None:
                if getattr(spawn_cfg, "random_color", False):
                    color_label, color = sample_dict(surface_cfg.diffuse_color_dict, env_ids.shape[0], self.device)
                else:
                    color = None
                if getattr(spawn_cfg, "random_roughness", False):
                    roughness = math_utils.sample_uniform(surface_cfg.roughness_range[0], surface_cfg.roughness_range[1], (env_ids.shape[0], 1), self.device)
                else:
                    roughness = torch.full((env_ids.shape[0], 1), 0.5, device=self.device)
                if getattr(spawn_cfg, "random_metallic", False):
                    metallic = math_utils.sample_uniform(surface_cfg.metallic_range[0], surface_cfg.metallic_range[1], (env_ids.shape[0], 1), self.device)
                else:
                    metallic = torch.full((env_ids.shape[0], 1), 0.5, device=self.device)
                randomize_surface(prims, color, roughness, metallic)
                description = get_surface_description(roughness, metallic)
            lang_label = get_lang_label(object_label, color_label, description)
            # print("language label: ", lang_label)
            if isinstance(lang_label, (list, tuple)):
                for i, env_id in enumerate(env_ids.tolist()):
                    self.language_instruction_buf[env_id] = lang_label[i]
            else:
                for env_id in env_ids.tolist():
                    self.language_instruction_buf[env_id] = str(lang_label)

        self._post_reset_process(env_ids)

    def _post_reset_process(self, env_ids):
        """
        Post reset process for the environment. Call to reset the information at step 1 to ensure the correctness of the simulation.
        """
        for obj_id in range(self.num_object):
            self.object_init_pos[obj_id, env_ids, :3] = self.scene[f"object_{obj_id}"].data.root_pos_w[env_ids, :3]
            self.object_init_orient[obj_id, env_ids, :4] = self.scene[f"object_{obj_id}"].data.root_quat_w[env_ids, :4]
        if not self.cfg.hydra_cfg.task.symmetry.symmetric_envs:
            info = self.command_manager.reset(env_ids)
            self.extras["log"].update(info)
    
    def _reset_symmetry(self, symmetric_env_ids):
        reset_symmetry = dict(self.cfg.hydra_cfg.task.symmetry.reset_symmetry)
        for key, value in reset_symmetry.items():
            if value["type"] == "articulation" or value["type"] == "rigid_object":
                asset = self.scene[key]
                if "position" in value["symmetry"]:
                    cur_position = asset.data.root_pos_w[symmetric_env_ids] - self.scene.env_origins[symmetric_env_ids]
                    symmetry_position = cur_position @ self.rep_Rd.T
                    symmetry_position = symmetry_position + self.scene.env_origins[symmetric_env_ids]
                else:
                    symmetry_position = asset.data.root_pos_w[symmetric_env_ids]
                if "quat" in value["symmetry"]:
                    cur_R = math_utils.matrix_from_quat(asset.data.root_quat_w[symmetric_env_ids])
                    cur_R_flat = cur_R.transpose(1, 2).reshape(-1, 9)
                    symmetry_R_flat = cur_R_flat @ self.rep_SO3_flat.T
                    symmetry_R = symmetry_R_flat.view(-1, 3, 3).transpose(1, 2)
                    symmetry_quat = math_utils.quat_from_matrix(symmetry_R)
                else:
                    symmetry_quat = asset.data.root_quat_w[symmetric_env_ids]
                asset.write_root_pose_to_sim(torch.cat([symmetry_position, symmetry_quat], dim=-1), env_ids=symmetric_env_ids)
            elif value["type"] == "command":
                command = self.command_manager.get_term(key)
                if "position" in value["symmetry"]:
                    cur_pos_command_e = command.pos_command_e[symmetric_env_ids]
                    command.pos_command_e[symmetric_env_ids] = cur_pos_command_e @ self.rep_Rd
                    command.pos_command_w[symmetric_env_ids] = command.pos_command_e[symmetric_env_ids] + self.scene.env_origins[symmetric_env_ids]
                if "quat" in value["symmetry"]:
                    cur_R = math_utils.matrix_from_quat(command.quat_command_w[symmetric_env_ids])
                    cur_R_flat = cur_R.transpose(1, 2).reshape(-1, 9)
                    symmetry_R_flat = cur_R_flat @ self.rep_SO3_flat.T
                    symmetry_R = symmetry_R_flat.view(-1, 3, 3).transpose(1, 2)
                    symmetry_quat = math_utils.quat_from_matrix(symmetry_R)
                    command.quat_command_w[symmetric_env_ids] = symmetry_quat
            else:
                raise ValueError(f"Unknown symmetry type: {value['type']}")

    def _process_camera_cfg(self, env_ids):
        for cam_name, cam_cfg in self.cfg.hydra_cfg.task.cam.cam_params.items():
            if not cam_cfg.enable or "wrist" in cam_name:
                continue
            # set camera pose
            eye = torch.tensor(cam_cfg.pos, device=self.device).unsqueeze(0).repeat(self.num_envs, 1)
            if cam_cfg.randomize:
                eye += math_utils.sample_uniform(torch.tensor([-0.1, -0.2, -0.2], device=self.device), torch.tensor([0.1, 0.1, 0.2], device=self.device), eye.shape, self.device)
            target = torch.tensor(cam_cfg.target, device=self.device).unsqueeze(0).repeat(self.num_envs, 1)
            if cam_name in self.camera_offset.keys():
                cur_pos = self.camera_offset[cam_name]['pos']
                cur_pos[env_ids] = eye[env_ids]
                eye = cur_pos.clone()
            self.scene[cam_name].set_world_poses_from_view(eyes=self.scene.env_origins + eye,
                                                           targets=self.scene.env_origins + target)
            # compute camera offset
            up_axis = stage_utils.get_stage_up_axis()
            rotation_matrix = create_rotation_matrix_from_view(eye, target, up_axis, device=self.device)
            rotation_matrix = rotation_matrix @ torch.tensor(CV_TO_GL, device=self.device, dtype=torch.float32).unsqueeze(0).repeat(self.num_envs, 1, 1)
            orientations = quat_from_matrix(rotation_matrix)
            if cam_name in self.camera_offset.keys():
                cur_orientation = self.camera_offset[cam_name]['orientation']
                cur_orientation[env_ids] = orientations[env_ids]
                orientations = cur_orientation.clone()
            self.camera_offset[cam_name] = {}
            self.camera_offset[cam_name]['pos'] = eye
            self.camera_offset[cam_name]['orientation'] = orientations