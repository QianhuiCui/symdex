from __future__ import annotations

from isaaclab.assets import RigidObjectCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveGaussianNoiseCfg as Gnoise
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.sensors.frame_transformer import OffsetCfg
import isaaclab.envs.mdp as mdp
from isaaclab.markers.config import FRAME_MARKER_CFG 

import symdex
from symdex.env.tasks.manager_based_env_cfg import *
from symdex.env.mdps.obs_mdps import *
from symdex.env.mdps.reset_mdps import *
from symdex.env.mdps.reward_mdps import *
from symdex.env.mdps.termination_mdps import *
from symdex.env.mdps.command_mdps.grasp_command_cfg import TargetPositionCommandCfg
from symdex.env.action_managers.actions_cfg import EMACumulativeRelativeJointPositionActionCfg
from symdex.utils.random_cfg import MultiUsdCfg, RandomPreviewSurfaceCfg, COLOR_DICT_20
import symdex.env.tasks.Pouring.mdps as pouring


FRAME_MARKER_SMALL_CFG = FRAME_MARKER_CFG.copy()
FRAME_MARKER_SMALL_CFG.markers["frame"].scale = (0.10, 0.10, 0.10)

@configclass
class PouringSceneCfg(BaseSceneCfg):
    # robots
    robot = ArticulationCfg(
        prim_path="/World/envs/env_.*/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{symdex.LIB_PATH}/assets/ufactory850/uf850_allegro_right.usd",  # assets/ufactory850/uf850_allegro_right_colored.usd",  # assets/ufactory850/uf850_allegro_right.usd",
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True,
                max_depenetration_velocity=1000.0,
                max_linear_velocity=1000,
                max_angular_velocity=1000,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False, solver_position_iteration_count=16, solver_velocity_iteration_count=1,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            joint_pos={
                "joint1": 0.5,
                "joint2": 0.0,
                "joint3": -0.5,
                "joint4": 0.0,
                "joint5": -0.3,
                "joint6": -1.57,
                # hand 
                "jif1": 0.0,
                "jif2": 0.4,
                "jif3": 0.4,
                "jif4": 0.0,
                "jmf1": 0.0,
                "jmf2": 0.4,
                "jmf3": 0.4,
                "jmf4": 0.0,
                "jpf1": 0.0,
                "jpf2": 0.4,
                "jpf3": 0.4,
                "jpf4": 0.0,
                "jth1": 1.3,
                "jth2": 0.0,
                "jth3": 0.2,
                "jth4": 0.0,
            },
            pos=(-0.274, -0.475, 0.01),
        ),
        actuators={
            "xArm_1-6": ImplicitActuatorCfg(
                joint_names_expr=["joint[1-6]"],
                stiffness=2000.0,
                damping=16.0,
            ),
            "allegro_hand_1": ImplicitActuatorCfg(
                joint_names_expr=["j.*f1"],
                stiffness=325.0,
                damping=20.0,
            ),
            "allegro_hand_2": ImplicitActuatorCfg(
                joint_names_expr=["j.*f2"],
                stiffness=425.0,
                damping=25.0,
            ),
            "allegro_hand_3": ImplicitActuatorCfg(
                joint_names_expr=["j.*f3"],
                stiffness=245.0,
                damping=15.0,
            ),
            "allegro_hand_4": ImplicitActuatorCfg(
                joint_names_expr=["j.*f4"],
                stiffness=1050.0,
                damping=65.0,
            ),
            "allegro_hand_thumb_1": ImplicitActuatorCfg(
                joint_names_expr=["jth1"],
                stiffness=100.0,
                damping=5.0,
            ),
            "allegro_hand_thumb_2": ImplicitActuatorCfg(
                joint_names_expr=["jth2"],
                stiffness=300.0,
                damping=15.0,
            ),
            "allegro_hand_thumb_3": ImplicitActuatorCfg(
                joint_names_expr=["jth3"],
                stiffness=1270.0,
                damping=100.0,
            ),
            "allegro_hand_thumb_4": ImplicitActuatorCfg(
                joint_names_expr=["jth4"],
                stiffness=1000.0,
                damping=50.0,
            ),
        },
    )

    robot_left = ArticulationCfg(
        prim_path="/World/envs/env_.*/Robot_left",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{symdex.LIB_PATH}/assets/ufactory850/uf850_allegro_left.usd",  # assets/ufactory850/uf850_allegro_left_colored.usd",  # assets/ufactory850/uf850_allegro_left.usd",
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True,
                max_depenetration_velocity=1000.0,
                max_linear_velocity=1000,
                max_angular_velocity=1000,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False, solver_position_iteration_count=16, solver_velocity_iteration_count=1,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            joint_pos={
                "joint1": -0.5,
                "joint2": 0.0,
                "joint3": -0.5,
                "joint4": 0.0,
                "joint5": -0.3,
                "joint6": 1.57,
                # hand 
                "jif1": 0.0,
                "jif2": 0.4,
                "jif3": 0.4,
                "jif4": 0.0,
                "jmf1": 0.0,
                "jmf2": 0.4,
                "jmf3": 0.4,
                "jmf4": 0.0,
                "jpf1": 0.0,
                "jpf2": 0.4,
                "jpf3": 0.4,
                "jpf4": 0.0,
                "jth1": 1.3,
                "jth2": 0.0,
                "jth3": 0.2,
                "jth4": 0.0,
            },
            pos=(-0.274, 0.475, 0.01),
        ),
        actuators={
            "xArm_1-6": ImplicitActuatorCfg(
                joint_names_expr=["joint[1-6]"],
                stiffness=2000.0,
                damping=16.0,
            ),
            "allegro_hand_1": ImplicitActuatorCfg(
                joint_names_expr=["j.*f1"],
                stiffness=325.0,
                damping=20.0,
            ),
            "allegro_hand_2": ImplicitActuatorCfg(
                joint_names_expr=["j.*f2"],
                stiffness=425.0,
                damping=25.0,
            ),
            "allegro_hand_3": ImplicitActuatorCfg(
                joint_names_expr=["j.*f3"],
                stiffness=245.0,
                damping=15.0,
            ),
            "allegro_hand_4": ImplicitActuatorCfg(
                joint_names_expr=["j.*f4"],
                stiffness=1050.0,
                damping=65.0,
            ),
            "allegro_hand_thumb_1": ImplicitActuatorCfg(
                joint_names_expr=["jth1"],
                stiffness=100.0,
                damping=5.0,
            ),
            "allegro_hand_thumb_2": ImplicitActuatorCfg(
                joint_names_expr=["jth2"],
                stiffness=300.0,
                damping=15.0,
            ),
            "allegro_hand_thumb_3": ImplicitActuatorCfg(
                joint_names_expr=["jth3"],
                stiffness=1270.0,
                damping=100.0,
            ),
            "allegro_hand_thumb_4": ImplicitActuatorCfg(
                joint_names_expr=["jth4"],
                stiffness=1000.0,
                damping=50.0,
            ),
        },
    )

    # objects
    object_0 = RigidObjectCfg(  # cup
        prim_path="/World/envs/env_.*/Object_0",
        spawn=MultiUsdCfg(
            usd_path="cup",
            random_choice=True,
            obj_label=True,
            preview_surface=RandomPreviewSurfaceCfg(
                diffuse_color_dict=COLOR_DICT_20,
                roughness_range=(0.2, 0.8),
                metallic_range=(0.2, 0.8),
            ),
            random_color=False,
            random_roughness=False,
            random_metallic=False,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=False,
                disable_gravity=False,
                max_linear_velocity=1000,
                max_angular_velocity=1000,
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=1,
                max_depenetration_velocity=1000.0,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.1),
            activate_contact_sensors=True,
            scale=(0.0874, 0.155, 0.0880),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            lin_vel=(0.0, 0.0, 0.0),
            ang_vel=(0.0, 0.0, 0.0),
            pos=(0.0, 0.0, 0.0),
        ),
    )

    object_1 = RigidObjectCfg(  # bowl
        prim_path="/World/envs/env_.*/Object_1",
        spawn=MultiUsdCfg(
            usd_path="bowl",
            random_choice=True,
            obj_label=True,
            preview_surface=RandomPreviewSurfaceCfg(
                diffuse_color_dict=COLOR_DICT_20,
                roughness_range=(0.2, 0.8),
                metallic_range=(0.2, 0.8),
            ),
            random_color=False,
            random_roughness=False,
            random_metallic=False,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=False,
                disable_gravity=False,
                max_linear_velocity=1000,
                max_angular_velocity=1000,
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=1,
                max_depenetration_velocity=1000.0,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.25),
            activate_contact_sensors=True,
            scale=(0.1, 0.1, 0.1),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            lin_vel=(0.0, 0.0, 0.0),
            ang_vel=(0.0, 0.0, 0.0),
            pos=(0.0, 0.0, 0.0),
        ),
    )

    # cameras
    cam_1 = CameraCfg(
        prim_path="/World/envs/env_.*/Cameras_1",
        width=128, height=128,
        data_types=["rgb", "depth"],
        spawn=sim_utils.PinholeCameraCfg(
                focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 1.0e5)
            ),  # default parameters
        offset=CameraCfg.OffsetCfg(convention="opengl"),
    )

    contact_sensors_0 = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/if5",  # index
        update_period=0.0, 
        debug_vis=True,
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Object_0"],
    )
    contact_sensors_1 = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/mf5",  # middle
        update_period=0.0, 
        debug_vis=True,
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Object_0"],
    )
    contact_sensors_2 = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/pf5",  # pinky
        update_period=0.0, 
        debug_vis=True,
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Object_0"],
    )
    contact_sensors_3 = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/th5",  # thumb
        update_period=0.0, 
        debug_vis=True,
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Object_0"],
    )
    contact_sensors_robot = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/link1|link2|link3|link4|link5",  # thumb
        update_period=0.0, 
        debug_vis=True,
        # filter_prim_paths_expr=["{ENV_REGEX_NS}/Table"],
    )
    contact_sensors_0_left = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot_left/if5",  # index
        update_period=0.0, 
        debug_vis=True,
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Object_1"],
    )
    contact_sensors_1_left = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot_left/mf5",  # middle
        update_period=0.0, 
        debug_vis=True,
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Object_1"],
    )
    contact_sensors_2_left = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot_left/pf5",  # pinky
        update_period=0.0, 
        debug_vis=True,
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Object_1"],
    )
    contact_sensors_3_left = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot_left/th5",  # thumb
        update_period=0.0, 
        debug_vis=True,
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Object_1"],
    )
    contact_sensors_robot_left = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot_left/link1|link2|link3|link4|link5",  # thumb
        update_period=0.0, 
        debug_vis=True,
    )
    palm_pose_1 = FrameTransformerCfg(
        prim_path="{ENV_REGEX_NS}/Object_0",
        debug_vis=True,
        visualizer_cfg=FRAME_MARKER_SMALL_CFG.replace(prim_path="/Visuals/Palm/PalmPose1FrameTransformer"),
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/Object_0",
                name="palm_frame",
                offset=OffsetCfg(
                    pos=(0.0, 0.0, 0.04),
                    rot=(0.5, 0.5, 0.5, 0.5),
                )
            )
        ]
    )
    palm_pose_2 = FrameTransformerCfg(
        prim_path="{ENV_REGEX_NS}/Object_1",
        debug_vis=True,
        visualizer_cfg=FRAME_MARKER_SMALL_CFG.replace(prim_path="/Visuals/Palm/PalmPose2FrameTransformer"),
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/Object_1",
                name="palm_frame",
                offset=OffsetCfg(
                    pos=(0.0, 0.06, -0.12),
                    rot=(0.0, -0.707, 0.0, -0.707),
                )
            )
        ]
    )


@configclass
class PouringEventCfg(BaseEventCfg):
    """Configuration for events."""

    reset_robot_joints_left = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (1.0, 1.0),
            "velocity_range": (0.0, 0.0),
            "asset_cfg": SceneEntityCfg("robot_left")
        },
    )

    reset_object_cup = EventTerm(
        func=reset_object,
        mode="reset",
        params={
            "pose_range": {"x":[0.15, 0.25], "y":[-0.15, -0.25], "z":[0.05, 0.05], "roll": [1.57, 1.57]},
            "velocity_range": {},
            "object_id": 0,
        }
    )
    
    reset_object_bowl = EventTerm(
        func=reset_object,
        mode="reset",
        params={
            "pose_range": {"x": [0.05, 0.2], "y": [0.1, 0.3], "z": [0.0, 0.0], "roll": [1.57, 1.57]},
            "velocity_range": {},
            "object_id": 1,
        },
    )


@configclass
class PouringCommandsCfg(BaseCommandsCfg):
    """Command specifications for the MDP."""

    # waiting_pos_cup = TargetPositionCommandCfg(
    #     object_id=0,
    #     success_threshold=0.05,
    #     success_threshold_orient=0.5, # 60 degree 
    #     pose_range={"x": [0.2, 0.2], "y": [0.0, 0.0], "z": [0.28, 0.28], "roll": [1.57, 1.57]},
    #     # pose_range={"x": [0.15, 0.15], "y": [-0.1, -0.1], "z": [0.28, 0.28], "roll": [1.57, 1.57]},
    #     debug_vis=True,
    # )

    target_pos_cup = TargetPositionCommandCfg(
        object_id=0,
        success_threshold=0.05,
        success_threshold_orient=0.5,  # 60 degree 
        pose_range={"x": [0.2, 0.2], "y": [-0.1, -0.1], "z": [0.18, 0.18]},
        debug_vis=True,
    )

    target_pos_bowl = TargetPositionCommandCfg(
        object_id=1,
        success_threshold=0.05,
        axis="y",
        success_threshold_orient=0.97,  # 111 degree
        pose_range={"x": [0.1, 0.1], "y": [0.05, 0.05], "z": [0.0, 0.0], "roll": [1.57, 1.57]},
        debug_vis=True,
    )


@configclass
class PouringObservationsCfg(BaseObservationsCfg):
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""
        # -- robot terms (order preserved)
        ee_pose_right = ObsTerm(func=ee_pose, params={"ee_name": "palm_link"})
        hand_joint_pos_right = ObsTerm(func=joint_pos_limit_normalized, 
                                       params={"joints": ["j.*f1", "j.*f2", "j.*f3", "j.*f4", "jth1", "jth2", "jth3", "jth4"], 
                                               "joint_lower_limit": JOINT_LOWER_LIMIT[6:], 
                                               "joint_upper_limit": JOINT_UPPER_LIMIT[6:]}, 
                                       noise=Gnoise(std=0.005))
        # joint_vel_right = ObsTerm(func=joint_vel, params={"joints": None}, noise=Gnoise(std=0.01))
        # cmd_right = ObsTerm(func=pouring.generated_commands_right, noise=Gnoise(std=0.01))
        ee_pose_left = ObsTerm(func=ee_pose, params={"ee_name": "palm_link", "asset_cfg": SceneEntityCfg("robot_left")})
        hand_joint_pos_left = ObsTerm(func=joint_pos_limit_normalized, 
                                      params={"joints": ["j.*f1", "j.*f2", "j.*f3", "j.*f4", "jth1", "jth2", "jth3", "jth4"], 
                                              "joint_lower_limit": JOINT_LOWER_LIMIT_LEFT[6:], 
                                              "joint_upper_limit": JOINT_UPPER_LIMIT_LEFT[6:],
                                              "asset_cfg": SceneEntityCfg("robot_left")}, 
                                      noise=Gnoise(std=0.005))
        # joint_vel_left = ObsTerm(func=joint_vel, params={"joints": None, "asset_cfg": SceneEntityCfg("robot_left")}, noise=Gnoise(std=0.01))
        # -- object terms
        cup_pos = ObsTerm(func=object_pos, params={"object_id": 0}, noise=Gnoise(std=0.01))
        cup_quat = ObsTerm(func=object_quat, params={"object_id": 0}, noise=Gnoise(std=0.01))
        bowl_pos = ObsTerm(func=object_pos, params={"object_id": 1}, noise=Gnoise(std=0.01))
        # bowl_quat = ObsTerm(func=object_quat, params={"object_id": 1}, noise=Gnoise(std=0.01))
        last_action = ObsTerm(func=last_action)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class VisionCfg(ObsGroup):
        """Observations for vision group."""

        # -- robot terms (order preserved)
        rgb_image = ObsTerm(func=rgb_image, params={"camera_name": ["cam_1"]})
        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()
    vision: VisionCfg = VisionCfg()


@configclass
class PouringActionsCfg:
    arm_hand_action = EMACumulativeRelativeJointPositionActionCfg(
        asset_name="robot",
        joint_names=[".*"],
        scale=1.0,
        use_default_offset=False,
        joint_lower_limit=JOINT_LOWER_LIMIT,
        joint_upper_limit=JOINT_UPPER_LIMIT,
        alpha=0.2
    )

    arm_hand_action_left = EMACumulativeRelativeJointPositionActionCfg(
        asset_name="robot_left",
        joint_names=[".*"],
        scale=1.0,
        use_default_offset=False,
        joint_lower_limit=JOINT_LOWER_LIMIT_LEFT,
        joint_upper_limit=JOINT_UPPER_LIMIT_LEFT,
        alpha=0.2
    )


@configclass
class PouringTerminationsCfg(BaseTerminationsCfg):
    max_consecutive_success = DoneTerm(
        func=pouring.max_consecutive_success, params={"num_success": 1}
    )


@configclass
class PouringRewardsCfg(BaseRewardsCfg):
    """Reward terms for the MDP."""
    # -- right hand (cup, object_0)
    align_pos_right = RewTerm(func=align_palm_to_pos,
                              params={"link_name": ["palm_link"],
                                      "frame_name": "palm_pose_1",
                                      "asset_cfg": SceneEntityCfg("robot")},
                              weight=0.0)
    align_ori_right = RewTerm(func=align_palm_to_quat,
                              params={"link_name": ["palm_link"],
                                      "frame_name": "palm_pose_1",
                                      "asset_cfg": SceneEntityCfg("robot")},
                              weight=0.0)
    reach_right = RewTerm(func=object_robot_distance,
                          params={"object_id": 0,
                                  "weight": [1.0, 1.0, 1.0, 1.5],
                                  "link_name": ["if5", "mf5", "pf5", "th5"],
                                  "asset_cfg": SceneEntityCfg("robot")},
                          weight=0.0)
    lift_right = RewTerm(func=lift_distance,
                         params={"command_name": "target_pos_cup",
                                 "object_id": 0,
                                 "sensor_names": ["contact_sensors_0", "contact_sensors_1", "contact_sensors_2", "contact_sensors_3"]},
                         weight=0.0)
    move_right = RewTerm(func=object_goal_distance,
                         params={"command_name": "target_pos_cup",
                                 "object_id": 0,
                                 "sensor_names": ["contact_sensors_0", "contact_sensors_1", "contact_sensors_2", "contact_sensors_3"]},
                         weight=0.0)
    move_ori_right = RewTerm(func=object_goal_distance_orient,
                             params={"command_name": "target_pos_cup",
                                     "object_id": 0,
                                     "axis": None,
                                     "sensor_names": ["contact_sensors_0", "contact_sensors_1", "contact_sensors_2", "contact_sensors_3"]},
                             weight=0.0)
    cmd_wait_right = RewTerm(func=pouring.cmd_wait,
                             params={},
                             weight=0.0)
    cmd_success_right = RewTerm(func=pouring.object_success,
                                params={"object_id": 0},
                                weight=0.0)
    # -- left hand (bowl, object_1)
    align_pos_left = RewTerm(func=align_palm_to_pos,
                             params={"link_name": ["palm_link"],
                                     "frame_name": "palm_pose_2",
                                     "asset_cfg": SceneEntityCfg("robot_left")},
                             weight=0.0)
    align_ori_left = RewTerm(func=align_palm_to_quat,
                             params={"link_name": ["palm_link"],
                                     "frame_name": "palm_pose_2",
                                     "asset_cfg": SceneEntityCfg("robot_left")},
                             weight=0.0)
    move_left = RewTerm(func=object_goal_distance,
                        params={"command_name": "target_pos_bowl",
                                "object_id": 1,
                                "sensor_names": ["contact_sensors_0_left", "contact_sensors_1_left", "contact_sensors_2_left", "contact_sensors_3_left"]},
                        weight=0.0)
    move_ori_left = RewTerm(func=object_goal_distance_orient,
                            params={"command_name": "target_pos_bowl",
                                    "object_id": 1,
                                    "axis": "y",
                                    "sensor_names": ["contact_sensors_0_left", "contact_sensors_1_left", "contact_sensors_2_left", "contact_sensors_3_left"]},
                            weight=0.0)
    cmd_success_left = RewTerm(func=pouring.object_success,
                               params={"object_id": 1},
                               weight=0.0)
    success_bonus = RewTerm(func=success_bonus,
                            params={"command_names": ["target_pos_cup", "target_pos_bowl"]},
                            weight=0.0)  

   
@configclass
class PouringEnvCfg(BaseEnvCfg):
    name: str = "Pouring"
    scene = PouringSceneCfg(num_envs=4096, env_spacing=3.0)
    events = PouringEventCfg()
    commands = PouringCommandsCfg()
    observations = PouringObservationsCfg()
    actions = PouringActionsCfg()
    terminations = PouringTerminationsCfg()
    rewards = PouringRewardsCfg()
    num_object = 2
    action_dim = 44 # arm + hand
    action_scale: list = [0.05, 0.05, 0.05, 0.05, 0.05, 0.05,
                            0.03, 0.03, 0.03, 0.03, 
                            0.03, 0.03, 0.03, 0.03, 
                            0.03, 0.03, 0.03, 0.015,
                            0.03, 0.03, 0.03, 0.03,
                          0.05, 0.05, 0.05, 0.05, 0.05, 0.05,
                            0.03, 0.03, 0.03, 0.03, 
                            0.03, 0.03, 0.03, 0.03, 
                            0.03, 0.03, 0.03, 0.015,
                            0.03, 0.03, 0.03, 0.03]  # jth3 needs smaller rate
    visualize_marker: bool = False
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        self.viewer.eye = (-0.8, 0.0, 0.7)