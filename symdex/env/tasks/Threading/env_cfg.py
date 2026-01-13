from __future__ import annotations

from isaaclab.assets import RigidObjectCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveGaussianNoiseCfg as Gnoise
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.sensors.frame_transformer import OffsetCfg
from isaaclab.envs.mdp.actions import JointPositionActionCfg
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
from symdex.env.tasks.StirBowl import mdps as bowl
from symdex.env.tasks.Threading import mdps as threading

FRAME_MARKER_SMALL_CFG = FRAME_MARKER_CFG.copy()
FRAME_MARKER_SMALL_CFG.markers["frame"].scale = (0.10, 0.10, 0.10)

@configclass
class ThreadingSceneCfg(BaseSceneCfg):
    # robots
    robot = ArticulationCfg(
        prim_path="/World/envs/env_.*/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{symdex.LIB_PATH}/assets/ufactory850/uf850_allegro_right.usd",
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
                "joint1": 0.6,
                "joint2": 0.3,
                "joint3": -0.6,
                "joint4": 0.0,
                "joint5": -0.8,
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
            usd_path=f"{symdex.LIB_PATH}/assets/ufactory850/uf850_allegro_left.usd",
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
                "joint1": 0.25,
                "joint2": 0.0,
                "joint3": -0.5,
                "joint4": -1.4,
                "joint5": -1.0,
                "joint6": 3.14,
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

    object_0 = RigidObjectCfg(
        prim_path=f"/World/envs/env_.*/Object_0",
        spawn=MultiUsdCfg(
        # sim_utils.UsdFileCfg(
            # usd_path=f"{symdex.LIB_PATH}/assets/object/cube_with_hole.usd",
            usd_path="cube_with_hole",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=False,
                disable_gravity=False,
                max_linear_velocity=1000,
                max_angular_velocity=1000,
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=1,
                max_depenetration_velocity=1000.0,
            ),
            preview_surface=RandomPreviewSurfaceCfg(
                diffuse_color_dict=COLOR_DICT_20,
                roughness_range=(0.2, 0.8),
                metallic_range=(0.2, 0.8),
            ),
            random_color=True,
            random_roughness=True,
            random_metallic=True,
            mass_props=sim_utils.MassPropertiesCfg(mass=0.2),  # 0.15
            activate_contact_sensors=True,
            scale=(0.75, 0.75, 0.75),
        ), # wait for initialization
        init_state=RigidObjectCfg.InitialStateCfg(
            lin_vel=(0.0, 0.0, 0.0),
            ang_vel=(0.0, 0.0, 0.0),
            pos=(0.0, 0.0, 0.0),
            # rot=(0.500, -0.500, 0.500, -0.500)
        ),
    )

    object_1 = RigidObjectCfg(
        prim_path=f"/World/envs/env_.*/Object_1",
        spawn=MultiUsdCfg(
        # sim_utils.UsdFileCfg(
            # usd_path=f"{symdex.LIB_PATH}/assets/object/drill.usd",
            usd_path="drill",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=False,
                disable_gravity=False,
                max_linear_velocity=1000,
                max_angular_velocity=1000,
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=1,
                max_depenetration_velocity=1000.0,
            ),
            preview_surface=RandomPreviewSurfaceCfg(
                diffuse_color_dict=COLOR_DICT_20,
                roughness_range=(0.2, 0.8),
                metallic_range=(0.2, 0.8),
            ),
            random_color=True,
            random_roughness=True,
            random_metallic=True,
            mass_props=sim_utils.MassPropertiesCfg(mass=0.5),  # 0.85
            activate_contact_sensors=True,
            scale=(1.0, 1.0, 1.0),
        ), # wait for initialization
        init_state=RigidObjectCfg.InitialStateCfg(
            lin_vel=(0.0, 0.0, 0.0),
            ang_vel=(0.0, 0.0, 0.0),
            pos=(0.0, 0.0, 0.0),
            # rot=(0.500, 0.500, -0.500, -0.500)
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

    # sensors
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
    contact_sensors_0_left = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot_left/if5",  # index
        update_period=0.0, 
        debug_vis=True,
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Object_0"],
    )
    contact_sensors_1_left = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot_left/mf5",  # middle
        update_period=0.0, 
        debug_vis=True,
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Object_0"],
    )
    contact_sensors_2_left = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot_left/pf5",  # pinky
        update_period=0.0, 
        debug_vis=True,
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Object_0"],
    )
    contact_sensors_3_left = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot_left/th5",  # thumb
        update_period=0.0, 
        debug_vis=True,
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Object_0"],
    )

    drill_head_frame = FrameTransformerCfg(
        prim_path="{ENV_REGEX_NS}/Object_1",
        debug_vis=False,
        visualizer_cfg=FRAME_MARKER_SMALL_CFG.replace(prim_path="/Visuals/ObjectApproachFrameTransformer"),
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/Object_1",
                name="approach_frame",
                offset=OffsetCfg(
                    pos=(-0.15, 0.0, 0.07),
                    rot=(1.0, 0.0, 0.0, 0.0),
                ),
            ),
        ],
    )

    object_approach_frame = FrameTransformerCfg(
        prim_path="{ENV_REGEX_NS}/Object_1",
        debug_vis=False,
        visualizer_cfg=FRAME_MARKER_SMALL_CFG.replace(prim_path="/Visuals/ObjectApproachFrameTransformer"),
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/Object_1",
                name="approach_frame",
                offset=OffsetCfg(
                    pos=(0.05, 0.0, 0.0),
                    rot=(0.0, 0.0, 0.7071, -0.7071),
                ),
            ),
        ],
    )

    # object_approach_frame_symmetry = FrameTransformerCfg(
    #     prim_path="{ENV_REGEX_NS}/Object_1",
    #     debug_vis=False,
    #     visualizer_cfg=FRAME_MARKER_SMALL_CFG.replace(prim_path="/Visuals/ObjectApproachFrameTransformer"),
    #     target_frames=[
    #         FrameTransformerCfg.FrameCfg(
    #             prim_path="{ENV_REGEX_NS}/Object_1",
    #             name="approach_frame",
    #             offset=OffsetCfg(
    #                 pos=(0.05, 0.0, 0.0),
    #                 rot=(0.0, 0.0, 0.7071, 0.7071),
    #             ),
    #         ),
    #     ],
    # )

@configclass
class ThreadingEventCfg(BaseEventCfg):
    """Configuration for events."""
    reset_robot_joints_left = EventTerm(
        func=reset_joints_by_symmetry,
        mode="reset",
        params={
            "position_range": (1.0, 1.0),
            "velocity_range": (0.0, 0.0),
            "asset_cfg": SceneEntityCfg("robot_left")
        },
    )
    
    reset_object_cube = EventTerm(
        func=reset_object,
        mode="reset",
        params={
            "pose_range": {"x": [0.1, 0.1], "y": [-0.3, -0.3], "z": [0.05, 0.05], "roll": [-1.57, -1.57]}, # 0.785 = 45 degree, 2.356 = 135 degree
            "velocity_range": {},
            "object_id": 0,
        },
    )

    reset_object_drill = EventTerm(
        func=reset_object,
        mode="reset",
        params={
            "pose_range": {"x": [0.15, 0.15], "y": [0.2, 0.2], "z": [0.06, 0.06], "yaw": [1.57, 1.57]}, # 0.785 = 45 degree, 2.356 = 135 degree
            "velocity_range": {},
            "object_id": 1,
        },
    )

@configclass
class ThreadingCommandsCfg(BaseCommandsCfg):
    """Command specifications for the MDP."""

    cube_target_pos = TargetPositionCommandCfg(
        object_id=0,
        success_threshold=0.1,
        success_threshold_orient=0.97, # 30 degree 
        pose_range={"x": [0.0, 0.0], "y": [-0.15, -0.15], "z": [0.3, 0.3], "roll": [-1.57, -1.57]},
        update_goal_on_success=False,
        debug_vis=True,
    )

    drill_target_pos = TargetPositionCommandCfg(
        object_id=1,
        success_threshold=0.1,
        success_threshold_orient=0.97, # 60 degree 
        pose_range={"x": [0.0, 0.0], "y": [0.05, 0.05], "z": [0.23, 0.23], "yaw": [1.57, 1.57]},
        update_goal_on_success=False,
        debug_vis=True,
    )

@configclass
class ThreadingObservationsCfg(BaseObservationsCfg):
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # -- robot terms (order preserved)
        ee_pose_right = ObsTerm(func=ee_pose, params={"ee_name": "palm_link"})
        joint_pos_right = ObsTerm(func=joint_pos_limit_normalized, params={"joints": None,
                                                                           "joint_lower_limit": JOINT_LOWER_LIMIT, 
                                                                           "joint_upper_limit": JOINT_UPPER_LIMIT}, noise=Gnoise(std=0.005))
        joint_vel_right = ObsTerm(func=joint_vel, params={"joints": None},)
        ee_pose_left = ObsTerm(func=ee_pose, params={"ee_name": "palm_link", "asset_cfg": SceneEntityCfg("robot_left")})
        joint_pos_left = ObsTerm(func=joint_pos_limit_normalized, params={"joints": None, 
                                                                          "joint_lower_limit": JOINT_LOWER_LIMIT_LEFT, 
                                                                          "joint_upper_limit": JOINT_UPPER_LIMIT_LEFT, 
                                                                          "asset_cfg": SceneEntityCfg("robot_left")}, noise=Gnoise(std=0.005))
        joint_vel_left = ObsTerm(func=joint_vel, params={"joints": None, "asset_cfg": SceneEntityCfg("robot_left")},)
        # -- object terms
        cube_pos = ObsTerm(
            func=object_pos, params={"object_id": 0}, noise=Unoise(n_min=0.0, n_max=0.015)
        )
        cube_quat = ObsTerm(
            func=object_quat, params={"object_id": 0, "symmetry": True}, noise=Gnoise(std=0.005)
        )
        drill_pos = ObsTerm( # bowl position
            func=object_pos, params={"object_id": 1}, noise=Unoise(n_min=0.0, n_max=0.015)
        )
        drill_quat = ObsTerm(
            func=object_quat, params={"object_id": 1, "symmetry": True}, noise=Gnoise(std=0.005)
        )
        # # -- action terms
        last_action = ObsTerm(func=last_action)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class ThreadingActionsCfg:
    # arm_hand_action = EMACumulativeRelativeJointPositionActionCfg(
    #     asset_name="robot",
    #     joint_names=[".*"],
    #     scale=1.0,
    #     use_default_offset=False,
    #     joint_lower_limit=JOINT_LOWER_LIMIT,
    #     joint_upper_limit=JOINT_UPPER_LIMIT,
    #     alpha=0.2
    # )

    # arm_hand_action_left = EMACumulativeRelativeJointPositionActionCfg(
    #     asset_name="robot_left",
    #     joint_names=[".*"],
    #     scale=1.0,
    #     use_default_offset=False,
    #     joint_lower_limit=JOINT_LOWER_LIMIT_LEFT,
    #     joint_upper_limit=JOINT_UPPER_LIMIT_LEFT,
    #     alpha=0.2
    # )
    arm_hand_action = JointPositionActionCfg(
        asset_name="robot",
        joint_names=[".*"],
        scale=1.0,
        use_default_offset=False,
    )
    arm_hand_action_left = JointPositionActionCfg(
        asset_name="robot_left",
        joint_names=[".*"],
        scale=0.95,
        use_default_offset=False,
    )

@configclass
class ThreadingTerminationsCfg(BaseTerminationsCfg):
    out_of_space = DoneTerm(
        func=threading.obj_out_space, params={"asset_cfg": SceneEntityCfg("robot"), "object_id": 0,}
    )
    out_of_space_left = DoneTerm(
        func=threading.obj_out_space, params={"asset_cfg": SceneEntityCfg("robot_left"), "object_id": 1,}
    )
    max_consecutive_success = DoneTerm(
        func=threading.max_consecutive_success, params={"num_success": 1}
    )

@configclass
class ThreadingRewardsCfg(BaseRewardsCfg):
    """Reward terms for the MDP."""
    reaching_object = RewTerm(func=object_robot_distance, 
                              params={"weight": [1.0, 1.0, 1.0, 1.5], 
                                      "link_name": ["if5", "mf5", "pf5", "th5"], 
                                      "object_id": 0}, 
                                      weight=0.0)
    object_lifting = RewTerm(func=lift_distance,
                             params={"command_name": "cube_target_pos", "object_id": 0, "sensor_names": ["contact_sensors_0", "contact_sensors_1", "contact_sensors_2", "contact_sensors_3"]},
                             weight=0.0,
                             )
    cube_goal_tracking = RewTerm(func=threading.object_goal_distance,
                                   params={"command_name": "cube_target_pos", "object_id": 0, "sensor_names": ["contact_sensors_0", "contact_sensors_1", "contact_sensors_2", "contact_sensors_3"],},
                                   weight=0.0,
                                   )
    cube_goal_orient_tracking = RewTerm(func=object_goal_distance_orient,
                                   params={"command_name": "cube_target_pos", 
                                           "object_id": 0, 
                                           "axis": "z", 
                                           "sensor_names": ["contact_sensors_0", "contact_sensors_1", "contact_sensors_2", "contact_sensors_3"],
                                           "pos_success_threshold": 0.1,
                                           },
                                   weight=0.0,
                                   )
    cube_success_bonus = RewTerm(func=bowl.cmd_success_bonus,
                            params={"command_names": "cube_target_pos", "num_success": 1},
                            weight=0.0,
                            )
    align_hand_to_pos = RewTerm(func=align_palm_to_pos,
                                   params={"link_name": ["palm_link"], "frame_name": "object_approach_frame", "asset_cfg": SceneEntityCfg("robot_left")},
                                   weight=0.0,
                                   )
    align_hand_to_quat = RewTerm(func=align_palm_to_quat,
                                   params={"link_name": ["palm_link"], "frame_name": "object_approach_frame", "asset_cfg": SceneEntityCfg("robot_left")},
                                   weight=0.0,
                                   )
    reaching_drill = RewTerm(func=object_robot_distance, 
                              params={"weight": [1.0, 1.0, 1.0, 1.5], 
                                      "link_name": ["if5", "mf5", "pf5", "th5"], 
                                      "object_id": 1,
                                      "asset_cfg": SceneEntityCfg("robot_left")
                                      }, 
                                      weight=0.0)
    drill_goal_tracking = RewTerm(func=threading.object_goal_distance,
                                   params={"command_name": "drill_target_pos", "object_id": 1},
                                   weight=0.0,
                                   )
    drill_goal_orient_tracking = RewTerm(func=threading.drill_goal_orient_distance,
                                   params={"command_name": "drill_target_pos", "object_id": 1},
                                   weight=0.0,
                                   )
    drill_success_bonus = RewTerm(func=bowl.cmd_success_bonus,
                            params={"command_names": "drill_target_pos", "num_success": 1},
                            weight=0.0,
                            )
    drill_cube_distance = RewTerm(func=threading.drill_cube_distance,
                                   params={"frame_name": "drill_head_frame", "cube_id": 0, "drill_id": 1},
                                   weight=0.0,
                                   )
    success_bonus = RewTerm(func=threading.success_bonus,
                            params={"num_success": 20, "object_id": 0, "frame_name": "drill_head_frame"},
                            weight=0.0,
                            )
    
    # symmetry
    reaching_object_symmetry = RewTerm(func=object_robot_distance, 
                              params={"weight": [1.0, 1.0, 1.0, 1.5], 
                                      "link_name": ["if5", "mf5", "pf5", "th5"], 
                                      "object_id": 0,
                                      "asset_cfg": SceneEntityCfg("robot_left")
                                      }, 
                                      weight=0.0)
    object_lifting_symmetry = RewTerm(func=lift_distance,
                             params={"command_name": "cube_target_pos", "object_id": 0, "sensor_names": ["contact_sensors_0_left", "contact_sensors_1_left", "contact_sensors_2_left", "contact_sensors_3_left"]},
                             weight=0.0,
                             )
    cube_goal_tracking_symmetry = RewTerm(func=threading.object_goal_distance,
                                   params={"command_name": "cube_target_pos", "object_id": 0, "sensor_names": ["contact_sensors_0_left", "contact_sensors_1_left", "contact_sensors_2_left", "contact_sensors_3_left"],},
                                   weight=0.0,
                                   )
    cube_goal_orient_tracking_symmetry = RewTerm(func=object_goal_distance_orient,
                                   params={"command_name": "cube_target_pos", 
                                           "object_id": 0, 
                                           "axis": "z", 
                                           "sensor_names": ["contact_sensors_0_left", "contact_sensors_1_left", "contact_sensors_2_left", "contact_sensors_3_left"],
                                           "pos_success_threshold": 0.1,
                                           },
                                   weight=0.0,
                                   )
    align_hand_to_pos_symmetry = RewTerm(func=align_palm_to_pos,
                                   params={"link_name": ["palm_link"], "frame_name": "object_approach_frame_symmetry", "asset_cfg": SceneEntityCfg("robot")},
                                   weight=0.0,
                                   )
    align_hand_to_quat_symmetry = RewTerm(func=align_palm_to_quat,
                                   params={"link_name": ["palm_link"], "frame_name": "object_approach_frame_symmetry", "asset_cfg": SceneEntityCfg("robot")},
                                   weight=0.0,
                                   )
    reaching_drill_symmetry = RewTerm(func=object_robot_distance, 
                              params={"weight": [1.0, 1.0, 1.0, 1.5], 
                                      "link_name": ["if5", "mf5", "pf5", "th5"], 
                                      "object_id": 1,
                                      "asset_cfg": SceneEntityCfg("robot")
                                      }, 
                                      weight=0.0)
    energy = RewTerm(func=energy_punishment,
                                  weight=0.0,
                                  params={"asset_cfg": SceneEntityCfg("robot"), "actuator_name": ["allegro_hand_1", "allegro_hand_2", "allegro_hand_3", "allegro_hand_4", 
                                                                                                  "allegro_hand_thumb_1", "allegro_hand_thumb_2", "allegro_hand_thumb_3", "allegro_hand_thumb_4"]},
                                  )
    energy_left = RewTerm(func=energy_punishment,
                                  weight=0.0,
                                  params={"asset_cfg": SceneEntityCfg("robot_left"), "actuator_name": ["allegro_hand_1", "allegro_hand_2", "allegro_hand_3", "allegro_hand_4", 
                                                                                                  "allegro_hand_thumb_1", "allegro_hand_thumb_2", "allegro_hand_thumb_3", "allegro_hand_thumb_4"]},
                                  )
    collision_to_table = RewTerm(func=collision_penalty,
                                params={"sensor_names": ["contact_sensors_0", "contact_sensors_1", "contact_sensors_2", "contact_sensors_3"]},
                                weight=0.0,
                                )
    collision_to_table_symmetry = RewTerm(func=collision_penalty,
                                params={"sensor_names": ["contact_sensors_0_left", "contact_sensors_1_left", "contact_sensors_2_left", "contact_sensors_3_left"]},
                                weight=0.0,
                                )


@configclass
class ThreadingEnvCfg(BaseEnvCfg):
    name: str = "Threading"
    scene = ThreadingSceneCfg(num_envs=4096, env_spacing=3.0)
    events = ThreadingEventCfg()
    commands = ThreadingCommandsCfg()
    observations = ThreadingObservationsCfg()
    actions = ThreadingActionsCfg()
    terminations = ThreadingTerminationsCfg()
    rewards = ThreadingRewardsCfg()
    num_object = 2
    action_dim = 44 # arm + hand
    action_scale: list = [1.0] * action_dim
    # action_scale: list = [0.05, 0.05, 0.05, 0.05, 0.05, 0.05,
    #                         0.03, 0.03, 0.03, 0.03, 
    #                         0.03, 0.03, 0.03, 0.03, 
    #                         0.03, 0.03, 0.03, 0.015,
    #                         0.03, 0.03, 0.03, 0.03,
    #                         0.05, 0.05, 0.05, 0.05, 0.05, 0.05,
    #                         0.03, 0.03, 0.03, 0.03, 
    #                         0.03, 0.03, 0.03, 0.03, 
    #                         0.03, 0.03, 0.03, 0.015,
    #                         0.03, 0.03, 0.03, 0.03]  # jth3 needs smaller rate

    visualize_marker: bool = False

    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        self.viewer.eye = (-1.5, 0.0, 1.5)