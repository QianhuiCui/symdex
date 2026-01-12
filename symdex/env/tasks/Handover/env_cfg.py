from __future__ import annotations

from isaaclab.assets import RigidObjectCfg
from isaaclab.utils import configclass
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.sensors.frame_transformer import OffsetCfg
from isaaclab.markers.config import FRAME_MARKER_CFG 
from isaaclab.envs.mdp.actions import JointPositionActionCfg

import symdex
from symdex.env.tasks.manager_based_env_cfg import *
from symdex.env.mdps.obs_mdps import *
from symdex.env.mdps.reset_mdps import *
from symdex.env.mdps.reward_mdps import *
from symdex.env.mdps.termination_mdps import *
from symdex.env.mdps.command_mdps.grasp_command_cfg import TargetPositionCommandCfg
from symdex.env.mdps.command_mdps.reach_command_cfg import TargetPositionCommandCfg as ReachCommandCfg
from symdex.env.action_managers.actions_cfg import EMACumulativeRelativeJointPositionActionCfg
from symdex.utils.random_cfg import MultiUsdCfg, RandomPreviewSurfaceCfg, COLOR_DICT_20
import symdex.env.tasks.Handover.mdps as handover

FRAME_MARKER_SMALL_CFG = FRAME_MARKER_CFG.copy()
FRAME_MARKER_SMALL_CFG.markers["frame"].scale = (0.10, 0.10, 0.10)

@configclass
class HandoverSceneCfg(BaseSceneCfg):
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
                "joint1": 0.05,
                "joint2": 0.0,
                "joint3": -0.5,
                "joint4": 1.4,
                "joint5": -1.0,
                "joint6": -3.14,
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
                "jpf4": 1.2,
                "jth1": 0.364,
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
                "joint1": -0.8,
                "joint2": 0.3,
                "joint3": -0.6,
                "joint4": 0.0,
                "joint5": -0.8,
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
                "jth3": -0.1,
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
            # usd_path=f"{symdex.LIB_PATH}/assets/object/orange_bottle.usd",
            usd_path="bottle",
            random_choice=True,
            obj_label=True,
            preview_surface=RandomPreviewSurfaceCfg(
                diffuse_color_dict=COLOR_DICT_20,
                roughness_range=(0.2, 0.8),
                metallic_range=(0.2, 0.8),
            ),
            random_color=True,
            random_roughness=True,
            random_metallic=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=False,
                disable_gravity=False,
                max_linear_velocity=1000,
                max_angular_velocity=1000,
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=1,
                max_depenetration_velocity=1000.0,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
            activate_contact_sensors=True,
            # scale=(1.2, 1.2, 1.2),
            scale=(1.0, 1.0, 1.0),
        ), # wait for initialization
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
    contact_sensors_0_4 = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/if4",  # index
        update_period=0.0, 
        debug_vis=True,
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Object_0"],
    )
    contact_sensors_1_4 = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/mf4",  # middle
        update_period=0.0, 
        debug_vis=True,
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Object_0"],
    )
    contact_sensors_2_4 = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/pf4",  # pinky
        update_period=0.0, 
        debug_vis=True,
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Object_0"],
    )
    contact_sensors_3_4 = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/th4",  # thumb
        update_period=0.0, 
        debug_vis=True,
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Object_0"],
    )
    contact_sensors_0_left_4 = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot_left/if4",  # index
        update_period=0.0, 
        debug_vis=True,
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Object_0"],
    )
    # left
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
    contact_sensors_0_4_left = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot_left/if4",  # index
        update_period=0.0, 
        debug_vis=True,
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Object_0"],
    )
    contact_sensors_1_4_left = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot_left/mf4",  # middle
        update_period=0.0, 
        debug_vis=True,
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Object_0"],
    )
    contact_sensors_2_4_left = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot_left/pf4",  # pinky
        update_period=0.0, 
        debug_vis=True,
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Object_0"],
    )
    contact_sensors_3_4_left = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot_left/th4",  # thumb
        update_period=0.0, 
        debug_vis=True,
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Object_0"],
    )

    bottle_top = FrameTransformerCfg(
        prim_path="{ENV_REGEX_NS}/Object_0",
        debug_vis=True,
        visualizer_cfg=FRAME_MARKER_SMALL_CFG.replace(prim_path="/Visuals/BottomTopFrameTransformer"),
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/Object_0",
                name="approach_frame",
                offset=OffsetCfg(
                    pos=(0.0, 0.0, 0.07),
                    rot=(1.0, 0.0, 0.0, 0.0),
                ),
            ),
        ],
    )

    bottle_bottom = FrameTransformerCfg(
        prim_path="{ENV_REGEX_NS}/Object_0",
        debug_vis=True,
        visualizer_cfg=FRAME_MARKER_SMALL_CFG.replace(prim_path="/Visuals/BottomBottomFrameTransformer"),
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/Object_0",
                name="approach_frame",
                offset=OffsetCfg(
                    pos=(0.0, 0.0, -0.08),
                    rot=(1.0, 0.0, 0.0, 0.0),
                ),
            ),
        ],
    )

@configclass
class HandoverEventCfg(BaseEventCfg):
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
    
    reset_object = EventTerm(
        func=reset_object,
        mode="reset",
        params={
            "pose_range": {"x": [0.2, 0.2], "y": [-0.25, -0.25], "z": [0.12, 0.12], "yaw": [-3.14, 3.14]},  # 
            "velocity_range": {},
            "object_id": 0,
        },
    )

    object_mass = EventTerm(
        func=randomize_rigid_body_mass,
        mode="startup",
        params={
            "mass_distribution_params": (0.05, 0.9),
            "operation": "scale",
        },
    )

@configclass
class HandoverCommandsCfg(BaseCommandsCfg):
    """Command specifications for the MDP."""

    target_pos = TargetPositionCommandCfg(
        object_id=0,
        success_threshold=0.05,
        success_threshold_orient=0.95, # 60 degree 
        pose_range={"x": [0.15, 0.15], "y": [0.0, 0.0], "z": [0.25, 0.25], "roll": [-1.57, -1.57]},
        update_goal_on_success=False,
        debug_vis=True,
    )

    left_hand_target_pos = ReachCommandCfg(
        asset_cfg=SceneEntityCfg("robot_left"),
        target_link="palm_link",
        success_threshold=0.05,
        success_threshold_orient=0.95, # 60 degree 
        pose_range={"x": [0.15, 0.15], "y": [0.07, 0.07], "z": [0.37, 0.37], "pitch": [1.57, 1.57]},
        update_goal_on_success=False,
        debug_vis=True,
    )

@configclass
class HandoverObservationsCfg(BaseObservationsCfg):
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # -- robot terms (order preserved)
        ee_pose_right = ObsTerm(func=ee_pose, params={"ee_name": "palm_link"})
        joint_pos_right = ObsTerm(func=joint_pos_limit_normalized, params={"joints": None,
                                                                           "joint_lower_limit": JOINT_LOWER_LIMIT, 
                                                                           "joint_upper_limit": JOINT_UPPER_LIMIT}, ) # noise=Gnoise(std=0.002)
        joint_vel_right = ObsTerm(func=joint_vel, params={"joints": None},) # noise=Gnoise(std=0.002)
        ee_pose_left = ObsTerm(func=ee_pose, params={"ee_name": "palm_link", "asset_cfg": SceneEntityCfg("robot_left")})
        joint_pos_left = ObsTerm(func=joint_pos_limit_normalized, params={"joints": None, 
                                                                          "joint_lower_limit": JOINT_LOWER_LIMIT_LEFT, 
                                                                          "joint_upper_limit": JOINT_UPPER_LIMIT_LEFT, 
                                                                          "asset_cfg": SceneEntityCfg("robot_left")}, ) # noise=Gnoise(std=0.002)
        joint_vel_left = ObsTerm(func=joint_vel, params={"joints": None, "asset_cfg": SceneEntityCfg("robot_left")},) # noise=Gnoise(std=0.002)
        bottle_pos = ObsTerm(
            func=object_pos, # noise=Gnoise(std=0.01)
            params={"object_id": 0}
        )
        bottle_quat = ObsTerm(
            func=object_quat, params={"object_id": 0, "symmetry": True}, # noise=Gnoise(std=0.01)
        )
        handover_pos = ObsTerm(func=generated_commands, params={"command_name": "target_pos"})
        last_action = ObsTerm(func=last_action)
        symmetry_tracker = ObsTerm(func=symmetry_tracker)
        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class HandoverActionsCfg:
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
        scale=1.0,
        use_default_offset=False,
    )


@configclass
class HandoverTerminationsCfg(BaseTerminationsCfg):
    out_of_space = DoneTerm(
        func=handover.obj_out_space, params={"asset_cfg": SceneEntityCfg("robot")}
    )
    out_of_space_left = DoneTerm(
        func=handover.obj_out_space, params={"asset_cfg": SceneEntityCfg("robot_left")}
    ) 
    max_consecutive_success = DoneTerm(
        func=handover.max_consecutive_success, 
        params={"num_success": 1},
    )


@configclass
class HandoverRewardsCfg(BaseRewardsCfg):
    """Reward terms for the MDP."""
    reaching_object = RewTerm(func=handover.frame_marker_robot_distance, 
                              params={"weight": [1.0, 1.0, 1.0, 1.5, 2.0], 
                                      "link_name": ["if5", "mf5", "pf5", "th5", "palm_link"], 
                                      "frame_name": "bottle_bottom"}, 
                                      weight=0.0)
    object_goal_tracking = RewTerm(func=handover.object_goal_distance,
                                   params={"command_name": "target_pos", "object_id": 0},
                                   weight=0.0,
                                   )
    object_goal_orient_tracking = RewTerm(func=handover.object_goal_orient_distance,
                                   params={"command_name": "target_pos", "object_id": 0, "axis": "z"},
                                   weight=0.0,
                                   )
    middle_success_bonus = RewTerm(func=handover.cmd_success_bonus,
                                   params={"command_names": "target_pos", "num_success": 1, "if_right": True},
                                   weight=0.0,
                                   )
    contact_bottle_punish = RewTerm(func=handover.contact_bottle_punish,
                                   params={"sensor_names": ["contact_sensors_0", "contact_sensors_1", "contact_sensors_2", "contact_sensors_3",
                                                            "contact_sensors_0_4", "contact_sensors_1_4", "contact_sensors_2_4", "contact_sensors_3_4"]},
                                   weight=0.0,
                                   )
    reset_robot_joint_pos = RewTerm(func=handover.robot_goal_distance, 
                              params={"target_pos": [0.0462, -0.5045, 0.4468], 
                                      "target_link": "palm_link",
                                      "sensor_names": ["contact_sensors_0", "contact_sensors_1", "contact_sensors_2", "contact_sensors_3",
                                                            "contact_sensors_0_4", "contact_sensors_1_4", "contact_sensors_2_4", "contact_sensors_3_4"]}, 
                                      weight=0.0)   
    left_align_hand_pose = RewTerm(func=handover.align_hand_pose,
                                   params={"link_name": "palm_link", "command_name": "left_hand_target_pos", "asset_cfg": SceneEntityCfg("robot_left")},
                                   weight=0.0,
                                   )
    left_align_finger_joint = RewTerm(func=handover.align_finger_joint,
                                   params={"link_name": ["jif1", "jif2", "jif3", "jif4", "jmf1", "jmf2", "jmf3", "jmf4", "jpf1", "jpf2", "jpf3", "jpf4", "jth1", "jth2", "jth3", "jth4"], 
                                           "asset_cfg": SceneEntityCfg("robot_left")},
                                   weight=0.0,
                                   )
    left_reaching_object = RewTerm(func=handover.frame_marker_robot_distance, 
                              params={"weight": [1.5, 1.0, 1.0, 2.0], 
                                      "link_name": ["if5", "mf5", "pf5", "th5"], 
                                      "frame_name": "bottle_top",
                                      "if_left": True,
                                      "asset_cfg": SceneEntityCfg("robot_left")}, 
                                      weight=0.0)
    left_object_goal_tracking = RewTerm(func=handover.object_goal_distance,
                                   params={"command_name": "target_pos", 
                                           "object_id": 0, 
                                           "if_left": True,
                                           "sensor_names": ["contact_sensors_0_left", "contact_sensors_1_left", "contact_sensors_2_left", "contact_sensors_3_left",
                                                            "contact_sensors_0_4_left", "contact_sensors_1_4_left", "contact_sensors_2_4_left", "contact_sensors_3_4_left"]},
                                   weight=0.0,
                                   )
    left_object_goal_orient_tracking = RewTerm(func=handover.object_goal_orient_distance,
                                   params={"command_name": "target_pos", 
                                           "object_id": 0, 
                                           "axis": "z", 
                                           "if_left": True,
                                           "sensor_names": ["contact_sensors_0_left", "contact_sensors_1_left", "contact_sensors_2_left", "contact_sensors_3_left",
                                                            "contact_sensors_0_4_left", "contact_sensors_1_4_left", "contact_sensors_2_4_left", "contact_sensors_3_4_left"]},
                                   weight=0.0,
                                   )
    middle_success_bonus_left = RewTerm(func=handover.cmd_success_bonus,
                                   params={"command_names": "target_pos", "num_success": 1, "if_left": True, 
                                           "sensor_names": ["contact_sensors_0_left", "contact_sensors_1_left", "contact_sensors_2_left", "contact_sensors_3_left",
                                                            "contact_sensors_0_4_left", "contact_sensors_1_4_left", "contact_sensors_2_4_left", "contact_sensors_3_4_left"]},
                                   weight=0.0,
                                   )
    success_bonus = RewTerm(func=handover.success_bonus,
                            params={"command_names": "target_pos", 
                                    "num_success": 10,
                                    "not_contact_sensor_names": ["contact_sensors_0", "contact_sensors_1", "contact_sensors_2", "contact_sensors_3",
                                                                "contact_sensors_0_4", "contact_sensors_1_4", "contact_sensors_2_4", "contact_sensors_3_4"],
                                    "is_contact_sensor_names": ["contact_sensors_0_left", "contact_sensors_1_left", "contact_sensors_2_left", "contact_sensors_3_left",
                                                                "contact_sensors_0_4_left", "contact_sensors_1_4_left", "contact_sensors_2_4_left", "contact_sensors_3_4_left"]},
                            weight=0.0,
                            )
                            

@configclass
class HandoverEnvCfg(BaseEnvCfg):
    name: str = "Handover"
    scene = HandoverSceneCfg(num_envs=4096, env_spacing=3.0)
    events = HandoverEventCfg()
    commands = HandoverCommandsCfg()
    observations = HandoverObservationsCfg()
    actions = HandoverActionsCfg()
    terminations = HandoverTerminationsCfg()
    rewards = HandoverRewardsCfg()
    num_object = 1
    action_dim = 44 # arm + hand
    action_scale: list = [1.0] * action_dim
                        # [0.05, 0.05, 0.05, 0.05, 0.05, 0.05,
                        #     0.03, 0.03, 0.03, 0.03, 
                        #     0.03, 0.03, 0.03, 0.03, 
                        #     0.03, 0.03, 0.03, 0.015,
                        #     0.03, 0.03, 0.03, 0.03,
                        #     0.05, 0.05, 0.05, 0.05, 0.05, 0.05,
                        #     0.03, 0.03, 0.03, 0.03, 
                        #     0.03, 0.03, 0.03, 0.03, 
                        #     0.03, 0.03, 0.03, 0.015,
                        #     0.03, 0.03, 0.03, 0.03]  # jth3 needs smaller rate

    visualize_marker: bool = False

    def __post_init__(self):
        # post init of parent
        super().__post_init__()
