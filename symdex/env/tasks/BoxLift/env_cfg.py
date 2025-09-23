from __future__ import annotations

from isaaclab.assets import RigidObjectCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveGaussianNoiseCfg as Gnoise
import isaaclab.envs.mdp as mdp
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.sensors.frame_transformer import OffsetCfg
from isaaclab.markers.config import FRAME_MARKER_CFG 

import symdex
from symdex.env.tasks.manager_based_env_cfg import *
from symdex.env.mdps.obs_mdps import *
from symdex.env.mdps.reset_mdps import *
from symdex.env.mdps.reward_mdps import *
from symdex.env.mdps.termination_mdps import *
from symdex.env.mdps.command_mdps.grasp_command_cfg import TargetPositionCommandCfg
from symdex.env.action_managers.actions_cfg import EMACumulativeRelativeJointPositionActionCfg
from symdex.env.tasks.BoxLift import mdps as lift

FRAME_MARKER_SMALL_CFG = FRAME_MARKER_CFG.copy()
FRAME_MARKER_SMALL_CFG.markers["frame"].scale = (0.10, 0.10, 0.10)

@configclass
class BoxLiftSceneCfg(BaseSceneCfg):
    # robots
    robot = ArticulationCfg(
        prim_path="/World/envs/env_.*/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{symdex.LIB_PATH}/assets/ufactory850/uf850_allegro_right_colored.usd",
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
                "jpf4": 0.0,
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
            usd_path=f"{symdex.LIB_PATH}/assets/ufactory850/uf850_allegro_left_colored.usd",
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
                "joint1": -0.05,
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
                "jth1": 0.364,
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

    object_0 = RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Object_0",
        spawn=sim_utils.MultiAssetSpawnerCfg(
            assets_cfg=[
                # sim_utils.CuboidCfg(
                #     size=(0.25, 0.39, 0.155),
                # ),
                # sim_utils.CuboidCfg(
                #     size=(0.31, 0.405, 0.215),
                # ),
                # sim_utils.CuboidCfg(
                #     size=(0.19, 0.275, 0.215),
                # ),
                # sim_utils.CuboidCfg(
                #     size=(0.18, 0.26, 0.13),
                # ),
                sim_utils.UsdFileCfg(
                    usd_path=f"{symdex.LIB_PATH}/assets/object/tote.usd",
                    scale=(0.8, 0.6, 1.0),
                ),
            ],
            random_choice=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                solver_position_iteration_count=16, solver_velocity_iteration_count=1
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
            activate_contact_sensors=True,
            collision_props=sim_utils.CollisionPropertiesCfg(),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
    )

    # sensors
    contact_sensors_robot = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/link1|link2|link3|link4|link5",  # thumb
        update_period=0.0, 
        debug_vis=True,
    )
    contact_sensors_robot_left = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot_left/link1|link2|link3|link4|link5",  # thumb
        update_period=0.0, 
        debug_vis=True,
    )

    tote_right = FrameTransformerCfg(
        prim_path="{ENV_REGEX_NS}/Object_0",
        debug_vis=True,
        visualizer_cfg=FRAME_MARKER_SMALL_CFG.replace(prim_path="/Visuals/ToteRightFrameTransformer"),
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/Object_0",
                name="approach_frame",
                offset=OffsetCfg(
                    pos=(0.0, -0.2, 0.1),
                    rot=(0.500, 0.500, 0.500, 0.500),
                ),
            ),
        ],
    )

    tote_left = FrameTransformerCfg(
        prim_path="{ENV_REGEX_NS}/Object_0",
        debug_vis=True,
        visualizer_cfg=FRAME_MARKER_SMALL_CFG.replace(prim_path="/Visuals/ToteLeftFrameTransformer"),
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/Object_0",
                name="approach_frame",
                offset=OffsetCfg(
                    pos=(0.0, 0.2, 0.1),
                    rot=(-0.500, 0.500, -0.500, 0.500),
                ),
            ),
        ],
    )

@configclass
class BoxLiftEventCfg(BaseEventCfg):
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
    
    reset_object = EventTerm(
        func=reset_object,
        mode="reset",
        params={
            "pose_range": {"x": [0.0, 0.2], "y": [-0.1, 0.1], "z": [0.0, 0.0], "yaw": [-0.7, 0.7]},
            "velocity_range": {},
            "object_id": 0,
        },
    )

    compute_box_side_transform = EventTerm(
        func=lift.compute_box_side_transform,
        mode="startup",
        params={"object_id": 0, "axis": "y"},
    )

@configclass
class BoxLiftCommandsCfg(BaseCommandsCfg):
    """Command specifications for the MDP."""

    target_pos = TargetPositionCommandCfg(
        object_id=0,
        success_threshold=0.05,
        success_threshold_orient=0.9, # 60 degree 
        pose_range={"x": [0.15, 0.15], "y": [0.0, 0.0], "z": [0.18, 0.18]},
        update_goal_on_success=True,
        debug_vis=True,
        return_type="pos",
    )


@configclass
class BoxLiftObservationsCfg(BaseObservationsCfg):
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""
        # -- robot terms (order preserved)
        joint_pos_right = ObsTerm(func=joint_pos_limit_normalized, params={"joints": None,
                                                                           "joint_lower_limit": JOINT_LOWER_LIMIT, 
                                                                           "joint_upper_limit": JOINT_UPPER_LIMIT}, noise=Gnoise(std=0.005))
        joint_vel_right = ObsTerm(func=joint_vel, params={"joints": None},)
        joint_pos_left = ObsTerm(func=joint_pos_limit_normalized, params={"joints": None, 
                                                                          "joint_lower_limit": JOINT_LOWER_LIMIT_LEFT, 
                                                                          "joint_upper_limit": JOINT_UPPER_LIMIT_LEFT, 
                                                                          "asset_cfg": SceneEntityCfg("robot_left")}, noise=Gnoise(std=0.005))
        joint_vel_left = ObsTerm(func=joint_vel, params={"joints": None, "asset_cfg": SceneEntityCfg("robot_left")},)
        box_pos = ObsTerm(
            func=object_pos, noise=Unoise(n_min=0.0, n_max=0.01),
            params={"object_id": 0}
        )
        box_quat = ObsTerm(
            func=object_quat, params={"object_id": 0, "symmetry": True}, noise=Gnoise(std=0.01)
        )
        box_pos_right = ObsTerm(func=lift.box_side, params={"side": "right"})
        frame_quat_right = ObsTerm(func=frame_quat, params={"frame_name": "tote_right", "symmetry": True})
        box_pos_left = ObsTerm(func=lift.box_side, params={"side": "left"})
        frame_quat_left = ObsTerm(func=frame_quat, params={"frame_name": "tote_left", "symmetry": True})
        box_length = ObsTerm(func=lift.box_length, noise=Unoise(n_min=0.0, n_max=0.03))
        last_action = ObsTerm(func=last_action)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class BoxLiftActionsCfg:
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
class BoxLiftTerminationsCfg(BaseTerminationsCfg):
    max_consecutive_success = DoneTerm(
        func=max_consecutive_success, params={"num_success": 20, "command_names": "target_pos"}
    )


@configclass
class BoxLiftRewardsCfg(BaseRewardsCfg):
    """Reward terms for the MDP."""
    align_hand_to_pos = RewTerm(func=lift.align_palm_to_pos,
                                   params={"link_name": ["palm_link"], "side": "right", "asset_cfg": SceneEntityCfg("robot")},
                                   weight=0.0,
                                   )
    align_hand_to_quat = RewTerm(func=align_palm_to_quat,
                                   params={"link_name": ["palm_link"], "frame_name": "tote_right", "asset_cfg": SceneEntityCfg("robot")},
                                   weight=0.0,
                                   )
    align_hand_to_pos_left = RewTerm(func=lift.align_palm_to_pos,
                                   params={"link_name": ["palm_link"], "side": "left", "asset_cfg": SceneEntityCfg("robot_left")},
                                   weight=0.0,
                                   )
    align_hand_to_quat_left = RewTerm(func=align_palm_to_quat,
                                   params={"link_name": ["palm_link"], "frame_name": "tote_left", "asset_cfg": SceneEntityCfg("robot_left")},
                                   weight=0.0,
                                   )
    object_goal_tracking = RewTerm(func=lift.object_goal_distance,
                                   params={"command_name": "target_pos", "object_id": 0},
                                   weight=0.0,
                                   )
    object_goal_orient_tracking = RewTerm(func=lift.object_goal_orient_distance,
                                   params={"object_id": 0, "command_name": "target_pos"},
                                   weight=0.0,
                                   )
    punish_collision = RewTerm(func=lift.punish_collision,
                                   params={"sensor": "contact_sensors_robot"},
                                   weight=0.0,
                                   )
    punish_collision_left = RewTerm(func=lift.punish_collision,
                                   params={"sensor": "contact_sensors_robot_left"},
                                   weight=0.0,
                                   )
    success_bonus = RewTerm(func=success_bonus,
                            params={"command_names": "target_pos", "num_success": 20},
                            weight=0.0,
                            )

@configclass
class BoxLiftEnvCfg(BaseEnvCfg):
    name: str = "BoxLift"
    scene = BoxLiftSceneCfg(num_envs=4096, env_spacing=3.0)
    events = BoxLiftEventCfg()
    commands = BoxLiftCommandsCfg()
    observations = BoxLiftObservationsCfg()
    actions = BoxLiftActionsCfg()
    terminations = BoxLiftTerminationsCfg()
    rewards = BoxLiftRewardsCfg()
    num_object = 1
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
