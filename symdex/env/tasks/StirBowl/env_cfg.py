from __future__ import annotations

from isaaclab.assets import RigidObjectCfg
from isaaclab.utils import configclass
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
from symdex.env.tasks.StirBowl import mdps as bowl

FRAME_MARKER_SMALL_CFG = FRAME_MARKER_CFG.copy()
FRAME_MARKER_SMALL_CFG.markers["frame"].scale = (0.10, 0.10, 0.10)

@configclass
class StirBowlSceneCfg(BaseSceneCfg):
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
                "joint1": 0.5,
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
                "joint1": -0.5,
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
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{symdex.LIB_PATH}/assets/object/egg_beater.usd",
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
            scale=(0.001, 0.001, 0.001),
        ), # wait for initialization
        init_state=RigidObjectCfg.InitialStateCfg(
            lin_vel=(0.0, 0.0, 0.0),
            ang_vel=(0.0, 0.0, 0.0),
            pos=(0.0, 0.0, 0.0),
        ),
    )

    object_1 = RigidObjectCfg(
        prim_path=f"/World/envs/env_.*/Object_1",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{symdex.LIB_PATH}/assets/object/bowl.usd",
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
            scale=(0.035, 0.04, 0.035),
        ), # wait for initialization
        init_state=RigidObjectCfg.InitialStateCfg(
            lin_vel=(0.0, 0.0, 0.0),
            ang_vel=(0.0, 0.0, 0.0),
            pos=(0.0, 0.0, 0.0),
        ),
    )

    object_2 = RigidObjectCfg(
        prim_path=f"/World/envs/env_.*/Object_2",
        spawn=sim_utils.SphereCfg(
            radius=0.03,
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
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.0, 1.0), metallic=0.2),
        ), # wait for initialization
        init_state=RigidObjectCfg.InitialStateCfg(
            lin_vel=(0.0, 0.0, 0.0),
            ang_vel=(0.0, 0.0, 0.0),
            pos=(0.0, 0.0, 0.0),
        ),
    )

    object_3 = RigidObjectCfg(
        prim_path=f"/World/envs/env_.*/Object_3",
        spawn=sim_utils.SphereCfg(
            radius=0.03,
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
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.0, 1.0), metallic=0.2),
        ), # wait for initialization
        init_state=RigidObjectCfg.InitialStateCfg(
            lin_vel=(0.0, 0.0, 0.0),
            ang_vel=(0.0, 0.0, 0.0),
            pos=(0.0, 0.0, 0.0),
        ),
    )

    object_4 = RigidObjectCfg(
        prim_path=f"/World/envs/env_.*/Object_4",
        spawn=sim_utils.SphereCfg(
            radius=0.03,
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
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.0, 1.0), metallic=0.2),
        ), # wait for initialization
        init_state=RigidObjectCfg.InitialStateCfg(
            lin_vel=(0.0, 0.0, 0.0),
            ang_vel=(0.0, 0.0, 0.0),
            pos=(0.0, 0.0, 0.0),
        ),
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

    object_approach_frame = FrameTransformerCfg(
        prim_path="{ENV_REGEX_NS}/Object_1",
        debug_vis=True,
        visualizer_cfg=FRAME_MARKER_SMALL_CFG.replace(prim_path="/Visuals/ObjectApproachFrameTransformer"),
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/Object_1",
                name="approach_frame",
                offset=OffsetCfg(
                    pos=(0.0, 0.18, 0.0),
                    rot=(0.5, -0.5, 0.5, -0.5),
                ),
            ),
        ],
    )

    object_approach_frame_symmetry = FrameTransformerCfg(
        prim_path="{ENV_REGEX_NS}/Object_1",
        debug_vis=True,
        visualizer_cfg=FRAME_MARKER_SMALL_CFG.replace(prim_path="/Visuals/ObjectApproachFrameTransformerSymmetry"),
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/Object_1",
                name="approach_frame",
                offset=OffsetCfg(
                    pos=(0.0, -0.18, 0.0),
                    rot=(0.5, 0.5, 0.5, 0.5),
                ),
            ),
        ],
    )

@configclass
class StirBowlEventCfg(BaseEventCfg):
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
    
    reset_object_egg_beater = EventTerm(
        func=reset_object,
        mode="reset",
        params={
            "pose_range": {"x": [0.15, 0.05], "y": [-0.3, -0.35], "z": [0.01, 0.01], "roll": [1.57, 1.57]},
            "velocity_range": {},
            "object_id": 0,
        },
    )

    reset_object_bowl = EventTerm(
        func=reset_object,
        mode="reset",
        params={
            "pose_range": {"x": [0.1, 0.1], "y": [0.2, 0.2], "z": [0.06, 0.06]},
            "velocity_range": {},
            "object_id": 1,
        },
    )

    reset_ball_1 = EventTerm(
        func=reset_object,
        mode="reset",
        params={
            "pose_range": {"x": [0.1, 0.1], "y": [0.2, 0.2], "z": [0.05, 0.05]},
            "velocity_range": {},
            "object_id": 2,
        },
    )

    reset_ball_2 = EventTerm(
        func=reset_object,
        mode="reset",
        params={
            "pose_range": {"x": [0.14, 0.14], "y": [0.22, 0.22], "z": [0.05, 0.05]},
            "velocity_range": {},
            "object_id": 3,
        },
    )

    reset_ball_3 = EventTerm(
        func=reset_object,
        mode="reset",
        params={
            "pose_range": {"x": [0.06, 0.06], "y": [0.18, 0.18], "z": [0.05, 0.05]},
            "velocity_range": {},
            "object_id": 4,
        },
    )

@configclass
class StirBowlCommandsCfg(BaseCommandsCfg):
    """Command specifications for the MDP."""

    target_pos = TargetPositionCommandCfg(
        object_id=0,
        success_threshold=0.1,
        success_threshold_orient=0.86,
        pose_range={"x": [-0.1, -0.1], "y": [0.0, 0.0], "z": [0.28, 0.28]},
        update_goal_on_success=False,
        debug_vis=True,
    )

    bowl_target_pos = TargetPositionCommandCfg(
        object_id=1,
        success_threshold=0.05,
        success_threshold_orient=0.97,
        pose_range={"x": [-0.1, -0.1], "y": [0.0, 0.0], "z": [0.06, 0.06]},
        update_goal_on_success=False,
        debug_vis=True,
    )

@configclass
class StirBowlObservationsCfg(BaseObservationsCfg):
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""
        # -- robot terms (order preserved)
        ee_pose_right = ObsTerm(func=ee_pose, params={"ee_name": "palm_link"})
        joint_pos_right = ObsTerm(func=joint_pos_limit_normalized, params={"joints": None,
                                                                           "joint_lower_limit": JOINT_LOWER_LIMIT, 
                                                                           "joint_upper_limit": JOINT_UPPER_LIMIT}, )
        joint_vel_right = ObsTerm(func=joint_vel, params={"joints": None},)
        ee_pose_left = ObsTerm(func=ee_pose, params={"ee_name": "palm_link", "asset_cfg": SceneEntityCfg("robot_left")})
        joint_pos_left = ObsTerm(func=joint_pos_limit_normalized, params={"joints": None, 
                                                                          "joint_lower_limit": JOINT_LOWER_LIMIT_LEFT, 
                                                                          "joint_upper_limit": JOINT_UPPER_LIMIT_LEFT, 
                                                                          "asset_cfg": SceneEntityCfg("robot_left")}, )
        joint_vel_left = ObsTerm(func=joint_vel, params={"joints": None, "asset_cfg": SceneEntityCfg("robot_left")},)
        # -- object terms
        egg_beater_pos = ObsTerm(
            func=object_pos, params={"object_id": 0},
        )
        egg_beater_quat = ObsTerm(
            func=object_quat, params={"object_id": 0, "symmetry": True},
        )
        bowl_pos = ObsTerm(
            func=object_pos, params={"object_id": 1},
        )
        bowl_quat = ObsTerm(
            func=object_quat, params={"object_id": 1, "symmetry": True},
        )
        bowl_lin_vel = ObsTerm(
            func=object_lin_vel, params={"object_id": 1},
        )
        ball_1_pos = ObsTerm(
            func=object_pos, params={"object_id": 2},
        )
        ball_1_lin_vel = ObsTerm(
            func=object_lin_vel, params={"object_id": 2},
        )
        ball_2_pos = ObsTerm(
            func=object_pos, params={"object_id": 3},
        )
        ball_2_lin_vel = ObsTerm(
            func=object_lin_vel, params={"object_id": 3},
        )
        ball_3_pos = ObsTerm(
            func=object_pos, params={"object_id": 4},
        )
        ball_3_lin_vel = ObsTerm(
            func=object_lin_vel, params={"object_id": 4},
        )
        goal_pos_egg_beater = ObsTerm(func=generated_commands, params={"command_name": "target_pos"})
        goal_pos_bowl = ObsTerm(func=generated_commands, params={"command_name": "bowl_target_pos"})
        # -- action terms
        last_action = ObsTerm(func=last_action)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class StirBowlActionsCfg:
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
class StirBowlTerminationsCfg(BaseTerminationsCfg):
    pass


@configclass
class StirBowlRewardsCfg(BaseRewardsCfg):
    """Reward terms for the MDP."""
    reaching_object = RewTerm(func=object_robot_distance, 
                              params={"weight": [1.0, 1.0, 1.0, 1.5], 
                                      "link_name": ["if5", "mf5", "pf5", "th5"], 
                                      "object_id": 0}, 
                                      weight=0.0)
    object_lifting = RewTerm(func=lift_distance,
                             params={"command_name": "target_pos", "object_id": 0, "sensor_names": ["contact_sensors_0", "contact_sensors_1", "contact_sensors_2", "contact_sensors_3"]},
                             weight=0.0,
                             )
    egg_beater_goal_tracking = RewTerm(func=object_goal_distance,
                                   params={"command_name": "target_pos", "object_id": 0, "sensor_names": ["contact_sensors_0", "contact_sensors_1", "contact_sensors_2", "contact_sensors_3"],},
                                   weight=0.0,
                                   )
    egg_beater_goal_orient_tracking = RewTerm(func=object_goal_distance_orient,
                                   params={"command_name": "target_pos", "object_id": 0, "axis": "z", "sensor_names": ["contact_sensors_0", "contact_sensors_1", "contact_sensors_2", "contact_sensors_3"],},
                                   weight=0.0,
                                   )
    ball_velocity = RewTerm(func=bowl.object_vel, 
                            params={"object_id": [2, 3, 4], 
                                    "sensor_names": ["contact_sensors_0", "contact_sensors_1", "contact_sensors_2", "contact_sensors_3"]},
                                    weight=0.0)
    bowl_goal_tracking = RewTerm(func=bowl.object_goal_distance,
                                   params={"command_name": "bowl_target_pos", "object_id": 1,},
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
    bowl_success_bonus = RewTerm(func=bowl.cmd_success_bonus,
                            params={"command_names": "bowl_target_pos", "num_success": 1},
                            weight=0.0,
                            )
    success_bonus = RewTerm(func=bowl.success_bonus,
                            params={"num_success": 5},
                            weight=0.0,
                            )
    
    # symmetry
    reaching_object_symmetry = RewTerm(func=object_robot_distance, 
                              params={"weight": [1.0, 1.0, 1.0, 1.5], 
                                      "link_name": ["if5", "mf5", "pf5", "th5"], 
                                      "object_id": 0,
                                      "asset_cfg": SceneEntityCfg("robot_left")}, 
                                      weight=0.0)
    object_lifting_symmetry = RewTerm(func=lift_distance,
                             params={"command_name": "target_pos", "object_id": 0, "sensor_names": ["contact_sensors_0_left", "contact_sensors_1_left", "contact_sensors_2_left", "contact_sensors_3_left"]},
                             weight=0.0,
                             )
    egg_beater_goal_tracking_symmetry = RewTerm(func=object_goal_distance,
                                   params={"command_name": "target_pos", "object_id": 0, "sensor_names": ["contact_sensors_0_left", "contact_sensors_1_left", "contact_sensors_2_left", "contact_sensors_3_left"],},
                                   weight=0.0,
                                   )
    egg_beater_goal_orient_tracking_symmetry = RewTerm(func=object_goal_distance_orient,
                                   params={"command_name": "target_pos", "object_id": 0, "axis": "z", "sensor_names": ["contact_sensors_0_left", "contact_sensors_1_left", "contact_sensors_2_left", "contact_sensors_3_left"],},
                                   weight=0.0,
                                   )
    ball_velocity_symmetry = RewTerm(func=bowl.object_vel, 
                            params={"object_id": [2, 3, 4], 
                                    "sensor_names": ["contact_sensors_0_left", "contact_sensors_1_left", "contact_sensors_2_left", "contact_sensors_3_left"]},
                                    weight=0.0)
    align_hand_to_pos_symmetry = RewTerm(func=align_palm_to_pos,
                                   params={"link_name": ["palm_link"], "frame_name": "object_approach_frame_symmetry", "asset_cfg": SceneEntityCfg("robot")},
                                   weight=0.0,
                                   )
    align_hand_to_quat_symmetry = RewTerm(func=align_palm_to_quat,
                                   params={"link_name": ["palm_link"], "frame_name": "object_approach_frame_symmetry", "asset_cfg": SceneEntityCfg("robot")},
                                   weight=0.0,
                                   )



@configclass
class StirBowlEnvCfg(BaseEnvCfg):
    name: str = "StirBowl"
    scene = StirBowlSceneCfg(num_envs=4096, env_spacing=3.0)
    events = StirBowlEventCfg()
    commands = StirBowlCommandsCfg()
    observations = StirBowlObservationsCfg()
    actions = StirBowlActionsCfg()
    terminations = StirBowlTerminationsCfg()
    rewards = StirBowlRewardsCfg()
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
        self.viewer.eye = (-3.5, 0.0, 3.5)