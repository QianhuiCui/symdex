from __future__ import annotations

from isaaclab.assets import RigidObjectCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveGaussianNoiseCfg as Gnoise
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
from symdex.env.tasks.PickObject import mdps as pick
FRAME_MARKER_SMALL_CFG = FRAME_MARKER_CFG.copy()
FRAME_MARKER_SMALL_CFG.markers["frame"].scale = (0.10, 0.10, 0.10)

@configclass
class PickObjectSceneCfg(BaseSceneCfg):
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
            usd_path=f"{symdex.LIB_PATH}/assets/object/tote_collision.usd",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True,
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
            scale=(0.6, 0.45, 1.0),
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
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/DexCube/dex_cube_instanceable.usd",
            # usd_path=f"{symdex.LIB_PATH}/assets/object/dog.usd",
            # usd_path=f"{symdex.LIB_PATH}/assets/object/can.usd",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=False,
                disable_gravity=False,
                max_linear_velocity=1000,
                max_angular_velocity=1000,
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=1,
                max_depenetration_velocity=1000.0,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.11),
            activate_contact_sensors=True,
            scale=(1.2, 1.2, 1.2),
            # scale=(1.0, 1.0, 1.0),
        ), # wait for initialization
        init_state=RigidObjectCfg.InitialStateCfg(
            lin_vel=(0.0, 0.0, 0.0),
            ang_vel=(0.0, 0.0, 0.0),
            pos=(0.0, 0.0, 0.0),
        ),
    )

    object_2 = RigidObjectCfg(
        prim_path=f"/World/envs/env_.*/Object_2",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/DexCube/dex_cube_instanceable.usd",
            # usd_path=f"{symdex.LIB_PATH}/assets/object/dog.usd",
            # usd_path=f"{symdex.LIB_PATH}/assets/object/can.usd",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=False,
                disable_gravity=False,
                max_linear_velocity=1000,
                max_angular_velocity=1000,
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=1,
                max_depenetration_velocity=1000.0,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.11),
            activate_contact_sensors=True,
            scale=(1.2, 1.2, 1.2),
            # scale=(1.0, 1.0, 1.0),
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
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Object_1"],
    )
    contact_sensors_1 = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/mf5",  # middle
        update_period=0.0, 
        debug_vis=True,
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Object_1"],
    )
    contact_sensors_2 = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/pf5",  # pinky
        update_period=0.0, 
        debug_vis=True,
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Object_1"],
    )
    contact_sensors_3 = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/th5",  # thumb
        update_period=0.0, 
        debug_vis=True,
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Object_1"],
    )
    contact_sensors_0_symmetry = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot_left/if5",  # index
        update_period=0.0, 
        debug_vis=True,
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Object_1"],
    )
    contact_sensors_1_symmetry = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot_left/mf5",  # middle
        update_period=0.0, 
        debug_vis=True,
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Object_1"],
    )
    contact_sensors_2_symmetry = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot_left/pf5",  # pinky
        update_period=0.0, 
        debug_vis=True,
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Object_1"],
    )
    contact_sensors_3_symmetry = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot_left/th5",  # thumb
        update_period=0.0, 
        debug_vis=True,
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Object_1"],
    )
    contact_sensors_robot = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/link1|link2|link3|link4|link5|link6",  # thumb
        update_period=0.0, 
        debug_vis=True,
    )
    contact_sensors_0_left = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot_left/if5",  # index
        update_period=0.0, 
        debug_vis=True,
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Object_2"],
    )
    contact_sensors_1_left = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot_left/mf5",  # middle
        update_period=0.0, 
        debug_vis=True,
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Object_2"],
    )
    contact_sensors_2_left = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot_left/pf5",  # pinky
        update_period=0.0, 
        debug_vis=True,
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Object_2"],
    )
    contact_sensors_3_left = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot_left/th5",  # thumb
        update_period=0.0, 
        debug_vis=True,
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Object_2"],
    )
    contact_sensors_robot_left = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot_left/link1|link2|link3|link4|link5|link6",  # thumb
        update_period=0.0, 
        debug_vis=True,
    )
    contact_sensors_0_left_symmetry = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/if5",  # index
        update_period=0.0, 
        debug_vis=True,
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Object_2"],
    )
    contact_sensors_1_left_symmetry = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/mf5",  # middle
        update_period=0.0, 
        debug_vis=True,
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Object_2"],
    )
    contact_sensors_2_left_symmetry = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/pf5",  # pinky
        update_period=0.0, 
        debug_vis=True,
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Object_2"],
    )
    contact_sensors_3_left_symmetry = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/th5",  # thumb
        update_period=0.0, 
        debug_vis=True,
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Object_2"],
    )


@configclass
class PickObjectEventCfg(BaseEventCfg):
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
    
    reset_tote = EventTerm(
        func=reset_object,
        mode="reset",
        params={
            "pose_range": {"x": [0.1, 0.1], "y": [0.0, 0.0], "z": [0.0, 0.0]},
            "velocity_range": {},
            "object_id": 0,
        },
    )

    reset_object_1 = EventTerm(
        func=reset_object,
        mode="reset",
        params={
            "pose_range": {"x": [0.1, 0.1], "y": [-0.35, -0.35], "z": [0.0, 0.0], "yaw": [-3.14, 3.14]}, 
            "velocity_range": {},
            "object_id": 1,
        },
    )

    reset_object_2 = EventTerm(
        func=reset_object,
        mode="reset",
        params={
            "pose_range": {"x": [0.1, 0.1], "y": [0.35, 0.35], "z": [0.0, 0.0], "yaw": [-3.14, 3.14]},
            "velocity_range": {},
            "object_id": 2,
        },
    )

@configclass
class PickObjectCommandsCfg(BaseCommandsCfg):
    """Command specifications for the MDP."""

    target_pos = TargetPositionCommandCfg(
        object_id=0,
        success_threshold=0.05,
        z_height=0.3,
        debug_vis=True,
        use_initial_pose=True,
    )

    waiting_pos = TargetPositionCommandCfg(
        object_id=0,
        success_threshold=0.05,
        success_threshold_orient=1.0,
        pose_range={"x": [0.0, 0.0], "y": [0.2, 0.2], "z": [0.3, 0.3]},
        return_type="pos",
        debug_vis=True,
        offset=True,
        use_initial_pose=True,
    )


@configclass
class PickObjectObservationsCfg(BaseObservationsCfg):
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # -- robot terms (order preserved)
        ee_pose_right = ObsTerm(func=ee_pose, params={"ee_name": "palm_link"})
        joint_pos_right = ObsTerm(func=joint_pos_limit_normalized, params={"joints": None, 
                                                                           "joint_lower_limit": JOINT_LOWER_LIMIT, 
                                                                           "joint_upper_limit": JOINT_UPPER_LIMIT,}, noise=Gnoise(std=0.005))
        joint_vel_right = ObsTerm(func=joint_vel, params={"joints": None},)
        object_pos_1 = ObsTerm(
            func=object_pos,
            params={"object_id": 1}, noise=Unoise(n_min=0.0, n_max=0.015)
        )
        ee_pose_left = ObsTerm(func=ee_pose, params={"ee_name": "palm_link", "asset_cfg": SceneEntityCfg("robot_left")})
        joint_pos_left = ObsTerm(func=joint_pos_limit_normalized, params={"joints": None, 
                                                                          "joint_lower_limit": JOINT_LOWER_LIMIT_LEFT, 
                                                                          "joint_upper_limit": JOINT_UPPER_LIMIT_LEFT, 
                                                                          "asset_cfg": SceneEntityCfg("robot_left")}, noise=Gnoise(std=0.005))
        joint_vel_left = ObsTerm(func=joint_vel, params={"joints": None, "asset_cfg": SceneEntityCfg("robot_left")},)
        object_pos_2 = ObsTerm(
            func=object_pos,
            params={"object_id": 2}, noise=Unoise(n_min=0.0, n_max=0.015)
        )
        tote_pos = ObsTerm(
            func=object_pos,
            params={"object_id": 0}
        )
        last_action = ObsTerm(func=last_action)
        waiting_pos = ObsTerm(func=pick.generated_commands, params={"object_id": 2})
        target_pos = ObsTerm(func=pick.generated_commands, params={"object_id": 0})

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class PickObjectActionsCfg:
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
class PickObjectTerminationsCfg(BaseTerminationsCfg):
    pass


@configclass
class PickObjectRewardsCfg(BaseRewardsCfg):
    """Reward terms for the MDP."""
    reaching_object = RewTerm(func=object_robot_distance, 
                              params={"weight": [1.0, 1.0, 1.0, 1.5], 
                                      "link_name": ["if5", "mf5", "pf5", "th5"], 
                                      "object_id": 1}, 
                                      weight=0.0)
    object_lifting = RewTerm(func=lift_distance,
                             params={"command_name": "target_pos", "object_id": 1, "sensor_names": ["contact_sensors_0", "contact_sensors_1", "contact_sensors_2", "contact_sensors_3"]},
                             weight=0.0,
                             )
    object_goal_tracking = RewTerm(func=pick.object_goal_distance,
                                   params={"command_name": "target_pos", "object_id": 1, "sensor_names": ["contact_sensors_0", "contact_sensors_1", "contact_sensors_2", "contact_sensors_3"],},
                                   weight=0.0,
                                   )
    object_1_in_tote = RewTerm(func=pick.if_in_tote,
                             params={"object_id": 1, "sensor_names": ["contact_sensors_0", "contact_sensors_1", "contact_sensors_2", "contact_sensors_3"], "distance_threshold": 0.15},
                             weight=0.0,
                             )
    reset_robot_joint_pos = RewTerm(func=pick.robot_goal_distance, 
                              params={"object_id": 1,
                                      "target_pos": [0.0462, -0.3045, 0.4468], 
                                      "target_link": "palm_link"}, 
                                      weight=0.0)
    reaching_object_left = RewTerm(func=object_robot_distance, 
                              params={"weight": [1.0, 1.0, 1.0, 1.5], 
                                      "link_name": ["if5", "mf5", "pf5", "th5"], 
                                      "object_id": 2,
                                      "asset_cfg": SceneEntityCfg("robot_left")}, 
                                      weight=0.0)
    object_lifting_left = RewTerm(func=lift_distance,
                             params={"command_name": "target_pos", "object_id": 2, "sensor_names": ["contact_sensors_0_left", "contact_sensors_1_left", "contact_sensors_2_left", "contact_sensors_3_left"]},
                             weight=0.0,
                             )
    object_goal_tracking_left = RewTerm(func=pick.object_goal_distance,
                                   params={"command_name": "waiting_pos", 
                                           "object_id": 2, 
                                           "sensor_names": ["contact_sensors_0_left", "contact_sensors_1_left", "contact_sensors_2_left", "contact_sensors_3_left"],
                                           "delay": False,
                                           "switch": True},
                                   weight=0.0,
                                   )
    object_goal_tracking_left_delay = RewTerm(func=pick.object_goal_distance,
                                   params={"command_name": "target_pos", 
                                           "object_id": 2, 
                                           "sensor_names": ["contact_sensors_0_left", "contact_sensors_1_left", "contact_sensors_2_left", "contact_sensors_3_left"],
                                           "delay": True,
                                           "switch": False,
                                           "distance_threshold": 0.2},
                                   weight=0.0,
                                   )
    object_2_in_tote = RewTerm(func=pick.if_in_tote,
                             params={"object_id": 2, 
                                     "sensor_names": ["contact_sensors_0_left", "contact_sensors_1_left", "contact_sensors_2_left", "contact_sensors_3_left"], 
                                     "distance_threshold": 0.2,
                                     "delay": True},
                             weight=0.0,
                             )
    reset_robot_joint_pos_left = RewTerm(func=pick.robot_goal_distance, 
                              params={"object_id": 2, 
                                      "target_pos": [0.0462, 0.3045, 0.4468], 
                                      "target_link": "palm_link",
                                      "asset_cfg": SceneEntityCfg("robot_left")}, 
                                      weight=0.0)
    success_bonus = RewTerm(func=pick.success_bonus,
                            params={"num_success": 5},
                            weight=1.0,
                            )
    
    # symmetry
    reaching_object_symmetry = RewTerm(func=object_robot_distance, 
                              params={"weight": [1.0, 1.0, 1.0, 1.5], 
                                      "link_name": ["if5", "mf5", "pf5", "th5"], 
                                      "object_id": 1,
                                      "asset_cfg": SceneEntityCfg("robot_left")}, 
                                      weight=0.0)
    object_lifting_symmetry = RewTerm(func=lift_distance,
                             params={"command_name": "target_pos", "object_id": 1, "sensor_names": ["contact_sensors_0_symmetry", "contact_sensors_1_symmetry", "contact_sensors_2_symmetry", "contact_sensors_3_symmetry"]},
                             weight=0.0,
                             )
    object_goal_tracking_symmetry = RewTerm(func=pick.object_goal_distance,
                                   params={"command_name": "target_pos", "object_id": 1, "sensor_names": ["contact_sensors_0_symmetry", "contact_sensors_1_symmetry", "contact_sensors_2_symmetry", "contact_sensors_3_symmetry"],},
                                   weight=0.0,
                                   )
    object_1_in_tote_symmetry = RewTerm(func=pick.if_in_tote,
                             params={"object_id": 1, 
                                     "sensor_names": ["contact_sensors_0_symmetry", "contact_sensors_1_symmetry", "contact_sensors_2_symmetry", "contact_sensors_3_symmetry"], 
                                     "distance_threshold": 0.15,
                                     "symmetry": True},
                             weight=0.0,
                             )
    reset_robot_joint_pos_symmetry = RewTerm(func=pick.robot_goal_distance, 
                              params={"object_id": 1,
                                      "target_pos": [0.0462, 0.3045, 0.4468], 
                                      "target_link": "palm_link",
                                      "asset_cfg": SceneEntityCfg("robot_left")}, 
                                      weight=0.0)
    reaching_object_left_symmetry = RewTerm(func=object_robot_distance, 
                              params={"weight": [1.0, 1.0, 1.0, 1.5], 
                                      "link_name": ["if5", "mf5", "pf5", "th5"], 
                                      "object_id": 2}, 
                                      weight=0.0)
    object_lifting_left_symmetry = RewTerm(func=lift_distance,
                             params={"command_name": "target_pos", "object_id": 2, "sensor_names": ["contact_sensors_0_left_symmetry", "contact_sensors_1_left_symmetry", "contact_sensors_2_left_symmetry", "contact_sensors_3_left_symmetry"]},
                             weight=0.0,
                             )
    object_goal_tracking_left_symmetry = RewTerm(func=pick.object_goal_distance,
                                   params={"command_name": "waiting_pos", 
                                           "object_id": 2, 
                                           "sensor_names": ["contact_sensors_0_left_symmetry", "contact_sensors_1_left_symmetry", "contact_sensors_2_left_symmetry", "contact_sensors_3_left_symmetry"],
                                           "delay": False,
                                           "switch": True},
                                   weight=0.0,
                                   )
    object_goal_tracking_left_delay_symmetry = RewTerm(func=pick.object_goal_distance,
                                   params={"command_name": "target_pos", 
                                           "object_id": 2, 
                                           "sensor_names": ["contact_sensors_0_left_symmetry", "contact_sensors_1_left_symmetry", "contact_sensors_2_left_symmetry", "contact_sensors_3_left_symmetry"],
                                           "delay": True,
                                           "switch": False,
                                           "distance_threshold": 0.2},
                                   weight=0.0,
                                   )
    object_2_in_tote_symmetry = RewTerm(func=pick.if_in_tote,
                             params={"object_id": 2, 
                                     "sensor_names": ["contact_sensors_0_left_symmetry", "contact_sensors_1_left_symmetry", "contact_sensors_2_left_symmetry", "contact_sensors_3_left_symmetry"], 
                                     "distance_threshold": 0.2,
                                     "delay": True,
                                     "symmetry": True},
                             weight=0.0,
                             )
    reset_robot_joint_pos_left_symmetry = RewTerm(func=pick.robot_goal_distance, 
                              params={"object_id": 2, 
                                      "target_pos": [0.0462, -0.3045, 0.4468], 
                                      "target_link": "palm_link"}, 
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
                                params={"sensor_names": ["contact_sensors_0_symmetry", "contact_sensors_1_symmetry", "contact_sensors_2_symmetry", "contact_sensors_3_symmetry"]},
                                weight=0.0,
                                )
    collision_to_table_left = RewTerm(func=collision_penalty,
                                params={"sensor_names": ["contact_sensors_0_left", "contact_sensors_1_left", "contact_sensors_2_left", "contact_sensors_3_left"]},
                                weight=0.0,
                                )
    collision_to_table_left_symmetry = RewTerm(func=collision_penalty,
                                params={"sensor_names": ["contact_sensors_0_left_symmetry", "contact_sensors_1_left_symmetry", "contact_sensors_2_left_symmetry", "contact_sensors_3_left_symmetry"]},
                                weight=0.0,
                                )

@configclass
class PickObjectEnvCfg(BaseEnvCfg):
    name: str = "PickObject"
    scene = PickObjectSceneCfg(num_envs=4096, env_spacing=3.0)
    events = PickObjectEventCfg()
    commands = PickObjectCommandsCfg()
    observations = PickObjectObservationsCfg()
    actions = PickObjectActionsCfg()
    terminations = PickObjectTerminationsCfg()
    rewards = PickObjectRewardsCfg()
    num_object = 3
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
