from __future__ import annotations

from isaaclab.assets import RigidObjectCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveGaussianNoiseCfg as Gnoise
import isaaclab.envs.mdp as mdp
from isaaclab.markers.config import FRAME_MARKER_CFG 
from isaaclab.envs.mdp.actions import JointPositionActionCfg
from isaaclab.sensors import CameraCfg

import symdex
from symdex.env.tasks.manager_based_env_cfg import *
from symdex.env.mdps.obs_mdps import *
from symdex.env.mdps.reset_mdps import *
from symdex.env.mdps.reward_mdps import *
from symdex.env.mdps.termination_mdps import *
from symdex.env.mdps.command_mdps.grasp_command_cfg import TargetPositionCommandCfg
from symdex.env.action_managers.actions_cfg import EMACumulativeRelativeJointPositionActionCfg
from symdex.utils.random_cfg import MultiUsdCfg, RandomPreviewSurfaceCfg, COLOR_DICT_20
import symdex.env.tasks.InsertDrawer.mdps as drawer

FRAME_MARKER_SMALL_CFG = FRAME_MARKER_CFG.copy()
FRAME_MARKER_SMALL_CFG.markers["frame"].scale = (0.10, 0.10, 0.10)

@configclass
class InsertDrawerSceneCfg(BaseSceneCfg):
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
                "joint1": 0.8,
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
            # usd_path=f"{symdex.LIB_PATH}/assets/grasp/dog_coacd.usd",
            usd_path="grasp",
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
            mass_props=sim_utils.MassPropertiesCfg(mass=0.11),
            activate_contact_sensors=True,
            scale=(1.0, 1.0, 1.0),
        ), # wait for initialization
        init_state=RigidObjectCfg.InitialStateCfg(
            lin_vel=(0.0, 0.0, 0.0),
            ang_vel=(0.0, 0.0, 0.0),
            pos=(0.0, 0.0, 0.0),
        ),
    )

    drawer = ArticulationCfg(
        prim_path=f"/World/envs/env_.*/Drawer",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{symdex.LIB_PATH}/assets/drawer/drawer.usd",
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
            scale=(0.3, 0.6, 0.5),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            joint_pos={
                "base_drawer_joint": 0.0,
            },
            pos=(0.0, 0.0, 0.0),
            rot=(0.0, 0.0, 0.0, 1.0),
        ),
        actuators={
            "joint": ImplicitActuatorCfg(
                joint_names_expr=["base_drawer_joint"],
                effort_limit=87.0,
                velocity_limit=100.0,
                stiffness=10.0,
                damping=1.0,
                friction=1.0,
            ),
        },
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
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Drawer/handle_grip"],
    )
    contact_sensors_1_left = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot_left/mf5",  # middle
        update_period=0.0, 
        debug_vis=True,
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Drawer/handle_grip"],
    )
    contact_sensors_2_left = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot_left/pf5",  # pinky
        update_period=0.0, 
        debug_vis=True,
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Drawer/handle_grip"],
    )
    contact_sensors_3_left = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot_left/th5",  # thumb
        update_period=0.0, 
        debug_vis=True,
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Drawer/handle_grip"],
    )
    drawer_sensor = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Drawer/handle_grip",
        update_period=0.0,
        debug_vis=True,
    )
    # symmetric sensors
    # contact_sensors_0_symmetry = ContactSensorCfg(
    #     prim_path="/World/envs/env_.*/Robot_left/if5",  # index
    #     update_period=0.0, 
    #     debug_vis=True,
    #     filter_prim_paths_expr=["{ENV_REGEX_NS}/Object_0"],
    # )
    # contact_sensors_1_symmetry = ContactSensorCfg(
    #     prim_path="/World/envs/env_.*/Robot_left/mf5",  # middle
    #     update_period=0.0, 
    #     debug_vis=True,
    #     filter_prim_paths_expr=["{ENV_REGEX_NS}/Object_0"],
    # )
    # contact_sensors_2_symmetry = ContactSensorCfg(
    #     prim_path="/World/envs/env_.*/Robot_left/pf5",  # pinky
    #     update_period=0.0, 
    #     debug_vis=True,
    #     filter_prim_paths_expr=["{ENV_REGEX_NS}/Object_0"],
    # )
    # contact_sensors_3_symmetry = ContactSensorCfg(
    #     prim_path="/World/envs/env_.*/Robot_left/th5",  # thumb
    #     update_period=0.0, 
    #     debug_vis=True,
    #     filter_prim_paths_expr=["{ENV_REGEX_NS}/Object_0"],
    # )
    # contact_sensors_0_left_symmetry = ContactSensorCfg(
    #     prim_path="/World/envs/env_.*/Robot/if5",  # index
    #     update_period=0.0, 
    #     debug_vis=True,
    #     filter_prim_paths_expr=["{ENV_REGEX_NS}/Drawer/handle_grip"],
    # )
    # contact_sensors_1_left_symmetry = ContactSensorCfg(
    #     prim_path="/World/envs/env_.*/Robot/mf5",  # middle
    #     update_period=0.0, 
    #     debug_vis=True,
    #     filter_prim_paths_expr=["{ENV_REGEX_NS}/Drawer/handle_grip"],
    # )
    # contact_sensors_2_left_symmetry = ContactSensorCfg(
    #     prim_path="/World/envs/env_.*/Robot/pf5",  # pinky
    #     update_period=0.0, 
    #     debug_vis=True,
    #     filter_prim_paths_expr=["{ENV_REGEX_NS}/Drawer/handle_grip"],
    # )
    # contact_sensors_3_left_symmetry = ContactSensorCfg(
    #     prim_path="/World/envs/env_.*/Robot/th5",  # thumb
    #     update_period=0.0, 
    #     debug_vis=True,
    #     filter_prim_paths_expr=["{ENV_REGEX_NS}/Drawer/handle_grip"],
    # )


@configclass
class InsertDrawerEventCfg(BaseEventCfg):
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

    reset_drawer = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (1.0, 1.0),
            "velocity_range": (0.0, 0.0),
            "asset_cfg": SceneEntityCfg("drawer")
        },
    )

    reset_drawer_pos = EventTerm(
        func=reset_articulation,
        mode="reset",
        params={
            "pose_range": {"x": [0.4, 0.4], "y": [0.1, 0.1], "z": [0.1, 0.1]}, 
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("drawer")
        },
    )
    
    reset_object_right = EventTerm(
        func=reset_object,
        mode="reset",
        params={
            "pose_range": {"x": [0.05, 0.05], "y": [-0.35, -0.35], "z": [0.0, 0.0], "yaw": [-3.14, 3.14]},
            "velocity_range": {},
            "object_id": 0,
        },
    )

@configclass
class InsertDrawerCommandsCfg(BaseCommandsCfg):
    """Command specifications for the MDP."""

    target_pos = TargetPositionCommandCfg(
        object_id="drawer",
        success_threshold=0.05,
        success_threshold_orient=-1, # 30 degree 
        pose_range={"x": [-0.3, -0.3], "y": [0.0, 0.0], "z": [0.18, 0.18]},
        update_goal_on_success=True,
        debug_vis=True,
        offset=True,
    )


@configclass
class InsertDrawerObservationsCfg(BaseObservationsCfg):
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # -- robot terms (order preserved)
        ee_pose_right = ObsTerm(func=ee_pose, params={"ee_name": "palm_link"})
        joint_pos_right = ObsTerm(func=joint_pos_limit_normalized, params={"joints": None, 
                                                                           "joint_lower_limit": JOINT_LOWER_LIMIT, 
                                                                           "joint_upper_limit": JOINT_UPPER_LIMIT,},
                                                                           noise=Gnoise(std=0.005))
        joint_vel_right = ObsTerm(func=joint_vel, params={"joints": None},)
        object_pos = ObsTerm(
            func=object_pos,
            noise=Unoise(n_min=0.0, n_max=0.01),
            params={"object_id": 0}
        )
        ee_pose_left = ObsTerm(func=ee_pose, params={"ee_name": "palm_link", "asset_cfg": SceneEntityCfg("robot_left")})
        joint_pos_left = ObsTerm(func=joint_pos_limit_normalized, params={"joints": None, 
                                                                          "joint_lower_limit": JOINT_LOWER_LIMIT_LEFT, 
                                                                          "joint_upper_limit": JOINT_UPPER_LIMIT_LEFT, 
                                                                          "asset_cfg": SceneEntityCfg("robot_left")},
                                                                          noise=Gnoise(std=0.005))
        joint_vel_left = ObsTerm(func=joint_vel, params={"joints": None, "asset_cfg": SceneEntityCfg("robot_left")},)
        drawer_handle_pose = ObsTerm(func=ee_pose, params={"ee_name": "handle_grip", "asset_cfg": SceneEntityCfg("drawer")})
        drawer_joint_pos = ObsTerm(func=joint_pos_limit_normalized, params={"joints": None, "asset_cfg": SceneEntityCfg("drawer")}, )
        last_action = ObsTerm(func=last_action)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True
            
    # @configclass
    # class CriticCfg(ObsGroup):
    #     """Observations for policy group."""

    #     # -- robot terms (order preserved)
    #     ee_pose_right = ObsTerm(func=ee_pose, params={"ee_name": "palm_link"})
    #     joint_pos_right = ObsTerm(func=joint_pos_limit_normalized, params={"joints": None, 
    #                                                                        "joint_lower_limit": JOINT_LOWER_LIMIT, 
    #                                                                        "joint_upper_limit": JOINT_UPPER_LIMIT,},
    #                                                                        noise=Gnoise(std=0.005))
    #     joint_vel_right = ObsTerm(func=joint_vel, params={"joints": None},)
    #     object_pos = ObsTerm(
    #         func=object_pos,
    #         noise=Unoise(n_min=0.0, n_max=0.01),
    #         params={"object_id": 0}
    #     )
    #     ee_pose_left = ObsTerm(func=ee_pose, params={"ee_name": "palm_link", "asset_cfg": SceneEntityCfg("robot_left")})
    #     joint_pos_left = ObsTerm(func=joint_pos_limit_normalized, params={"joints": None, 
    #                                                                       "joint_lower_limit": JOINT_LOWER_LIMIT_LEFT, 
    #                                                                       "joint_upper_limit": JOINT_UPPER_LIMIT_LEFT, 
    #                                                                       "asset_cfg": SceneEntityCfg("robot_left")},
    #                                                                       noise=Gnoise(std=0.005))
    #     joint_vel_left = ObsTerm(func=joint_vel, params={"joints": None, "asset_cfg": SceneEntityCfg("robot_left")},)
    #     drawer_handle_pos = ObsTerm(func=ee_pose, params={"ee_name": "handle_grip", "asset_cfg": SceneEntityCfg("drawer")})
    #     drawer_joint_pos = ObsTerm(func=joint_pos_limit_normalized, params={"joints": None, "asset_cfg": SceneEntityCfg("drawer")}, )
    #     last_action = ObsTerm(func=last_action)

    #     def __post_init__(self):
    #         self.enable_corruption = False
    #         self.concatenate_terms = True

    # @configclass
    # class VisionCfg(ObsGroup):
    #     """Observations for vision group."""

    #     # -- robot terms (order preserved)
        # rgb_image = ObsTerm(func=rgb_image, params={"camera_name": ["cam_1"]})
    #     def __post_init__(self):
    #         self.enable_corruption = False
    #         self.concatenate_terms = True

    # @configclass
    # class PointCloudCfg(ObsGroup):
    #     """Observations for point cloud group."""

    #     # -- robot terms (order preserved)
    #     point_cloud = ObsTerm(func=point_cloud, params={"camera_name": ["cam_1"], 
    #                                                     "wrist_cam_name": [], 
    #                                                     "crop_range": [[-0.14, 0.4], [-0.6, 0.6], [0.015, 0.5]],
    #                                                     "max_points": 2048, 
    #                                                     "downsample": "random",
    #                                                     "add_noise": True})
    #     # point_cloud_right = ObsTerm(func=point_cloud, params={"camera_name": ["cam_1"], 
    #     #                                                 "wrist_cam_name": [], 
    #     #                                                 "crop_range": [[-0.14, 0.4], [-0.6, 0.1], [0.015, 0.5]],
    #     #                                                 "max_points": 2048, 
    #     #                                                 "downsample": "random",
    #     #                                                 "add_noise": True})
    #     # point_cloud_left = ObsTerm(func=point_cloud, params={"camera_name": ["cam_1"], 
    #     #                                                 "wrist_cam_name": [], 
    #     #                                                 "crop_range": [[-0.14, 0.4], [-0.0, 0.6], [0.015, 0.5]],
    #     #                                                 "max_points": 2048, 
    #     #                                                 "downsample": "random",
    #     #                                                 "add_noise": True})
    #     def __post_init__(self):
    #         self.enable_corruption = False
    #         self.concatenate_terms = True

    # @configclass
    # class PolicyCfg(ObsGroup):
    #     """Observations for policy group."""

    #     # -- robot terms (order preserved)
    #     joint_pos_right = ObsTerm(func=joint_pos_limit_normalized, params={"joints": None, 
    #                                                                        "joint_lower_limit": JOINT_LOWER_LIMIT, 
    #                                                                        "joint_upper_limit": JOINT_UPPER_LIMIT,},
    #                                                                        noise=Gnoise(std=0.01))
    #     last_action_right = ObsTerm(func=last_action_side, params={"side": "right"})
    #     joint_pos_left = ObsTerm(func=joint_pos_limit_normalized, params={"joints": None, 
    #                                                                       "joint_lower_limit": JOINT_LOWER_LIMIT_LEFT, 
    #                                                                       "joint_upper_limit": JOINT_UPPER_LIMIT_LEFT, 
    #                                                                       "asset_cfg": SceneEntityCfg("robot_left")},
    #                                                                       noise=Gnoise(std=0.01))
    #     last_action_left = ObsTerm(func=last_action_side, params={"side": "left"})
    #     def __post_init__(self):
    #         self.enable_corruption = True
    #         self.concatenate_terms = True

    # observation groups
    # critic: CriticCfg = CriticCfg()
    # vision: VisionCfg = VisionCfg()
    # point_cloud: PointCloudCfg = PointCloudCfg()
    policy: PolicyCfg = PolicyCfg()


@configclass
class InsertDrawerActionsCfg:
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
class InsertDrawerTerminationsCfg(BaseTerminationsCfg):
    out_of_space = DoneTerm(
        func=drawer.obj_out_space, params={"asset_cfg": SceneEntityCfg("robot")}
    )
    max_consecutive_success = DoneTerm(
        func=drawer.max_consecutive_success, params={"num_success": 1}
    )


@configclass
class InsertDrawerRewardsCfg(BaseRewardsCfg):
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
    object_goal_tracking = RewTerm(func=object_goal_distance,
                                   params={"command_name": "target_pos", "object_id": 0, "sensor_names": ["contact_sensors_0", "contact_sensors_1", "contact_sensors_2", "contact_sensors_3"],},
                                   weight=0.0,
                                   )
    object_in_drawer = RewTerm(func=drawer.if_in_drawer,
                             params={"object_id": 0, "sensor_names": ["contact_sensors_0", "contact_sensors_1", "contact_sensors_2", "contact_sensors_3"]},
                             weight=0.0,
                             )
    reset_robot_joint_pos = RewTerm(func=drawer.robot_goal_distance, 
                              params={"target_pos": [-0.1277, -0.3174,  1.2583], 
                                      "target_link": "palm_link"}, 
                                      weight=0.0)           
    
    reaching_handle = RewTerm(func=drawer.drawer_handle_robot_distance, 
                              params={"weight": [1.0, 1.0], 
                                      "link_name": ["if5", "mf5"], 
                                      "asset_cfg": SceneEntityCfg("robot_left")}, 
                                      weight=0.0)
    moving_drawer = RewTerm(func=drawer.drawer_move, 
                            params={"joints": ["base_drawer_joint"], "asset_cfg": SceneEntityCfg("drawer"), "sensor_names": ["contact_sensors_0_left"]},
                            weight=0.0)
    moving_drawer_inside = RewTerm(func=drawer.drawer_move_inside, 
                            params={"joints": ["base_drawer_joint"], "asset_cfg": SceneEntityCfg("drawer"), "sensor_names": ["contact_sensors_0_left"]},
                            weight=0.0)
    success_bonus = RewTerm(func=drawer.success_bonus,
                            params={},
                            weight=0.0,
                            )
    collision_to_table = RewTerm(func=collision_penalty,
                                params={"sensor_names": ["contact_sensors_0", "contact_sensors_1", "contact_sensors_2", "contact_sensors_3"]},
                                weight=0.0,
                                )
    collision_to_drawer = RewTerm(func=collision_penalty,
                                params={"sensor_names": ["contact_sensors_0_left", "contact_sensors_1_left", "contact_sensors_2_left", "contact_sensors_3_left"]},
                                weight=0.0,
                                )
    # symmetry terms
    reaching_object_symmetry = RewTerm(func=object_robot_distance,
                                      params={"weight": [1.0, 1.0, 1.0, 1.5], 
                                      "link_name": ["if5", "mf5", "pf5", "th5"], 
                                      "object_id": 0,
                                      "asset_cfg": SceneEntityCfg("robot_left")}, 
                                      weight=0.0)
    object_lifting_symmetry = RewTerm(func=lift_distance,
                             params={"command_name": "target_pos", "object_id": 0, "sensor_names": ["contact_sensors_0_symmetry", "contact_sensors_1_symmetry", "contact_sensors_2_symmetry", "contact_sensors_3_symmetry"]},
                             weight=0.0,
                             )
    object_goal_tracking_symmetry = RewTerm(func=object_goal_distance,
                                   params={"command_name": "target_pos", "object_id": 0, "sensor_names": ["contact_sensors_0_symmetry", "contact_sensors_1_symmetry", "contact_sensors_2_symmetry", "contact_sensors_3_symmetry"],},
                                   weight=0.0,
                                   )
    object_in_drawer_symmetry = RewTerm(func=drawer.if_in_drawer,
                             params={"object_id": 0, "sensor_names": ["contact_sensors_0_symmetry", "contact_sensors_1_symmetry", "contact_sensors_2_symmetry", "contact_sensors_3_symmetry"]},
                             weight=0.0,
                             )
    reset_robot_joint_pos_symmetry = RewTerm(func=drawer.robot_goal_distance, 
                              params={"target_pos": [-0.1277, 0.3174,  1.2583], 
                                      "target_link": "palm_link",
                                      "asset_cfg": SceneEntityCfg("robot_left")}, 
                                      weight=0.0)           
    
    reaching_handle_symmetry = RewTerm(func=drawer.drawer_handle_robot_distance, 
                              params={"weight": [1.0, 1.0], 
                                      "link_name": ["if5", "mf5"], 
                                      "asset_cfg": SceneEntityCfg("robot")}, 
                                      weight=0.0)
    moving_drawer_symmetry = RewTerm(func=drawer.drawer_move, 
                            params={"joints": ["base_drawer_joint"], "asset_cfg": SceneEntityCfg("drawer"), "sensor_names": ["contact_sensors_0_left_symmetry"]},
                            weight=0.0)
    moving_drawer_inside_symmetry = RewTerm(func=drawer.drawer_move_inside, 
                            params={"joints": ["base_drawer_joint"], "asset_cfg": SceneEntityCfg("drawer"), "sensor_names": ["contact_sensors_0_left_symmetry"]},
                            weight=0.0)
    collision_to_table_symmetry = RewTerm(func=collision_penalty,
                                params={"sensor_names": ["contact_sensors_0_symmetry", "contact_sensors_1_symmetry", "contact_sensors_2_symmetry", "contact_sensors_3_symmetry"]},
                                weight=0.0,
                                )
    collision_to_drawer_symmetry = RewTerm(func=collision_penalty,
                                params={"sensor_names": ["contact_sensors_0_left_symmetry", "contact_sensors_1_left_symmetry", "contact_sensors_2_left_symmetry", "contact_sensors_3_left_symmetry"]},
                                weight=0.0,
                                )

@configclass
class InsertDrawerEnvCfg(BaseEnvCfg):
    name: str = "InsertDrawer"
    scene = InsertDrawerSceneCfg(num_envs=4096, env_spacing=3.0)
    events = InsertDrawerEventCfg()
    commands = InsertDrawerCommandsCfg()
    observations = InsertDrawerObservationsCfg()
    actions = InsertDrawerActionsCfg()
    terminations = InsertDrawerTerminationsCfg()
    rewards = InsertDrawerRewardsCfg()
    num_object = 1
    action_dim = 44 # arm + hand
    action_scale: list = [1.0] * action_dim
    #                      [0.05, 0.05, 0.05, 0.05, 0.05, 0.05,
    #                         0.03, 0.03, 0.03, 0.03, 
    #                         0.03, 0.03, 0.03, 0.03, 
    #                         0.03, 0.03, 0.03, 0.015,
    #                         0.03, 0.03, 0.03, 0.03,
    #                       0.05, 0.05, 0.05, 0.05, 0.05, 0.05,
    #                         0.03, 0.03, 0.03, 0.03, 
    #                         0.03, 0.03, 0.03, 0.03, 
    #                         0.03, 0.03, 0.03, 0.015,
    #                         0.03, 0.03, 0.03, 0.03]  # jth3 needs smaller rate

    visualize_marker: bool = False

    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        self.viewer.eye = (-1.5, 0.0, 1.5)