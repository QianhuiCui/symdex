
from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.actuators.actuator_cfg import ImplicitActuatorCfg, IdealPDActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObject, RigidObjectCfg
from isaaclab.sensors import CameraCfg, ContactSensorCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise
import isaaclab.envs.mdp as mdp
from isaaclab.utils.noise import AdditiveGaussianNoiseCfg as Gnoise
from isaaclab.sim.simulation_cfg import PhysxCfg, SimulationCfg
from isaaclab.sim.spawners.materials.physics_materials_cfg import RigidBodyMaterialCfg

import symdex
from symdex.env.mdps.reset_mdps import *
from symdex.env.mdps.reward_mdps import *


JOINT_LOWER_LIMIT = [-6.283, -2.304, -4.224, -6.283, -2.164, -6.283,
                    # jif1, jmf1, jpf1, jth1
                    -0.05, -0.05, -0.570, 0.364,
                    # jif2, jmf2, jpf2, jth2
                    -0.296, -0.296, -0.296, -0.205,
                    # jif3, jmf3, jpf3, jth3
                    -0.274, -0.274, -0.274, -0.290,
                    # jif4, jmf4, jpf4, jth4
                    -0.327, -0.327, -0.327, -0.262]
JOINT_UPPER_LIMIT = [6.283, 2.304, 0.061, 6.283, 2.164, 6.283,
                    # jif1, jmf1, jpf1, jth1
                    0.570, 0.05, 0.05, 1.497,
                    # jif2, jmf2, jpf2, jth2
                    1.710, 1.710, 1.710, 1.130, 
                    # jif3, jmf3, jpf3, jth3
                    1.809, 1.809, 1.809, 1.633, 
                    # jif4, jmf4, jpf4, jth4
                    1.718, 1.718, 1.718, 1.820]
JOINT_LOWER_LIMIT_LEFT = [-6.283, -2.304, -4.224, -6.283, -2.164, -6.283,
                    # jif1, jmf1, jpf1, jth1
                    -0.570, -0.05, -0.05, 0.364,
                    # jif2, jmf2, jpf2, jth2
                    -0.296, -0.296, -0.296, -0.205,
                    # jif3, jmf3, jpf3, jth3
                    -0.274, -0.274, -0.274, -0.290,
                    # jif4, jmf4, jpf4, jth4
                    -0.327, -0.327, -0.327, -0.262]
JOINT_UPPER_LIMIT_LEFT = [6.283, 2.304, 0.061, 6.283, 2.164, 6.283,
                    # jif1, jmf1, jpf1, jth1
                    0.05, 0.05, 0.570, 1.497,
                    # jif2, jmf2, jpf2, jth2
                    1.710, 1.710, 1.710, 1.130, 
                    # jif3, jmf3, jpf3, jth3
                    1.809, 1.809, 1.809, 1.633, 
                    # jif4, jmf4, jpf4, jth4
                    1.718, 1.718, 1.718, 1.820]

@configclass
class BaseSceneCfg(InteractiveSceneCfg):
    """Configuration for the scene with a robotic arm."""

    # world
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -0.82)),
    )

    # lights
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=2500.0),
    )

    # table
    table: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Table",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{symdex.LIB_PATH}/assets/object/table.usd",
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True,
                disable_gravity=False,
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=1,
                max_depenetration_velocity=10.0,
            ),
            scale=(1.0, 1.0, 1.0),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0), rot=(0.70710678, 0, 0., 0.70710678)),
    )

    replicate_physics = False

@configclass
class BaseCommandsCfg:
    """Command terms for the MDP."""
    pass

@configclass
class BaseActionsCfg:
    """Action specifications for the MDP."""
    pass


@configclass
class BaseObservationsCfg:
    """Observation specifications for the MDP."""
    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""
        # -- robot terms (order preserved)
        joint_pos = ObsTerm(func=mdp.joint_pos_limit_normalized, noise=Gnoise(std=0.005))
        # -- action terms
        last_action = ObsTerm(func=mdp.last_action)
        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True
    # observation groups
    policy: PolicyCfg = PolicyCfg()

@configclass
class BaseEventCfg:
    """Configuration for events."""
    reset_robot_joints = EventTerm(
        func=reset_joints_by_symmetry,
        mode="reset",
        params={
            "position_range": (1.0, 1.0),
            "velocity_range": (0.0, 0.0),
        },
    )

@configclass
class BaseRewardsCfg:
    """Reward terms for the MDP."""
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

@configclass
class BaseTerminationsCfg:
    """Termination terms for the MDP."""
    time_out = DoneTerm(func=mdp.time_out, time_out=True)

@configclass
class BaseCurriculumCfg:
    """Curriculum terms for the MDP."""
    pass

@configclass
class BaseEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the reach end-effector pose tracking environment."""
    name: str = "Base"
    # Scene settings
    scene: BaseSceneCfg = BaseSceneCfg(num_envs=4096, env_spacing=2.0)
    # Basic settings
    observations: BaseObservationsCfg = BaseObservationsCfg()
    actions: BaseActionsCfg = BaseActionsCfg()
    # EE positions, EE rot, Hand joint angle
    action_scale = None

    commands: BaseCommandsCfg = BaseCommandsCfg()
    # MDP settings
    rewards: BaseRewardsCfg = BaseRewardsCfg()
    terminations: BaseTerminationsCfg = BaseTerminationsCfg()
    events: BaseEventCfg = BaseEventCfg()
    curriculum: BaseCurriculumCfg = BaseCurriculumCfg()

    visualize_marker = False
    visualizer_scale = (0.1, 0.1, 0.1)

    num_object = None

    sim: SimulationCfg = SimulationCfg(
        physics_material=RigidBodyMaterialCfg(
            static_friction=1.5,
            dynamic_friction=1.0,
            restitution=0.0,
            restitution_combine_mode=min,
        ),
        physx=PhysxCfg(
            gpu_max_rigid_contact_count=2**24,
            gpu_max_rigid_patch_count=2**24,
        ),
    )

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 6
        self.episode_length_s = 8.3333
        self.viewer.eye = (3.5, 3.5, 3.5)
        # simulation settings
        self.sim.dt = 1.0 / 120.0
        self.sim.render_interval = self.decimation

