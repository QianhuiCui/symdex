from isaaclab.app import AppLauncher

app_launcher = AppLauncher({"headless": False, "enable_cameras": True, "raw_data": True})
simulation_app = app_launcher.app

import torch
import hydra
from omegaconf import DictConfig, OmegaConf
import gymnasium as gym
import threading
import time 
import hashlib 
import json

from symdex.utils.common import set_random_seed, capture_keyboard_interrupt, preprocess_cfg
from symdex.utils.trajectory_utils import get_obs, now_ms, as_flag, to_str
from symdex.utils.trajectory_logger import TrajectoryLogger, SphereWriter
from symdex.env.tasks.manager_based_env_cfg import * 
from symdex.utils.rl_env_wrapper import VecEnvWrapper
# from symdex.utils.delta_action_scaler import DeltaActionScaler
# from symdex.utils.action_scaler import ActionScaler, ActionScalerCfg
from symdex.utils.symmetry import SymmetryManager
import sys
sys.path.append("/home/qianhui/Desktop/dex_bimanual_telep/scripts")
from zmq_utils import recv_msg, send_msg

import symdex


teleop_joint_data = {"value": None, "timestamp": None, "recv_time": None}
teleop_target_poses = {"right": None, "left": None}
match_control = {"start": 0, "recv_time": None}
_data_lock = threading.Lock()

def recv_teleop():
    print("[Teleop Receiver] Listening for teleoperation data...")
    while simulation_app.is_running():
        topic, msg = recv_msg(["teleop_joint_state", "human_wrist_poses", "match_control"])
        if msg is None:
            time.sleep(0.01)
            continue

        now_local = time.monotonic()
        with _data_lock:
            if topic == "human_wrist_poses":
                teleop_target_poses["right"] = msg.get("right", None)
                teleop_target_poses["left"] = msg.get("left", None)
            elif topic == "teleop_joint_state":
                teleop_joint_data["value"] = msg.get("value", None)
                teleop_joint_data["timestamp"] = msg.get("timestamp", None) 
                teleop_joint_data["recv_time"] = now_local
            elif topic == "match_control":
                if int(msg.get("start", 0) or 0) == 1:
                    match_control["start"] = 1
                    match_control["recv_time"] = now_local

        time.sleep(0.001)

# def get_action_terms(env):
#     action_manager = env.unwrapped.action_manager
#     right_term = action_manager.get_term("arm_hand_action")
#     left_term = action_manager.get_term("arm_hand_action_left")
#     return right_term, left_term

def get_action_manager_state(env):
    # right_term, left_term = get_action_terms(env)
    action_manager = env.unwrapped.action_manager
    right_term = action_manager.get_term("arm_hand_action")
    left_term = action_manager.get_term("arm_hand_action_left")
    q_init_right = right_term.init_joint_pos[0].detach().cpu().numpy().astype(np.float32)
    q_init_left = left_term.init_joint_pos[0].detach().cpu().numpy().astype(np.float32)
    del_right = right_term.del_action[0].detach().cpu().numpy().astype(np.float32)
    del_left = left_term.del_action[0].detach().cpu().numpy().astype(np.float32)
    q_init = np.concatenate([q_init_right, q_init_left], axis=0)
    del_action = np.concatenate([del_right, del_left], axis=0)
    return q_init, del_action

def clip_qtarget_to_joint_limits(q_target: np.ndarray) -> np.ndarray:
    joint_lower = np.asarray(JOINT_LOWER_LIMIT + JOINT_LOWER_LIMIT_LEFT, dtype=np.float32)
    joint_upper = np.asarray(JOINT_UPPER_LIMIT + JOINT_UPPER_LIMIT_LEFT, dtype=np.float32)
    return np.clip(q_target, joint_lower, joint_upper).astype(np.float32)

def make_clean_env_action_from_qtarget(env, q_target: np.ndarray):
    """
    Convert absolute teleop joint target into the normalized action expected by:
    VecEnvWrapper -> BaseEnv.action_scale -> EMACumulativeRelativeJointPositionAction.
    Returns:
        action_exec: action to send into env.step and save into H5 offline_data/actions.
    """
    q_target = np.asarray(q_target, dtype=np.float32).reshape(-1)
    if q_target.shape[0] != 44:
        raise RuntimeError(f"q_target must have dim 44, got {q_target.shape}")

    q_target = clip_qtarget_to_joint_limits(q_target)
    q_init, del_action = get_action_manager_state(env)
    action_scale = env.unwrapped._scale.detach().cpu().numpy().astype(np.float32)
    # action_scale = np.maximum(np.abs(action_scale), 1e-6)

    # Required scaled increment before BaseEnv action_scale.
    scaled_increment = q_target - q_init - del_action
    # Convert to normalized env action.
    action_raw = scaled_increment / action_scale
    # Actual action sent to VecEnvWrapper / env.step.
    action_exec = np.clip(action_raw, -1.0, 1.0).astype(np.float32)
    return action_exec

def get_episode_init_meta(env, cfg):
    meta = {}
    for obj_id in [0, 1, 2]:
        obj = env.unwrapped.scene[f"object_{obj_id}"]
        meta[f"object_{obj_id}_init_pos_w"] = obj.data.root_pos_w[0, :3].detach().cpu().numpy().astype(np.float32)
        meta[f"object_{obj_id}_init_quat_w"] = obj.data.root_quat_w[0, :4].detach().cpu().numpy().astype(np.float32)
    robot = env.unwrapped.scene["robot"]
    robot_left = env.unwrapped.scene["robot_left"]
    meta["robot_init_qpos"] = robot.data.joint_pos[0, :].detach().cpu().numpy().astype(np.float32)
    meta["robot_left_init_qpos"] = robot_left.data.joint_pos[0, :].detach().cpu().numpy().astype(np.float32)

    if hasattr(env.unwrapped, "camera_offset") and "cam_1" in env.unwrapped.camera_offset:
        meta["cam_1_init_pos"] = env.unwrapped.camera_offset["cam_1"]["pos"][0].detach().cpu().numpy().astype(np.float32)
        meta["cam_1_init_quat"] = env.unwrapped.camera_offset["cam_1"]["orientation"][0].detach().cpu().numpy().astype(np.float32)

    meta["seed"] = np.asarray([cfg.seed if cfg.seed is not None else -1], dtype=np.int64)
    return meta
    
def get_current_qpos(env):
    robot = env.unwrapped.scene["robot"]
    robot_left = env.unwrapped.scene["robot_left"]
    qpos = robot.data.joint_pos[0, :].detach().cpu().numpy().astype(np.float32)
    qpos_left = robot_left.data.joint_pos[0, :].detach().cpu().numpy().astype(np.float32)
    return np.concatenate([qpos, qpos_left], axis=0)

@hydra.main(
    config_path=symdex.LIB_PATH_PATH.joinpath("cfg").as_posix(),
    config_name="default"
)
def main(cfg: DictConfig):
    torch.set_printoptions(sci_mode=False, precision=3)
    if cfg.seed is None or int(cfg.seed) < 0:
        cfg.seed = int(time.time() * 1000) % (2**31 - 1)
    print(f"[Teleop] Using seed: {cfg.seed}")
    set_random_seed(cfg.seed)
    capture_keyboard_interrupt()

    cfg, env_cfg = preprocess_cfg(cfg)

    env = gym.make(cfg.env_name, cfg=env_cfg)
    env = VecEnvWrapper(env, rl_device=cfg.rl_device)
    symmetry_manager = SymmetryManager(cfg.task.multi.TD3BC, cfg.task.symmetry.symmetric_envs)
    initial_obs, _ = env.reset()
    pending_init_meta = get_episode_init_meta(env, cfg)
    send_msg("robot_reset", {"t": time.time()})
    prev_obs = get_obs(initial_obs)
    # delta_scaler = DeltaActionScaler()
    # delta_scaler.reset()
    # absolute actions 
    # joint_lower = np.concatenate([JOINT_LOWER_LIMIT, JOINT_LOWER_LIMIT_LEFT])
    # joint_upper = np.concatenate([JOINT_UPPER_LIMIT, JOINT_UPPER_LIMIT_LEFT])
    # action_scaler_cfg = ActionScalerCfg(max_delta=0.03, deadband=0.005, ema_alpha=0.75)
    # action_scaler = ActionScaler(env=env, env_cfg=env_cfg, cfg=action_scaler_cfg, joint_lower=joint_lower, joint_upper=joint_upper)
    # action_scaler.reset()

    with _data_lock:
        teleop_joint_data["value"] = None
        teleop_joint_data["timestamp"] = None
        teleop_joint_data["recv_time"] = None 
        match_control["start"] = 0
        match_control["recv_time"] = None

    last_reset_recv_time = time.monotonic()
    action_buf = torch.empty((env.num_envs, env.action_space.shape[1]), device=cfg.rl_device, dtype=torch.float32)

    threading.Thread(target=recv_teleop, daemon=True).start()

    # --- logger setup ---
    use_logger = cfg.logger.enable
    logger = None
    if use_logger:
        logger = TrajectoryLogger(task_name=cfg.task.env_name)
        print("[Teleop] Logger enabled.")
    # Recording state (independent from teleop)
    episodes_saved = 0
    episode_started = False
    cur_lang = None

    rew_term_dict = env_cfg.rewards.to_dict()
    rew_names = list(rew_term_dict.keys())
    rew_weights = np.asarray([float(rew_term_dict[name].get("weight", 0.0)) for name in rew_names], dtype=np.float32)
    sig = json.dumps({"rew_names": rew_names, "rew_weights": rew_weights.tolist()},
                     separators=(",", ":"),
                     sort_keys=True)
    rew_cfg_hash = hashlib.sha256(sig.encode("utf-8")).hexdigest()

    # Visualization
    sphere_writer = SphereWriter(device=cfg.rl_device)

    print("[IsaacLab Teleop] Started. Waiting for teleop input...")

    while simulation_app.is_running():
        with _data_lock:
            q_value = teleop_joint_data["value"]
            q_ts_recv = teleop_joint_data["recv_time"]
            pose_right = teleop_target_poses["right"] 
            pose_left = teleop_target_poses["left"]
            match_control["start"] = 0  # consume trigger once
            # consume joint packet immediately (so we never replay old data if sender stalls)
            teleop_joint_data["value"] = None
            teleop_joint_data["timestamp"] = None
            teleop_joint_data["recv_time"] = None

        # ---- visualization spheres ----
        if pose_right is not None:
            sphere_writer.write(env, "target_sphere", np.array(pose_right, dtype=np.float32))
        if pose_left is not None:
            sphere_writer.write(env, "target_sphere_left", np.array(pose_left, dtype=np.float32))
                    
        if q_value is None or q_ts_recv is None:
            time.sleep(0.001)
            continue
        # Gate
        if q_ts_recv < last_reset_recv_time:  # or q_ts_sender < last_reset_recv_time:
            time.sleep(0.001)
            continue

        if (q_value is not None) and (q_ts_recv is not None) and (q_ts_recv >= last_reset_recv_time):
            # delta_cmd
            q_target = np.asarray(q_value, dtype=np.float32).reshape(-1)
            q_curr = get_current_qpos(env)
            if len(q_target) != len(q_curr):
                raise RuntimeError(f"teleop_joint_state length {len(q_target)} != expected {len(q_curr)}.")

            # delta_cmd = (q_target - q_curr).astype(np.float32)
            # delta_cmd = delta_scaler.process(q_target, q_curr)
            delta_cmd = make_clean_env_action_from_qtarget(env, q_target)
            action_buf.zero_()
            action_buf[0].copy_(torch.from_numpy(delta_cmd).to(action_buf.device))
            action_to_log = delta_cmd
            # absolute actions 
            # abs_cmd = q_target.astype(np.float32)
            # abs_cmd = action_scaler.process(abs_cmd)
            # action_buf.zero_()
            # action_buf[0].copy_(torch.from_numpy(abs_cmd).to(action_buf.device))
            # action_to_log = abs_cmd
        else:
            action_buf.zero_()
            action_to_log = np.zeros(env.action_space.shape[1], dtype=np.float32)

        next_obs, rew, reset, extras = env.step(action_buf)

        # ---- obs ----
        obs_vec = get_obs(next_obs)

        # ---- reward & flags ----
        # reward = float(rew[0].item())
        reward_right, reward_left = symmetry_manager.get_multi_agent_rew(extras["detailed_reward"], env.unwrapped.symmetry_tracker)
        reward_right = reward_right[0].item()
        reward_left = reward_left[0].item()
        terminated = as_flag(extras.get("terminated"))
        truncated = as_flag(extras.get("time_outs"))

        # ---- reward terms ----
        detailed_reward = extras["detailed_reward"]  # dict: name -> tensor([B])
        rew_terms = np.asarray([float(detailed_reward[name][0].item()) for name in rew_names], dtype=np.float32)
        if rew_terms.shape[0] != rew_weights.shape[0]:
            raise RuntimeError(f"[Teleop] reward_terms K={rew_terms.shape[0]} != rew_weights K={rew_weights.shape[0]}")

        # ---- start recording if trigger arrives & episode metadata ----
        # if trig_start == 1 and use_logger and (not episode_started):
        if use_logger and (not episode_started):
            episode_started = True
            cur_lang = to_str(extras.get("language_instruction", ""))
            logger.start_episode(
                language_instruction=cur_lang,
                rew_cfg_hash=rew_cfg_hash,
                rew_names=rew_names,
                rew_weights=rew_weights,
                init_meta=pending_init_meta,
            )
            print("[Teleop] Recording started.")
                
        if use_logger and episode_started:
            # if cur_lang == "":
            #     cur_lang = to_str(extras.get("language_instruction", ""))
            logger.add_transition(
                observation=prev_obs,
                # action=delta_cmd,
                action=action_to_log,
                # reward=reward,
                reward_right=reward_right,
                reward_left=reward_left,
                next_observation=obs_vec,
                terminated=bool(terminated),
                truncated=bool(truncated),
                rew_terms=rew_terms,
            )
        prev_obs = obs_vec

        if bool(terminated or truncated or as_flag(reset)):
            if terminated and not truncated:
                print("[Teleop] Episode finished: SUCCESS or terminal condition reached.")
            elif truncated:
                print("[Teleop] Episode finished: TIMEOUT (time limit reached).")
            else:
                print("[Teleop] Episode finished: ENV RESET triggered.")

            initial_obs, _ = env.reset()
            pending_init_meta = get_episode_init_meta(env, cfg)
            send_msg("robot_reset", {"t": time.time()})
            prev_obs = get_obs(initial_obs)
            # delta_scaler.reset()
            # action_scaler.reset()
            last_reset_recv_time = time.monotonic()

            with _data_lock:
                teleop_joint_data["value"] = None
                teleop_joint_data["timestamp"] = None
                teleop_joint_data["recv_time"] = None
                match_control["start"] = 0
                match_control["recv_time"] = None

            if use_logger and episode_started:
                # logger.save_episode()
                logger.save_episode(success=bool(terminated and not truncated))
                episodes_saved += 1
                if episodes_saved >= cfg.max_episodes:
                    break
            episode_started = False
            cur_lang = None

    if use_logger:
        logger.close()
    env.close()
    print("[IsaacLab Teleop] Closed cleanly.")    


if __name__ == "__main__":
    main()
    simulation_app.close()