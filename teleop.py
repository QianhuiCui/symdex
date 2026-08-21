from isaaclab.app import AppLauncher

app_launcher = AppLauncher({"headless": False, "enable_cameras": True, "raw_data": True})
simulation_app = app_launcher.app

import torch
import hydra
from omegaconf import DictConfig, OmegaConf
import gymnasium as gym
import threading
import time

from symdex.utils.common import set_random_seed, capture_keyboard_interrupt, preprocess_cfg
from symdex.utils.trajectory_utils import get_obs, as_flag, to_str
from symdex.utils.trajectory_logger import TrajectoryLogger, SphereWriter
from symdex.env.tasks.manager_based_env_cfg import * 
from symdex.utils.rl_env_wrapper import VecEnvWrapper
from symdex.utils.symmetry import SymmetryManager
import sys
sys.path.append("/home/qianhui/dex_bimanual_telep/scripts")
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
    pending_init_meta = get_episode_init_meta(env, cfg)
    send_msg("robot_reset", {"t": time.time()})

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

            delta_cmd = make_clean_env_action_from_qtarget(env, q_target)
            action_buf.zero_()
            action_buf[0].copy_(torch.from_numpy(delta_cmd).to(action_buf.device))
            action_to_log = delta_cmd 
        else:
            action_buf.zero_()
            action_to_log = np.zeros(env.action_space.shape[1], dtype=np.float32)

        obs, _, reset, extras = env.step(action_buf)

        # ---- flags ----
        terminated = as_flag(extras.get("terminated"))
        truncated = as_flag(extras.get("time_outs"))

        # ---- start recording if trigger arrives & episode metadata ----
        if use_logger and (not episode_started):
            episode_started = True
            cur_lang = to_str(extras.get("language_instruction", ""))
            logger.start_episode(
                language_instruction=cur_lang,
                init_meta=pending_init_meta,
            )
            print("[Teleop] Recording started.")

        if use_logger and episode_started:
            logger.add_traj(
                observation=get_obs(obs),
                action=action_to_log,
                terminated=bool(terminated),
                truncated=bool(truncated),
            )

        if bool(terminated or truncated or as_flag(reset)):
            if terminated and not truncated:
                print("[Teleop] Episode finished: SUCCESS or terminal condition reached.")
            elif truncated:
                print("[Teleop] Episode finished: TIMEOUT (time limit reached).")
            else:
                print("[Teleop] Episode finished: ENV RESET triggered.")

            pending_init_meta = get_episode_init_meta(env, cfg)
            send_msg("robot_reset", {"t": time.time()})
            last_reset_recv_time = time.monotonic()

            with _data_lock:
                teleop_joint_data["value"] = None
                teleop_joint_data["timestamp"] = None
                teleop_joint_data["recv_time"] = None
                match_control["start"] = 0
                match_control["recv_time"] = None

            if use_logger and episode_started:
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