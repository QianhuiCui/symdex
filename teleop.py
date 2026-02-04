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
from symdex.env.tasks.manager_based_env_cfg import * 
from symdex.utils.trajectory_logger import TrajectoryLogger, SphereWriter
from symdex.utils.action_scaler import ActionScaler, ActionScalerCfg
from symdex.utils.rl_env_wrapper import VecEnvWrapper
import sys
sys.path.append("/home/qianhui/Desktop/dex_bimanual_telep/scripts")
from zmq_utils import recv_msg

import symdex


teleop_joint_data = {"value": None, "timestamp": None, "recv_time": None}
teleop_target_poses = {"right": None, "left": None}
_data_lock = threading.Lock()

def recv_teleop():
    print("[Teleop Receiver] Listening for teleoperation joint data...")
    while simulation_app.is_running():
        topic, msg = recv_msg(["teleop_joint_state", "human_wrist_poses"])
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

        time.sleep(0.001)

def do_reset(env, scaler, get_obs_fn, data_lock, teleop_joint_data):
    initial_obs, _ = env.reset()
    scaler.reset()
    prev_obs = get_obs_fn(initial_obs)

    with data_lock:
        teleop_joint_data["value"] = None
        teleop_joint_data["timestamp"] = None
        teleop_joint_data["recv_time"] = None

    reset_recv_time = time.monotonic()
    return prev_obs, reset_recv_time


@hydra.main(
    config_path=symdex.LIB_PATH_PATH.joinpath("cfg").as_posix(),
    config_name="default"
)
def main(cfg: DictConfig):
    torch.set_printoptions(sci_mode=False, precision=3)
    set_random_seed(cfg.seed)
    capture_keyboard_interrupt()

    cfg, env_cfg = preprocess_cfg(cfg)

    env = gym.make(cfg.env_name, cfg=env_cfg)
    env = VecEnvWrapper(env, rl_device=cfg.rl_device) 
    scaler = ActionScaler(env, env_cfg, ActionScalerCfg(warmup_steps=20, max_delta=0.03, deadband=0.005, ema_alpha=0.75), 
                          joint_lower=np.concatenate((JOINT_LOWER_LIMIT, JOINT_LOWER_LIMIT_LEFT)),
                          joint_upper=np.concatenate((JOINT_UPPER_LIMIT, JOINT_UPPER_LIMIT_LEFT))) 

    prev_obs, last_reset_recv_time = do_reset(env, scaler, get_obs, _data_lock, teleop_joint_data)
    action_buf = torch.empty((env.num_envs, env.action_space.shape[1]), device=cfg.rl_device, dtype=torch.float32)

    threading.Thread(target=recv_teleop, daemon=True).start()

    # Initialize logger if enabled 
    episodes_saved = 0 
    episode_started = False
    cur_lang = None
    max_episodes = getattr(cfg, "max_episodes", None)
    use_logger = getattr(cfg, "logger", False)
    if use_logger:
        logger = TrajectoryLogger(task_name=cfg.task.env_name)
        print("[Teleop] Logger enabled.")

    print("[IsaacLab Teleop] Started. Waiting for teleop input...") 
    rew_term_dict = env_cfg.rewards.to_dict()
    rew_names = list(rew_term_dict.keys())
    rew_weights = np.asarray([float(rew_term_dict[name].get("weight", 0.0)) for name in rew_names],
                             dtype=np.float32)
    sig = json.dumps({"rew_names": rew_names, "rew_weights": rew_weights.tolist()},
                     separators=(",", ":"),
                     sort_keys=True)
    rew_cfg_hash = hashlib.sha256(sig.encode("utf-8")).hexdigest()

    sphere_writer = SphereWriter(device=cfg.rl_device)
    
    while simulation_app.is_running():
        with _data_lock:
            q_value = teleop_joint_data["value"]
            q_ts_sender = teleop_joint_data["timestamp"]
            q_ts_recv = teleop_joint_data["recv_time"]
            pose_right = teleop_target_poses["right"] 
            pose_left = teleop_target_poses["left"]
        
        if q_value is None or q_ts_recv is None:
            time.sleep(0.001)
            continue
        # Gate
        if q_ts_recv < last_reset_recv_time:  # or q_ts_sender < last_reset_recv_time:
            time.sleep(0.001)
            continue

        q_np = scaler.process(np.asarray(q_value, dtype=np.float32))
        action_buf[0].copy_(torch.from_numpy(q_np).to(action_buf.device))
        # q_tensor = torch.as_tensor(q_np[None, :], device=cfg.rl_device)

        next_obs, rew, reset, extras = env.step(action_buf)

        # ---- reward & flags ----
        reward = float(rew[0].item())
        terminated = as_flag(extras.get("terminated"))
        truncated = as_flag(extras.get("time_outs"))

        # ---- obs ----
        obs_vec = get_obs(next_obs)

        # ---- episode metadata ----
        if (not episode_started) and use_logger:
            cur_lang = to_str(extras.get("language_instruction", ""))
            if use_logger:
                logger.start_episode(
                    language_instruction=cur_lang,
                    rew_cfg_hash=rew_cfg_hash,
                    rew_names=rew_names,
                    rew_weights=rew_weights
                )
            episode_started = True
        
        # ---- reward terms ----
        detailed_reward = extras["detailed_reward"]  # dict: name -> tensor([B])
        rew_terms = np.asarray([float(detailed_reward[name][0].item()) for name in rew_names], dtype=np.float32)
        if rew_terms.shape[0] != rew_weights.shape[0]:
            raise RuntimeError(f"[Teleop] reward_terms K={rew_terms.shape[0]} != rew_weights K={rew_weights.shape[0]}")
        
        if use_logger:
            logger.add_transition(
                observation=prev_obs,
                action=q_np,
                reward=reward,
                next_observation=obs_vec,
                terminated=bool(terminated),
                truncated=bool(truncated),
                rew_terms=rew_terms,
            )
        prev_obs = obs_vec
        
        if bool(terminated or truncated or as_flag(reset)):
            # initial_obs, _ = env.reset()
            # scaler.reset()
            # prev_obs = get_obs(initial_obs)
            prev_obs, last_reset_recv_time = do_reset(env, scaler, get_obs, _data_lock, teleop_joint_data)

            if use_logger:
                logger.save_episode()
                episodes_saved += 1
                if (max_episodes is not None) and (episodes_saved >= max_episodes):
                    break
            episode_started = False
            cur_lang = None

        # ---- visualization spheres ----
        if pose_right is not None:
            sphere_writer.write(env, "target_sphere", np.array(pose_right, dtype=np.float32))
        if pose_left is not None:
            sphere_writer.write(env, "target_sphere_left", np.array(pose_left, dtype=np.float32))

    if use_logger:
        logger.close()
    env.close()
    print("[IsaacLab Teleop] Closed cleanly.")    


if __name__ == "__main__":
    main()
    simulation_app.close()