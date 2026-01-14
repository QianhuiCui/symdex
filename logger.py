from isaaclab.app import AppLauncher

app_launcher = AppLauncher({"headless": False, "enable_cameras": True})
simulation_app = app_launcher.app

import time
import numpy as np
import torch
import hydra
from hydra.utils import to_absolute_path
from omegaconf import DictConfig
import gymnasium as gym

from symdex.utils.common import set_random_seed, capture_keyboard_interrupt, preprocess_cfg
from symdex.utils.trajectory_utils import build_observation_dict, as_flag, to_str
from symdex.utils.action_scaler import ActionScaler, ActionScalerCfg
from symdex.env.tasks.manager_based_env_cfg import *
from symdex.utils.rl_env_wrapper import VecEnvWrapper
import symdex


def load_trajectories(action_file: str): 
    if action_file.endswith(".npz"):
        z = np.load(action_file, allow_pickle=False)
        traj_keys = sorted(z.files)
        trajectories = []
        for k in traj_keys:
            tr = z[k].astype(np.float32)
            trajectories.append(tr)
        print(f"[Offline Playback] Loaded NPZ: {action_file}, num_traj={len(trajectories)}")
        return trajectories, traj_keys
    # default: npy
    actions = np.load(action_file, allow_pickle=False).astype(np.float32)
    print(f"[Offline Playback] Loaded NPY: {action_file}, steps={actions.shape[0]}")
    return [actions], ["single_npy"]

@hydra.main(
    config_path=symdex.LIB_PATH_PATH.joinpath("cfg").as_posix(),
    config_name="default"
)
def main(cfg: DictConfig):
    torch.set_printoptions(sci_mode=False, precision=3)
    set_random_seed(cfg.seed)
    capture_keyboard_interrupt()

    cfg, env_cfg = preprocess_cfg(cfg)

    # Load trajectories
    action_file = to_absolute_path(cfg.dataset_path)
    trajectories, traj_keys = load_trajectories(action_file)
    traj_idx, step_idx = 0, 0
    cur_traj = trajectories[traj_idx]
    num_steps = cur_traj.shape[0]
    action_dim = cur_traj.shape[1]
    print(f"[Offline Playback] Current traj={traj_keys[traj_idx]} shape={cur_traj.shape}")

    env = gym.make(cfg.env_name, cfg=env_cfg)
    #  Action Scaler
    scaler = ActionScaler(env, env_cfg, ActionScalerCfg(warmup_steps=20, max_delta=0.05))
    
    env = VecEnvWrapper(env, rl_device=cfg.rl_device)
    env.reset()
    scaler.reset()
    # Logger
    use_logger = getattr(cfg, "logger", False)
    if use_logger:
        from symdex.utils.trajectory_logger import TrajectoryLogger
        logger = TrajectoryLogger(task_name=cfg.task.env_name)
        max_episodes = getattr(cfg, "max_episodes", None)
        episodes_saved = 0
        print("[Offline Playback] Logger enabled.")

    just_reset = True
    cur_lang = None

    print("[Offline Playback] Start stepping env with trajectory actions...")

    while simulation_app.is_running():
        # ---- if reach end of current trajectory: reset + switch ----
        if step_idx >= num_steps:
            print("[Offline Playback] Reached end of trajectory. Reset and switch.")
            env.reset()
            scaler.reset()
            just_reset = True
            cur_lang = None
            if use_logger:
                logger.save_episode()
                episodes_saved += 1
                if (max_episodes is not None) and (episodes_saved >= max_episodes):
                    print(f"[Offline Playback] Reached max_episodes={max_episodes}. Stopping.")
                    break
            traj_idx = (traj_idx + 1) % len(trajectories)
            cur_traj = trajectories[traj_idx]
            num_steps = cur_traj.shape[0]
            step_idx = 0
            print(f"[Offline Playback] Switch traj (traj_end) -> {traj_keys[traj_idx]} shape={cur_traj.shape}")
            continue
        
        traj_last_step = (step_idx == (num_steps - 1))

        # ---- get action ----
        q_np = np.asarray(cur_traj[step_idx], dtype=np.float32)
        q_np = scaler.scale(q_np)
        q_tensor = torch.as_tensor(q_np[None, ...], dtype=torch.float32, device=cfg.rl_device)
        step_idx += 1

        obs, rew, reset, extras = env.step(q_tensor)

        reward_scalar = float(rew[0].item() if torch.is_tensor(rew) else rew)
        succeed = as_flag(extras.get("success"))
        failed = (as_flag(reset) or traj_last_step) and (not succeed)

        obs_dict = build_observation_dict(env, obs)
        action = q_np
        action_dict = {
            "arm_hand_action_right": q_np[:22] if action_dim >= 22 else q_np,
            "arm_hand_action_left": q_np[22:44] if action_dim >= 44 else q_np[22:],
        }

        if just_reset or (cur_lang is None):
            lang_field = extras.get("language_instruction", "")
            cur_lang = to_str(lang_field)
        language_instruction = cur_lang

        # ---- log ----
        if use_logger:
            logger.add_step(
                action=action,
                action_dict=action_dict,
                observation=obs_dict,
                reward=reward_scalar,
                is_first=bool(just_reset),
                failed=bool(failed),
                succeed=bool(succeed),
                language_instruction=language_instruction,
                discount=1.0,
            )

        # Episode management
        if succeed:
            print("[Offline Playback] Task success, resetting environment.")
            env.reset()
            scaler.reset()
            just_reset = True
            cur_lang = None
            if use_logger:
                logger.save_episode()
                episodes_saved += 1
                if (max_episodes is not None) and (episodes_saved >= max_episodes):
                    print(f"[Offline Playback] Reached max_episodes={max_episodes}. Stopping.")
                    break
            traj_idx = (traj_idx + 1) % len(trajectories)
            cur_traj = trajectories[traj_idx]
            num_steps = cur_traj.shape[0]
            step_idx = 0
            print(f"[Offline Playback] Switch traj (success) -> {traj_keys[traj_idx]} shape={cur_traj.shape}")
            continue
        elif reset.any():
            print("[Offline Playback] Environment reset triggered auto-reset internally.")
            env.reset()
            scaler.reset()
            just_reset = True
            cur_lang = None
            if use_logger:
                logger.save_episode()
                episodes_saved += 1
                if (max_episodes is not None) and (episodes_saved >= max_episodes):
                    print(f"[Offline Playback] Reached max_episodes={max_episodes}. Stopping.")
                    break
            traj_idx = (traj_idx + 1) % len(trajectories)
            cur_traj = trajectories[traj_idx]
            num_steps = cur_traj.shape[0]
            step_idx = 0
            print(f"[Offline Playback] Switch traj (reset) -> {traj_keys[traj_idx]} shape={cur_traj.shape}")
            continue
        else:
            just_reset = False

    if use_logger:
        logger.close()
    env.close()
    print("[Offline Playback] Closed cleanly.")


if __name__ == "__main__":
    main()
    simulation_app.close()
