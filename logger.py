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
from symdex.utils.trajectory_utils import build_observation_dict, now_ms, rgb_to_HWC, depth_to_gray, as_flag, to_str
from symdex.env.tasks.manager_based_env_cfg import *
from symdex.utils.rl_env_wrapper import VecEnvWrapper
import symdex


@hydra.main(
    config_path=symdex.LIB_PATH_PATH.joinpath("cfg").as_posix(),
    config_name="default"
)
def main(cfg: DictConfig):
    """Launch teleoperation control in Isaac Lab using TeleopDifferentialIKAction."""
    torch.set_printoptions(sci_mode=False, precision=3)
    set_random_seed(cfg.seed)
    capture_keyboard_interrupt()

    cfg, env_cfg = preprocess_cfg(cfg)

    # Load actions from file
    action_file = to_absolute_path(cfg.dataset_path)
    actions = np.load(action_file, allow_pickle=False)

    num_actions, action_dim = actions.shape
    print(f"[Offline Playback] actions.shape={actions.shape}, dtype={actions.dtype}")

    env = gym.make(cfg.env_name, cfg=env_cfg)
    env = VecEnvWrapper(env, rl_device=cfg.rl_device)
    env.reset()

    # Logger / episode control
    use_logger = getattr(cfg, "logger", False)
    if use_logger:
        from symdex.utils.trajectory_logger import TrajectoryLogger
        logger = TrajectoryLogger(task_name=cfg.task.env_name)
        max_episodes = getattr(cfg, "max_episodes", None)
        episodes_saved = 0
        print("[Offline Playback] Logger enabled.")

    # Playback pacing (seconds). 0.0 = run as fast as possible
    playback_dt = float(getattr(cfg, "playback_dt", 0.0))

    step_idx = 0       # index inside file (cycles)
    global_step = 0    # monotonically increasing

    just_reset = True

    print("[Offline Playback] Start stepping env with cyclic actions...")

    while simulation_app.is_running():

        q_np = np.asarray(actions[step_idx], dtype=np.float32)
        q_tensor = torch.as_tensor(q_np[None, ...], dtype=torch.float32, device=cfg.rl_device)

        step_idx = (step_idx + 1) % num_actions
        global_step += 1

        obs, rew, reset, extras = env.step(q_tensor)

        reward_scalar = float(rew[0].item() if torch.is_tensor(rew) else rew)
        success_flag = as_flag(extras.get("success"))
        done_flag = as_flag(reset)

        obs_dict = build_observation_dict(env, obs)

        action = q_np
        action_dict = {
            "arm_hand_action_right": q_np[:22] if action_dim >= 22 else q_np,
            "arm_hand_action_left": q_np[22:] if action_dim >= 44 else q_np[22:],
        }

        if just_reset or (cur_lang is None):
            lang_field = extras.get("language_instruction", "")
            cur_lang = to_str(lang_field)
        language_instruction = cur_lang

        if use_logger:
            logger.add_step(
                action=action,
                action_dict=action_dict,
                observation=obs_dict,
                reward=reward_scalar,
                is_first=bool(just_reset),
                is_last=done_flag,
                is_terminal=bool(success_flag),
                language_instruction=language_instruction,
                discount=1.0,
            )

        if success_flag:
            print("[Offline Playback] Task success, resetting environment.")
            env.reset()
            just_reset = True
            cur_lang = None
            if use_logger:
                logger.save_episode()
                episodes_saved += 1
                if (max_episodes is not None) and (episodes_saved >= max_episodes):
                    print(f"[Offline Playback] Reached max_episodes={max_episodes}. Stopping.")
                    break
        elif reset.any():
            print("[Offline Playback] Environment reset triggered auto-reset internally.")
            env.reset()
            just_reset = True
            cur_lang = None
            if use_logger:
                logger.save_episode()
                episodes_saved += 1
                if (max_episodes is not None) and (episodes_saved >= max_episodes):
                    print(f"[Offline Playback] Reached max_episodes={max_episodes}. Stopping.")
                    break
        else:
            just_reset = False

        if playback_dt > 0.0:
            time.sleep(playback_dt)

    if use_logger:
        logger.close()
    env.close()
    print("[Offline Playback] Closed cleanly.")


if __name__ == "__main__":
    main()
    simulation_app.close()
