from isaaclab.app import AppLauncher

app_launcher = AppLauncher({"headless": False, "enable_cameras": True})
simulation_app = app_launcher.app

import os
import glob
import h5py
import time
import numpy as np
import torch
import hydra
import gymnasium as gym
from omegaconf import DictConfig

import symdex
from symdex.algo import alg_name_to_path
from symdex.utils.common import set_random_seed, capture_keyboard_interrupt, preprocess_cfg, load_class_from_path
from symdex.env.tasks.manager_based_env_cfg import *
from symdex.utils.rl_env_wrapper import VecEnvWrapper


def compute_state_norm_from_h5(root_dir: str, pattern: str, eps: float = 1e-3):
    """
    Compute the same normalization statistics as OfflineBuffer.normalize_states(),
    but without loading point cloud or action data.
    """
    file_paths = sorted(glob.glob(os.path.join(root_dir, pattern)))
    if len(file_paths) == 0:
        raise ValueError(f"No h5 files found in {root_dir} with pattern {pattern}")

    total_count = 0
    state_sum = None
    state_sq_sum = None

    for file_path in file_paths:
        with h5py.File(file_path, "r") as f:
            episode_keys = sorted([k for k in f.keys() if k.startswith("episode_")])

            for ep_key in episode_keys:
                state = np.asarray(f[f"{ep_key}/offline_data/observations/policy"], dtype=np.float32)
                if state_sum is None:
                    state_dim = state.shape[1]
                    state_sum = np.zeros((state_dim,), dtype=np.float64)
                    state_sq_sum = np.zeros((state_dim,), dtype=np.float64)

                state_sum += state.sum(axis=0, dtype=np.float64)
                state_sq_sum += np.square(state, dtype=np.float64).sum(axis=0)
                total_count += state.shape[0]

    if total_count == 0:
        raise RuntimeError("No states found when computing state normalization.")

    mean = state_sum / total_count
    var = state_sq_sum / total_count - np.square(mean)
    var = np.maximum(var, 0.0)

    mean = mean.reshape(1, -1).astype(np.float32)
    std = (np.sqrt(var).reshape(1, -1) + eps).astype(np.float32)

    print(
        f"[StateNorm] files={len(file_paths)}, states={total_count}, "
        f"state_dim={mean.shape[1]}, std_min={float(std.min()):.6f}, "
        f"std_mean={float(std.mean()):.6f}"
    )

    return mean, std


def normalize_state(state: torch.Tensor, state_mean, state_std):
    if state_mean is None or state_std is None:
        return state

    mean = torch.as_tensor(state_mean, dtype=state.dtype, device=state.device)
    std = torch.as_tensor(state_std, dtype=state.dtype, device=state.device)
    return (state - mean) / std


def build_policy_input(obs, cfg, state_mean, state_std):
    """
    Same input convention as EvaluatorTD3BC._build_policy_input().
    """
    if isinstance(obs, dict):
        batch = {}
        batch["state"] = normalize_state(obs["policy"], state_mean, state_std)

        if getattr(cfg.algo.observation, "vision", False):
            batch["vision"] = obs["vision"]
        if getattr(cfg.algo.observation, "pc", False):
            batch["pc"] = obs["point_cloud"]

        return batch

    return {"state": normalize_state(obs, state_mean, state_std)}


def get_state_dim_from_obs(obs):
    if isinstance(obs, dict):
        return obs["policy"].shape[-1]
    return obs.shape[-1]


def get_action_dim_from_env(env):
    return env.action_space.shape[-1]


@torch.no_grad()
def run_visualization(policy, env, cfg, state_mean, state_std, max_episodes: int):
    device = torch.device(cfg.device)

    obs, _ = env.reset()

    current_returns = torch.zeros(cfg.num_envs, dtype=torch.float32, device=device)
    current_lengths = torch.zeros(cfg.num_envs, dtype=torch.float32, device=device)

    return_list = []
    length_list = []
    success_list = []
    action_abs_list = []
    action_mean_list = []
    action_sat_list = []

    episodes_done = 0
    env_steps = 0
    start_time = time.time()

    while simulation_app.is_running() and episodes_done < max_episodes:
        batch = build_policy_input(obs, cfg, state_mean, state_std)
        action = policy.select_action(batch)

        action_abs = action.detach().abs()
        action_abs_list.append(float(action_abs.mean().item()))
        action_mean_list.append(float(action.detach().mean().item()))
        action_sat_list.append(float((action_abs > 0.99 * policy.max_action).float().mean().item()))

        next_obs, reward, done, info = env.step(action)

        current_returns += reward
        current_lengths += 1
        env_steps += 1

        done_indices = torch.where(done > 0)[0]

        if len(done_indices) > 0:
            for env_id in done_indices.detach().cpu().tolist():
                ep_return = float(current_returns[env_id].item())
                ep_length = float(current_lengths[env_id].item())

                if isinstance(info, dict) and "success" in info:
                    ep_success = float(info["success"][env_id].item())
                else:
                    ep_success = 0.0

                return_list.append(ep_return)
                length_list.append(ep_length)
                success_list.append(ep_success)

                episodes_done += 1

                print(
                    f"[Episode {episodes_done:04d}] "
                    f"return={ep_return:.3f}, "
                    f"length={ep_length:.0f}, "
                    f"success={ep_success:.0f}"
                )

                if episodes_done >= max_episodes:
                    break

            current_returns[done_indices] = 0.0
            current_lengths[done_indices] = 0.0

        obs = next_obs

    elapsed = time.time() - start_time

    returns_np = np.asarray(return_list, dtype=np.float32)
    lengths_np = np.asarray(length_list, dtype=np.float32)
    success_np = np.asarray(success_list, dtype=np.float32)

    print("========== TD3_BC Visualizer Summary ==========")
    print(f"episodes              : {len(return_list)}")
    print(f"env_steps             : {env_steps}")
    print(f"elapsed_sec           : {elapsed:.2f}")

    if len(returns_np) > 0:
        print(f"return_mean           : {float(returns_np.mean()):.6f}")
        print(f"return_std            : {float(returns_np.std()):.6f}")
        print(f"return_min            : {float(returns_np.min()):.6f}")
        print(f"return_max            : {float(returns_np.max()):.6f}")
        print(f"episode_length_mean   : {float(lengths_np.mean()):.6f}")
        print(f"success_rate          : {float(success_np.mean()):.6f}")
        print(f"success_std           : {float(success_np.std()):.6f}")
    else:
        print("No completed episodes were collected.")

    if len(action_abs_list) > 0:
        print(f"action_abs_mean       : {float(np.mean(action_abs_list)):.6f}")
        print(f"action_mean           : {float(np.mean(action_mean_list)):.6f}")
        print(f"action_saturation_099 : {float(np.mean(action_sat_list)):.6f}")


@hydra.main(config_path=symdex.LIB_PATH_PATH.joinpath('cfg').as_posix(), config_name="default")
def main(cfg: DictConfig):
    set_random_seed(cfg.seed)
    capture_keyboard_interrupt()

    if cfg.algo.checkpoint.load_path is None:
        raise ValueError(
            "Missing checkpoint path. Use: "
            "algo.checkpoint.load_path=/path/to/model_final.pth"
        )

    cfg, env_cfg = preprocess_cfg(cfg)

    env = gym.make(cfg.env_name, cfg=env_cfg)
    env = VecEnvWrapper(env, rl_device=cfg.rl_device, clip_obs=50.0)

    if cfg.task.randomize.eval:
        env.unwrapped.update_randomization(1.0)
    else:
        env.unwrapped.update_randomization(0.0)

    obs, _ = env.reset()

    state_dim = get_state_dim_from_obs(obs)
    action_dim = get_action_dim_from_env(env)

    state_mean, state_std = None, None
    if cfg.algo.normalize_states:
        state_mean, state_std = compute_state_norm_from_h5(
            root_dir=cfg.algo.offline.data_dir,
            pattern=cfg.algo.offline.pattern,
            eps=1e-3,
        )

    algo_class = load_class_from_path(cfg.algo.name, alg_name_to_path[cfg.algo.name])

    policy = algo_class(
        state_dim=state_dim,
        action_dim=action_dim,
        max_action=cfg.algo.offline.max_action,
        device=cfg.device,
        use_vision=cfg.algo.observation.vision,
        use_pc=cfg.algo.observation.pc,
        discount=cfg.algo.discount,
        tau=cfg.algo.tau,
        policy_noise=cfg.algo.policy_noise,
        noise_clip=cfg.algo.noise_clip,
        policy_freq=cfg.algo.policy_freq,
        alpha=cfg.algo.alpha,
        actor_lr=cfg.algo.actor_lr,
        critic_lr=cfg.algo.critic_lr,
        reward_scale=cfg.algo.reward_scale,
    )

    policy.load(cfg.algo.checkpoint.load_path)
    policy.actor.eval()
    policy.critic.eval()
    policy.actor_target.eval()
    policy.critic_target.eval()

    print("========== TD3_BC Visualizer ==========")
    print(f"checkpoint            : {cfg.algo.checkpoint.load_path}")
    print(f"env_name              : {cfg.env_name}")
    print(f"num_envs              : {cfg.num_envs}")
    print(f"state_dim             : {state_dim}")
    print(f"action_dim            : {action_dim}")
    print(f"use_pc                : {cfg.algo.observation.pc}")
    print(f"use_vision            : {cfg.algo.observation.vision}")
    print(f"max_action            : {policy.max_action}")
    print(f"max_episode_length    : {env.max_episode_length}")

    max_episodes = int(getattr(cfg, "max_episodes", 10))

    run_visualization(
        policy=policy,
        env=env,
        cfg=cfg,
        state_mean=state_mean,
        state_std=state_std,
        max_episodes=max_episodes,
    )

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()