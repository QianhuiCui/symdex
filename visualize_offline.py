from isaaclab.app import AppLauncher

# Must be before importing torch/gym/isaac/symdex modules.
app_launcher = AppLauncher({"headless": False, "enable_cameras": True})
simulation_app = app_launcher.app

import os
import glob
import time
import h5py
import numpy as np
import torch
import hydra
import gymnasium as gym
from omegaconf import DictConfig, OmegaConf
from hydra.utils import to_absolute_path

import symdex
from symdex.algo import alg_name_to_path
from symdex.utils.common import set_random_seed, capture_keyboard_interrupt, preprocess_cfg, load_class_from_path
from symdex.env.tasks.manager_based_env_cfg import *
from symdex.utils.rl_env_wrapper import VecEnvWrapper


def load_state_norm_from_checkpoint_dir(checkpoint_path: str):
    ckpt_dir = os.path.dirname(to_absolute_path(checkpoint_path))
    norm_path = os.path.join(ckpt_dir, "state_norm.pth")
    if not os.path.exists(norm_path):
        raise FileNotFoundError(f"state_norm.pth not found: {norm_path}. ")

    norm_ckpt = torch.load(norm_path, map_location="cpu")
    state_mean = norm_ckpt["state_mean"]
    state_std = norm_ckpt["state_std"]

    print(f"[StateNorm] loaded from: {norm_path}")
    print(f"[StateNorm] mean shape: {state_mean.shape}")
    print(f"[StateNorm] std shape : {state_std.shape}")
    print(f"[StateNorm] std min/mean: {float(state_std.min()):.6f}, {float(state_std.mean()):.6f}")
    return state_mean, state_std

def normalize_state(state: torch.Tensor, state_mean, state_std):
    if state_mean is None or state_std is None:
        return state

    mean = torch.as_tensor(state_mean, dtype=state.dtype, device=state.device)
    std = torch.as_tensor(state_std, dtype=state.dtype, device=state.device)
    return (state - mean) / std

def build_policy_input(obs, cfg, state_mean, state_std):
    if isinstance(obs, dict):
        batch = {}
        batch["state"] = normalize_state(obs["policy"], state_mean, state_std)
        if getattr(cfg.algo.observation, "vision", False):
            batch["vision"] = obs["vision"]
        if getattr(cfg.algo.observation, "pc", False):
            batch["pc"] = obs["point_cloud"]
        return batch

    return {"state": normalize_state(obs, state_mean, state_std)}

# =========================
# Debug helpers
# =========================
def get_state_dim_from_obs(obs):
    if isinstance(obs, dict):
        return obs["policy"].shape[-1]
    return obs.shape[-1]

def get_action_dim_from_env(env):
    return env.action_space.shape[-1]

def action_group_values(action: torch.Tensor):
    """
    Assumes PickObject action layout:
      0:6      right arm
      6:22     right hand
      22:28    left arm
      28:44    left hand
    """
    return {"right_arm": action[..., 0:6],
            "right_hand": action[..., 6:22],
            "left_arm": action[..., 22:28],
            "left_hand": action[..., 28:44],
            "all": action,}

def print_action_groups(action: torch.Tensor, prefix: str, max_action: float = 1.0):
    action = action.detach()
    groups = action_group_values(action)

    for name, x in groups.items():
        x_abs = x.abs()
        print(
            f"{prefix}/{name}: "
            f"mean={float(x.mean().item()):.6f}, "
            f"abs_mean={float(x_abs.mean().item()):.6f}, "
            f"abs_max={float(x_abs.max().item()):.6f}, "
            f"sat099={float((x_abs > 0.99 * max_action).float().mean().item()):.6f}"
        )

def pc_debug_stats(name: str, pc: torch.Tensor):
    """
    pc shape:
      (B, N, 6): right xyz + left xyz
    """
    if not torch.is_tensor(pc):
        pc = torch.as_tensor(pc)

    pc = pc.detach()
    print(f"[{name}] shape: {tuple(pc.shape)}")
    print(
        f"[{name}] all: "
        f"min={float(pc.min().item()):.6f}, "
        f"max={float(pc.max().item()):.6f}, "
        f"mean={float(pc.mean().item()):.6f}, "
        f"std={float(pc.std(unbiased=False).item()):.6f}, "
        f"abs_max={float(pc.abs().max().item()):.6f}"
    )

    right = pc[..., :3]
    left = pc[..., 3:6]
    right_valid = (right.abs().sum(dim=-1) > 1e-8).float()
    left_valid = (left.abs().sum(dim=-1) > 1e-8).float()
    print(
        f"[{name}] right: "
        f"valid_ratio={float(right_valid.mean().item()):.6f}, "
        f"empty_ratio={float((right_valid.sum(dim=1) == 0).float().mean().item()):.6f}, "
        f"mean={float(right.mean().item()):.6f}, "
        f"std={float(right.std(unbiased=False).item()):.6f}, "
        f"min={float(right.min().item()):.6f}, "
        f"max={float(right.max().item()):.6f}"
    )
    print(
        f"[{name}] left : "
        f"valid_ratio={float(left_valid.mean().item()):.6f}, "
        f"empty_ratio={float((left_valid.sum(dim=1) == 0).float().mean().item()):.6f}, "
        f"mean={float(left.mean().item()):.6f}, "
        f"std={float(left.std(unbiased=False).item()):.6f}, "
        f"min={float(left.min().item()):.6f}, "
        f"max={float(left.max().item()):.6f}"
    )

def collect_h5_paths(cfg):
    dataset_path = getattr(cfg, "dataset_path", None)
    if dataset_path is not None:
        dataset_path = to_absolute_path(str(dataset_path))
        if os.path.isfile(dataset_path):
            return [dataset_path]
        if os.path.isdir(dataset_path):
            return sorted(glob.glob(os.path.join(dataset_path, "*.h5")))

    root_dir = to_absolute_path(str(cfg.algo.offline.data_dir))
    pattern = str(cfg.algo.offline.pattern)
    paths = sorted(glob.glob(os.path.join(root_dir, pattern)))
    if len(paths) == 0:
        raise FileNotFoundError(f"No h5 files found under {root_dir} with pattern {pattern}")
    return paths

# =========================
# Offline checkpoint validation on h5
# =========================
@torch.no_grad()
def validate_checkpoint_on_h5(policy, cfg, state_mean, state_std, h5_paths, device, max_files: int = -1, max_transitions: int = -1, chunk_size: int = 256):
    """
    Validate policy(recorded_obs) against recorded clipped action.
    This does not step the environment. It only tests whether the loaded checkpoint reproduces h5 actions on h5 observations.
    """
    policy.actor.eval()
    h5_paths = h5_paths[:max_files]
    mse_all = []
    mae_all = []
    group_mse = {"right_arm": [],
                 "right_hand": [],
                 "left_arm": [],
                 "left_hand": [],}
    total_seen = 0

    print("========== Offline H5 Policy Validation ==========")
    print(f"files: {len(h5_paths)}")
    print(f"use_pc: {cfg.algo.observation.pc}")
    print(f"chunk_size: {chunk_size}")

    for h5_path in h5_paths:
        with h5py.File(h5_path, "r") as f:
            ep_keys = sorted([k for k in f.keys() if k.startswith("episode_")])

            for ep_key in ep_keys:
                base = f[f"{ep_key}/offline_data"]
                state = np.asarray(base["observations"]["policy"], dtype=np.float32)
                actions = np.asarray(base["actions"], dtype=np.float32)
                actions = np.clip(actions, -float(cfg.algo.offline.max_action), float(cfg.algo.offline.max_action)).astype(np.float32)
                if getattr(cfg.algo.observation, "pc", False):
                    if "point_cloud" not in base["observations"]:
                        raise RuntimeError(f"{h5_path}/{ep_key} has no observations/point_cloud")
                    pc = np.asarray(base["observations"]["point_cloud"], dtype=np.float32)
                else:
                    pc = None

                T = state.shape[0]
                for start in range(0, T, chunk_size):
                    end = min(start + chunk_size, T)
                    if max_transitions > 0 and total_seen >= max_transitions:
                        break
                    if max_transitions > 0:
                        end = min(end, start + (max_transitions - total_seen))

                    state_t = torch.as_tensor(state[start:end], dtype=torch.float32, device=device)
                    state_t = normalize_state(state_t, state_mean, state_std)
                    batch = {"state": state_t}
                    if pc is not None:
                        batch["pc"] = torch.as_tensor(pc[start:end], dtype=torch.float32, device=device)
                    action_t = torch.as_tensor(actions[start:end], dtype=torch.float32, device=device)
                    pred = policy.actor(batch)

                    mse = ((pred - action_t) ** 2).mean(dim=-1)
                    mae = (pred - action_t).abs().mean(dim=-1)
                    mse_all.append(mse.detach().cpu())
                    mae_all.append(mae.detach().cpu())
                    if pred.shape[-1] >= 44:
                        group_mse["right_arm"].append(((pred[:, 0:6] - action_t[:, 0:6]) ** 2).mean(dim=-1).detach().cpu())
                        group_mse["right_hand"].append(((pred[:, 6:22] - action_t[:, 6:22]) ** 2).mean(dim=-1).detach().cpu())
                        group_mse["left_arm"].append(((pred[:, 22:28] - action_t[:, 22:28]) ** 2).mean(dim=-1).detach().cpu())
                        group_mse["left_hand"].append(((pred[:, 28:44] - action_t[:, 28:44]) ** 2).mean(dim=-1).detach().cpu())
                    total_seen += end - start

                if max_transitions > 0 and total_seen >= max_transitions:
                    break

        if max_transitions > 0 and total_seen >= max_transitions:
            break

    if len(mse_all) == 0:
        print("[Offline H5 Validation] No transitions validated.")
        return

    mse_all = torch.cat(mse_all)
    mae_all = torch.cat(mae_all)
    print(f"validated_transitions: {mse_all.numel()}")
    print(f"mse_mean             : {float(mse_all.mean().item()):.6f}")
    print(f"mse_std              : {float(mse_all.std(unbiased=False).item()):.6f}")
    print(f"mse_max              : {float(mse_all.max().item()):.6f}")
    print(f"mae_mean             : {float(mae_all.mean().item()):.6f}")
    print(f"mae_max              : {float(mae_all.max().item()):.6f}")

    for k, vals in group_mse.items():
        if len(vals) > 0:
            vals = torch.cat(vals)
            print(f"{k}_mse_mean       : {float(vals.mean().item()):.6f}")
            print(f"{k}_mse_max        : {float(vals.max().item()):.6f}")

    print("==================================================")

@torch.no_grad()
def run_visualization(policy, env, cfg, state_mean, state_std, max_episodes: int):
    device = torch.device(cfg.device)

    obs, _ = env.reset()

    current_returns = torch.zeros(cfg.num_envs, dtype=torch.float32, device=device)
    current_lengths = torch.zeros(cfg.num_envs, dtype=torch.float32, device=device)
    detailed_returns = {}
    return_list = []
    length_list = []
    success_list = []
    action_abs_list = []
    action_mean_list = []
    action_sat_list = []
    group_abs_lists = {
        "right_arm": [],
        "right_hand": [],
        "left_arm": [],
        "left_hand": [],
    }
    group_sat_lists = {
        "right_arm": [],
        "right_hand": [],
        "left_arm": [],
        "left_hand": [],
    }
    last_action_diff_list = []
    last_action_diff_max_list = []
    scaled_action_abs_list = []
    detailed_episode_returns = {}
    episodes_done = 0
    env_steps = 0
    start_time = time.time()

    print_every = int(OmegaConf.select(cfg, "visualizer.print_every", default=100))
    printed_pc_once = False

    while simulation_app.is_running() and episodes_done < max_episodes:
        batch = build_policy_input(obs, cfg, state_mean, state_std)
        if getattr(cfg.algo.observation, "pc", False) and "pc" in batch and not printed_pc_once:
            pc_debug_stats("EvalPC", batch["pc"])
            printed_pc_once = True

        action = policy.select_action(batch)
        action_abs = action.detach().abs()
        action_abs_list.append(float(action_abs.mean().item()))
        action_mean_list.append(float(action.detach().mean().item()))
        action_sat_list.append(float((action_abs > 0.99 * policy.max_action).float().mean().item()))

        groups = action_group_values(action)
        for name in ["right_arm", "right_hand", "left_arm", "left_hand"]:
            if name in groups:
                g_abs = groups[name].detach().abs()
                group_abs_lists[name].append(float(g_abs.mean().item()))
                group_sat_lists[name].append(float((g_abs > 0.99 * policy.max_action).float().mean().item()))

        if env_steps < 10 or env_steps % print_every == 0:
            print_action_groups(action, prefix=f"[PolicyAction step={env_steps}]", max_action=float(policy.max_action))

        next_obs, reward, done, info = env.step(action)

        if hasattr(env.unwrapped, "last_action"):
            env_last_action = env.unwrapped.last_action.to(action.device)
            diff = (env_last_action - action).abs()
            last_action_diff_list.append(float(diff.mean().item()))
            last_action_diff_max_list.append(float(diff.max().item()))

            if env_steps < 10 or env_steps % print_every == 0:
                print_action_groups(env_last_action, prefix=f"[BaseEnv.last_action step={env_steps}]", max_action=float(policy.max_action))
                print(
                    f"[BaseEnv.last_action_diff step={env_steps}] "
                    f"mean={float(diff.mean().item()):.8f}, "
                    f"max={float(diff.max().item()):.8f}"
                )

        if hasattr(env.unwrapped, "last_scaled_action"):
            scaled_action = env.unwrapped.last_scaled_action
            scaled_action_abs_list.append(float(scaled_action.abs().mean().item()))
            if env_steps < 10 or env_steps % print_every == 0:
                print_action_groups(scaled_action, prefix=f"[BaseEnv.last_scaled_action step={env_steps}]", max_action=1.0)

        current_returns += reward
        current_lengths += 1
        env_steps += 1

        if isinstance(info, dict) and "detailed_reward" in info:
            for k, v in info["detailed_reward"].items():
                v = v.to(device=device).float()
                if k not in detailed_returns:
                    detailed_returns[k] = torch.zeros(cfg.num_envs, dtype=torch.float32, device=device)
                detailed_returns[k] += v

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

                if len(detailed_returns) > 0:
                    detail_pairs = []
                    for k, v in detailed_returns.items():
                        val = float(v[env_id].item())
                        detail_pairs.append((k, val))
                        if k not in detailed_episode_returns:
                            detailed_episode_returns[k] = []
                        detailed_episode_returns[k].append(val)
                    detail_pairs = sorted(detail_pairs, key=lambda x: abs(x[1]), reverse=True)[:12]
                    print(f"[DetailedReward top] {detail_pairs}")

                if episodes_done >= max_episodes:
                    break

            current_returns[done_indices] = 0.0
            current_lengths[done_indices] = 0.0

            for k in detailed_returns:
                detailed_returns[k][done_indices] = 0.0

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

    for k in ["right_arm", "right_hand", "left_arm", "left_hand"]:
        if len(group_abs_lists[k]) > 0:
            print(f"action_abs/{k}        : {float(np.mean(group_abs_lists[k])):.6f}")
            print(f"action_sat/{k}        : {float(np.mean(group_sat_lists[k])):.6f}")

    if len(last_action_diff_list) > 0:
        print(f"env_last_action_diff_mean: {float(np.mean(last_action_diff_list)):.8f}")
        print(f"env_last_action_diff_max : {float(np.mean(last_action_diff_max_list)):.8f}")

    if len(scaled_action_abs_list) > 0:
        print(f"scaled_action_abs_mean   : {float(np.mean(scaled_action_abs_list)):.8f}")

    if len(detailed_episode_returns) > 0:
        print("========== Detailed Reward Mean ==========")
        for k, vals in sorted(detailed_episode_returns.items()):
            print(f"{k}: {float(np.mean(vals)):.6f}")


@hydra.main(config_path=symdex.LIB_PATH_PATH.joinpath("cfg").as_posix(), config_name="default")
def main(cfg: DictConfig):
    set_random_seed(cfg.seed)
    capture_keyboard_interrupt()

    if cfg.algo.checkpoint.load_path is None:
        raise ValueError("Missing checkpoint path. Use: algo.checkpoint.load_path=/path/to/model_latest.pth")

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
        state_mean, state_std = load_state_norm_from_checkpoint_dir(cfg.algo.checkpoint.load_path)

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

    do_validate_h5 = bool(OmegaConf.select(cfg, "visualizer.validate_h5", default=True))
    if do_validate_h5:
        h5_paths = collect_h5_paths(cfg)
        max_files = int(OmegaConf.select(cfg, "visualizer.validate_max_files", default=-1))
        max_transitions = int(OmegaConf.select(cfg, "visualizer.validate_max_transitions", default=10000))
        chunk_size = int(OmegaConf.select(cfg, "visualizer.validate_chunk_size", default=256))
        validate_checkpoint_on_h5(
            policy=policy,
            cfg=cfg,
            state_mean=state_mean,
            state_std=state_std,
            h5_paths=h5_paths,
            device=torch.device(cfg.device),
            max_files=max_files,
            max_transitions=max_transitions,
            chunk_size=chunk_size,
        )

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