#!/usr/bin/env python
# -*- coding: utf-8 -*-

from isaaclab.app import AppLauncher

app_launcher = AppLauncher({"headless": False, "enable_cameras": True})
simulation_app = app_launcher.app

import argparse  
import os
import h5py
import torch
import hydra
import numpy as np
from omegaconf import DictConfig
import gymnasium as gym
from hydra.utils import to_absolute_path

from symdex.utils.common import set_random_seed, capture_keyboard_interrupt, preprocess_cfg
import symdex
from symdex.env.tasks.manager_based_env_cfg import *
from symdex.utils.rl_env_wrapper import VecEnvWrapper


def load_all_episodes_from_h5(path: str):
    path = to_absolute_path(path)
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not exists: {path}")

    episodes_actions = []
    episodes_obs = []

    with h5py.File(path, "r") as f:
        episode_names = sorted([k for k in f.keys() if k.startswith("episode_")])
        if not episode_names:
            raise RuntimeError(f"There is no episode_* group under {path}")

        print(f"Found {len(episode_names)} episodes: {episode_names}")

        for epi_name in episode_names:
            ep = f[epi_name]
            if "steps" not in ep:
                print(f"[WARN] There is no 'steps' group under {epi_name}")
                continue

            steps_grp = ep["steps"]
            step_names = sorted(list(steps_grp.keys()))
            T = len(step_names)
            if T == 0:
                print(f"[WARN] {epi_name} has 0 steps")
                continue

            s0 = steps_grp[step_names[0]]

            if "action" not in s0:
                print(f"[WARN] {epi_name}/{step_names[0]} has no 'action' dataset")
                continue

            action0 = s0["action"][()]
            if action0.ndim != 1:
                raise ValueError(
                    f"Expected dim=1 for {epi_name}/{step_names[0]}/action, actual shape={action0.shape}"
                )
            action_dim = int(action0.shape[0])

            if "observation" not in s0 or "policy" not in s0["observation"]:
                print(
                    f"[WARN] {epi_name}/{step_names[0]} has no 'observation/policy',"
                    f"only read action in this episode"
                )
                obs_dim = None
            else:
                policy0 = s0["observation"]["policy"][()]
                if policy0.ndim == 2 and policy0.shape[0] == 1:
                    obs_dim = int(policy0.shape[1])
                elif policy0.ndim == 1:
                    obs_dim = int(policy0.shape[0])
                else:
                    raise ValueError(
                        f"Unsupported shape of {epi_name}/{step_names[0]}/observation/policy: {policy0.shape}"
                    )

            actions = np.zeros((T, action_dim), dtype=np.float32)
            obs_policy = None
            if obs_dim is not None:
                obs_policy = np.zeros((T, obs_dim), dtype=np.float32)

            for t, sname in enumerate(step_names):
                s = steps_grp[sname]

                a = s["action"][()]
                if a.shape != (action_dim,):
                    raise ValueError(
                        f"{epi_name}/{sname}/action shape mismatch: "
                        f"Expected shape {(action_dim,)}, actual {a.shape}"
                    )
                actions[t] = a.astype(np.float32)

                if obs_dim is not None:
                    p = s["observation"]["policy"][()]
                    if p.ndim == 2 and p.shape[0] == 1:
                        p = p[0]
                    if p.shape != (obs_dim,):
                        raise ValueError(
                            f"{epi_name}/{sname}/observation/policy shape mismatch: "
                            f"Expected shape {(obs_dim,)}, actual {p.shape}"
                        )
                    obs_policy[t] = p.astype(np.float32)

            episodes_actions.append(torch.from_numpy(actions))
            if obs_policy is not None:
                episodes_obs.append(torch.from_numpy(obs_policy))
            else:
                episodes_obs.append(None)

            print(
                f"[load_all_episodes_from_h5] {epi_name}: "
                f"steps={T}, action_dim={action_dim}, obs_dim={obs_dim}"
            )

    if not episodes_actions:
        raise RuntimeError(f"There is no episode action sequence in {path}")

    return episodes_actions, episodes_obs


def make_env(cfg: DictConfig):
    cfg, env_cfg = preprocess_cfg(cfg)
    env = gym.make(cfg.env_name, cfg=env_cfg)
    env = VecEnvWrapper(env, rl_device=cfg.rl_device)
    return env


@hydra.main(config_path=symdex.LIB_PATH_PATH.joinpath("cfg").as_posix(), config_name="default")
def main(cfg: DictConfig):
    torch.set_printoptions(sci_mode=False, precision=3)
    set_random_seed(cfg.seed)
    capture_keyboard_interrupt()

    dataset_path = cfg.dataset_path
    print(f"[INFO] Dataset: {dataset_path}")

    episodes_actions, episodes_obs = load_all_episodes_from_h5(dataset_path)

    cfg, env_cfg = preprocess_cfg(cfg)
    env = gym.make(cfg.env_name, cfg=env_cfg)
    env = VecEnvWrapper(env, rl_device=cfg.rl_device)

    action_space_shape = env.action_space.shape
    if len(action_space_shape) == 1:
        num_envs = 1
        action_dim_env = action_space_shape[0]
    elif len(action_space_shape) == 2:
        num_envs, action_dim_env = action_space_shape
    else:
        raise ValueError(f"Unsupported env.action_space.shape: {action_space_shape}")

    env.reset()
    env.unwrapped.update_randomization(0.0)

    episode_idx = 0
    step_idx = 0

    device = torch.device(cfg.rl_device)
    episodes_actions = [ep.to(device) for ep in episodes_actions]

    print(
        "[INFO] Start to replay dataset, "
        f"{len(episodes_actions)} episodes loaded."
    )

    while simulation_app.is_running():
        actions_ep = episodes_actions[episode_idx]
        T = actions_ep.shape[0]
        action_dim = actions_ep.shape[1]

        if action_dim != action_dim_env:
            raise ValueError(
                f"Dimension of action in H5 file is {action_dim} mismatch with action in Env: {action_dim_env}"
            )

        if step_idx >= T:
            episode_idx = (episode_idx + 1) % len(episodes_actions)
            step_idx = 0
            env.reset()
            print("-" * 80)
            print(f"[INFO] Start next episode: index={episode_idx}")
            continue

        a_single = actions_ep[step_idx]  # torch.Tensor

        if num_envs == 1:
            actions = a_single.unsqueeze(0)  # (1, action_dim)
        else:
            actions = a_single.unsqueeze(0).expand(num_envs, -1)  # (num_envs, action_dim)

        env.step(actions)
        step_idx += 1

    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
