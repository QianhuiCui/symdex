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

    with h5py.File(path, "r") as file:
        episode_names = sorted([k for k in file.keys() if k.startswith("episode_")])
        if not episode_names:
            raise RuntimeError(f"There is no episode_* group under {path}")

        print(f"Found {len(episode_names)} episodes: {episode_names}")

        for epi_name in episode_names:
            ep = file[epi_name]

            steps_grp = ep["steps"]
            step_names = sorted(list(steps_grp.keys()))
            num_steps = len(step_names)

            if num_steps > 400:
                print(f"[INFO] Skip {epi_name} with too many steps: {num_steps} > 400")
                continue
            flag = steps_grp[step_names[-1]].attrs["is_terminal"]
            if not flag:
                print(f"[INFO] Skip {epi_name} since it is failed.")
                continue

            step_0 = steps_grp[step_names[0]]
            action_0 = step_0["action"][()]
            action_dim = int(action_0.shape[0])
            policy_0 = step_0["observation"]["policy"][()]
            if policy_0.ndim == 2 and policy_0.shape[0] == 1:
                obs_dim = int(policy_0.shape[1])
            elif policy_0.ndim == 1:
                obs_dim = int(policy_0.shape[0])
            else:
                raise ValueError(
                    f"Unsupported shape of {epi_name}/{step_names[0]}/observation/policy: {policy_0.shape}"
                )

            actions = np.zeros((num_steps, action_dim), dtype=np.float32)
            obs_policy = np.zeros((num_steps, obs_dim), dtype=np.float32)

            for t, sname in enumerate(step_names):
                step = steps_grp[sname]

                action = step["action"][()]
                actions[t] = action.astype(np.float32)

                policy = step["observation"]["policy"][()]
                if policy.ndim == 2 and policy.shape[0] == 1:
                    policy = policy[0]
                obs_policy[t] = policy.astype(np.float32)

            episodes_actions.append(torch.from_numpy(actions))
            episodes_obs.append(torch.from_numpy(obs_policy))
            print(
                f"[load_all_episodes_from_h5] {epi_name}: "
                f"steps={num_steps}, action_dim={action_dim}, obs_dim={obs_dim}"
            )

    if not episodes_actions:
        raise RuntimeError(f"There is no episode action sequence in {path}")

    return episodes_actions, episodes_obs


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

    env.reset()
    env.unwrapped.update_randomization(0.0)

    episode_idx = 2
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

        if step_idx >= T:
            episode_idx = (episode_idx + 1) % len(episodes_actions)
            step_idx = 0
            env.reset()
            print("-" * 80)
            print(f"[INFO] Start next episode: index={episode_idx}")
            continue

        a_single = actions_ep[step_idx]  # torch.Tensor

        if cfg.num_envs == 1:
            actions = a_single.unsqueeze(0)  # (1, action_dim)
        else:
            actions = a_single.unsqueeze(0).expand(cfg.num_envs, -1)  # (num_envs, action_dim)

        env.step(actions)
        step_idx += 1

    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
