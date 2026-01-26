from isaaclab.app import AppLauncher

app_launcher = AppLauncher({"headless": False, "enable_cameras": True})
simulation_app = app_launcher.app

import os
import h5py
import torch
import hydra
import numpy as np
import gymnasium as gym
from omegaconf import DictConfig
from hydra.utils import to_absolute_path

from symdex.utils.common import set_random_seed, capture_keyboard_interrupt, preprocess_cfg
from symdex.env.tasks.manager_based_env_cfg import *
import symdex
from symdex.utils.rl_env_wrapper import VecEnvWrapper


def _load_episode_actions_from_h5(file: h5py.File, epi_name: str) -> np.ndarray:
    ep = file[epi_name]
    steps_grp = ep["steps"]
    step_names = sorted(list(steps_grp.keys()))
    num_steps = len(step_names)

    if num_steps > 400:
        raise RuntimeError(f"{epi_name} too long: {num_steps} > 400")
    last_flag = bool(steps_grp[step_names[-1]].attrs.get("is_terminal", False))
    if not last_flag:
        raise RuntimeError(f"{epi_name} is not terminal(success)")

    a0 = np.array(steps_grp[step_names[0]]["action"][()], dtype=np.float32).reshape(-1)
    action_dim = int(a0.shape[0])

    actions = np.zeros((num_steps, action_dim), dtype=np.float32)
    for t, sname in enumerate(step_names):
        actions[t] = np.array(steps_grp[sname]["action"][()], dtype=np.float32).reshape(-1)
    return actions


def load_episodes_actions(path: str):
    path = to_absolute_path(path)
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not exists: {path}")

    ext = os.path.splitext(path)[1].lower()
    episodes_actions = []

    if ext == ".npy":
        arr = np.load(path)
        arr = np.asarray(arr, dtype=np.float32)
        if arr.ndim != 2:
            raise ValueError(f".npy actions must be 2D (T, action_dim), got {arr.shape}")
        episodes_actions.append(torch.from_numpy(arr))
        print(f"[load_episodes_actions] npy: steps={arr.shape[0]}, action_dim={arr.shape[1]}")

    elif ext == ".npz":
        data = np.load(path)
        keys = sorted(list(data.keys()))
        if "actions" in data:
            arr = np.asarray(data["actions"], dtype=np.float32)
            if arr.ndim != 2:
                raise ValueError(f".npz['actions'] must be 2D, got {arr.shape}")
            episodes_actions.append(torch.from_numpy(arr))
            print(f"[load_episodes_actions] npz(actions): steps={arr.shape[0]}, action_dim={arr.shape[1]}")
        else:
            for k in keys:
                arr = np.asarray(data[k], dtype=np.float32)
                if arr.ndim != 2:
                    raise ValueError(f".npz['{k}'] must be 2D, got {arr.shape}")
                episodes_actions.append(torch.from_numpy(arr))
                print(f"[load_episodes_actions] npz({k}): steps={arr.shape[0]}, action_dim={arr.shape[1]}")

    elif ext in [".h5", ".hdf5"]:
        with h5py.File(path, "r") as file:
            episode_names = sorted([k for k in file.keys() if k.startswith("episode_")])
            if not episode_names:
                raise RuntimeError(f"There is no episode_* group under {path}")

            print(f"[load_episodes_actions] Found {len(episode_names)} episodes: {episode_names}")

            for epi_name in episode_names:
                try:
                    actions = _load_episode_actions_from_h5(file, epi_name)
                except Exception as e:
                    print(f"[INFO] Skip {epi_name}: {e}")
                    continue
                episodes_actions.append(torch.from_numpy(actions))
                print(f"[load_episodes_actions] {epi_name}: steps={actions.shape[0]}, action_dim={actions.shape[1]}")

    elif ext in [".pt", ".pth"]:
        # 你的数据是 torch.save(list[dict]) 形式；每个 dict 至少有 'actions'
        try:
            obj = torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:
            # 兼容旧 torch 没有 weights_only 参数的情况
            obj = torch.load(path, map_location="cpu")

        if isinstance(obj, dict) and "actions" in obj:
            obj = [obj]

        if not isinstance(obj, list) or len(obj) == 0:
            raise RuntimeError(f".pt/.pth expected non-empty list (or dict with 'actions'), got {type(obj)}")

        for i, ep in enumerate(obj):
            if not isinstance(ep, dict) or "actions" not in ep:
                print(f"[INFO] Skip item {i}: not a dict with 'actions' key")
                continue

            a = ep["actions"]
            if isinstance(a, np.ndarray):
                a = torch.from_numpy(a)
            if not isinstance(a, torch.Tensor):
                print(f"[INFO] Skip item {i}: actions type={type(a)}")
                continue

            if a.ndim == 1:
                a = a.unsqueeze(0)  # (action_dim,) -> (1, action_dim)
            if a.ndim != 2:
                print(f"[INFO] Skip item {i}: actions must be 2D (T, action_dim), got {tuple(a.shape)}")
                continue

            a = a.detach().to(dtype=torch.float32, device="cpu")
            episodes_actions.append(a)
            if i < 5:
                print(f"[load_episodes_actions] pt item {i}: steps={a.shape[0]}, action_dim={a.shape[1]}")

        print(f"[load_episodes_actions] pt: loaded episodes={len(episodes_actions)}")

    else:
        raise ValueError(f"Unsupported file extension: {ext}")

    if not episodes_actions:
        raise RuntimeError(f"No episode action sequence loaded from {path}")

    return episodes_actions



@hydra.main(config_path=symdex.LIB_PATH_PATH.joinpath("cfg").as_posix(), config_name="default")
def main(cfg: DictConfig):
    torch.set_printoptions(sci_mode=False, precision=3)
    set_random_seed(cfg.seed)
    capture_keyboard_interrupt()

    action_path = cfg.dataset_path
    print(f"[INFO] Action source: {action_path}")

    episodes_actions = load_episodes_actions(action_path)

    cfg, env_cfg = preprocess_cfg(cfg)
    env = gym.make(cfg.env_name, cfg=env_cfg)
    env = VecEnvWrapper(env, rl_device=cfg.rl_device)

    env.reset()
    env.unwrapped.update_randomization(0.0)

    device = torch.device(cfg.rl_device)
    episodes_actions = [ep.to(device) for ep in episodes_actions]

    episode_idx = 0
    step_idx = 0

    print(f"[INFO] Start replay. episodes={len(episodes_actions)}, num_envs={cfg.num_envs}, device={cfg.rl_device}")

    while simulation_app.is_running():
        actions_ep = episodes_actions[episode_idx]
        T = int(actions_ep.shape[0])

        if step_idx >= T:
            episode_idx = (episode_idx + 1) % len(episodes_actions)
            step_idx = 0
            env.reset()
            print("-" * 80)
            print(f"[INFO] Start next episode: index={episode_idx}")
            continue

        a_single = actions_ep[step_idx]  # (action_dim,)

        if cfg.num_envs == 1:
            actions = a_single.unsqueeze(0)  # (1, action_dim)
        else:
            actions = a_single.unsqueeze(0).expand(cfg.num_envs, -1)  # (num_envs, action_dim)

        env.step(actions)
        step_idx += 1

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
