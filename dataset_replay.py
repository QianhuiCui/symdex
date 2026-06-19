from isaaclab.app import AppLauncher

app_launcher = AppLauncher({"headless": False, "enable_cameras": True, "raw_data": True})
simulation_app = app_launcher.app

import torch
import hydra
import h5py
import numpy as np
import gymnasium as gym
from omegaconf import DictConfig

from symdex.utils.common import set_random_seed, capture_keyboard_interrupt, preprocess_cfg
from symdex.utils.trajectory_utils import get_obs
from symdex.utils.rl_env_wrapper import VecEnvWrapper
from symdex.utils.symmetry import SymmetryManager
from symdex.env.tasks.manager_based_env_cfg import *
import symdex


def load_episode(file, episode_key):
    grp = file[episode_key]
    meta = {}
    
    for k in grp["epi_meta"].keys():
        v = grp["epi_meta"][k][()]
        meta[k] = [x.decode("utf-8") for x in v] if v.dtype.kind == "S" else v
    
    data = grp["offline_data"]
    episode = {
        "state": data["observations"]["policy"][()],
        "next_state": data["next_observations"]["policy"][()],
        # "vision": data["observations"]["vision"][()],
        # "next_vision": data["next_observations"]["vision"][()],
        # "pc": data["observations"]["point_cloud"][()],
        # "next_pc": data["next_observations"]["point_cloud"][()],
        "actions": data["actions"][()],
        "rew_right": data["rewards_right"][()],
        "rew_left": data["rewards_left"][()],
        "terminals": data["terminals"][()],
        "timeouts": data["timeouts"][()],
    }
    if "reward_terms" in data:
        episode["reward_terms"] = data["reward_terms"][()]
    if "reward_names" in grp["epi_meta"]:
        episode["reward_names"] = meta.pop("reward_names")
    return meta, episode

def set_initial_state(env_raw, meta, device):
    env_ids = torch.zeros(1, dtype=torch.long, device=device)

    q_right = torch.tensor(meta["robot_init_qpos"], dtype=torch.float32, device=device).unsqueeze(0)
    env_raw.scene["robot"].write_joint_state_to_sim(q_right, torch.zeros_like(q_right), env_ids=env_ids)
    q_left = torch.tensor(meta["robot_left_init_qpos"], dtype=torch.float32, device=device).unsqueeze(0)
    env_raw.scene["robot_left"].write_joint_state_to_sim(q_left, torch.zeros_like(q_left), env_ids=env_ids)

    action_manager = env_raw.action_manager
    action_right = action_manager.get_term("arm_hand_action")
    action_right.init_joint_pos[0].copy_(q_right[0])
    action_right.del_action[0].zero_()
    action_left = action_manager.get_term("arm_hand_action_left")
    action_left.init_joint_pos[0].copy_(q_left[0])
    action_left.del_action[0].zero_()

    obj_idx = 0
    while f"object_{obj_idx}_init_pos_w" in meta:
        obj  = env_raw.scene[f"object_{obj_idx}"]
        pos  = torch.tensor(meta[f"object_{obj_idx}_init_pos_w"], dtype=torch.float32, device=device).unsqueeze(0)
        quat = torch.tensor(meta[f"object_{obj_idx}_init_quat_w"], dtype=torch.float32, device=device).unsqueeze(0)
        root_state = torch.cat([pos, quat, torch.zeros(1, 6, device=device)], dim=-1)
        obj.write_root_state_to_sim(root_state, env_ids=env_ids)
        obj_idx += 1
    
    env_raw.scene.write_data_to_sim()
    env_raw.scene.update(dt=env_raw.physics_dt)

def get_initial_sim_obs(env_raw) -> np.ndarray:
    obs_dict = env_raw.observation_manager.compute()
    return obs_dict["policy"][0].detach().cpu().numpy().astype(np.float32)

def replay_episode(env: VecEnvWrapper, symmetry_manager: SymmetryManager, meta: dict, episode: dict, cfg: DictConfig):
    state_stored = episode["state"]
    next_state_stored = episode["next_state"]
    actions_stored = episode["actions"]
    rew_right_stored = episode["rew_right"]
    rew_left_stored = episode["rew_left"]
    steps = min(cfg.max_episode_length, len(actions_stored))
    print_every = int(cfg.replay.print_every)

    env.reset()
    set_initial_state(env.unwrapped, meta, env.device)

    state_init = get_initial_sim_obs(env.unwrapped)
    state_err = np.abs(state_init - state_stored[0])
    print(f"[t=0] init state err mean={state_err.mean():.4f}, max={state_err.max():.4f}")

    state_errs = [state_err]
    next_state_errs = []
    rew_errs_right = []
    rew_errs_left = []

    state = state_init
    for i in range(steps):
        if i > 0:
            state_errs.append(np.abs(state - state_stored[i]))
        
        action = torch.tensor(actions_stored[i], dtype=torch.float32, device=env.device).unsqueeze(0)
        next_state, rew, done, extras = env.step(action)
        rew_right, rew_left = symmetry_manager.get_multi_agent_rew(extras["detailed_reward"], env.unwrapped.symmetry_tracker)
        next_state = get_obs(next_state)["policy"]
        rew_right = float(rew_right[0].item())
        rew_left = float(rew_left[0].item())

        next_state_err = np.abs(next_state - next_state_stored[i])
        rew_err_right = abs(rew_right - rew_right_stored[i])
        rew_err_left = abs(rew_left - rew_left_stored[i])

        next_state_errs.append(next_state_err)
        rew_errs_right.append(rew_err_right)
        rew_errs_left.append(rew_err_left)

        if (i + 1) % print_every == 0 or i == steps - 1:
            cur_state_err = state_errs[-1]
            print(
                f"  [t={i+1:4d}/{steps}]"
                f"  state_err(mean={cur_state_err.mean():.5f} max={cur_state_err.max():.5f})"
                f"  nxt_state_err(mean={next_state_err.mean():.5f} max={next_state_err.max():.5f})"
                f"  reward_right={rew_right:.5f} stored={rew_right_stored[i]:.5f} err={rew_err_right:.5f}"
                f"  reward_left={rew_left:.5f} stored={rew_left_stored[i]:.5f} err={rew_err_left:.5f}"
            )

        state = next_state  # was incorrectly inside the print block

        if float(done[0].item()) > 0:
            print(f" Done at step {i + 1}")
            break

    state_errs = np.stack(state_errs)
    next_state_errs = np.stack(next_state_errs)
    rew_errs_right = np.array(rew_errs_right)
    rew_errs_left = np.array(rew_errs_left)
    return {
        "steps": len(rew_errs_right),
        "state_err_mean": float(state_errs.mean()),
        "state_err_max": float(state_errs.max()),
        "next_state_err_mean": float(next_state_errs.mean()),
        "next_state_err_max": float(next_state_errs.max()),
        "rew_err_right_mean": float(rew_errs_right.mean()),
        "rew_err_left_mean": float(rew_errs_left.mean()),
    }


@hydra.main(
    config_path=symdex.LIB_PATH_PATH.joinpath("cfg").as_posix(),
    config_name="default",
)
def main(cfg: DictConfig):
    set_random_seed(cfg.seed)
    capture_keyboard_interrupt()
    cfg, env_cfg = preprocess_cfg(cfg)
    env = gym.make(cfg.env_name, cfg=env_cfg)
    env = VecEnvWrapper(env, rl_device=cfg.rl_device)
    symmetry_manager = SymmetryManager(cfg.task.multi.TD3BC, cfg.task.symmetry.symmetric_envs)
    dataset_path = cfg.replay.dataset_path
    if dataset_path is None:
        raise ValueError("Provide dataset_path=<path/to/file.h5> on the CLI.")

    print(f"[Replay] Dataset: {dataset_path}")
    f = h5py.File(dataset_path, "r")
    episode_keys = sorted(f.keys())
    print(f"[Replay] Episodes: {episode_keys}")

    all_stats = []
    for ep_key in episode_keys:
        print(f"\n{'='*60}\n  {ep_key}  ({f[ep_key]['offline_data']['actions'].shape[0]} steps)\n{'='*60}")
        meta, episode = load_episode(f, ep_key)
        stats = replay_episode(env, symmetry_manager, meta, episode, cfg)
        all_stats.append(stats)
        print(
            f"\n  [{ep_key}] steps={stats['steps']}"
            f"  state_err(mean={stats['state_err_mean']:.5f} max={stats['state_err_max']:.5f})"
            f"  nxt_state_err(mean={stats['next_state_err_mean']:.5f} max={stats['next_state_err_max']:.5f})"
            f"  rew_err right={stats['rew_err_right_mean']:.5f} left={stats['rew_err_left_mean']:.5f}"
        )

    f.close()

    print(f"\n{'='*60}\n  GLOBAL SUMMARY\n{'='*60}")
    for key in ["state_err_mean", "state_err_max", "next_state_err_mean", "next_state_err_max",
                "rew_err_right_mean", "rew_err_left_mean"]:
        vals = [s[key] for s in all_stats]
        print(f"  {key:30s}  avg={np.mean(vals):.5f}  max={np.max(vals):.5f}")

    simulation_app.close()


if __name__ == "__main__":
    main()
