import time

from isaaclab.app import AppLauncher

app_launcher = AppLauncher({"headless": False, "enable_cameras": True})
simulation_app = app_launcher.app

import cv2
import torch
import hydra
import h5py
import numpy as np
import gymnasium as gym
from pathlib import Path
from omegaconf import DictConfig
from hydra.utils import to_absolute_path

from symdex.utils.common import set_random_seed, capture_keyboard_interrupt, preprocess_cfg
from symdex.utils.trajectory_utils import get_obs
from symdex.utils.rl_env_wrapper import VecEnvWrapper
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
        "vision": data["observations"]["vision"][()],
        "actions": data["actions"][()],
        "terminals": data["terminals"][()],
        "timeouts": data["timeouts"][()],
    }
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

def get_initial_sim_state(env_raw) -> np.ndarray:
    obs_dict = env_raw.observation_manager.compute()
    return obs_dict["policy"][0].detach().cpu().numpy().astype(np.float32)

def get_initial_sim_vision(env_raw) -> np.ndarray | None:
    """Raw (unwrapped) camera obs is (num_envs, num_cams, H, W, frame_stack*3), HWC layout.
    Take the most recent frame in the stack for env 0, cam 0."""
    obs_dict = env_raw.observation_manager.compute()
    if "vision" not in obs_dict:
        return None
    img = obs_dict["vision"][0, 0, ..., -3:].detach().cpu().numpy()
    if img.dtype != np.uint8:
        mn, mx = float(img.min()), float(img.max())
        if mn >= -1.0 and mx <= 1.0:
            img = ((img + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
        elif mx <= 1.0:
            img = (img * 255.0).clip(0, 255).astype(np.uint8)
        else:
            img = img.clip(0, 255).astype(np.uint8)
    return img

def make_comparison_frame(recorded: np.ndarray, replayed: np.ndarray | None) -> np.ndarray:
    """Stacks recorded / replayed / abs-diff camera frames side by side (BGR, for cv2)."""
    recorded_bgr = cv2.cvtColor(recorded, cv2.COLOR_RGB2BGR)
    panels = [recorded_bgr]
    labels = ["recorded"]

    if replayed is not None:
        replayed_bgr = cv2.cvtColor(replayed, cv2.COLOR_RGB2BGR)
        if replayed_bgr.shape[:2] != recorded_bgr.shape[:2]:
            replayed_bgr = cv2.resize(replayed_bgr, (recorded_bgr.shape[1], recorded_bgr.shape[0]))
        diff = cv2.absdiff(recorded_bgr, replayed_bgr)
        panels += [replayed_bgr, diff]
        labels += ["replayed", "abs diff"]

    canvas = cv2.hconcat(panels)
    canvas = cv2.copyMakeBorder(canvas, 20, 0, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    panel_w = recorded_bgr.shape[1]
    for idx, label in enumerate(labels):
        cv2.putText(canvas, label, (idx * panel_w + 5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return canvas

def replay_episode(env: VecEnvWrapper, meta: dict, episode: dict, cfg: DictConfig, ep_key: str):
    state_stored = episode["state"]
    vision_stored = episode["vision"]
    actions_stored = episode["actions"]
    steps = min(cfg.max_episode_length, len(actions_stored))
    print_every = int(cfg.replay.print_every)

    show_vision = bool(cfg.replay.get("show_vision", True)) and vision_stored is not None
    vision_out_dir = cfg.replay.get("vision_out_dir", None)
    video_writer = None
    if show_vision and vision_out_dir:
        out_dir = Path(to_absolute_path(vision_out_dir))
        out_dir.mkdir(parents=True, exist_ok=True)
        video_path = out_dir / f"{ep_key}.mp4"
        h, w = vision_stored.shape[1], vision_stored.shape[2] * 3
        video_writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (w, h))

    env.reset()
    set_initial_state(env.unwrapped, meta, env.device)

    state_init = get_initial_sim_state(env.unwrapped)
    vision_init = get_initial_sim_vision(env.unwrapped)

    state_err = np.abs(state_init - state_stored[0])
    print(f"[t=0] init state err mean={state_err.mean():.4f}, max={state_err.max():.4f}")

    state_errs = [state_err]
    vision_errs = []

    if show_vision:
        frame = make_comparison_frame(vision_stored[0], vision_init)
        cv2.imshow(f"dataset_replay: {ep_key}", frame)
        cv2.waitKey(1)
        if video_writer is not None:
            video_writer.write(frame)

    state = state_init
    for i in range(steps):
        if i > 0:
            state_errs.append(np.abs(state - state_stored[i]))

        action = torch.tensor(actions_stored[i], dtype=torch.float32, device=env.device).unsqueeze(0)
        next_state_raw, _, done, _ = env.step(action)
        next_obs = get_obs(next_state_raw, raw_data=False)
        state = next_obs["policy"]
        vision_live = next_obs.get("vision")

        if show_vision and (i + 1) < len(vision_stored):
            vision_recorded_next = vision_stored[i + 1]
            if vision_live is not None:
                vision_err = float(np.abs(vision_recorded_next.astype(np.float32) - vision_live.astype(np.float32)).mean())
                vision_errs.append(vision_err)
            frame = make_comparison_frame(vision_recorded_next, vision_live)
            cv2.imshow(f"dataset_replay: {ep_key}", frame)
            cv2.waitKey(1)
            if video_writer is not None:
                video_writer.write(frame)

        if (i + 1) % print_every == 0 or i == steps - 1:
            cur_state_err = state_errs[-1]
            msg = (
                f"  [t={i+1:4d}/{steps}]"
                f"  state_err(mean={cur_state_err.mean():.5f} max={cur_state_err.max():.5f})"
            )
            if vision_errs:
                msg += f"  vision_err(mean_abs_pixel={vision_errs[-1]:.3f})"
            print(msg)

        if float(done[0].item()) > 0:
            print(f" Done at step {i + 1}")
            break

    if video_writer is not None:
        video_writer.release()
        print(f"[Replay] Saved comparison video: {video_path}")

    state_errs = np.stack(state_errs)
    stats = {
        "steps": len(state_errs),
        "state_err_mean": float(state_errs.mean()),
        "state_err_max": float(state_errs.max()),
    }
    if vision_errs:
        vision_errs = np.array(vision_errs)
        stats["vision_err_mean"] = float(vision_errs.mean())
        stats["vision_err_max"] = float(vision_errs.max())
    return stats


@hydra.main(
    config_path=symdex.LIB_PATH_PATH.joinpath("cfg").as_posix(),
    config_name="default",
)
def main(cfg: DictConfig):
    if cfg.seed is None or int(cfg.seed) < 0:
        cfg.seed = int(time.time() * 1000) % (2**31 - 1)
    print(f"[Teleop] Using seed: {cfg.seed}")
    set_random_seed(cfg.seed)
    capture_keyboard_interrupt()
    cfg, env_cfg = preprocess_cfg(cfg)
    env = gym.make(cfg.env_name, cfg=env_cfg)
    env = VecEnvWrapper(env, rl_device=cfg.rl_device, raw_data=False)
    dataset_path = cfg.replay.dataset_path
    if dataset_path is None:
        raise ValueError("Provide dataset_path=<path/to/file.h5> on the CLI.")
    dataset_path = to_absolute_path(dataset_path)

    print(f"[Replay] Dataset: {dataset_path}")
    f = h5py.File(dataset_path, "r")
    episode_keys = sorted(f.keys())
    print(f"[Replay] Episodes: {episode_keys}")

    all_stats = []
    for ep_key in episode_keys:
        print(f"\n{'='*60}\n  {ep_key}  ({f[ep_key]['offline_data']['actions'].shape[0]} steps)\n{'='*60}")
        meta, episode = load_episode(f, ep_key)
        stats = replay_episode(env, meta, episode, cfg, ep_key)
        all_stats.append(stats)
        msg = (
            f"\n  [{ep_key}] steps={stats['steps']}"
            f"  state_err(mean={stats['state_err_mean']:.5f} max={stats['state_err_max']:.5f})"
        )
        if "vision_err_mean" in stats:
            msg += f"  vision_err(mean={stats['vision_err_mean']:.3f} max={stats['vision_err_max']:.3f})"
        print(msg)

    f.close()
    cv2.destroyAllWindows()

    print(f"\n{'='*60}\n  GLOBAL SUMMARY\n{'='*60}")
    for key in ["state_err_mean", "state_err_max", "vision_err_mean", "vision_err_max"]:
        vals = [s[key] for s in all_stats if key in s]
        if vals:
            print(f"  {key:20s}  avg={np.mean(vals):.5f}  max={np.max(vals):.5f}")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
