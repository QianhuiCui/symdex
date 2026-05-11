from isaaclab.app import AppLauncher

# Keep this before importing torch/gym/isaac/symdex modules.
app_launcher = AppLauncher({"headless": False, "enable_cameras": True})
simulation_app = app_launcher.app

import os
from dataclasses import dataclass
from typing import Any, Dict, Optional

import h5py
import hydra
import gymnasium as gym
import numpy as np
import torch
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf

from symdex.utils.common import set_random_seed, capture_keyboard_interrupt, preprocess_cfg
from symdex.env.tasks.manager_based_env_cfg import *
import symdex
from symdex.utils.rl_env_wrapper import VecEnvWrapper


@dataclass
class EpisodeReplayData:
    name: str
    actions: torch.Tensor
    observations: Dict[str, np.ndarray]
    next_observations: Dict[str, np.ndarray]
    rewards: Optional[np.ndarray]
    terminals: Optional[np.ndarray]
    timeouts: Optional[np.ndarray]
    meta: Dict[str, Any]


def _decode_h5_attr(value):
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return value


def _read_dataset_if_exists(group: h5py.Group, key: str):
    if key in group:
        return group[key][()]
    return None


def _read_obs_group(
    group: h5py.Group,
    load_vision: bool = True,
    load_point_cloud: bool = False,
) -> Dict[str, np.ndarray]:
    obs = {}

    for key in group.keys():
        value = group[key]

        if not isinstance(value, h5py.Dataset):
            continue

        if key == "vision" and not load_vision:
            continue

        if key == "point_cloud" and not load_point_cloud:
            continue

        obs[key] = value[()]

    return obs


def _load_offline_episode_from_h5(
    file: h5py.File,
    epi_name: str,
    max_steps: int = -1,
    load_vision: bool = True,
    load_point_cloud: bool = False,
) -> EpisodeReplayData:
    ep = file[epi_name]

    if "offline_data" not in ep:
        raise RuntimeError(f"{epi_name} does not contain offline_data")

    offline = ep["offline_data"]

    if "actions" not in offline:
        raise RuntimeError(f"{epi_name}/offline_data/actions not found")

    actions_np = np.asarray(offline["actions"][()], dtype=np.float32)

    if actions_np.ndim != 2:
        raise RuntimeError(f"{epi_name} actions must be 2D, got {actions_np.shape}")

    total_steps = int(actions_np.shape[0])

    if max_steps > 0:
        total_steps = min(total_steps, int(max_steps))
        actions_np = actions_np[:total_steps]

    observations = {}
    next_observations = {}

    if "observations" in offline:
        observations = _read_obs_group(
            offline["observations"],
            load_vision=load_vision,
            load_point_cloud=load_point_cloud,
        )
        observations = {k: v[:total_steps] for k, v in observations.items()}

    if "next_observations" in offline:
        next_observations = _read_obs_group(
            offline["next_observations"],
            load_vision=load_vision,
            load_point_cloud=load_point_cloud,
        )
        next_observations = {k: v[:total_steps] for k, v in next_observations.items()}

    rewards = _read_dataset_if_exists(offline, "rewards")
    terminals = _read_dataset_if_exists(offline, "terminals")
    timeouts = _read_dataset_if_exists(offline, "timeouts")

    if rewards is not None:
        rewards = rewards[:total_steps]
    if terminals is not None:
        terminals = terminals[:total_steps]
    if timeouts is not None:
        timeouts = timeouts[:total_steps]

    meta = {}

    if "epi_meta" in ep:
        epi_meta = ep["epi_meta"]

        for k, v in epi_meta.attrs.items():
            meta[k] = _decode_h5_attr(v)

        if "reward_names" in epi_meta:
            reward_names = epi_meta["reward_names"][()]
            meta["reward_names"] = [
                x.decode("utf-8") if isinstance(x, bytes) else str(x)
                for x in reward_names
            ]

        if "reward_weights" in epi_meta:
            meta["reward_weights"] = epi_meta["reward_weights"][()].tolist()

    return EpisodeReplayData(
        name=epi_name,
        actions=torch.from_numpy(actions_np),
        observations=observations,
        next_observations=next_observations,
        rewards=rewards,
        terminals=terminals,
        timeouts=timeouts,
        meta=meta,
    )


def load_h5_replay_data(
    path: str,
    max_steps: int = -1,
    load_vision: bool = True,
    load_point_cloud: bool = False,
):
    path = to_absolute_path(path)

    if path is None:
        raise RuntimeError("cfg.dataset_path is None")

    if not os.path.exists(path):
        raise FileNotFoundError(f"File does not exist: {path}")

    ext = os.path.splitext(path)[1].lower()
    if ext not in [".h5", ".hdf5"]:
        raise ValueError(f"This script expects .h5 or .hdf5, got: {ext}")

    episodes = []

    with h5py.File(path, "r") as file:
        episode_names = sorted([k for k in file.keys() if k.startswith("episode_")])

        if not episode_names:
            raise RuntimeError(f"No episode_* group found under {path}")

        print(f"[load_h5_replay_data] Found {len(episode_names)} episodes: {episode_names}")

        for epi_name in episode_names:
            try:
                ep = _load_offline_episode_from_h5(
                    file=file,
                    epi_name=epi_name,
                    max_steps=max_steps,
                    load_vision=load_vision,
                    load_point_cloud=load_point_cloud,
                )
            except Exception as e:
                print(f"[INFO] Skip {epi_name}: {e}")
                continue

            obs_keys = sorted(list(ep.observations.keys()))
            next_obs_keys = sorted(list(ep.next_observations.keys()))

            print(
                f"[load_h5_replay_data] {epi_name}: "
                f"steps={ep.actions.shape[0]}, "
                f"action_dim={ep.actions.shape[1]}, "
                f"obs_keys={obs_keys}, "
                f"next_obs_keys={next_obs_keys}"
            )

            if "language_instruction" in ep.meta:
                print(f"[load_h5_replay_data] {epi_name} language: {ep.meta['language_instruction']}")

            episodes.append(ep)

    if len(episodes) == 0:
        raise RuntimeError(f"No valid episode loaded from {path}")

    return episodes


class RecordedVisionViewer:
    def __init__(
        self,
        enabled: bool = True,
        window_name: str = "recorded_h5_vision",
        scale: int = 4,
    ):
        self.enabled = bool(enabled)
        self.window_name = window_name
        self.scale = int(scale)
        self.cv2 = None
        self._warned = False

        if self.enabled:
            try:
                import cv2

                self.cv2 = cv2
            except Exception as e:
                self.enabled = False
                print(f"[WARN] cv2 is not available, vision viewer disabled: {e}")

    def show(self, image_rgb: np.ndarray, title: str = ""):
        if not self.enabled or self.cv2 is None:
            return

        if image_rgb is None:
            return

        img = np.asarray(image_rgb)

        if img.ndim != 3 or img.shape[-1] != 3:
            if not self._warned:
                print(f"[WARN] Expected RGB image shape (H, W, 3), got {img.shape}")
                self._warned = True
            return

        if img.dtype != np.uint8:
            img = np.clip(img, 0, 255).astype(np.uint8)

        img_bgr = self.cv2.cvtColor(img, self.cv2.COLOR_RGB2BGR)

        if self.scale > 1:
            h, w = img_bgr.shape[:2]
            img_bgr = self.cv2.resize(
                img_bgr,
                (w * self.scale, h * self.scale),
                interpolation=self.cv2.INTER_NEAREST,
            )

        if title:
            self.cv2.putText(
                img_bgr,
                title,
                (8, 22),
                self.cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
                self.cv2.LINE_AA,
            )

        self.cv2.imshow(self.window_name, img_bgr)
        self.cv2.waitKey(1)

    def close(self):
        if self.enabled and self.cv2 is not None:
            try:
                self.cv2.destroyWindow(self.window_name)
            except Exception:
                pass


class RecordedPointCloudViewer:
    def __init__(
        self,
        enabled: bool = True,
        window_name: str = "recorded_h5_point_cloud",
        point_size: float = 3.0,
        every_n_steps: int = 1,
        center_for_display: bool = True,
        show_frame: bool = True,
        bimanual_concat_xyz: bool = True,
    ):
        self.enabled = bool(enabled)
        self.window_name = window_name
        self.point_size = float(point_size)
        self.every_n_steps = max(1, int(every_n_steps))
        self.center_for_display = bool(center_for_display)
        self.show_frame = bool(show_frame)
        self.bimanual_concat_xyz = bool(bimanual_concat_xyz)

        self.o3d = None
        self.vis = None
        self.pcd = None
        self.frame = None
        self.initialized = False

        self._warned_shape = False
        self._warned_empty = False

        if self.enabled:
            try:
                import open3d as o3d
                self.o3d = o3d
            except Exception as e:
                self.enabled = False
                print(f"[WARN] open3d is not available, point cloud viewer disabled: {e}")

    def _prepare_xyz_rgb(self, point_cloud: np.ndarray):
        pc = np.asarray(point_cloud, dtype=np.float64)

        if pc.ndim != 2:
            if not self._warned_shape:
                print(f"[WARN] Expected point cloud shape (N, C), got {pc.shape}")
                self._warned_shape = True
            return None, None

        if self.bimanual_concat_xyz and pc.shape[1] == 6:
            # Your dataset format:
            #   pc[:, 0:3] = right xyz, crop y in [-0.6, 0.0]
            #   pc[:, 3:6] = left xyz,  crop y in [ 0.0, 0.6]
            right_xyz = pc[:, 0:3]
            left_xyz = pc[:, 3:6]

            right_valid = np.isfinite(right_xyz).all(axis=1)
            left_valid = np.isfinite(left_xyz).all(axis=1)

            right_xyz = right_xyz[right_valid]
            left_xyz = left_xyz[left_valid]

            xyz = np.concatenate([right_xyz, left_xyz], axis=0)

            # Use two fixed colors only for visualization:
            # right = red-ish, left = blue-ish.
            right_color = np.tile(np.asarray([[1.0, 0.25, 0.25]], dtype=np.float64), (right_xyz.shape[0], 1))
            left_color = np.tile(np.asarray([[0.25, 0.45, 1.0]], dtype=np.float64), (left_xyz.shape[0], 1))
            colors = np.concatenate([right_color, left_color], axis=0)

        elif pc.shape[1] >= 3:
            # Generic point cloud format: pc[:, 0:3] is xyz.
            xyz = pc[:, 0:3]
            valid = np.isfinite(xyz).all(axis=1)
            xyz = xyz[valid]

            colors = np.ones_like(xyz) * np.asarray([[0.85, 0.85, 0.85]], dtype=np.float64)

        else:
            if not self._warned_shape:
                print(f"[WARN] Expected point cloud shape (N, >=3), got {pc.shape}")
                self._warned_shape = True
            return None, None

        if xyz.shape[0] == 0:
            if not self._warned_empty:
                print("[WARN] Point cloud has no valid finite xyz points")
                self._warned_empty = True
            return None, None

        if self.center_for_display:
            xyz = xyz - xyz.mean(axis=0, keepdims=True)

        return xyz, colors

    def _debug_ranges(self, point_cloud: np.ndarray, step_idx: int):
        pc = np.asarray(point_cloud, dtype=np.float64)

        if pc.ndim != 2:
            return

        if pc.shape[1] == 6:
            right_xyz = pc[:, 0:3]
            left_xyz = pc[:, 3:6]

            right_xyz = right_xyz[np.isfinite(right_xyz).all(axis=1)]
            left_xyz = left_xyz[np.isfinite(left_xyz).all(axis=1)]

            if right_xyz.shape[0] > 0:
                print(
                    f"[PC DEBUG] step={step_idx} right_xyz "
                    f"min={right_xyz.min(axis=0)} "
                    f"max={right_xyz.max(axis=0)} "
                    f"mean={right_xyz.mean(axis=0)}"
                )

            if left_xyz.shape[0] > 0:
                print(
                    f"[PC DEBUG] step={step_idx} left_xyz  "
                    f"min={left_xyz.min(axis=0)} "
                    f"max={left_xyz.max(axis=0)} "
                    f"mean={left_xyz.mean(axis=0)}"
                )

        elif pc.shape[1] >= 3:
            xyz = pc[:, 0:3]
            xyz = xyz[np.isfinite(xyz).all(axis=1)]

            if xyz.shape[0] > 0:
                print(
                    f"[PC DEBUG] step={step_idx} xyz "
                    f"min={xyz.min(axis=0)} "
                    f"max={xyz.max(axis=0)} "
                    f"mean={xyz.mean(axis=0)}"
                )

    def show(self, point_cloud: np.ndarray, step_idx: int = 0):
        if not self.enabled or self.o3d is None:
            return

        if step_idx % self.every_n_steps != 0:
            return

        if step_idx % 50 == 0:
            self._debug_ranges(point_cloud, step_idx)

        xyz, colors = self._prepare_xyz_rgb(point_cloud)

        if xyz is None:
            return

        if not self.initialized:
            self.vis = self.o3d.visualization.Visualizer()
            self.vis.create_window(
                window_name=self.window_name,
                width=960,
                height=720,
                visible=True,
            )

            self.pcd = self.o3d.geometry.PointCloud()
            self.pcd.points = self.o3d.utility.Vector3dVector(xyz)
            self.pcd.colors = self.o3d.utility.Vector3dVector(colors)

            self.vis.add_geometry(self.pcd)

            if self.show_frame:
                self.frame = self.o3d.geometry.TriangleMesh.create_coordinate_frame(
                    size=0.1,
                    origin=[0.0, 0.0, 0.0],
                )
                self.vis.add_geometry(self.frame)

            render_option = self.vis.get_render_option()
            render_option.point_size = self.point_size
            render_option.background_color = np.asarray([0.0, 0.0, 0.0])

            view_control = self.vis.get_view_control()
            view_control.set_front([0.0, 0.0, -1.0])
            view_control.set_lookat([0.0, 0.0, 0.0])
            view_control.set_up([0.0, -1.0, 0.0])
            view_control.set_zoom(0.7)

            self.initialized = True

        else:
            self.pcd.points = self.o3d.utility.Vector3dVector(xyz)
            self.pcd.colors = self.o3d.utility.Vector3dVector(colors)
            self.vis.update_geometry(self.pcd)

        self.vis.poll_events()
        self.vis.update_renderer()

    def close(self):
        if self.enabled and self.vis is not None:
            try:
                self.vis.destroy_window()
            except Exception:
                pass


def _extract_policy_obs_from_env_output(env_output):
    """
    Best-effort extraction.

    Possible forms:
      env.step(...) -> obs
      env.step(...) -> obs, reward, done, info
      obs -> Tensor
      obs -> dict containing "policy"
    """
    if isinstance(env_output, tuple):
        obs = env_output[0]
    else:
        obs = env_output

    if isinstance(obs, torch.Tensor):
        return obs

    if isinstance(obs, dict):
        for key in ["policy", "obs", "observations"]:
            if key in obs and isinstance(obs[key], torch.Tensor):
                return obs[key]
            if key in obs and isinstance(obs[key], np.ndarray):
                return torch.from_numpy(obs[key])

    return None


def _compare_policy_observation(
    live_policy_obs,
    recorded_policy_np: np.ndarray,
    device: torch.device,
):
    if live_policy_obs is None or recorded_policy_np is None:
        return None

    if not isinstance(live_policy_obs, torch.Tensor):
        return None

    live = live_policy_obs.detach()

    if live.ndim >= 2:
        live = live[0]

    recorded = torch.as_tensor(recorded_policy_np, dtype=torch.float32, device=device)

    if live.shape != recorded.shape:
        return None

    live = live.to(device=device, dtype=torch.float32)

    mse = torch.mean((live - recorded) ** 2).item()
    max_abs = torch.max(torch.abs(live - recorded)).item()

    return mse, max_abs


def _make_action_batch(
    a_single: torch.Tensor,
    num_envs: int,
):
    if num_envs == 1:
        return a_single.unsqueeze(0)

    return a_single.unsqueeze(0).expand(num_envs, -1)


@hydra.main(config_path=symdex.LIB_PATH_PATH.joinpath("cfg").as_posix(), config_name="default")
def main(cfg: DictConfig):
    torch.set_printoptions(sci_mode=False, precision=3)

    set_random_seed(cfg.seed)
    capture_keyboard_interrupt()

    if cfg.dataset_path is None:
        raise RuntimeError("cfg.dataset_path is None. Please pass dataset_path=/path/to/file.h5")

    action_path = cfg.dataset_path
    print(f"[INFO] Replay source: {action_path}")

    show_vision = bool(OmegaConf.select(cfg, "replay.show_vision", default=False))
    show_pc = bool(OmegaConf.select(cfg, "replay.show_pc", default=False))
    point_size = float(OmegaConf.select(cfg, "replay.point_size", default=2.0))
    compare_policy = bool(OmegaConf.select(cfg, "replay.compare_policy", default=False))
    max_steps = int(OmegaConf.select(cfg, "replay.max_steps", default=-1))
    vision_scale = int(OmegaConf.select(cfg, "replay.vision_scale", default=4))

    # Optional extra field. You do not need to add it to default.yaml.
    pc_every_n_steps = int(OmegaConf.select(cfg, "replay.pc_every_n_steps", default=1))
    print_every = int(OmegaConf.select(cfg, "replay.print_every", default=50))

    episodes = load_h5_replay_data(
        path=action_path,
        max_steps=max_steps,
        load_vision=show_vision,
        load_point_cloud=show_pc,
    )

    cfg, env_cfg = preprocess_cfg(cfg)

    env = gym.make(cfg.env_name, cfg=env_cfg)
    env = VecEnvWrapper(env, rl_device=cfg.rl_device)

    env.reset()

    if hasattr(env.unwrapped, "update_randomization"):
        env.unwrapped.update_randomization(0.0)

    device = torch.device(cfg.rl_device)

    for ep in episodes:
        ep.actions = ep.actions.to(device=device, dtype=torch.float32)

    vision_viewer = RecordedVisionViewer(
        enabled=show_vision,
        window_name="recorded_h5_vision",
        scale=vision_scale,
    )

    pc_viewer = RecordedPointCloudViewer(
        enabled=show_pc,
        window_name="recorded_h5_point_cloud",
        point_size=point_size,
        every_n_steps=pc_every_n_steps,
        center_for_display=True,
        show_frame=True,
        bimanual_concat_xyz=True,
    )

    max_episodes = int(OmegaConf.select(cfg, "max_episodes", default=-1))

    episode_idx = 0
    step_idx = 0
    completed_episodes = 0

    print(
        f"[INFO] Start replay. "
        f"loaded_episodes={len(episodes)}, "
        f"max_episodes={max_episodes}, "
        f"num_envs={cfg.num_envs}, "
        f"device={cfg.rl_device}, "
        f"show_vision={show_vision}, "
        f"show_pc={show_pc}, "
        f"compare_policy={compare_policy}"
    )

    while simulation_app.is_running():
        if max_episodes > 0 and completed_episodes >= max_episodes:
            print(f"[INFO] Reached max_episodes={max_episodes}. Stop replay.")
            break

        ep = episodes[episode_idx]
        actions_ep = ep.actions
        T = int(actions_ep.shape[0])

        if step_idx >= T:
            completed_episodes += 1

            if max_episodes > 0 and completed_episodes >= max_episodes:
                print(f"[INFO] Reached max_episodes={max_episodes}. Stop replay.")
                break

            episode_idx = (episode_idx + 1) % len(episodes)
            step_idx = 0

            env.reset()

            if hasattr(env.unwrapped, "update_randomization"):
                env.unwrapped.update_randomization(0.0)

            print("-" * 80)
            print(
                f"[INFO] Start next episode: "
                f"episode_idx={episode_idx}, "
                f"name={episodes[episode_idx].name}, "
                f"completed={completed_episodes}"
            )
            continue

        recorded_obs = ep.observations
        recorded_next_obs = ep.next_observations

        # 1. Show recorded observation[t].
        if show_vision and "vision" in recorded_obs:
            title = f"{ep.name} obs[{step_idx}]"
            vision_viewer.show(recorded_obs["vision"][step_idx], title=title)

        if show_pc and "point_cloud" in recorded_obs:
            pc_viewer.show(recorded_obs["point_cloud"][step_idx], step_idx=step_idx)

        # 2. Step current IsaacLab/Symdex env with recorded action[t].
        a_single = actions_ep[step_idx]
        actions = _make_action_batch(a_single, int(cfg.num_envs))

        env_output = env.step(actions)

        # 3. Optional: compare live env policy obs with recorded next_observations/policy[t].
        if compare_policy and "policy" in recorded_next_obs and step_idx % print_every == 0:
            live_policy = _extract_policy_obs_from_env_output(env_output)

            result = _compare_policy_observation(
                live_policy_obs=live_policy,
                recorded_policy_np=recorded_next_obs["policy"][step_idx],
                device=device,
            )

            reward_str = ""
            if ep.rewards is not None:
                reward_str = f", reward={float(ep.rewards[step_idx]):.4f}"

            terminal_str = ""
            if ep.terminals is not None:
                terminal_str = f", terminal={int(ep.terminals[step_idx])}"

            timeout_str = ""
            if ep.timeouts is not None:
                timeout_str = f", timeout={int(ep.timeouts[step_idx])}"

            if result is None:
                print(
                    f"[INFO] {ep.name} step={step_idx}/{T} "
                    f"policy_compare=unavailable"
                    f"{reward_str}"
                    f"{terminal_str}"
                    f"{timeout_str}"
                )
            else:
                mse, max_abs = result
                print(
                    f"[INFO] {ep.name} step={step_idx}/{T} "
                    f"next_policy_mse={mse:.6e}, "
                    f"next_policy_max_abs={max_abs:.6e}"
                    f"{reward_str}"
                    f"{terminal_str}"
                    f"{timeout_str}"
                )

        step_idx += 1

    vision_viewer.close()
    pc_viewer.close()
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()