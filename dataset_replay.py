from isaaclab.app import AppLauncher

app_launcher = AppLauncher({"headless": False, "enable_cameras": True})
simulation_app = app_launcher.app

import json
import time
from pathlib import Path

import h5py
import hydra
import numpy as np
import torch
import gymnasium as gym
from omegaconf import DictConfig

from symdex.utils.common import set_random_seed, capture_keyboard_interrupt, preprocess_cfg
from symdex.utils.trajectory_utils import get_obs

import symdex
from symdex.env.tasks.manager_based_env_cfg import *
from symdex.utils.rl_env_wrapper import VecEnvWrapper


def to_numpy(x):
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def scalar_bool(x):
    x = to_numpy(x)
    return bool(np.asarray(x).reshape(-1)[0])


def get_scene_entity(env, name):
    try:
        return env.unwrapped.scene[name]
    except Exception:
        return None


def get_current_qpos(env):
    robot = get_scene_entity(env, "robot")
    robot_left = get_scene_entity(env, "robot_left")

    if robot is None or robot_left is None:
        return None

    qpos = robot.data.joint_pos[0, :].detach().cpu().numpy().astype(np.float32)
    qpos_left = robot_left.data.joint_pos[0, :].detach().cpu().numpy().astype(np.float32)

    return np.concatenate([qpos, qpos_left], axis=0)


def restore_object_pose(env, object_name, pos, quat):
    obj = get_scene_entity(env, object_name)
    if obj is None:
        print(f"[Replay] Warning: scene object not found: {object_name}")
        return False

    device = env.unwrapped.device

    pos = np.asarray(pos, dtype=np.float32).reshape(3)
    quat = np.asarray(quat, dtype=np.float32).reshape(4)
    root_pose = np.concatenate([pos, quat], axis=0).reshape(1, 7)

    obj.write_root_pose_to_sim(
        torch.tensor(root_pose, device=device, dtype=torch.float32)
    )

    if hasattr(obj, "write_root_velocity_to_sim"):
        obj.write_root_velocity_to_sim(
            torch.zeros((1, 6), device=device, dtype=torch.float32)
        )

    return True


def restore_robot_qpos(env, robot_name, qpos):
    robot = get_scene_entity(env, robot_name)
    if robot is None:
        print(f"[Replay] Warning: scene robot not found: {robot_name}")
        return False

    if not hasattr(robot, "write_joint_state_to_sim"):
        raise RuntimeError(f"{robot_name} does not have write_joint_state_to_sim().")

    device = env.unwrapped.device

    qpos = np.asarray(qpos, dtype=np.float32).reshape(1, -1)
    qvel = np.zeros_like(qpos, dtype=np.float32)

    robot.write_joint_state_to_sim(
        torch.tensor(qpos, device=device, dtype=torch.float32),
        torch.tensor(qvel, device=device, dtype=torch.float32),
    )

    return True


def restore_camera_offset(env, meta):
    if not hasattr(env.unwrapped, "camera_offset"):
        return False

    if "cam_1" not in env.unwrapped.camera_offset:
        return False

    restored = False
    device = env.unwrapped.device

    if "cam_1_init_pos" in meta:
        pos = np.asarray(meta["cam_1_init_pos"][()], dtype=np.float32).reshape(3)
        env.unwrapped.camera_offset["cam_1"]["pos"][0].copy_(
            torch.tensor(pos, device=device, dtype=torch.float32)
        )
        restored = True

    if "cam_1_init_quat" in meta:
        quat = np.asarray(meta["cam_1_init_quat"][()], dtype=np.float32).reshape(4)
        env.unwrapped.camera_offset["cam_1"]["orientation"][0].copy_(
            torch.tensor(quat, device=device, dtype=torch.float32)
        )
        restored = True

    return restored


def restore_episode_initial_state(env, meta):
    restored = {}

    for object_id in [0, 1, 2]:
        object_name = f"object_{object_id}"
        pos_key = f"{object_name}_init_pos_w"
        quat_key = f"{object_name}_init_quat_w"

        if pos_key in meta and quat_key in meta:
            restored[object_name] = restore_object_pose(
                env,
                object_name,
                meta[pos_key][()],
                meta[quat_key][()],
            )
        else:
            restored[object_name] = False

    if "robot_init_qpos" in meta:
        restored["robot"] = restore_robot_qpos(
            env,
            "robot",
            meta["robot_init_qpos"][()],
        )
    else:
        restored["robot"] = False

    if "robot_left_init_qpos" in meta:
        restored["robot_left"] = restore_robot_qpos(
            env,
            "robot_left",
            meta["robot_left_init_qpos"][()],
        )
    else:
        restored["robot_left"] = False

    restored["cam_1"] = restore_camera_offset(env, meta)

    if hasattr(env.unwrapped.scene, "write_data_to_sim"):
        env.unwrapped.scene.write_data_to_sim()

    if hasattr(env.unwrapped, "sim") and hasattr(env.unwrapped.sim, "forward"):
        env.unwrapped.sim.forward()

    return restored


def compute_current_obs(env):
    base_env = env.unwrapped

    if hasattr(base_env, "_get_observations"):
        try:
            return get_obs(base_env._get_observations())
        except Exception as exc:
            print(f"[Replay] Warning: _get_observations() failed: {exc}")

    if hasattr(base_env, "get_observations"):
        try:
            return get_obs(base_env.get_observations())
        except Exception as exc:
            print(f"[Replay] Warning: get_observations() failed: {exc}")

    if hasattr(base_env, "observation_manager"):
        try:
            return get_obs(base_env.observation_manager.compute())
        except Exception as exc:
            print(f"[Replay] Warning: observation_manager.compute() failed: {exc}")

    return None


def read_obs_at(obs_group, step):
    obs = {}

    if "policy" in obs_group:
        obs["policy"] = obs_group["policy"][step]

    if "vision" in obs_group:
        obs["vision"] = obs_group["vision"][step]

    if "point_cloud" in obs_group:
        obs["point_cloud"] = obs_group["point_cloud"][step]

    return obs


def compare_array(actual, recorded, atol, rtol):
    actual = np.asarray(actual)
    recorded = np.asarray(recorded)

    if actual.shape != recorded.shape:
        return {
            "shape_match": False,
            "actual_shape": list(actual.shape),
            "recorded_shape": list(recorded.shape),
            "allclose": False,
            "max_abs": None,
            "mean_abs": None,
            "rmse": None,
        }

    diff = actual.astype(np.float64) - recorded.astype(np.float64)
    abs_diff = np.abs(diff)

    return {
        "shape_match": True,
        "actual_shape": list(actual.shape),
        "recorded_shape": list(recorded.shape),
        "allclose": bool(np.allclose(actual, recorded, atol=atol, rtol=rtol)),
        "max_abs": float(abs_diff.max()) if abs_diff.size > 0 else 0.0,
        "mean_abs": float(abs_diff.mean()) if abs_diff.size > 0 else 0.0,
        "rmse": float(np.sqrt(np.mean(diff * diff))) if diff.size > 0 else 0.0,
    }


def compare_vision(actual, recorded, pixel_tol=2):
    actual = np.asarray(actual)
    recorded = np.asarray(recorded)

    if actual.shape != recorded.shape:
        return {
            "shape_match": False,
            "actual_shape": list(actual.shape),
            "recorded_shape": list(recorded.shape),
            "exact": False,
            "within_pixel_tol": False,
            "max_abs_pixel": None,
            "mean_abs_pixel": None,
            "mismatch_rate": None,
        }

    diff = np.abs(actual.astype(np.int16) - recorded.astype(np.int16))
    mismatch = diff > int(pixel_tol)

    return {
        "shape_match": True,
        "actual_shape": list(actual.shape),
        "recorded_shape": list(recorded.shape),
        "exact": bool(np.array_equal(actual, recorded)),
        "within_pixel_tol": bool(not np.any(mismatch)),
        "max_abs_pixel": int(diff.max()) if diff.size > 0 else 0,
        "mean_abs_pixel": float(diff.mean()) if diff.size > 0 else 0.0,
        "mismatch_rate": float(mismatch.mean()) if mismatch.size > 0 else 0.0,
    }


def normalize_point_cloud_layout(point_cloud):
    pc = np.asarray(point_cloud)

    while pc.ndim > 2 and pc.shape[0] == 1:
        pc = pc[0]

    if pc.ndim > 2:
        pc = pc.reshape(-1, pc.shape[-1])

    return pc


def extract_right_left_xyz(point_cloud):
    pc = normalize_point_cloud_layout(point_cloud)

    if pc.ndim != 2:
        return None, None

    if pc.shape[1] < 6:
        print(f"[Replay][PC] Expected point_cloud shape [N, 6+] for right/left xyz, got {pc.shape}")
        return None, None

    right_xyz = pc[:, 0:3].astype(np.float32)
    left_xyz = pc[:, 3:6].astype(np.float32)

    right_valid = np.isfinite(right_xyz).all(axis=1)
    left_valid = np.isfinite(left_xyz).all(axis=1)

    right_xyz = right_xyz[right_valid]
    left_xyz = left_xyz[left_valid]

    return right_xyz, left_xyz


def compare_point_cloud(actual_pc, recorded_pc):
    actual_right, actual_left = extract_right_left_xyz(actual_pc)
    recorded_right, recorded_left = extract_right_left_xyz(recorded_pc)

    result = {}

    if actual_right is not None and recorded_right is not None:
        result["right"] = compare_array(
            actual_right,
            recorded_right,
            atol=5e-3,
            rtol=5e-3,
        )

    if actual_left is not None and recorded_left is not None:
        result["left"] = compare_array(
            actual_left,
            recorded_left,
            atol=5e-3,
            rtol=5e-3,
        )

    return result


def compare_obs(actual_obs, recorded_obs, cfg):
    result = {}

    if cfg.replay.compare_policy:
        if "policy" in actual_obs and "policy" in recorded_obs:
            result["policy"] = compare_array(
                actual_obs["policy"],
                recorded_obs["policy"],
                atol=1e-4,
                rtol=1e-4,
            )

    if "vision" in actual_obs and "vision" in recorded_obs:
        result["vision"] = compare_vision(
            actual_obs["vision"],
            recorded_obs["vision"],
            pixel_tol=2,
        )

    if "point_cloud" in actual_obs and "point_cloud" in recorded_obs:
        result["point_cloud"] = compare_point_cloud(
            actual_obs["point_cloud"],
            recorded_obs["point_cloud"],
        )

    return result


def normalize_image_for_display(image):
    image = np.asarray(image)

    while image.ndim > 3 and image.shape[0] == 1:
        image = image[0]

    if image.ndim > 3:
        image = image.reshape((-1,) + image.shape[-3:])[0]

    if image.ndim == 3 and image.shape[0] in [1, 3, 4] and image.shape[-1] not in [1, 3, 4]:
        image = np.transpose(image, (1, 2, 0))

    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)

    if image.ndim == 2:
        image = image[..., None]

    if image.ndim == 3 and image.shape[-1] == 1:
        image = np.repeat(image, 3, axis=-1)

    if image.ndim == 3 and image.shape[-1] > 3:
        image = image[..., :3]

    return image


def maybe_show_vision(recorded_obs, replay_obs, cfg):
    if not cfg.replay.show_vision:
        return

    if "vision" not in recorded_obs:
        return

    try:
        import cv2
    except Exception:
        print("[Replay] Warning: cv2 is not available, cannot show recorded vision.")
        return

    recorded = normalize_image_for_display(recorded_obs["vision"])

    if replay_obs is not None and "vision" in replay_obs:
        replay = normalize_image_for_display(replay_obs["vision"])

        if recorded.shape == replay.shape:
            panel = np.concatenate([recorded, replay], axis=1)
            window_name = "recorded vision | replay vision"
        else:
            panel = recorded
            window_name = "recorded vision"
    else:
        panel = recorded
        window_name = "recorded vision"

    scale = int(cfg.replay.vision_scale)
    if scale > 1:
        h, w = panel.shape[:2]
        panel = cv2.resize(panel, (w * scale, h * scale), interpolation=cv2.INTER_NEAREST)

    if panel.ndim == 3 and panel.shape[-1] == 3:
        panel = panel[..., ::-1]

    cv2.imshow(window_name, panel)
    cv2.waitKey(1)


class IsaacDebugPointDrawer:
    def __init__(self):
        self.draw = None

        try:
            from omni.isaac.debug_draw import _debug_draw
            self.draw = _debug_draw.acquire_debug_draw_interface()
        except Exception:
            self.draw = None

        if self.draw is None:
            try:
                from isaacsim.util.debug_draw import _debug_draw
                self.draw = _debug_draw.acquire_debug_draw_interface()
            except Exception:
                self.draw = None

        if self.draw is None:
            print("[Replay] Warning: Isaac debug draw is unavailable. Point cloud visualization disabled.")

    def clear(self):
        if self.draw is None:
            return

        try:
            self.draw.clear_points()
        except Exception:
            pass

    def draw_points(self, xyz, rgba, point_size):
        if self.draw is None:
            return

        xyz = np.asarray(xyz, dtype=np.float32)

        if xyz.ndim != 2 or xyz.shape[1] != 3 or xyz.shape[0] == 0:
            return

        points = [tuple(p) for p in xyz]
        colors = [tuple(rgba)] * xyz.shape[0]
        sizes = [float(point_size)] * xyz.shape[0]

        try:
            self.draw.draw_points(points, colors, sizes)
        except Exception as exc:
            print(f"[Replay] Warning: draw_points failed: {exc}")


def downsample_xyz(xyz, max_points):
    if xyz is None:
        return None

    if xyz.shape[0] <= max_points:
        return xyz

    stride = max(1, xyz.shape[0] // max_points)
    return xyz[::stride]


def maybe_show_point_cloud(recorded_obs, replay_obs, step, cfg, pc_drawer):
    if not cfg.replay.show_pc:
        return

    if step % int(cfg.replay.pc_every_n_steps) != 0:
        return

    if "point_cloud" not in recorded_obs:
        return

    recorded_right, recorded_left = extract_right_left_xyz(recorded_obs["point_cloud"])

    if recorded_right is None or recorded_left is None:
        print(
            f"[Replay][PC] step={step:04d} invalid recorded point_cloud shape="
            f"{np.asarray(recorded_obs['point_cloud']).shape}"
        )
        return

    replay_right = None
    replay_left = None

    if replay_obs is not None and "point_cloud" in replay_obs:
        replay_right, replay_left = extract_right_left_xyz(replay_obs["point_cloud"])

    max_points = 20000

    recorded_right = downsample_xyz(recorded_right, max_points)
    recorded_left = downsample_xyz(recorded_left, max_points)
    replay_right = downsample_xyz(replay_right, max_points)
    replay_left = downsample_xyz(replay_left, max_points)

    pc_drawer.clear()

    pc_drawer.draw_points(
        recorded_right,
        rgba=(1.0, 0.1, 0.1, 1.0),
        point_size=float(cfg.replay.point_size),
    )

    pc_drawer.draw_points(
        recorded_left,
        rgba=(1.0, 0.6, 0.1, 1.0),
        point_size=float(cfg.replay.point_size),
    )

    if replay_right is not None:
        pc_drawer.draw_points(
            replay_right,
            rgba=(0.1, 0.8, 1.0, 1.0),
            point_size=float(cfg.replay.point_size),
        )

    if replay_left is not None:
        pc_drawer.draw_points(
            replay_left,
            rgba=(0.1, 1.0, 0.4, 1.0),
            point_size=float(cfg.replay.point_size),
        )

    print(
        f"[Replay][PC] step={step:04d} "
        f"recorded_right={recorded_right.shape[0]} "
        f"recorded_left={recorded_left.shape[0]} "
        f"replay_right={0 if replay_right is None else replay_right.shape[0]} "
        f"replay_left={0 if replay_left is None else replay_left.shape[0]}"
    )


def print_step_line(
    step,
    action_copy_error,
    recorded_action,
    qpos_before,
    pre_cmp,
    post_cmp,
    reward_env,
    reward_log,
    done_env,
    done_log,
):
    action_abs_max = float(np.max(np.abs(recorded_action)))
    action_l2 = float(np.linalg.norm(recorded_action))

    parts = [
        f"[Replay] step={step:04d}",
        f"action_copy_error={action_copy_error:.3e}",
        f"action_l2={action_l2:.6f}",
        f"action_abs_max={action_abs_max:.6f}",
        f"reward_env={float(reward_env):.6f}",
        f"reward_log={float(reward_log):.6f}",
        f"reward_err={abs(float(reward_env) - float(reward_log)):.3e}",
        f"done_env={bool(done_env)}",
        f"done_log={bool(done_log)}",
    ]

    if qpos_before is not None:
        parts.append(f"qpos_abs_max={float(np.max(np.abs(qpos_before))):.6f}")

    if "policy" in pre_cmp:
        m = pre_cmp["policy"]
        if m["shape_match"]:
            parts.append(f"pre_policy_max={m['max_abs']:.3e}")
        else:
            parts.append(f"pre_policy_shape={m['actual_shape']}!={m['recorded_shape']}")

    if "policy" in post_cmp:
        m = post_cmp["policy"]
        if m["shape_match"]:
            parts.append(f"post_policy_max={m['max_abs']:.3e}")
        else:
            parts.append(f"post_policy_shape={m['actual_shape']}!={m['recorded_shape']}")

    if "vision" in post_cmp:
        m = post_cmp["vision"]
        if m["shape_match"]:
            parts.append(f"post_vision_max_px={m['max_abs_pixel']}")
            parts.append(f"post_vision_mismatch={m['mismatch_rate']:.3e}")
        else:
            parts.append(f"post_vision_shape={m['actual_shape']}!={m['recorded_shape']}")

    if "point_cloud" in post_cmp:
        pc_cmp = post_cmp["point_cloud"]

        if "right" in pc_cmp:
            m = pc_cmp["right"]
            if m["shape_match"]:
                parts.append(f"post_pc_right_max={m['max_abs']:.3e}")
                parts.append(f"post_pc_right_mean={m['mean_abs']:.3e}")
            else:
                parts.append(f"post_pc_right_shape={m['actual_shape']}!={m['recorded_shape']}")

        if "left" in pc_cmp:
            m = pc_cmp["left"]
            if m["shape_match"]:
                parts.append(f"post_pc_left_max={m['max_abs']:.3e}")
                parts.append(f"post_pc_left_mean={m['mean_abs']:.3e}")
            else:
                parts.append(f"post_pc_left_shape={m['actual_shape']}!={m['recorded_shape']}")

    print(" | ".join(parts))


def summarize_step_results(step_results):
    summary = {
        "steps": len(step_results),
        "max_action_copy_error": 0.0,
        "max_reward_error": 0.0,
        "mean_reward_error": 0.0,
        "done_mismatch_count": 0,
    }

    if len(step_results) == 0:
        return summary

    action_copy_errors = np.asarray(
        [x["action_copy_error"] for x in step_results],
        dtype=np.float64,
    )

    reward_errors = np.asarray(
        [abs(x["reward_env"] - x["reward_log"]) for x in step_results],
        dtype=np.float64,
    )

    summary["max_action_copy_error"] = float(action_copy_errors.max())
    summary["max_reward_error"] = float(reward_errors.max())
    summary["mean_reward_error"] = float(reward_errors.mean())
    summary["done_mismatch_count"] = int(
        sum(x["done_env"] != x["done_log"] for x in step_results)
    )

    policy_max = []
    pc_right_max = []
    pc_left_max = []
    vision_mismatch = []

    for item in step_results:
        post_cmp = item["post_compare"]

        if "policy" in post_cmp and post_cmp["policy"]["shape_match"]:
            policy_max.append(post_cmp["policy"]["max_abs"])

        if "vision" in post_cmp and post_cmp["vision"]["shape_match"]:
            vision_mismatch.append(post_cmp["vision"]["mismatch_rate"])

        if "point_cloud" in post_cmp:
            pc_cmp = post_cmp["point_cloud"]

            if "right" in pc_cmp and pc_cmp["right"]["shape_match"]:
                pc_right_max.append(pc_cmp["right"]["max_abs"])

            if "left" in pc_cmp and pc_cmp["left"]["shape_match"]:
                pc_left_max.append(pc_cmp["left"]["max_abs"])

    if len(policy_max) > 0:
        summary["post_policy_max_abs_over_episode"] = float(np.max(policy_max))
        summary["post_policy_mean_max_abs_over_episode"] = float(np.mean(policy_max))

    if len(vision_mismatch) > 0:
        summary["post_vision_max_mismatch_rate"] = float(np.max(vision_mismatch))
        summary["post_vision_mean_mismatch_rate"] = float(np.mean(vision_mismatch))

    if len(pc_right_max) > 0:
        summary["post_pc_right_max_abs_over_episode"] = float(np.max(pc_right_max))
        summary["post_pc_right_mean_max_abs_over_episode"] = float(np.mean(pc_right_max))

    if len(pc_left_max) > 0:
        summary["post_pc_left_max_abs_over_episode"] = float(np.max(pc_left_max))
        summary["post_pc_left_mean_max_abs_over_episode"] = float(np.mean(pc_left_max))

    return summary


def replay_episode(env, episode_group, episode_name, cfg, pc_drawer):
    print("-" * 80)
    print(f"[Replay] Episode: {episode_name}")

    meta = episode_group["epi_meta"]
    offline = episode_group["offline_data"]

    actions_h5 = offline["actions"]
    rewards_h5 = offline["rewards"]
    terminals_h5 = offline["terminals"]
    timeouts_h5 = offline["timeouts"]
    obs_h5 = offline["observations"]
    next_obs_h5 = offline["next_observations"]

    total_steps = int(actions_h5.shape[0])

    if int(cfg.replay.max_steps) > 0:
        replay_steps = min(total_steps, int(cfg.replay.max_steps))
    else:
        replay_steps = total_steps

    print(f"[Replay] Recorded steps: {total_steps}")
    print(f"[Replay] Replay steps:   {replay_steps}")
    print(f"[Replay] H5 action shape: {actions_h5.shape}")
    print("[Replay] Action handling: direct copy from H5 to env.step(). No conversion. No action-manager modification.")

    env.reset()

    # if hasattr(env.unwrapped, "update_randomization"):
    #     env.unwrapped.update_randomization(0.0)

    restored = restore_episode_initial_state(env, meta)
    print(f"[Replay] Restored initial state: {restored}")

    current_obs = compute_current_obs(env)
    if current_obs is None:
        print("[Replay] Warning: pre-action observation compare is disabled.")

    action_buf = torch.zeros(env.action_space.shape, device=cfg.rl_device, dtype=torch.float32)

    if action_buf.ndim != 2:
        raise RuntimeError(
            f"Expected env.action_space.shape to be [num_envs, action_dim], got {tuple(action_buf.shape)}"
        )

    if action_buf.shape[1] != actions_h5.shape[1]:
        raise RuntimeError(
            f"Action dimension mismatch. H5 action dim={actions_h5.shape[1]}, "
            f"env action dim={action_buf.shape[1]}"
        )

    step_results = []

    for step in range(replay_steps):
        recorded_action = np.asarray(actions_h5[step], dtype=np.float32).reshape(-1)

        qpos_before = get_current_qpos(env)

        action_buf.zero_()
        action_buf[0].copy_(
            torch.from_numpy(recorded_action).to(device=action_buf.device, dtype=action_buf.dtype)
        )

        sent_action = action_buf[0].detach().cpu().numpy().astype(np.float32)
        action_copy_error = float(np.max(np.abs(sent_action - recorded_action)))

        if action_copy_error != 0.0:
            raise RuntimeError(
                f"Action changed before env.step at step {step}: "
                f"max_abs_error={action_copy_error}"
            )

        recorded_pre_obs = read_obs_at(obs_h5, step)

        if current_obs is None:
            pre_cmp = {}
        else:
            pre_cmp = compare_obs(current_obs, recorded_pre_obs, cfg)

        next_obs_raw, reward_env, done_env, extras = env.step(action_buf)
        next_obs = get_obs(next_obs_raw)

        recorded_next_obs = read_obs_at(next_obs_h5, step)
        post_cmp = compare_obs(next_obs, recorded_next_obs, cfg)

        reward_env_scalar = float(reward_env[0].item())
        reward_log_scalar = float(np.asarray(rewards_h5[step]).item())

        done_env_scalar = scalar_bool(done_env)
        terminal_log = bool(np.asarray(terminals_h5[step]).item())
        timeout_log = bool(np.asarray(timeouts_h5[step]).item())
        done_log = bool(terminal_log or timeout_log)

        if step % int(cfg.replay.print_every) == 0:
            print_step_line(
                step=step,
                action_copy_error=action_copy_error,
                recorded_action=recorded_action,
                qpos_before=qpos_before,
                pre_cmp=pre_cmp,
                post_cmp=post_cmp,
                reward_env=reward_env_scalar,
                reward_log=reward_log_scalar,
                done_env=done_env_scalar,
                done_log=done_log,
            )

        maybe_show_vision(
            recorded_obs=recorded_next_obs,
            replay_obs=next_obs,
            cfg=cfg,
        )

        maybe_show_point_cloud(
            recorded_obs=recorded_next_obs,
            replay_obs=next_obs,
            step=step,
            cfg=cfg,
            pc_drawer=pc_drawer,
        )

        step_results.append(
            {
                "step": int(step),
                "action_copy_error": action_copy_error,
                "reward_env": reward_env_scalar,
                "reward_log": reward_log_scalar,
                "done_env": done_env_scalar,
                "done_log": done_log,
                "terminal_log": terminal_log,
                "timeout_log": timeout_log,
                "pre_compare": pre_cmp,
                "post_compare": post_cmp,
            }
        )

        current_obs = next_obs

        if done_log:
            print(f"[Replay] Recorded episode ended at step {step}.")
            break

    summary = summarize_step_results(step_results)

    print(f"[Replay] Summary for {episode_name}:")
    print(json.dumps(summary, indent=2))

    return {
        "episode": episode_name,
        "summary": summary,
        "steps": step_results,
    }


@hydra.main(
    config_path=symdex.LIB_PATH_PATH.joinpath("cfg").as_posix(),
    config_name="default",
)
def main(cfg: DictConfig):
    torch.set_printoptions(sci_mode=False, precision=3)
    cfg.seed = 42
    set_random_seed(cfg.seed)
    capture_keyboard_interrupt()

    cfg, env_cfg = preprocess_cfg(cfg)

    if cfg.dataset_path is None:
        raise RuntimeError("dataset_path is None. Set dataset_path in default.yaml.")

    dataset_path = Path(str(cfg.dataset_path)).expanduser().resolve()

    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset file does not exist: {dataset_path}")

    env = gym.make(cfg.env_name, cfg=env_cfg)
    env = VecEnvWrapper(env, rl_device=cfg.rl_device)

    pc_drawer = IsaacDebugPointDrawer()

    print(f"[Replay] Dataset: {dataset_path}")
    print(f"[Replay] Env: {cfg.env_name}")
    print(f"[Replay] compare_policy: {cfg.replay.compare_policy}")
    print(f"[Replay] show_vision: {cfg.replay.show_vision}")
    print(f"[Replay] show_pc: {cfg.replay.show_pc}")
    print("[Replay] H5 actions are sent directly to env.step().")

    all_results = []

    with h5py.File(str(dataset_path), "r") as h5_file:
        episode_names = sorted([name for name in h5_file.keys() if name.startswith("episode_")])

        if len(episode_names) == 0:
            raise RuntimeError(f"No episode_* groups found in {dataset_path}")

        for episode_name in episode_names:
            result = replay_episode(
                env=env,
                episode_group=h5_file[episode_name],
                episode_name=episode_name,
                cfg=cfg,
                pc_drawer=pc_drawer,
            )
            all_results.append(result)

    summary_path = dataset_path.with_suffix("")
    summary_path = summary_path.parent / f"{summary_path.name}_replay_compare.json"

    with open(summary_path, "w") as f:
        json.dump(
            {
                "dataset_path": str(dataset_path),
                "env_name": str(cfg.env_name),
                "timestamp": time.strftime("%Y%m%d_%H%M%S"),
                "action_handling": "direct_h5_action_to_env_step_no_conversion",
                "episodes": all_results,
            },
            f,
            indent=2,
        )

    print("-" * 80)
    print(f"[Replay] Saved summary: {summary_path}")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()