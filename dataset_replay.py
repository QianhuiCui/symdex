from isaaclab.app import AppLauncher

app_launcher = AppLauncher({"headless": False, "enable_cameras": True})
simulation_app = app_launcher.app

import cv2
import torch
import hydra
import h5py
import omni.usd
import numpy as np
import gymnasium as gym
from omegaconf import DictConfig
from hydra.utils import to_absolute_path
from pxr import UsdGeom

from symdex.utils.common import set_random_seed, capture_keyboard_interrupt, preprocess_cfg
from symdex.utils.trajectory_utils import get_obs, as_flag, to_str, get_episode_init_meta
from symdex.utils.trajectory_logger import TrajectoryLogger, SAVE_DIR
from symdex.utils.rl_env_wrapper import VecEnvWrapper
from symdex.env.tasks.manager_based_env_cfg import *
import symdex


def get_recorded_seed(file, episode_key) -> int | None:
    epi_meta = file[episode_key]["epi_meta"]
    if "seed" not in epi_meta:
        return None
    seed = int(np.asarray(epi_meta["seed"][()]).reshape(-1)[0])
    return seed if seed >= 0 else None


def hide_debug_markers(env_raw, root: str = "/Visuals"):
    """Hide all debug-vis markers (frames, goal markers, contact spheres) from every render, incl. cameras.

    Hiding only `root` is not enough: VisualizationMarkers does PointInstancer.Define("/Visuals/..."), which
    auto-creates `/Visuals` as a typeless prim, and visibility authored on a non-Imageable prim is ignored by
    the renderer. So switch debug vis off at the source and hide every Imageable marker prim explicitly.
    """
    # 1) sensors (contact sensors, frame transformers): hides their markers + drops the per-frame callback
    for sensor in env_raw.scene.sensors.values():
        sensor.set_debug_vis(False)
    # 2) command terms (goal markers)
    if hasattr(env_raw, "command_manager"):
        env_raw.command_manager.set_debug_vis(False)
    # 3) task-level IK/ee markers (cfg.visualize_marker)
    if hasattr(env_raw, "markers"):
        env_raw.cfg.visualize_marker = False
        for arm_markers in env_raw.markers.values():
            for marker in arm_markers.values():
                marker.set_visibility(False)
    # 4) fallback for anything else under root: hide the top-level Imageable prims directly
    stage = omni.usd.get_context().get_stage()
    prim = stage.GetPrimAtPath(root)
    if not prim.IsValid():
        return
    num_hidden = 0
    stack = list(prim.GetChildren())
    while stack:
        child = stack.pop()
        if child.IsA(UsdGeom.Imageable):
            UsdGeom.Imageable(child).MakeInvisible()
            num_hidden += 1
        else:
            stack.extend(child.GetChildren())  # typeless group (e.g. /Visuals/Palm): look inside
    print(f"[Replay] Hid {num_hidden} debug-marker prims under {root}")


def load_episode(file, episode_key):
    grp = file[episode_key]
    meta = {}
    epi_meta_grp = grp["epi_meta"]
    for k in epi_meta_grp.keys():
        v = epi_meta_grp[k][()]
        meta[k] = [x.decode("utf-8") for x in v] if v.dtype.kind == "S" else v
    # scalar meta (e.g. language_instruction) is stored as HDF5 attrs by TrajectoryLogger
    for k, v in epi_meta_grp.attrs.items():
        meta[k] = v

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

    # mirror EMACumulativeRelativeJointPositionAction.reset() with the recorded joint positions,
    # incl. the EMA history (_prev_applied_actions), otherwise the first targets blend toward the reset pose
    action_manager = env_raw.action_manager
    for term_name, q in (("arm_hand_action", q_right), ("arm_hand_action_left", q_left)):
        term = action_manager.get_term(term_name)
        q_term = q[0, term._joint_ids]
        term.init_joint_pos[0].copy_(q_term)
        term._prev_applied_actions[0].copy_(q_term)
        term.del_action[0].zero_()

    restore_camera_pose(env_raw, meta, device)

    obj_idx = 0
    while f"object_{obj_idx}_init_pos_w" in meta:
        obj  = env_raw.scene[f"object_{obj_idx}"]
        pos  = torch.tensor(meta[f"object_{obj_idx}_init_pos_w"], dtype=torch.float32, device=device).unsqueeze(0)
        quat = torch.tensor(meta[f"object_{obj_idx}_init_quat_w"], dtype=torch.float32, device=device).unsqueeze(0)
        root_state = torch.cat([pos, quat, torch.zeros(1, 6, device=device)], dim=-1)
        obj.write_root_state_to_sim(root_state, env_ids=env_ids)
        obj_idx += 1

    # same sequence as ManagerBasedEnv.reset(): push state, refresh physics views, re-render so cameras see the new scene
    env_raw.scene.write_data_to_sim()
    env_raw.sim.forward()
    if env_raw.sim.has_rtx_sensors():
        env_raw.sim.render()
    env_raw.scene.update(dt=env_raw.physics_dt)


def restore_camera_pose(env_raw, meta, device):
    """Put cam_1 back to the recorded (possibly randomized) pose and keep env.camera_offset in sync."""
    if "cam_1_init_pos" not in meta or not hasattr(env_raw, "camera_offset") or "cam_1" not in env_raw.camera_offset:
        return
    env_ids = torch.zeros(1, dtype=torch.long, device=device)
    eye = torch.tensor(meta["cam_1_init_pos"], dtype=torch.float32, device=device).unsqueeze(0)
    quat = torch.tensor(meta["cam_1_init_quat"], dtype=torch.float32, device=device).unsqueeze(0)
    target = torch.tensor(env_raw.cfg.hydra_cfg.task.cam.cam_params["cam_1"].target, dtype=torch.float32, device=device).unsqueeze(0)
    origin = env_raw.scene.env_origins[env_ids]
    env_raw.scene["cam_1"].set_world_poses_from_view(eyes=origin + eye, targets=origin + target, env_ids=env_ids)
    env_raw.camera_offset["cam_1"]["pos"][0].copy_(eye[0])
    env_raw.camera_offset["cam_1"]["orientation"][0].copy_(quat[0])


def get_initial_obs(env: VecEnvWrapper) -> dict:
    """Observation right after set_initial_state, run through the same pipeline as env.step() obs.
    Computed once: ObservationManager.compute() advances history buffers on every call."""
    obs_dict = env.unwrapped.observation_manager.compute()
    return get_obs(env._process_obs(obs_dict), raw_data=False)


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


def replay_episode(env: VecEnvWrapper, meta: dict, episode: dict, cfg: DictConfig, ep_key: str, logger: TrajectoryLogger | None = None):
    state_stored = episode["state"]
    vision_stored = episode["vision"]
    actions_stored = episode["actions"]
    # the env times out on its own limit (episode_length_s), so respect it as well as cfg.max_episode_length
    env_max_len = int(env.max_episode_length)
    steps = min(int(cfg.max_episode_length), env_max_len, len(actions_stored))
    print_every = int(cfg.replay.print_every)
    rerecord = logger is not None
    show_vision = bool(cfg.replay.get("show_vision", True)) and vision_stored is not None

    env.reset()
    set_initial_state(env.unwrapped, meta, env.device)
    init_obs = get_initial_obs(env)

    # recorded index i = obs AFTER action i (teleop.py logs post-step obs), so there is no recorded
    # counterpart of the pre-step initial obs; errors are compared step by step inside the loop
    state_errs = []
    vision_errs = []

    if rerecord:
        if "vision" not in init_obs:
            raise RuntimeError("[Replay] replay.logging=True requires camera obs (task.cam.enable=True).")
        init_meta = get_episode_init_meta(env, cfg)
        # cfg.seed is the replay seed; keep the seed the episode was originally recorded with
        if "seed" in meta:
            init_meta["seed"] = meta["seed"]
        logger.start_episode(
            language_instruction=to_str(meta.get("language_instruction", "")),
            init_meta=init_meta,
        )

    terminated = truncated = False
    # after the recorded actions run out, hold (zero action) up to N steps so the success check can fire;
    # capped so the settle phase alone never pushes the episode into the env's time-out
    settle_steps = int(cfg.replay.get("settle_steps", 50))
    settle_steps = max(0, min(settle_steps, env_max_len - 1 - steps))
    hold_action = np.zeros_like(actions_stored[0])
    for i in range(steps + settle_steps):
        replaying = i < steps  # False -> settle phase, hold position

        action_np = actions_stored[i] if replaying else hold_action
        action = torch.tensor(action_np, dtype=torch.float32, device=env.device).unsqueeze(0)
        next_state_raw, _, done, extras = env.step(action)
        next_obs = get_obs(next_state_raw, raw_data=False)
        state = next_obs["policy"]
        vision_live = next_obs.get("vision")
        if replaying and i < len(state_stored):
            state_errs.append(np.abs(state - state_stored[i]))

        # ---- flags of THIS replay (same as teleop.py) ----
        terminated = as_flag(extras.get("terminated"))
        truncated = as_flag(extras.get("time_outs"))

        if rerecord:
            # mirror teleop.py: log (action_i, obs_after_step_i, flags_after_step_i)
            logger.add_traj(
                observation=next_obs,
                action=action_np,
                terminated=bool(terminated),
                truncated=bool(truncated),
            )

        if show_vision and replaying and i < len(vision_stored):
            vision_recorded = vision_stored[i]
            if vision_live is not None:
                vision_err = float(np.abs(vision_recorded.astype(np.float32) - vision_live.astype(np.float32)).mean())
                vision_errs.append(vision_err)
            frame = make_comparison_frame(vision_recorded, vision_live)
            cv2.imshow(f"dataset_replay: {ep_key}", frame)
            cv2.waitKey(1)
            # if video_writer is not None:
            #     video_writer.write(frame)

        if replaying and state_errs and ((i + 1) % print_every == 0 or i == steps - 1):
            cur_state_err = state_errs[-1]
            msg = (
                f"  [t={i+1:4d}/{steps}]"
                f"  state_err(mean={cur_state_err.mean():.5f} max={cur_state_err.max():.5f})"
            )
            if vision_errs:
                msg += f"  vision_err(mean_abs_pixel={vision_errs[-1]:.3f})"
            print(msg)

        if float(done[0].item()) > 0:
            phase = "" if replaying else f" (settle phase, +{i + 1 - steps} hold steps)"
            print(f" Done at step {i + 1}{phase}")
            break

    if rerecord:
        # label by what happened in THIS replay, not the source file's label
        if as_flag(meta.get("aborted", False)):
            logger.mark_aborted()
            success = False
        else:
            success = bool(terminated and not truncated)
        print(f"[Replay] Replayed outcome: {'SUCCESS' if success else 'FAILED'}")
        logger.save_episode(success=success)

    stats = {"steps": len(state_errs)}
    if state_errs:
        state_errs = np.stack(state_errs)
        stats["state_err_mean"] = float(state_errs.mean())
        stats["state_err_max"] = float(state_errs.max())
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
    dataset_path = cfg.replay.dataset_path
    if dataset_path is None:
        raise ValueError("Provide dataset_path=<path/to/file.h5> on the CLI.")
    dataset_path = to_absolute_path(dataset_path)
    print(f"[Replay] Dataset: {dataset_path}")

    f = h5py.File(dataset_path, "r")
    episode_keys = sorted(f.keys())
    print(f"[Replay] Episodes: {episode_keys}")

    recorded_seeds = {k: get_recorded_seed(f, k) for k in episode_keys}
    first_seed = recorded_seeds[episode_keys[0]]
    if first_seed is not None:
        cfg.seed = first_seed
    elif cfg.seed is None or int(cfg.seed) < 0:
        raise ValueError("[Replay] No seed in dataset and no valid cfg.seed provided; replay may be non-deterministic.")
    print(f"[Replay] Using seed: {cfg.seed}")
    set_random_seed(cfg.seed)
    capture_keyboard_interrupt()
    cfg, env_cfg = preprocess_cfg(cfg)
    env = gym.make(cfg.env_name, cfg=env_cfg)
    env = VecEnvWrapper(env, rl_device=cfg.rl_device, raw_data=False)
    hide_debug_markers(env.unwrapped)

    # the recorded operator is authoritative; fall back to cfg.logger.operator for old files
    operator = f[episode_keys[0]]["epi_meta"].attrs.get("operator", cfg.logger.operator)
    print(f"[Replay] Operator: {operator}")
    if operator is None:
        print("[Replay] WARNING: no operator in dataset or cfg — saving under mixture/")

    # re-log the replayed trajectories as new demonstrations (clean images, debug_vis=False)
    logger = None
    if bool(cfg.replay.get("logging", False)):
        logger = TrajectoryLogger(save_dir=SAVE_DIR / "no_frame", task_name=cfg.task.env_name, operator=operator)
        print(f"[Replay] Re-recording enabled, operator={operator}")

    all_stats = []
    for idx, ep_key in enumerate(episode_keys):
        print(f"\n{'='*60}\n  {ep_key}  ({f[ep_key]['offline_data']['actions'].shape[0]} steps)\n{'='*60}")
        # reseed per episode with its own recorded seed (first episode was already seeded before env creation)
        if idx > 0 and recorded_seeds[ep_key] is not None:
            set_random_seed(recorded_seeds[ep_key])
        meta, episode = load_episode(f, ep_key)
        stats = replay_episode(env, meta, episode, cfg, ep_key, logger=logger)
        all_stats.append(stats)
        msg = f"\n  [{ep_key}] steps={stats['steps']}"
        if "state_err_mean" in stats:
            msg += f"  state_err(mean={stats['state_err_mean']:.5f} max={stats['state_err_max']:.5f})"
        if "vision_err_mean" in stats:
            msg += f"  vision_err(mean={stats['vision_err_mean']:.3f} max={stats['vision_err_max']:.3f})"
        print(msg)

    f.close()
    if logger is not None:
        logger.close()
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
