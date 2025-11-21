from isaaclab.app import AppLauncher

app_launcher = AppLauncher({"headless": False, "enable_cameras": True})
simulation_app = app_launcher.app

import torch
import hydra
from omegaconf import DictConfig
import gymnasium as gym
import threading
import time 

from isaaclab.utils.math import quat_from_matrix

from symdex.utils.common import set_random_seed, capture_keyboard_interrupt, preprocess_cfg
from symdex.utils.trajectory_utils import build_observation_dict, now_ms, rgb_to_HWC, depth_to_gray
from symdex.env.tasks.manager_based_env_cfg import *
from symdex.utils.rl_env_wrapper import VecEnvWrapper
import sys
sys.path.append("/home/qianhui/Desktop/dex_bimanual_telep/scripts")
from zmq_utils import recv_msg

import symdex


teleop_joint_data = {"value": None, "timestamp": None}
teleop_target_poses = {"right": None, "left": None}
_data_lock = threading.Lock()

def recv_teleop():
    print("[Teleop Receiver] Listening for teleoperation joint data...")
    while simulation_app.is_running():
        topic, msg = recv_msg(["teleop_joint_state", "human_wrist_poses"])
        if msg is None:
            time.sleep(0.01)
            continue
        # if msg is not None:
        #     print(f"[Teleop Receiver] Got message: {list(msg.keys())}, len(value)={len(msg.get('value', []))}")

        with _data_lock:
            if topic == "human_wrist_poses":
                teleop_target_poses["right"] = msg.get("right", None)
                teleop_target_poses["left"] = msg.get("left", None)
            elif topic == "teleop_joint_state":
                teleop_joint_data["value"] = msg.get("value", None)
                teleop_joint_data["timestamp"] = msg.get("timestamp", None)

        # latency = time.time() - teleop_joint_data["timestamp"]
        # print(f"[Teleop Receiver] Received joint data. Latency: {latency*1000:.2f} ms")
        time.sleep(0.001)

@hydra.main(
    config_path=symdex.LIB_PATH_PATH.joinpath("cfg").as_posix(),
    config_name="default"
)
def main(cfg: DictConfig):
    """Launch teleoperation control in Isaac Lab using TeleopDifferentialIKAction."""
    torch.set_printoptions(sci_mode=False, precision=3)
    set_random_seed(cfg.seed)
    capture_keyboard_interrupt()

    cfg, env_cfg = preprocess_cfg(cfg)

    env = gym.make(cfg.env_name, cfg=env_cfg)
    env = VecEnvWrapper(env, rl_device=cfg.rl_device)
    env.reset()

    # ---- Debug prints ----
    print("[DEBUG] Config max_episode_length:", env.unwrapped.max_episode_length)

    threading.Thread(target=recv_teleop, daemon=True).start()

    # Initialize logger if enabled
    use_logger = getattr(cfg, "logger", False)
    if use_logger:
        from symdex.utils.trajectory_logger import TrajectoryLogger
        logger = TrajectoryLogger(task_name=cfg.task.env_name)
        max_episodes = getattr(cfg, "max_episodes", None)
        episodes_saved = 0
        print("[Teleop] Logger enabled.")

    print("[IsaacLab Teleop] Started. Waiting for teleop input...") 
    just_reset = True

    while simulation_app.is_running():
        with _data_lock:
            q_value = teleop_joint_data["value"]
            pose_right = teleop_target_poses["right"] 
            pose_left = teleop_target_poses["left"]
            src_timestamp = teleop_joint_data["timestamp"]
        
        if q_value is None:
            time.sleep(0.01)
            continue

        # ---- control timing ----
        t_step_start = now_ms()

        q_np = np.asarray(q_value, dtype=np.float32)
        q_tensor = torch.as_tensor(q_np[None, ...], dtype=torch.float32, device=cfg.rl_device)

        t_policy_start = now_ms()

        obs, rew, reset, extras = env.step(q_tensor)
        print("[DEBUG] Observation Terms: ", obs)

        t_sleep_start = now_ms()
        t_control_start = t_sleep_start
        t_step_end = now_ms()

        # ---- flags & reward ----
        reward_scalar = float(rew[0].item() if torch.is_tensor(rew) else rew)
        success_flag = bool(extras.get("success", False))
        done_flag = bool(reset.any() or success_flag)

        # ---- controller info ----
        io_latency_ms = None
        ts_src_ms = None
        if src_timestamp is not None:
            try:
                ts = float(src_timestamp)
                ts_src_ms = ts * 1000.0 if ts < 1e12 else (ts / 1e6 if ts > 1e14 else ts)
                io_latency_ms = max(0.0, t_step_start - ts_src_ms)
            except Exception:
                io_latency_ms = None
                ts_src_ms = None
        controller_info = {"has_teleop": q_value is not None, "io_latency_ms": io_latency_ms}

        # ---- build observation dict ----
        obs_dict = build_observation_dict(env, obs)
        # ---- extract heavy data: vision frames -> video ----
        if use_logger and "vision" in obs_dict:
            v = obs_dict["vision"]
            if isinstance(v, dict) and "rgb_image" in v:
                frame_np = rgb_to_HWC(v["rgb_image"])
                logger.add_video_frame(camera_id="cam_1", frame=frame_np)
                v.pop("rgb_image")
                obs_dict["vision_meta"]["rgb"] = {"camera_ids": ["cam_1"], "shape": tuple(frame_np.shape)}
            if isinstance(v, dict) and "depth" in v:
                d_u8 = depth_to_gray(v["depth"])
                logger.add_video_frame(camera_id="cam_1", frame=d_u8)
                v.pop("depth")
                obs_dict["vision_meta"]["depth"] = {"camera_ids": ["cam_1"], "shape": tuple(d_u8.shape)}
                # ---- extract heavy data: point cloud -> per-step dataset ----
        if use_logger and "point_cloud" in obs_dict:
            pc = obs_dict["point_cloud"]
            if isinstance(pc, dict) and "point_cloud" in pc:
                point_cloud_arr = np.asarray(pc["point_cloud"], dtype=np.float32)
                pc.pop("point_cloud")
            elif isinstance(pc, np.ndarray):
                point_cloud_arr = np.asarray(pc, dtype=np.float32)
                obs_dict.pop("point_cloud")
            logger.add_point_cloud(point_cloud_arr)
            obs_dict["point_cloud_meta"] = {"num_points": int(point_cloud_arr.shape[0]),
                                            "dim": int(point_cloud_arr.shape[1]) if point_cloud_arr.ndim == 2 else None,}
        # ---- timestamps & intent ----
        obs_dict["timestamp"] = {
            "control": {
                "step_start": int(t_step_start),
                "policy_start": int(t_policy_start),
                "sleep_start": int(t_sleep_start),
                "control_start": int(t_control_start),
                "step_end": int(t_step_end),
            },
            "source": {"teleop_input_ms": None if ts_src_ms is None else float(ts_src_ms)},
        }
        obs_dict["controller_info"] = controller_info
        # ---- target ---- 
        targets = {}
        if pose_right is not None:
            targets["ee_target_pose_right"] = np.asarray(pose_right, dtype=np.float32)
        if pose_left is not None:
            targets["ee_target_pose_left"] = np.asarray(pose_left, dtype=np.float32)
        if targets:
            obs_dict["targets"] = targets

        # ---- action & action_dict ----
        action = q_np
        action_dict = {
            "arm_hand_action_right": q_np[:22],
            "arm_hand_action_left": q_np[22:],
        }

        if use_logger:
            logger.add_step(
                action=action,
                action_dict=action_dict,
                observation=obs_dict,
                reward=reward_scalar,
                is_first=bool(just_reset),
                is_last=done_flag,
                is_terminal=bool(success_flag),
                language_instruction="",
                discount=1.0,
            )

        # Episode management
        if success_flag:
            print("[Teleop] Task success, resetting environment.")
            env.reset()
            just_reset = True
            if use_logger:
                logger.save_episode()
                episodes_saved += 1
                if (max_episodes is not None) and (episodes_saved >= max_episodes):
                    print(f"[Teleop] Reached max_episodes={max_episodes}. Stopping.")
                    break
        elif reset.any():
            print("[Teleop] Environment reset triggered auto-reset internally.")
            env.reset()
            just_reset = True
            if use_logger:
                logger.save_episode()
                episodes_saved += 1
                if (max_episodes is not None) and (episodes_saved >= max_episodes):
                    print(f"[Teleop] Reached max_episodes={max_episodes}. Stopping.")
                    break
        else:
            just_reset = False

        if pose_right is not None:
            H = np.array(pose_right)
            R = torch.as_tensor(H[:3, :3]).unsqueeze(0)
            pos = torch.as_tensor(H[:3, 3]).unsqueeze(0)
            quat = quat_from_matrix(R)
            env.unwrapped.scene["target_sphere"].write_root_pose_to_sim(torch.cat([pos, quat], dim=-1))

        if pose_left is not None:
            H = np.array(pose_left)
            R = torch.as_tensor(H[:3, :3]).unsqueeze(0)
            pos = torch.as_tensor(H[:3, 3]).unsqueeze(0)
            quat = quat_from_matrix(R)
            env.unwrapped.scene["target_sphere_left"].write_root_pose_to_sim(torch.cat([pos, quat], dim=-1))

    if use_logger:
        logger.close()
    env.close()
    print("[IsaacLab Teleop] Closed cleanly.")


if __name__ == "__main__":
    main()
    simulation_app.close()