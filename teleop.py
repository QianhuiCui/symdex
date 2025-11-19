from isaaclab.app import AppLauncher

app_launcher = AppLauncher({"headless": False})
simulation_app = app_launcher.app

import torch
import hydra
from omegaconf import DictConfig
import gymnasium as gym
import threading
import time 

from isaaclab.utils.math import quat_from_matrix

from symdex.utils.common import set_random_seed, capture_keyboard_interrupt, preprocess_cfg
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
    # env.unwrapped.update_randomization(0.0)

    # ---- Debug prints ----
    # print("[DEBUG] Registered actions:", list(env.unwrapped.action_manager.active_terms))
    # print("[DEBUG] Total action dim:", env.unwrapped.action_manager.total_action_dim)
    # print("[DEBUG] Controlled joints for right arm action dim:", env.unwrapped.action_manager.get_term("arm_hand_action").action_dim)
    # print("[DEBUG] Current joint pos (first 6):", env.unwrapped.scene["robot"].data.joint_pos[0, :6].cpu().numpy())
    # print("[DEBUG] IsaacLab right arm joint nums:", env.unwrapped.scene["robot"].num_joints)
    # print("[DEBUG] Right robot joint order:", env.unwrapped.scene["robot"].data.joint_names)
    # print("[DEBUG] Left robot joint order:", env.unwrapped.scene["robot_left"].data.joint_names)
    # print("Right arm joint limits:", env.unwrapped.scene["robot"].data.joint_limits)
    # print("Left arm joint limits:", env.unwrapped.scene["robot_left"].data.joint_limits)
    # print("[IO] slices:", slices)
    # print("Effective action_scale:", env.unwrapped.cfg.action_scale)
    print("[DEBUG] Config max_episode_length:", env.unwrapped.max_episode_length)

    threading.Thread(target=recv_teleop, daemon=True).start()

    # Initialize logger if enabled
    use_logger = getattr(cfg, "logger", False)
    if use_logger:
        from symdex.utils.trajectory_logger import TrajectoryLogger
        logger = TrajectoryLogger(task_name=cfg.task.env_name)
        print("[Teleop] Logger enabled.")

    print("[IsaacLab Teleop] Started. Waiting for teleop input...") 
    just_reset = True

    while simulation_app.is_running():
        with _data_lock:
            q_value = teleop_joint_data["value"]
            pose_right = teleop_target_poses["right"] 
            pose_left = teleop_target_poses["left"]
            timestamp = teleop_joint_data["timestamp"]
        
        if q_value is None:
            time.sleep(0.01)
            continue
        # print(f"[DEBUG] q_value right: {q_value[:22]}, \n[DEBUG] q_value left: {q_value[22:]}")
        q_tensor = torch.tensor([q_value], dtype=torch.float32, device=cfg.rl_device)
        obs, rew, reset, extras = env.step(q_tensor)
        # print("[DEBUG]: observation keys: ", obs.keys())
        success_flag = bool(extras.get("success", False))

        # ---- Debug prints ----
        # q_current_right = env.unwrapped.scene["robot"].data.joint_pos[0].cpu().numpy()
        # print("[DEBUG] Step right robot joint pos:", q_current_right)
        # q_current_left = env.unwrapped.scene["robot_left"].data.joint_pos[0].cpu().numpy()
        # print("[DEBUG] Step left robot joint pos:", q_current_left)
        # if reset.any():
        #     print("[DEBUG] Reset flags:", reset.cpu().numpy())

        if use_logger:
            # logger.add_sample(
            #     t=timestamp if timestamp is not None else time.time(),
            #     q=q_value,
            #     left_pose=pose_left,
            #     right_pose=pose_right,
            # )
            scene = env.unwrapped.scene
            robot_r = scene["robot"]
            robot_l = scene["robot_left"]
            # joint states
            joint_pos_r = robot_r.data.joint_pos[0].cpu().numpy()
            joint_vel_r = robot_r.data.joint_vel[0].cpu().numpy()
            joint_pos_l = robot_l.data.joint_pos[0].cpu().numpy()
            joint_vel_l = robot_l.data.joint_vel[0].cpu().numpy()
            # cartesian poses
            ee_pos_r = robot_r.data.body_state_w[:, robot_r.find_bodies("palm_link"), :3].cpu().numpy()
            ee_pos_l = robot_l.data.body_state_w[:, robot_l.find_bodies("palm_link"), :3].cpu().numpy()

            observation = {
                "joint_pos_right": joint_pos_r,
                "joint_pos_left": joint_pos_l,
                "joint_vel_right": joint_vel_r,
                "joint_vel_left": joint_vel_l,
                "ee_pose_right": ee_pos_r,
                "ee_pose_left": ee_pos_l,
                # 'exterior_image_1_left': Image(shape=(180, 320, 3), dtype=uint8),
                # 'exterior_image_2_left': Image(shape=(180, 320, 3), dtype=uint8),
                # 'wrist_image_left': Image(shape=(180, 320, 3), dtype=uint8),
            }

            action = q_value
            action_dict = {
                "arm_hand_action_right": q_value[:22],
                "arm_hand_action_left": q_value[22:],
                "ee_target_pose_right": pose_right,
                "ee_target_pose_left": pose_left,
            }

            logger.add_step(
                action=action,
                action_dict=action_dict,
                observation=observation,
                reward=float(rew[0].item() if torch.is_tensor(rew) else rew),
                is_first=just_reset,
                is_last=bool(reset.any() or success_flag),
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

        elif reset.any():
            print("[Teleop] Environment reset triggered auto-reset internally.")
            env.reset()
            just_reset = True
            if use_logger:
                logger.start_new_trial()
                count += 1

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

        if use_logger and count == 25:
            logger.h5_file.flush()
            print(f"[Teleop] Flushed data with {count} trials logged.")

    if use_logger:
        logger.close()
    env.close()
    print("[IsaacLab Teleop] Closed cleanly.")


if __name__ == "__main__":
    main()
    simulation_app.close()