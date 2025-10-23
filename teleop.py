from isaaclab.app import AppLauncher

app_launcher = AppLauncher({"headless": False})
simulation_app = app_launcher.app

import torch
import hydra
from omegaconf import DictConfig
import gymnasium as gym
import threading
import time 

from symdex.utils.common import set_random_seed, capture_keyboard_interrupt, preprocess_cfg
from symdex.env.tasks.manager_based_env_cfg import *
from symdex.utils.rl_env_wrapper import VecEnvWrapper
import sys
sys.path.append("/home/qianhui/Desktop/dex_bimanual_telep/scripts")
from zmq_utils import recv_msg

import symdex


teleop_joint_data = {"value": None, "timestamp": None}
_data_lock = threading.Lock()

def recv_teleop():
    print("[Teleop Receiver] Listening for teleoperation joint data...")
    while simulation_app.is_running():
        topic, msg = recv_msg(["teleop_joint_state"])
        if msg is None:
            time.sleep(0.01)
            continue
        if msg is not None:
            print(f"[Teleop Receiver] Got message: {list(msg.keys())}, len(value)={len(msg.get('value', []))}")

        with _data_lock:
            teleop_joint_data["value"] = msg.get("value", None)
            teleop_joint_data["timestamp"] = msg.get("timestamp", None)

        latency = time.time() - teleop_joint_data["timestamp"]
        print(f"[Teleop Receiver] Received joint data. Latency: {latency*1000:.2f} ms")
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
    count = 0
    env.reset()
    env.unwrapped.update_randomization(0.0)

    threading.Thread(target=recv_teleop, daemon=True).start()
    print("[IsaacLab Teleop] Started. Waiting for teleop input...")

    while simulation_app.is_running():
        with _data_lock:
            q_value = teleop_joint_data["value"]

        if q_value is not None:
            q_tensor = torch.tensor([q_value], dtype=torch.float32, device=cfg.rl_device)
            env.step(q_tensor)
        # else:
        #     action_dim = env.unwrapped.action_manager.total_action_dim
        #     env.step(torch.zeros((env.num_envs, action_dim), device=cfg.rl_device))

        count += 1
        if count % 10 == 0:
            count = 0
            env.reset()
            print("-" * 80)
            print("[IsaacLab Teleop] Environment reset (periodic).")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
