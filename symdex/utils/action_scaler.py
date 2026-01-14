from dataclasses import dataclass
import numpy as np
import torch


@dataclass
class ActionScalerCfg:
    warmup_steps: int = 20
    max_delta: float = 0.05
    

class ActionScaler:
    def __init__(self, env, env_cfg, cfg: ActionScalerCfg):
        self.env = env
        self.cfg = cfg
        self.action_dim = env_cfg.action_dim

        self.robot = env.unwrapped.scene["robot"]
        self.rotbot_left = env.unwrapped.scene["robot_left"]
        self.ids = self.robot.find_joints(".*")[0]
        self.ids_left = self.rotbot_left.find_joints(".*")[0]

        self.t = 0
        self.start = None
        self.prev = None

    def reset(self):
        joints = self.robot.data.joint_pos[0, self.ids].detach().cpu().numpy()
        joints_left = self.rotbot_left.data.joint_pos[0, self.ids_left].detach().cpu().numpy()

        self.start = np.concatenate([joints, joints_left], axis=0)
        self.prev = self.start.copy()
        self.t = 0

    def scale(self, action):
        alpha = min(1.0, (self.t + 1) / self.cfg.warmup_steps)
        action_sampled = (1.0 - alpha) * self.start + alpha * action

        delta = action_sampled - self.prev
        delta = np.clip(delta, -self.cfg.max_delta, self.cfg.max_delta)
        action_scaled = self.prev + delta

        self.prev = action_scaled
        self.t += 1
        return action_scaled
