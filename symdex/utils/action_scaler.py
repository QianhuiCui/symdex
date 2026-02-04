from dataclasses import dataclass
import numpy as np
import torch


@dataclass
class ActionScalerCfg:
    warmup_steps: int = 20
    max_delta: float | np.ndarray = 0.03
    deadband: float | np.ndarray = 0.005
    ema_alpha: float | np.ndarray = 0.75


class ActionScaler:
    def __init__(self, env, env_cfg, cfg: ActionScalerCfg, joint_lower=None, joint_upper=None):
        self.env = env
        self.scaler_cfg = cfg
        self.action_dim = env_cfg.action_dim

        self.robot = env.unwrapped.scene["robot"]
        self.robot_left = env.unwrapped.scene["robot_left"]
        self.ids = self.robot.find_joints(".*")[0]
        self.ids_left = self.robot_left.find_joints(".*")[0]

        self.t = 0
        self.start = None
        self.prev = None
        self.ema_state = None

        # expand scaler cfg to vectors
        self.max_delta = self._as_vec(cfg.max_delta, self.action_dim, name="max_delta")
        self.deadband = self._as_vec(cfg.deadband, self.action_dim, name="deadband")
        self.ema_alpha = self._as_vec(cfg.ema_alpha, self.action_dim, name="ema_alpha")

        self.joint_lower = np.asarray(joint_lower, dtype=np.float32).reshape(-1) if joint_lower is not None and joint_lower.shape == (self.action_dim,) else None
        self.joint_upper = np.asarray(joint_upper, dtype=np.float32).reshape(-1) if joint_upper is not None and joint_upper.shape == (self.action_dim,) else None

    def reset(self):
        joints = self.robot.data.joint_pos[0, self.ids].detach().cpu().numpy().astype(np.float32)
        joints_left = self.robot_left.data.joint_pos[0, self.ids_left].detach().cpu().numpy().astype(np.float32)

        self.start = np.concatenate([joints, joints_left], axis=0)
        self.prev = self.start.copy()
        self.ema_state = self.start.copy()
        self.t = 0

    def process(self, action):
        if self.prev is None or self.start is None or self.ema_state is None:
            self.reset()
        
        # warmup blending
        alpha = min(1.0, (self.t + 1) / self.scaler_cfg.warmup_steps)
        action = (1.0 - alpha) * self.start + alpha * action

        # deadband
        delta = action - self.prev
        mask_small = np.abs(delta) < self.deadband
        delta[mask_small] = 0.0
        action = self.prev + delta

        # per-dim rate limit
        delta_2 = action - self.prev
        delta_2_clipped = np.clip(delta_2, -self.max_delta, self.max_delta)
        action = self.prev + delta_2_clipped

        # EMA smoothing on action
        self.ema_state = self.ema_alpha * action + (1 - self.ema_alpha) * self.ema_state
        action = self.ema_state

        action = np.clip(action, self.joint_lower, self.joint_upper) if self.joint_lower is not None and self.joint_upper is not None else action

        # commit state
        self.prev = action
        self.t += 1
        return action

    # ---- Helper functions ----
    def _as_vec(self, x, dim: int, name: str):
        if isinstance(x, (float, int)):
            return np.full((dim,), float(x), dtype=np.float32)
        x_arr = np.asarray(x, dtype=np.float32)
        if x_arr.shape != (dim,):
            raise ValueError(f"Expected {name} to have shape ({dim},), but got {x_arr.shape}")
        return x_arr