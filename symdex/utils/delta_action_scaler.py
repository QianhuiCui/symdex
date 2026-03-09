from __future__ import annotations
import numpy as np
from dataclasses import dataclass


@dataclass
class ScalerCfg:
    arm_ramp_steps: int = 20
    arm_max_delta: float = 0.05
    arm_deadband: float = 0.01

    hand_ramp_steps: int = 2
    hand_max_delta: float = 0.03
    hand_deadband: float = 0.0


class DeltaActionScaler:
    def __init__(self, cfg: ScalerCfg = ScalerCfg()):
        self.cfg = cfg
        self._t = np.zeros(4)
        self.total_dof_expected = 44

    def reset(self):
        self._t[:] = 0

    def process(self, q_target, q_curr):
        if q_target.shape != q_curr.shape:
            raise ValueError(f"q_target shape {q_target.shape} != q_curr shape {q_curr.shape}")
        
        delta = (q_target - q_curr).astype(np.float32)
        delta_cmd = np.zeros_like(delta, dtype=np.float32)

        groups = [
            (slice(0, 6),   self.cfg.arm_ramp_steps,  self.cfg.arm_max_delta,  self.cfg.arm_deadband,  0),  # right arm
            (slice(6, 22),  self.cfg.hand_ramp_steps, self.cfg.hand_max_delta, self.cfg.hand_deadband, 1),  # right hand
            (slice(22, 28), self.cfg.arm_ramp_steps,  self.cfg.arm_max_delta,  self.cfg.arm_deadband,  2),  # left arm
            (slice(28, 44), self.cfg.hand_ramp_steps, self.cfg.hand_max_delta, self.cfg.hand_deadband, 3),  # left hand
        ]
            
        for idx, ramp_steps, max_delta, deadband, t_idx in groups:
            d = delta[idx]

            gain = min(1.0, float(self._t[t_idx] + 1) / float(ramp_steps))
            d_cmd = gain * d

            if deadband > 0:
                d_cmd[np.abs(d_cmd) < deadband] = 0.0
            d_cmd = np.clip(d_cmd, -max_delta, max_delta)

            delta_cmd[idx] = d_cmd
            self._t[t_idx] += 1

        return delta_cmd