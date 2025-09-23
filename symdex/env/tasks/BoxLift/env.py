from __future__ import annotations

import torch
from typing import Any, ClassVar

from isaacsim.core.version import get_version
from isaaclab.envs.common import VecEnvStepReturn

from symdex.env.tasks.manager_based_env import BaseEnv
from symdex.env.tasks.BoxLift.env_cfg import BoxLiftEnvCfg


class BoxLiftEnv(BaseEnv):
    is_vector_env: ClassVar[bool] = True
    """Whether the environment is a vectorized environment."""
    metadata: ClassVar[dict[str, Any]] = {
        "render_modes": [None, "human", "rgb_array"],
        "isaac_sim_version": get_version(),
    }
    """Metadata for the environment."""

    cfg: BoxLiftEnvCfg
    """Configuration for the environment."""
    
    def step(self, action: torch.Tensor) -> VecEnvStepReturn:
        super().step(action)
        # Debug only
        if self.cfg.visualize_marker:
            from symdex.env.tasks.BoxLift.mdps import compute_side_points
            right_frame = compute_side_points(self.scene["object_0"].data.root_state_w, self.side_points, side="right")
            self.markers['arm_r']['ee_marker'].visualize(right_frame, self.scene["tote_right"].data.target_quat_w.reshape(-1, 4))
            left_frame = compute_side_points(self.scene["object_0"].data.root_state_w, self.side_points, side="left")
            self.markers['arm_l']['ee_marker'].visualize(left_frame, self.scene["tote_left"].data.target_quat_w.reshape(-1, 4))

        # return observations, rewards, resets and extras
        return self.obs_buf, self.reward_buf, self.reset_terminated, self.reset_time_outs, self.extras