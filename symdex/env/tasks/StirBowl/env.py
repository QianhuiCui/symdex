from __future__ import annotations

import torch
from typing import Any, ClassVar

from isaacsim.core.version import get_version
from isaaclab.envs.common import VecEnvStepReturn

from symdex.env.tasks.manager_based_env import *
from symdex.env.tasks.StirBowl.env_cfg import StirBowlEnvCfg


class StirBowlEnv(BaseEnv):
    is_vector_env: ClassVar[bool] = True
    """Whether the environment is a vectorized environment."""
    metadata: ClassVar[dict[str, Any]] = {
        "render_modes": [None, "human", "rgb_array"],
        "isaac_sim_version": get_version(),
    }
    """Metadata for the environment."""

    cfg: StirBowlEnvCfg
    """Configuration for the environment."""
    
    def step(self, action: torch.Tensor) -> VecEnvStepReturn:
        super().step(action)
        # Debug only
        if self.cfg.visualize_marker:
            left_palm_idx = self.scene["robot_left"].find_bodies("palm_link")[0]
            left_palm = self.scene["robot_left"].data.body_state_w[:, left_palm_idx, :7].reshape(-1, 7)
            self.markers['arm_r']['ee_marker'].visualize(left_palm[:, :3], left_palm[:, 3:7])
            # self.markers['arm_r']['goal_marker'].visualize(self.scene["object_0"].data.root_state_w[:, :3], self.scene["object_0"].data.root_state_w[:, 3:7])
            self.markers['arm_l']['goal_marker'].visualize(self.scene["object_1"].data.root_state_w[:, :3], self.scene["object_1"].data.root_state_w[:, 3:7])

        # return observations, rewards, resets and extras
        return self.obs_buf, self.reward_buf, self.reset_terminated, self.reset_time_outs, self.extras
    
    """
    Helper functions.
    """