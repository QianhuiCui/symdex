from __future__ import annotations

import torch
from typing import Any, ClassVar
from functools import reduce
from isaacsim.core.version import get_version
from isaaclab.envs.common import VecEnvStepReturn

from symdex.env.tasks.manager_based_env import BaseEnv
from symdex.env.tasks.Pouring.env_cfg import PouringEnvCfg


class PouringEnv(BaseEnv):
    is_vector_env: ClassVar[bool] = True
    """Whether the environment is a vectorized environment."""
    metadata: ClassVar[dict[str, Any]] = {
        "render_modes": [None, "human", "rgb_array"],
        "isaac_sim_version": get_version(),
    }
    """Metadata for the environment."""

    cfg: PouringEnvCfg
    """Configuration for the environment."""
 
    def step(self, action: torch.Tensor) -> VecEnvStepReturn:
        super().step(action)
        
        # Debug only
        if self.cfg.visualize_marker:
            self.markers['arm_l']['goal_marker'].visualize(self.scene["object_0"].data.root_state_w[:, :3], self.scene["object_0"].data.root_state_w[:, 3:7])
            self.markers['arm_r']['goal_marker'].visualize(self.scene["object_1"].data.root_state_w[:, :3], self.scene["object_1"].data.root_state_w[:, 3:7])
            right_palm_idx = self.scene["robot"].find_bodies("palm_link")[0]
            right_palm = self.scene["robot"].data.body_state_w[:, right_palm_idx, :7].reshape(-1, 7)
            self.markers['arm_r']['ee_marker'].visualize(right_palm[:, :3], right_palm[:, 3:7])
            left_palm_idx = self.scene["robot_left"].find_bodies("palm_link")[0]
            left_palm = self.scene["robot_left"].data.body_state_w[:, left_palm_idx, :7].reshape(-1, 7)
            self.markers['arm_l']['ee_marker'].visualize(left_palm[:, :3], left_palm[:, 3:7])

        # return observations, rewards, resets and extras
        return self.obs_buf, self.reward_buf, self.reset_terminated, self.reset_time_outs, self.extras
    
    def _pre_init_process(self):
        super()._pre_init_process()
        # environment specific initialization
        self.success_tracker_step = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.object_success_tracker = torch.zeros(self.num_object, self.num_envs, device=self.device)
        self.object_bonus_tracker = torch.zeros(self.num_object, self.num_envs, device=self.device)
        self.object_lift_tracker = torch.zeros(self.num_object, self.num_envs, device=self.device)
        # specific to pouring
        self.cup_reach_first_target = torch.zeros(self.num_envs, device=self.device)

    def _post_reset_process(self, env_ids):
        super()._post_reset_process(env_ids)
        self.object_success_tracker[:, env_ids] = 0.0
        self.object_bonus_tracker[:, env_ids] = 0.0
        self.object_lift_tracker[:, env_ids] = 0.0
        self.success_tracker_step[env_ids] = 0.0
        self.cup_reach_first_target[env_ids] = 0.0