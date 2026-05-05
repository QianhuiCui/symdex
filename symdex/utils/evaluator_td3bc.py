import time
from copy import deepcopy
import torch
import numpy as np


class EvaluatorTD3BC:
    def __init__(self, cfg, env_cfg, env, wandb_run, state_mean, state_std):
        self.cfg = deepcopy(cfg)
        self.env_cfg = env_cfg
        self.env = env
        self.wandb_run = wandb_run
        self.state_mean = state_mean
        self.state_std = state_std

        self.start_time = time.time()

    def _normalize_state(self, state: torch.Tensor) -> torch.Tensor:
        if self.state_mean is None or self.state_std is None:
            return state
        mean = torch.as_tensor(self.state_mean, dtype=state.dtype, device=state.device)
        std = torch.as_tensor(self.state_std, dtype=state.dtype, device=state.device)
        return (state - mean) / std
    
    def _build_policy_input(self, obs):
        if isinstance(obs, dict):
            batch = {}

            # low_dim policy obs
            state = self._normalize_state(obs["policy"])
            batch["state"] = state

            # optional vision
            if getattr(self.cfg.algo.observation, "vision", False):
                batch["vision"] = obs["vision"]
            if getattr(self.cfg.algo.observation, "pc", False):
                batch["pc"] = obs["point_cloud"]

            return batch
        
        state = self._normalize_state(obs)
        return {"state": state}

    @torch.no_grad()
    def eval_policy(self, policy, success_max=None):
        num_envs = self.cfg.num_envs
        max_step = self.env.max_episode_length

        current_returns = torch.zeros(num_envs, dtype=torch.float32, device=self.cfg.device)
        current_lengths = torch.zeros(num_envs, dtype=torch.float32, device=self.cfg.device)
        if_done = torch.ones(num_envs, dtype=torch.float32, device=self.cfg.device)

        return_list = []
        step_list = []
        success_list = []
        action_abs_list = []
        action_mean_list = []
        
        obs, _ = self.env.reset()
        last_info = info

        for _ in range(max_step):
            batch = self._build_policy_input(obs)
            action = policy.select_action(batch)
            next_obs, reward, done, info = self.env.step(action)

            current_returns += reward
            current_lengths += 1
            env_done_indices = torch.where(done > 0)[0]
            first_done = torch.logical_and(done > 0, if_done > 0)
            first_done_indices = torch.where(first_done > 0)[0]

            if len(first_done_indices) > 0:
                return_list.extend(current_returns[first_done_indices].detach().cpu().tolist())
                step_list.extend(current_lengths[first_done_indices].detach().cpu().tolist())
                success_list.extend(info['success'][first_done_indices].detach().cpu().tolist())
                if_done[first_done_indices] = 0.0
            if len(env_done_indices) > 0:
                current_returns[env_done_indices] = 0.0
                current_lengths[env_done_indices] = 0.0
            
            obs = next_obs
        
        unfinished_indices = torch.where(if_done > 0)[0]
        if len(unfinished_indices) > 0:
            return_list.extend(current_returns[unfinished_indices].detach().cpu().tolist())
            step_list.extend(current_lengths[unfinished_indices].detach().cpu().tolist())
            if last_info is not None and "success" in last_info:
                success_list.extend(last_info["success"][unfinished_indices].detach().cpu().tolist())
            else:
                success_list.extend([0.0] * len(unfinished_indices))
        
        # return_mean = float(np.mean(return_list)) if len(return_list) > 0 else 0.0
        # step_mean = float(np.mean(step_list)) if len(step_list) > 0 else 0.0
        # success_mean = float(np.mean(success_list)) if len(success_list) > 0 else 0.0
        returns_np = np.asarray(return_list, dtype=np.float32)
        steps_np = np.asarray(step_list, dtype=np.float32)
        success_np = np.asarray(success_list, dtype=np.float32)

        if len(returns_np) > 0:
            return_mean = float(np.mean(returns_np))
            return_std = float(np.std(returns_np))
            return_min = float(np.min(returns_np))
            return_max = float(np.max(returns_np))
        else:
            return_mean = 0.0
            return_std = 0.0
            return_min = 0.0
            return_max = 0.0

        step_mean = float(np.mean(steps_np)) if len(steps_np) > 0 else 0.0
        success_mean = float(np.mean(success_np)) if len(success_np) > 0 else 0.0
        success_std = float(np.std(success_np)) if len(success_np) > 0 else 0.0

        # return_dict = {
        #     "eval/return": return_mean,
        #     "eval/episode_length": step_mean,
        #     "eval/success_rate": success_mean,
        # }
        return_dict = {
            "eval/return": return_mean,
            "eval/return_std": return_std,
            "eval/return_min": return_min,
            "eval/return_max": return_max,
            "eval/episode_length": step_mean,
            "eval/success_rate": success_mean,
            "eval/success_std": success_std,
            "eval/num_episodes": len(return_list),
            "eval/action_abs_mean": float(np.mean(action_abs_list)) if len(action_abs_list) > 0 else 0.0,
            "eval/action_mean": float(np.mean(action_mean_list)) if len(action_mean_list) > 0 else 0.0,
        }

        if success_max is not None:
            if success_mean > success_max:
                success_max = success_mean
                if self.cfg.save_model:
                    policy.save(f"{self.wandb_run.dir}/model_best.pth")
            return return_dict, success_max
        
        return return_dict
    
    def check_if_should_stop(self, step=None):
        if self.cfg.max_step is not None:
            return step > self.cfg.max_step
        else:
            return (time.time() - self.start_time) > self.cfg.max_time