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
                
                if not hasattr(self, "_printed_pc_stats"):
                    self._printed_pc_stats = True
                    print("[Eval] pc shape:", batch["pc"].shape)
                    print("[Eval] pc min/max:", batch["pc"].min().item(), batch["pc"].max().item())
                    print("[Eval] pc mean/std:", batch["pc"].mean().item(), batch["pc"].std().item())
                    print("[Eval] pc abs max:", batch["pc"].abs().max().item())

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

        # -------------------------
        # Action group diagnostics
        # -------------------------
        action_sat_list = []
        right_arm_abs_list = []
        right_hand_abs_list = []
        left_arm_abs_list = []
        left_hand_abs_list = []
        right_arm_sat_list = []
        right_hand_sat_list = []
        left_arm_sat_list = []
        left_hand_sat_list = []

        # -------------------------
        # PC diagnostics
        # -------------------------
        pc_right_valid_list = []
        pc_left_valid_list = []
        pc_right_empty_list = []
        pc_left_empty_list = []

        # -------------------------
        # Env action-path diagnostics
        # -------------------------
        last_action_diff_list = []
        last_action_diff_max_list = []
        scaled_action_abs_list = []

        # -------------------------
        # Detailed reward diagnostics
        # detailed_return:
        #   current episode cumulative reward term per env
        # detailed_return_list:
        #   finished episode reward term values
        # -------------------------
        detailed_return = {}
        detailed_return_list = {}

        obs, _ = self.env.reset()
        last_info = None

        for _ in range(max_step):
            batch = self._build_policy_input(obs)

            # -------------------------
            # PC valid ratio diagnostics
            # -------------------------
            if "pc" in batch:
                pc = batch["pc"]
                right = pc[..., :3]
                left = pc[..., 3:6]
                right_valid = (right.abs().sum(dim=-1) > 1e-8).float()
                left_valid = (left.abs().sum(dim=-1) > 1e-8).float()
                pc_right_valid_list.append(float(right_valid.mean().item()))
                pc_left_valid_list.append(float(left_valid.mean().item()))
                pc_right_empty_list.append(float((right_valid.sum(dim=1) == 0).float().mean().item()))
                pc_left_empty_list.append(float((left_valid.sum(dim=1) == 0).float().mean().item()))

            action = policy.select_action(batch)
            action_abs = action.detach().abs()
            action_abs_list.append(float(action_abs.mean().item()))
            action_mean_list.append(float(action.detach().mean().item()))
            action_sat_list.append(float((action_abs > 0.99 * policy.max_action).float().mean().item()))

            # -------------------------
            # Action group diagnostics
            # action layout:
            #   0:6      right arm
            #   6:22     right hand
            #   22:28    left arm
            #   28:44    left hand
            # -------------------------
            right_arm = action[:, 0:6]
            right_hand = action[:, 6:22]
            left_arm = action[:, 22:28]
            left_hand = action[:, 28:44]
            for x, abs_list, sat_list in [
                (right_arm, right_arm_abs_list, right_arm_sat_list),
                (right_hand, right_hand_abs_list, right_hand_sat_list),
                (left_arm, left_arm_abs_list, left_arm_sat_list),
                (left_hand, left_hand_abs_list, left_hand_sat_list),
            ]:
                x_abs = x.detach().abs()
                abs_list.append(float(x_abs.mean().item()))
                sat_list.append(float((x_abs > 0.99 * policy.max_action).float().mean().item()))

            next_obs, reward, done, info = self.env.step(action)
            last_info = info

            # -------------------------
            # Verify action path:
            # policy action -> VecEnvWrapper clamp -> BaseEnv.last_action
            # -------------------------
            if hasattr(self.env.unwrapped, "last_action"):
                env_last_action = self.env.unwrapped.last_action.to(action.device)
                diff = (env_last_action - action).abs()
                last_action_diff_list.append(float(diff.mean().item()))
                last_action_diff_max_list.append(float(diff.max().item()))
            # Optional: requires adding self.last_scaled_action in BaseEnv.step()
            if hasattr(self.env.unwrapped, "last_scaled_action"):
                scaled_action = self.env.unwrapped.last_scaled_action
                scaled_action_abs_list.append(float(scaled_action.abs().mean().item()))

            # -------------------------
            # Accumulate episode return and length
            # -------------------------
            current_returns += reward
            current_lengths += 1

            # -------------------------
            # Accumulate detailed reward per term
            # info["detailed_reward"] should be:
            #   dict[str, tensor(num_envs)]
            # -------------------------
            if isinstance(info, dict) and "detailed_reward" in info:
                for name, value in info["detailed_reward"].items():
                    value = value.to(device=self.cfg.device).float()
                    if name not in detailed_return:
                        detailed_return[name] = torch.zeros(num_envs, dtype=torch.float32, device=self.cfg.device)
                    detailed_return[name] += value

            env_done_indices = torch.where(done > 0)[0]

            # first_done means this env is done for the first time in this eval rollout.
            # Your evaluator only counts one episode per env.
            first_done = torch.logical_and(done > 0, if_done > 0)
            first_done_indices = torch.where(first_done > 0)[0]

            if len(first_done_indices) > 0:
                return_list.extend(current_returns[first_done_indices].detach().cpu().tolist())
                step_list.extend(current_lengths[first_done_indices].detach().cpu().tolist())
                if isinstance(info, dict) and "success" in info:
                    success_list.extend(info["success"][first_done_indices].detach().cpu().tolist())
                else:
                    success_list.extend([0.0] * len(first_done_indices))
                # Store detailed reward values for completed episodes.
                for name, values in detailed_return.items():
                    if name not in detailed_return_list:
                        detailed_return_list[name] = []
                    detailed_return_list[name].extend(values[first_done_indices].detach().cpu().tolist())
                if_done[first_done_indices] = 0.0
            if len(env_done_indices) > 0:
                current_returns[env_done_indices] = 0.0
                current_lengths[env_done_indices] = 0.0
                # Reset detailed reward accumulators for envs that reset.
                for name in detailed_return:
                    detailed_return[name][env_done_indices] = 0.0
            obs = next_obs

        # -------------------------
        # Handle unfinished episodes
        # -------------------------
        unfinished_indices = torch.where(if_done > 0)[0]
        if len(unfinished_indices) > 0:
            return_list.extend(current_returns[unfinished_indices].detach().cpu().tolist())
            step_list.extend(current_lengths[unfinished_indices].detach().cpu().tolist())
            if last_info is not None and "success" in last_info:
                success_list.extend(last_info["success"][unfinished_indices].detach().cpu().tolist())
            else:
                success_list.extend([0.0] * len(unfinished_indices))
            # Store detailed reward for unfinished episodes too.
            for name, values in detailed_return.items():
                if name not in detailed_return_list:
                    detailed_return_list[name] = []
                detailed_return_list[name].extend(values[unfinished_indices].detach().cpu().tolist())

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

        return_dict = {
            "eval/return": return_mean,
            # "eval/return_std": return_std,
            # "eval/return_min": return_min,
            # "eval/return_max": return_max,
            "eval/episode_length": step_mean,
            "eval/success_rate": success_mean,
            # "eval/success_std": success_std,
            # "eval/action_abs_mean": float(np.mean(action_abs_list)) if len(action_abs_list) > 0 else 0.0,
            # "eval/action_mean": float(np.mean(action_mean_list)) if len(action_mean_list) > 0 else 0.0,
            # "eval/action_saturation_099": float(np.mean(action_sat_list)) if len(action_sat_list) > 0 else 0.0,
        }

        # -------------------------
        # Action group logs
        # -------------------------
        if len(right_arm_abs_list) > 0:
            return_dict.update({
                "eval_action/abs/right_arm": float(np.mean(right_arm_abs_list)),
                "eval_action/abs/right_hand": float(np.mean(right_hand_abs_list)),
                "eval_action/abs/left_arm": float(np.mean(left_arm_abs_list)),
                "eval_action/abs/left_hand": float(np.mean(left_hand_abs_list)),

                "eval_action/sat/right_arm": float(np.mean(right_arm_sat_list)),
                "eval_action/sat/right_hand": float(np.mean(right_hand_sat_list)),
                "eval_action/sat/left_arm": float(np.mean(left_arm_sat_list)),
                "eval_action/sat/left_hand": float(np.mean(left_hand_sat_list)),
            })

        # -------------------------
        # PC logs
        # -------------------------
        if len(pc_right_valid_list) > 0:
            return_dict.update({
                "eval_pc/right_valid_ratio": float(np.mean(pc_right_valid_list)),
                "eval_pc/left_valid_ratio": float(np.mean(pc_left_valid_list)),
                "eval_pc/right_empty_ratio": float(np.mean(pc_right_empty_list)),
                "eval_pc/left_empty_ratio": float(np.mean(pc_left_empty_list)),
            })

        # -------------------------
        # Env action path logs
        # -------------------------
        # if len(last_action_diff_list) > 0:
            # return_dict.update({"eval/env_last_action_diff_mean": float(np.mean(last_action_diff_list)),
            #                     "eval/env_last_action_diff_max": float(np.mean(last_action_diff_max_list)),})
        if len(scaled_action_abs_list) > 0:
            return_dict["eval/scaled_action_abs_mean"] = float(np.mean(scaled_action_abs_list))

        # -------------------------
        # Detailed reward logs
        # Each value is average cumulative reward term per evaluated episode.
        # -------------------------
        for name, values in detailed_return_list.items():
            if len(values) > 0:
                return_dict[f"eval_reward/{name}"] = float(np.mean(values))

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