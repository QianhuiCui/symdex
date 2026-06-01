import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

from symdex.algo.network.mlp import MLPBlock, MLPNet
from symdex.algo.network.network_td3bc import StateEncoder, MultiModalEncoder


class Actor(nn.Module):
    def __init__(
        self,
        actor_cfg,
        action_dim: int,
        max_action: float,
        pc_max_points: int = 512,
        pretrain_ckpt_path: str = None,
    ):
        super().__init__()
        self.max_action = max_action
        self.encoder = MultiModalEncoder(actor_cfg, pc_max_points=pc_max_points)
        self.policy = MLPNet(
            in_dim=256,
            out_dim=action_dim,
            hidden_layers=[256, 256],
        )

        if pretrain_ckpt_path is not None:
            ckpt = torch.load(pretrain_ckpt_path, map_location="cpu", weights_only=False)
            self.encoder.pc_encoder.load_state_dict(ckpt["pc_encoder"])
            val_mse = ckpt.get("results", {}).get("pc_multilabel", {}).get("best_val_mse", None)
            mse_str = f"{val_mse:.6f}" if val_mse is not None else "N/A"
            print(f"[Actor] Loaded pretrained pc_encoder from {pretrain_ckpt_path}, val_mse={mse_str}")

    def forward(self, batch: dict) -> torch.Tensor:
        z = self.encoder(batch)                          # (B, 256)
        return self.policy(z).tanh() * self.max_action  # (B, action_dim)


class Critic(nn.Module):
    def __init__(self, critic_cfg, action_dim: int):
        super().__init__()
        self.encoder = StateEncoder(critic_cfg, out_dim=256)
        # self.q1 = nn.Sequential(
        #     nn.Linear(256 + action_dim, 256),
        #     nn.ELU(),
        #     nn.Linear(256, 256),
        #     nn.ELU(),
        #     nn.Linear(256, 1),
        # )
        # self.q2 = nn.Sequential(
        #     nn.Linear(256 + action_dim, 256),
        #     nn.ELU(),
        #     nn.Linear(256, 256),
        #     nn.ELU(),
        #     nn.Linear(256, 1),
        # )
        self.q1 = MLPNet(in_dim=256 + action_dim, out_dim=1, hidden_layers=[256, 256])
        self.q2 = MLPNet(in_dim=256 + action_dim, out_dim=1, hidden_layers=[256, 256])

    def _encode(self, batch: dict) -> torch.Tensor:
        return self.encoder(batch['state'])              # (B, 256)

    def forward(self, batch: dict, action: torch.Tensor):
        z  = self._encode(batch)
        za = torch.cat([z, action], dim=-1)              # (B, 256 + action_dim)
        return self.q1(za), self.q2(za)

    def Q1(self, batch: dict, action: torch.Tensor) -> torch.Tensor:
        z  = self._encode(batch)
        za = torch.cat([z, action], dim=-1)
        return self.q1(za)
    

class TD3BC:
    def __init__(
        self,
        actor_cfg,
        critic_cfg,
        action_dim,
        max_action,
        device,
        discount=0.99,
        tau=0.005,
        policy_noise=0.2,
        noise_clip=0.5,
        policy_freq=2,
        alpha=2.5,
        actor_lr=3e-4,
        critic_lr=3e-4,
        reward_scale=1.0,
        pc_max_points=512,
        pretrain_ckpt_path: str = None,
    ):
        self.device = torch.device(device)
        self.actor = Actor(actor_cfg, action_dim, max_action, pc_max_points, pretrain_ckpt_path).to(self.device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)

        self.critic = Critic(critic_cfg, action_dim).to(self.device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)

        self.max_action = max_action
        self.discount = discount
        self.tau = tau
        self.policy_noise = policy_noise * self.max_action
        self.noise_clip = noise_clip * self.max_action
        self.policy_freq = policy_freq
        self.alpha = alpha
        self.total_it = 0
        self.reward_scale = reward_scale
    
    @torch.no_grad()
    def select_action(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        return self.actor(batch)
    
    def train(self, replay_buffer, batch_size, log_diagnostics=False):
        self.total_it += 1
        batch = replay_buffer.sample(batch_size)
        actor_batch = replay_buffer.sample_success(batch_size)

        actions, not_done = batch['actions'], batch['not_done']
        rewards =  batch['rewards'] * self.reward_scale
        cur_obs = {"state": batch['state']}
        nxt_obs = {"state": batch['next_state']}
        if 'vision' in batch:
            cur_obs['vision'] = batch['vision']
            nxt_obs['vision'] = batch['next_vision']
        if 'pc' in batch:
            cur_obs['pc'] = batch['pc']
            nxt_obs['pc'] = batch['next_pc']
        
        actor_actions = actor_batch["actions"]
        actor_obs = {"state": actor_batch["state"]}
        if "vision" in actor_batch:
            actor_obs["vision"] = actor_batch["vision"]
        if "pc" in actor_batch:
            actor_obs["pc"] = actor_batch["pc"]

        log_info = {}

        # -------------------------
        # Batch statistics
        # -------------------------
        if log_diagnostics:
            with torch.no_grad():
                log_info.update(self._tensor_stats("batch/reward", rewards))
                log_info.update(self._tensor_stats("batch/action", actions))
                log_info.update(self._tensor_stats("batch/state", cur_obs["state"]))
                log_info.update(self._tensor_stats("batch/success_state", actor_obs["state"]))
                if "pc" in cur_obs:
                    log_info.update(self._pc_stats("pc/cur_obs", cur_obs["pc"]))
                if "pc" in actor_obs:
                    log_info.update(self._pc_stats("pc/actor_obs", actor_obs["pc"]))
                log_info["batch/done_mean"] = float((1.0 - not_done.float()).mean().item())
                log_info["batch/action_out_of_range"] = float((actions.detach().abs() > self.max_action).float().mean().item())

        # -------------------------
        # Critic target
        # -------------------------
        with torch.no_grad():
            noise = (torch.randn_like(actions) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)

            nxt_actions_raw = self.actor_target(nxt_obs)
            nxt_actions = (nxt_actions_raw + noise).clamp(-self.max_action, self.max_action)

            target_Q1, target_Q2 = self.critic_target(nxt_obs, nxt_actions)
            target_Q_min = torch.min(target_Q1, target_Q2)
            target_Q = rewards + not_done * self.discount * target_Q_min

            if log_diagnostics:
                log_info.update(self._tensor_stats("target/q", target_Q))
                log_info["target/q_gap_abs_mean"] = float((target_Q1 - target_Q2).abs().mean().item())
                log_info.update(self._action_group_stats("target/action_raw", nxt_actions_raw))
                # log_info.update(self._action_group_stats("target/action_noisy", nxt_actions))

        # -------------------------
        # Critic update
        # -------------------------
        current_Q1, current_Q2 = self.critic(cur_obs, actions)

        critic_loss_q1 = F.mse_loss(current_Q1, target_Q)
        critic_loss_q2 = F.mse_loss(current_Q2, target_Q)
        critic_loss = critic_loss_q1 + critic_loss_q2

        if log_diagnostics:
            with torch.no_grad():
                q_gap = current_Q1 - current_Q2

                log_info["critic/loss_total"] = float(critic_loss.item())
                log_info["critic/loss_q1"] = float(critic_loss_q1.item())
                log_info["critic/loss_q2"] = float(critic_loss_q2.item())
                log_info["critic/q_gap_abs_mean"] = float(q_gap.abs().mean().item())

        self.critic_optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        if log_diagnostics:
            log_info["optim/critic_grad_norm"] = self._grad_norm(self.critic)
        self.critic_optimizer.step()

        # -------------------------
        # Delayed actor update
        # -------------------------
        if self.total_it % self.policy_freq == 0:
            for p in self.critic.parameters():
                p.requires_grad_(False)
            pi = self.actor(cur_obs)
            Q_pi = self.critic.Q1(cur_obs, pi)

            q_abs_mean = Q_pi.abs().mean().detach()
            lmbda = self.alpha / (q_abs_mean + 1e-6)

            pi_bc = self.actor(actor_obs)  # only for bc
            bc_loss = F.mse_loss(pi_bc, actor_actions)
            actor_q_loss = -lmbda * Q_pi.mean()
            actor_loss = actor_q_loss + bc_loss

            self.actor_optimizer.zero_grad(set_to_none=True)
            actor_loss.backward()
            if log_diagnostics:
                log_info["optim/actor_grad_norm"] = self._grad_norm(self.actor)
            self.actor_optimizer.step()

            for p in self.critic.parameters():
                p.requires_grad_(True)

            # -------------------------
            # Actor / policy diagnostics
            # -------------------------
            if log_diagnostics:
                with torch.no_grad():
                    pi_after = self.actor(actor_obs)  # after actor update
                    Q_pi_after = self.critic.Q1(actor_obs, pi_after)
                    Q_data_after = self.critic.Q1(actor_obs, actor_actions)

                    log_info["actor/loss"] = float(actor_loss.item())
                    log_info["actor/q_loss"] = float(actor_q_loss.item())
                    log_info["actor/bc_loss"] = float(bc_loss.item())
                    log_info["actor/q_pi_mean"] = float(Q_pi.mean().item())
                    log_info["actor/lambda"] = float(lmbda.item())
                    
                    log_info.update(self._tensor_stats("q_compare/policy_action", Q_pi_after))
                    log_info.update(self._tensor_stats("q_compare/dataset_action", Q_data_after))

                    # log_info["policy/action_mse"] = float(F.mse_loss(pi_after, actor_actions).item())
                    # log_info.update(self._tensor_stats("policy/action", pi_after))   # compare with eval/action)abs_mean
                    log_info["q_compare/policy_minus_dataset"] = float((Q_pi_after - Q_data_after).mean().item())
                    log_info.update(self._action_group_stats("policy/action", pi_after))
                    # log_info.update(self._action_group_stats("dataset/success_action", actor_actions))
                    log_info.update(self._action_group_mse("policy_vs_success", pi_after, actor_actions))

            # -------------------------
            # Target network update
            # -------------------------
            # for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            #     target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            # for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            #     target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            with torch.no_grad():
                for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                    target_param.copy_(self.tau * param + (1.0 - self.tau) * target_param)
                for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                    target_param.copy_(self.tau * param + (1.0 - self.tau) * target_param)

        return log_info
    
    def save(self, path):
        torch.save({
            'actor': self.actor.state_dict(),
            'actor_target': self.actor_target.state_dict(),
            'critic': self.critic.state_dict(),
            'critic_target': self.critic_target.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict(),
            'total_it': self.total_it,
        }, path)
    
    def load(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor'])
        self.actor_target.load_state_dict(checkpoint['actor_target'])
        self.critic.load_state_dict(checkpoint['critic'])
        self.critic_target.load_state_dict(checkpoint['critic_target'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])
        self.total_it = checkpoint.get('total_it', 0)
    
    # ================ Helper functions ================
    def _tensor_stats(self, name: str, x: torch.Tensor) -> dict:
        x = x.detach()
        return {
            f"{name}/mean": float(x.mean().item()),
            # f"{name}/std": float(x.std(unbiased=False).item()),
            # f"{name}/min": float(x.min().item()),
            # f"{name}/max": float(x.max().item()),
            # f"{name}/abs_mean": float(x.abs().mean().item()),
            f"{name}/abs_max": float(x.abs().max().item()),
        }
    
    def _pc_stats(self, prefix: str, pc: torch.Tensor) -> dict:
        pc = pc.detach()
        out = {
            f"{prefix}/mean": float(pc.mean().item()),
            f"{prefix}/std": float(pc.std(unbiased=False).item()),
            # f"{prefix}/abs_max": float(pc.abs().max().item()),s
        }

        right = pc[..., :3]
        left = pc[..., 3:6]
        right_valid = (right.abs().sum(dim=-1) > 1e-8).float()
        left_valid = (left.abs().sum(dim=-1) > 1e-8).float()
        right_count = right_valid.sum(dim=1)
        left_count = left_valid.sum(dim=1)

        out[f"{prefix}/right_valid_ratio"] = float(right_valid.mean().item())
        out[f"{prefix}/left_valid_ratio"] = float(left_valid.mean().item())
        out[f"{prefix}/right_empty_ratio"] = float((right_count == 0).float().mean().item())
        out[f"{prefix}/left_empty_ratio"] = float((left_count == 0).float().mean().item())
        # out[f"{prefix}/right_valid_points_mean"] = float(right_count.mean().item())
        # out[f"{prefix}/left_valid_points_mean"] = float(left_count.mean().item())
        # out[f"{prefix}/right_abs_max"] = float(right.abs().max().item())
        # out[f"{prefix}/left_abs_max"] = float(left.abs().max().item())
        return out
    
    def _action_group_stats(self, prefix: str, action: torch.Tensor) -> dict:
        action = action.detach()
        groups = {
            "right_arm": action[:, 0:6],
            "right_hand": action[:, 6:22],
            "left_arm": action[:, 22:28],
            "left_hand": action[:, 28:44],
        }
        out = {}

        for name, x in groups.items():
            abs_x = x.abs()
            out[f"{prefix}/{name}_mean"] = float(x.mean().item())
            # out[f"{prefix}/{name}_abs_mean"] = float(abs_x.mean().item())
            out[f"{prefix}/{name}_abs_max"] = float(abs_x.max().item())
            # out[f"{prefix}/{name}_sat_095"] = float((abs_x > 0.95 * self.max_action).float().mean().item())
            # out[f"{prefix}/{name}_sat_099"] = float((abs_x > 0.99 * self.max_action).float().mean().item())

        out[f"{prefix}/all_mean"] = float(action.mean().item())
        # out[f"{prefix}/all_abs_mean"] = float(action.abs().mean().item())
        out[f"{prefix}/all_abs_max"] = float(action.abs().max().item())
        # out[f"{prefix}/all_sat_099"] = float((action.abs() > 0.99 * self.max_action).float().mean().item())
        return out

    def _action_group_mse(self, prefix: str, pred: torch.Tensor, target: torch.Tensor) -> dict:
        pred = pred.detach()
        target = target.detach()

        groups = {
            "right_arm": (pred[:, 0:6], target[:, 0:6]),
            "right_hand": (pred[:, 6:22], target[:, 6:22]),
            "left_arm": (pred[:, 22:28], target[:, 22:28]),
            "left_hand": (pred[:, 28:44], target[:, 28:44]),
        }
        out = {}

        for name, (p, t) in groups.items():
            out[f"{prefix}/{name}_mse"] = float(F.mse_loss(p, t).item())
            # out[f"{prefix}/{name}_mae"] = float(F.l1_loss(p, t).item())

        out[f"{prefix}/all_mse"] = float(F.mse_loss(pred, target).item())
        # out[f"{prefix}/all_mae"] = float(F.l1_loss(pred, target).item())
        return out

    def _grad_norm(self, module: nn.Module) -> float:
        total_sq = 0.0
        for p in module.parameters():
            if p.grad is not None:
                param_norm = p.grad.detach().norm(2).item()
                total_sq += param_norm ** 2
        return float(total_sq ** 0.5)

    # def _param_norm(self, module: nn.Module) -> float:
    #     total_sq = 0.0
    #     with torch.no_grad():
    #         for p in module.parameters():
    #             param_norm = p.detach().data.norm(2).item()
    #             total_sq += param_norm ** 2
    #     return float(total_sq ** 0.5)