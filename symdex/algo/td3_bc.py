import copy
import torch
import torch.nn as nn
import torch.nn.functional as F


class StateEncoder(nn.Module):
    def __init__(self, state_dim, out_dim=256):
        super().__init__() 
        self.net = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, out_dim),
            nn.ReLU(),
        )
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state)


class VisionEncoder(nn.Module):
    def __init__(self, in_channels=3, out_dim=256):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=8, stride=4, padding=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.fc = nn.Sequential(
            nn.Linear(128, out_dim),
            nn.ReLU(),
        )
    
    def forward(self, img: torch.Tensor) -> torch.Tensor:
        # img shape: (B, V, C, H, W)
        b, v, c, h, w = img.shape
        img = img.reshape(b * v, c, h, w)
        features = self.cnn(img).reshape(b * v, -1)
        features = self.fc(features)
        features = features.reshape(b, v, -1)
        features = features.mean(dim=1)
        return features


# class PCEncoder(nn.Module):
#     def __init__(self, pc_dim, out_dim=256):
#         super().__init__()
#         self.net = nn.Sequential(
#             nn.Linear(pc_dim, 64),
#             nn.ReLU(),
#             nn.Linear(64, 128),
#             nn.ReLU(),
#             nn.Linear(128, 256),
#             nn.ReLU(),
#         )
#         self.fc = nn.Sequential(
#             nn.Linear(256, out_dim),
#             nn.ReLU(),
#         )
    
#     def forward(self, pc: torch.Tensor) -> torch.Tensor:
#         features = self.net(pc)
#         features = features.max(dim=1).values
#         features = self.fc(features)
#         return features


class PointNetPCEncoder(nn.Module):
    def __init__(self, in_dim=3, out_dim=128):
        super().__init__()
        self.point_mlp = nn.Sequential(
            nn.Linear(in_dim * 2, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            # nn.Linear(128, 256),
            # nn.LayerNorm(256),
            # nn.ReLU(),
        )
        # max pool + mean pool + center + radius
        self.fc = nn.Sequential(
            nn.Linear(128 * 2 + 4, 128),
            nn.LayerNorm(128),
            nn.ReLU(),

            nn.Linear(128, out_dim),
            nn.LayerNorm(out_dim),
            nn.ReLU(),
        )
    
    def forward(self, pc_xyz: torch.Tensor) -> torch.Tensor:
        # pc_xyz: (B, N, 3), return: (B, out_dim)
        pc_xyz = torch.nan_to_num(pc_xyz, nan=0.0, posinf=0.0, neginf=0.0)
        valid = pc_xyz.abs().sum(dim=-1) > 1e-8  # (B, N)
        # Protect against empty crop.
        no_valid = valid.sum(dim=1) == 0
        if no_valid.any():
            valid = valid.clone()
            valid[no_valid, 0] = True
        valid_f = valid.float().unsqueeze(-1)  # (B, N, 1)
        count = valid_f.sum(dim=1, keepdim=True).clamp_min(1.0)  # (B, 1, 1)

        center = (pc_xyz * valid_f).sum(dim=1, keepdim=True) / count  # (B, 1, 3)
        pc_centered = pc_xyz - center  # relative position
        # keep shape and position
        radius = (pc_centered.norm(dim=-1, keepdim=True) * valid_f).amax(dim=1, keepdim=True).clamp_min(1e-6)  # (B, 1, 1)
        pc_norm = pc_centered / radius

        # raw xyz provides abs position info; norm xyz provides local geometry info
        x = torch.cat([pc_xyz, pc_norm], dim=-1)  # (B, N, 6)
        h = self.point_mlp(x)  # (B, N, 256)
        # masked max pooling
        h_max = h.masked_fill(~valid.unsqueeze(-1), -1e9).max(dim=1).values
        # masked mean pooling
        h_mean = h.masked_fill(~valid.unsqueeze(-1), 0.0).sum(dim=1) / count.squeeze(1)
        # center: (B, 3), radius: (B, 1)
        geom = torch.cat([center.squeeze(1), radius.squeeze(1)], dim=-1)
        return self.fc(torch.cat([h_max, h_mean, geom], dim=-1))


class PCEncoder(nn.Module):
    def __init__(self, pc_dim, out_dim=256):
        super().__init__()
        self.right_encoder = PointNetPCEncoder(in_dim=3, out_dim=128)
        self.left_encoder = PointNetPCEncoder(in_dim=3, out_dim=128)
        self.fusion = nn.Sequential(
            nn.Linear(128 * 2, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, out_dim),
            nn.LayerNorm(out_dim),
            nn.ReLU(),
        )
    
    def forward(self, pc: torch.Tensor) -> torch.Tensor:
        pc_right = pc[..., :3]
        pc_left = pc[..., 3: 6]

        feat_right = self.right_encoder(pc_right)
        feat_left = self.left_encoder(pc_left)

        return self.fusion(torch.cat([feat_right, feat_left], dim=-1))


class MultiModalEncoder(nn.Module):
    def __init__(self, state_dim, use_vision=False, use_pc=False, vision_channels=3, pc_dim=6, out_dim=256):
        super().__init__()
        self.use_vision = use_vision
        self.use_pc = use_pc
        self.state_encoder = StateEncoder(state_dim, out_dim)
        feat_dim = [out_dim]
        if use_vision:
            self.vision_encoder = VisionEncoder(vision_channels, out_dim)
            feat_dim.append(out_dim)
        else:
            self.vision_encoder = None
        if use_pc:
            self.pc_encoder = PCEncoder(pc_dim, out_dim)
            feat_dim.append(out_dim)
        else:
            self.pc_encoder = None
        
        self.fusion = nn.Sequential(
            nn.Linear(sum(feat_dim), 512),
            nn.ReLU(),
            nn.Linear(512, out_dim),
        )
    
    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        features = [self.state_encoder(batch['state'])]
        if self.use_vision:
            features.append(self.vision_encoder(batch['vision']))
        if self.use_pc:
            features.append(self.pc_encoder(batch['pc']))
        features = torch.cat(features, dim=-1)
        features = self.fusion(features)
        return features


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action, use_vision=False, use_pc=False):
        super().__init__()
        self.encoder = MultiModalEncoder(state_dim, use_vision, use_pc)
        self.net = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim),
            nn.Tanh(),
        )
        self.max_action = max_action

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        features = self.encoder(batch)
        actions = self.net(features)
        return actions * self.max_action


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, use_vision=False, use_pc=False):
        super().__init__()
        self.encoder1 = MultiModalEncoder(state_dim, use_vision, use_pc)
        self.encoder2 = MultiModalEncoder(state_dim, use_vision, use_pc)

        self.q1 = nn.Sequential(
            nn.Linear(256 + action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )
        self.q2 = nn.Sequential(
            nn.Linear(256 + action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )
    
    def forward(self, batch: dict[str, torch.Tensor], action: torch.Tensor):
        features1 = self.encoder1(batch)
        features2 = self.encoder2(batch)
        q1 = self.q1(torch.cat([features1, action], dim=-1))
        q2 = self.q2(torch.cat([features2, action], dim=-1))
        return q1, q2
    
    def Q1(self, batch: dict[str, torch.Tensor], action: torch.Tensor):
        features = self.encoder1(batch)
        return self.q1(torch.cat([features, action], dim=-1))
    

class TD3BC:
    def __init__(
        self, 
        state_dim,
		action_dim,
		max_action,
        device,
        use_vision = False,
        use_pc = False,
		discount=0.99,
		tau=0.005,
		policy_noise=0.2,
		noise_clip=0.5,
		policy_freq=2,
		alpha=2.5,
        actor_lr=3e-4,
        critic_lr=3e-4,
        reward_scale=1.0,
    ):
        self.device = torch.device(device)
        self.actor = Actor(state_dim, action_dim, max_action, use_vision, use_pc).to(self.device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)

        self.critic = Critic(state_dim, action_dim, use_vision, use_pc).to(self.device)
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