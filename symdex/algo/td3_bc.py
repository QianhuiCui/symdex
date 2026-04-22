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


class PCEncoder(nn.Module):
    def __init__(self, pc_dim, out_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(pc_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
        )
        self.fc = nn.Sequential(
            nn.Linear(256, out_dim),
            nn.ReLU(),
        )
    
    def forward(self, pc: torch.Tensor) -> torch.Tensor:
        features = self.net(pc)
        features = features.max(dim=1).values
        features = self.fc(features)
        return features


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
        self.policy_noise = policy_noise * max_action
        self.noise_clip = noise_clip * max_action
        self.policy_freq = policy_freq
        self.alpha = alpha
        self.total_it = 0
    
    @torch.no_grad()
    def select_action(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        return self.actor(batch)
    
    def train(self, replay_buffer, batch_size):
        self.total_it += 1
        batch = replay_buffer.sample(batch_size)

        actions, rewards, not_done = batch['actions'], batch['rewards'], batch['not_done']
        cur_obs = {"state": batch['state']}
        nxt_obs = {"state": batch['next_state']}
        if 'vision' in batch:
            cur_obs['vision'] = batch['vision']
            nxt_obs['vision'] = batch['next_vision']
        if 'pc' in batch:
            cur_obs['pc'] = batch['pc']
            nxt_obs['pc'] = batch['next_pc']
        
        with torch.no_grad():
            noise = (torch.randn_like(actions) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
            nxt_actions = (self.actor_target(nxt_obs) + noise).clamp(-self.max_action, self.max_action)

            target_Q1, target_Q2 = self.critic_target(nxt_obs, nxt_actions)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = rewards + not_done * self.discount * target_Q

        current_Q1, current_Q2 = self.critic(cur_obs, actions)
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
        self.critic_optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic_optimizer.step()

        log_info = {"train/critic_loss": float(critic_loss.item())}

        if self.total_it % self.policy_freq == 0:
            pi = self.actor(cur_obs)
            Q_pi = self.critic.Q1(cur_obs, pi)

            lmbda = self.alpha / Q_pi.abs().mean().detach()
            bc_loss = F.mse_loss(pi, actions)
            actor_loss = -lmbda * Q_pi.mean() + bc_loss

            self.actor_optimizer.zero_grad(set_to_none=True)
            actor_loss.backward()
            self.actor_optimizer.step()

            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            
            log_info.update({
                "train/actor_loss": float(actor_loss.item()),
                "train/bc_loss": float(bc_loss.item()),
                "train/Q_mean": float(Q_pi.mean().item()),
                "train/lambda": float(lmbda.item()),
            })
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