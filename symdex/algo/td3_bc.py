from copy import deepcopy
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from symdex.algo.ac_base import ActorCriticBase
from symdex.replay.nstep_replay import NStepReplay
from symdex.utils.noise import add_mixed_normal_noise, add_normal_noise
from symdex.utils.schedule_util import ExponentialSchedule, LinearSchedule
from symdex.utils.torch_util import soft_update
from symdex.utils.common import handle_timeout, load_class_from_path
from symdex.algo.network import model_name_to_path
from symdex.utils.symmetry import load_symmetric_system, SymmetryManager
from symdex.utils.offline_buffer import OfflineBuffer

@dataclass
class AgentITD3BC(ActorCriticBase):
    def __post_init__(self):
        super().__post_init__()
        self.algo_multi_cfg = self.cfg.task.multi.TD3BC
        self.single_obs_dim = list(self.cfg.task.multi.TD3BC.single_agent_obs_dim)
        self.single_action_dim = int(self.cfg.task.multi.TD3BC.single_agent_action_dim)
        self.obs_dim = self.cfg.task.multi.shared_obs_dim
        self.action_dim = self.single_action_dim * 2
        act_class = load_class_from_path(self.cfg.algo.act_class,
                                         model_name_to_path[self.cfg.algo.act_class])
        cri_class = load_class_from_path(self.cfg.algo.cri_class,
                                         model_name_to_path[self.cfg.algo.cri_class])
        if "Equivariant" in self.cfg.algo.act_class:
            self.G = load_symmetric_system(cfg=self.cfg.task.symmetry)
            self.actor = act_class(self.G, self.cfg.task.symmetry.actor_input_fields[0], self.cfg.task.symmetry.actor_output_fields[0], self.single_obs_dim[0], self.single_action_dim).to(self.cfg.device)
            self.actor_left = act_class(self.G, self.cfg.task.symmetry.actor_input_fields[1], self.cfg.task.symmetry.actor_output_fields[1], self.single_obs_dim[1], self.single_action_dim).to(self.cfg.device)
        else:
            self.actor = act_class(self.single_obs_dim[0], self.single_action_dim).to(self.cfg.device)
            self.actor_left = act_class(self.single_obs_dim[1], self.single_action_dim).to(self.cfg.device)
        if "Equivariant" in self.cfg.algo.cri_class:
            self.G = load_symmetric_system(cfg=self.cfg.task.symmetry)
            self.critic = cri_class(self.G, self.cfg.task.symmetry.critic_input_fields[0] + ['Q_js'], self.cfg.task.symmetry.critic_output_fields[0], self.single_obs_dim[0], self.single_action_dim).to(self.cfg.device)
            self.critic_left = cri_class(self.G, self.cfg.task.symmetry.critic_input_fields[1] + ['Q_js'], self.cfg.task.symmetry.critic_output_fields[1], self.single_obs_dim[1], self.single_action_dim).to(self.cfg.device)
        else:
            self.critic = cri_class(self.single_obs_dim[0], self.single_action_dim).to(self.cfg.device)
            self.critic_left = cri_class(self.single_obs_dim[1], self.single_action_dim).to(self.cfg.device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), self.cfg.algo.actor_lr)
        self.actor_optimizer_left = torch.optim.Adam(self.actor_left.parameters(), self.cfg.algo.actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), self.cfg.algo.critic_lr)
        self.critic_optimizer_left = torch.optim.Adam(self.critic_left.parameters(), self.cfg.algo.critic_lr)

        self.critic_target = deepcopy(self.critic)
        self.critic_target_left = deepcopy(self.critic_left)
        self.actor_target = deepcopy(self.actor) if not self.cfg.algo.no_tgt_actor else self.actor
        self.actor_target_left = deepcopy(self.actor_left) if not self.cfg.algo.no_tgt_actor else self.actor_left

        if self.cfg.algo.noise.decay == 'linear':
            self.noise_scheduler = LinearSchedule(start_val=self.cfg.algo.noise.std_max,
                                                  end_val=self.cfg.algo.noise.std_min,
                                                  total_iters=self.cfg.algo.noise.lin_decay_iters)
        elif self.cfg.algo.noise.decay == 'exp':
            self.noise_scheduler = ExponentialSchedule(start_val=self.cfg.algo.noise.std_max,
                                                       gamma=self.cfg.algo.exp_decay_rate,
                                                       end_val=self.cfg.algo.noise.std_min)
        else:
            self.noise_scheduler = None
        
        self.n_step_buffer = NStepReplay(self.obs_dim,
                                         self.action_dim,
                                         self.cfg.num_envs,
                                         self.cfg.algo.nstep,
                                         device=self.device,
                                         gamma=self.cfg.algo.gamma,
                                         left_agent=True)
        self.symmetry_manager = SymmetryManager(self.cfg.task.multi.TD3BC, self.cfg.task.symmetry.symmetric_envs)
        self._critic_update_count = 0   # for delayed actor update
        offline_ratio = self.cfg.algo.offline.offline_ratio
        if offline_ratio > 0.0:
            self.offline_buffer = OfflineBuffer(self.cfg)
            self.offline_buffer.load(self.cfg.algo.offline.data_dir, self.cfg.algo.offline.pattern)
        else:
            self.offline_buffer = None

    def get_noise_std(self):
        if self.noise_scheduler is None:
            return self.cfg.algo.noise.std_max
        else:
            return self.noise_scheduler.val()

    def update_noise(self):
        if self.noise_scheduler is not None:
            self.noise_scheduler.step()
        
    def get_actions(self, obs, cur_symmetry_tracker, sample=True):
        if self.cfg.algo.obs_norm:
            obs = self.obs_rms.normalize(obs)
        ob_right, ob_left = self.symmetry_manager.get_multi_agent_obs(obs, cur_symmetry_tracker)
        actions_right = self.actor(ob_right)
        actions_left = self.actor_left(ob_left)
        actions = self.symmetry_manager.get_execute_action(actions_right, actions_left, cur_symmetry_tracker)
        if sample:
            if self.cfg.algo.noise.type == 'fixed':
                actions = add_normal_noise(actions,
                                           std=self.get_noise_std(),
                                           out_bounds=[-1., 1.])
            elif self.cfg.algo.noise.type == 'mixed':
                actions = add_mixed_normal_noise(actions,
                                                 std_min=self.cfg.algo.noise.std_min,
                                                 std_max=self.cfg.algo.noise.std_max,
                                                 out_bounds=[-1., 1.])
            else:
                raise NotImplementedError
        return actions

    @torch.no_grad()
    def get_tgt_policy_actions(self, actor_target, obs, sample=True):
        actions = actor_target(obs)
        if sample:
            actions = add_normal_noise(actions,
                                       std=self.cfg.algo.noise.tgt_pol_std,
                                       noise_bounds=[-self.cfg.algo.noise.tgt_pol_noise_bound,
                                                      self.cfg.algo.noise.tgt_pol_noise_bound],
                                       out_bounds=[-1., 1.])
        return actions

    def explore_env(self, env, timesteps: int, random: bool = False, offline: bool = False) -> list:
        if offline:
            warmup = min(timesteps * self.cfg.num_envs, self.offline_buffer.offline_states.shape[0])
            idx = torch.randperm(self.offline_buffer.offline_states.shape[0], device=self.cfg.device)[:warmup]
            data = (
                self.offline_buffer.offline_states[idx],
                self.offline_buffer.offline_actions[idx],
                self.offline_buffer.offline_rewards[idx],
                self.offline_buffer.offline_next_states[idx],
                self.offline_buffer.offline_dones[idx],
                self.offline_buffer.offline_rewards_left[idx],
            )
            return data, warmup
        else:
            random = True  # fallback to random if no offline data

        obs_dim = (self.obs_dim,) if isinstance(self.obs_dim, int) else self.obs_dim
        traj_states = torch.empty((self.cfg.num_envs, timesteps) + (*obs_dim,), device=self.device)
        traj_actions = torch.empty((self.cfg.num_envs, timesteps) + (self.action_dim,), device=self.device)
        traj_rewards = torch.empty((self.cfg.num_envs, timesteps), device=self.device)
        traj_rewards_left = torch.empty((self.cfg.num_envs, timesteps), device=self.device)
        traj_next_states = torch.empty((self.cfg.num_envs, timesteps) + (*obs_dim,), device=self.device)
        traj_dones = torch.empty((self.cfg.num_envs, timesteps), device=self.device)

        obs = self.obs
        for i in range(timesteps):
            if self.cfg.algo.obs_norm:
                self.obs_rms.update(obs)
            cur_symmetry_tracker = env.unwrapped.symmetry_tracker
            if random:
                action = torch.rand((self.cfg.num_envs, self.action_dim),
                                    device=self.cfg.device) * 2.0 - 1.0
            else:
                action = self.get_actions(obs, cur_symmetry_tracker, sample=True)
            
            next_obs, reward, done, info = env.step(action)
            reward_right, reward_left = self.symmetry_manager.get_multi_agent_rew(info['detailed_reward'], cur_symmetry_tracker)
            self.update_tracker(reward_right + reward_left, done, info)
            if self.cfg.algo.handle_timeout:
                done = handle_timeout(done, info)

            traj_states[:, i] = obs
            traj_actions[:, i] = action
            traj_dones[:, i] = done
            traj_rewards[:, i] = reward_right
            traj_rewards_left[:, i] = reward_left
            traj_next_states[:, i] = next_obs
            obs = next_obs
        self.obs = obs

        traj_rewards = self.cfg.algo.reward_scale * traj_rewards.reshape(self.cfg.num_envs, timesteps, 1)
        traj_rewards_left = self.cfg.algo.reward_scale * traj_rewards_left.reshape(self.cfg.num_envs, timesteps, 1)
        traj_dones = traj_dones.reshape(self.cfg.num_envs, timesteps, 1)
        data = self.n_step_buffer.add_to_buffer(traj_states, traj_actions, traj_rewards, traj_next_states, traj_dones, reward_left=traj_rewards_left)
        return data, timesteps * self.cfg.num_envs
    
    def update_net(self, memory):
        total_bs = self.cfg.algo.batch_size
        if self.offline_buffer is not None:
            offline_bs = max(1, int(total_bs * self.cfg.algo.offline.offline_ratio))
            online_bs  = total_bs - offline_bs
        else:
            online_bs, offline_bs = total_bs, 0

        critic_loss_list = list()
        critic_loss_left_list = list()
        actor_loss_list = list()
        actor_loss_left_list = list()
        for _ in range(self.cfg.algo.update_times):
            obs, action, reward_right, next_obs, done, reward_left = memory.sample_batch(online_bs)
            if offline_bs > 0:
                dataset = self.offline_buffer.sample_batch(offline_bs)
                obs = torch.cat([obs, dataset[0]])
                action = torch.cat([action, dataset[1]])
                reward_right = torch.cat([reward_right, dataset[2]])
                next_obs = torch.cat([next_obs, dataset[3]])
                done = torch.cat([done, dataset[4]])
                reward_left = torch.cat([reward_left, dataset[5]])
            if self.cfg.algo.obs_norm:
                obs = self.obs_rms.normalize(obs)
                next_obs = self.obs_rms.normalize(next_obs)
            obs_right, obs_left = self.symmetry_manager.get_multi_agent_obs(obs, None)
            next_obs_right, next_obs_left = self.symmetry_manager.get_multi_agent_obs(next_obs, None)
            action_right, action_left = action[:, :22], action[:, 22:]

            critic_loss, critic_grad_norm = self.update_critic(self.critic, self.critic_target, self.critic_optimizer, self.actor_target,
                                                               obs_right, action_right, reward_right, next_obs_right, done)
            critic_loss_list.append(critic_loss)
            critic_loss_left, critic_grad_norm_left = self.update_critic(self.critic_left, self.critic_target_left, self.critic_optimizer_left, self.actor_target_left,
                                                               obs_left, action_left, reward_left, next_obs_left, done)
            critic_loss_left_list.append(critic_loss_left)
            self._critic_update_count += 1

            if self._critic_update_count % self.cfg.algo.policy_delay == 0:  # delayed actor update
                # Q-maximization uses full mixed obs; BC uses only offline slice
                obs_right_dataset, obs_left_dataset = obs_right[online_bs:], obs_left[online_bs:]
                action_right_dataset, action_left_dataset = action_right[online_bs:], action_left[online_bs:]

                actor_loss, actor_grad_norm = self.update_actor(self.actor, self.actor_optimizer, self.critic, obs_right, obs_right_dataset, action_right_dataset)
                actor_loss_list.append(actor_loss)
                actor_loss_left, actor_grad_norm_left = self.update_actor(self.actor_left, self.actor_optimizer_left, self.critic_left, obs_left, obs_left_dataset, action_left_dataset)
                actor_loss_left_list.append(actor_loss_left)
                
                if not self.cfg.algo.no_tgt_actor:
                    soft_update(self.actor_target, self.actor, self.cfg.algo.tau)
                    soft_update(self.actor_target_left, self.actor_left, self.cfg.algo.tau)
                
            soft_update(self.critic_target, self.critic, self.cfg.algo.tau)
            soft_update(self.critic_target_left, self.critic_left, self.cfg.algo.tau)
        
        log_info = {
            "train/critic_loss": np.mean(critic_loss_list),
            "train/critic_loss_left": np.mean(critic_loss_left_list),
            "train/actor_loss": np.mean(actor_loss_list),
            "train/actor_loss_left": np.mean(actor_loss_left_list),
            "train/return": self.return_tracker.mean(),
            "train/episode_length": self.step_tracker.mean(),
            "train/success_rate": self.success_tracker.mean(),
        }
        return log_info

    def update_critic(self, critic, critic_target, critic_optimizer, actor_target, obs, action, reward, next_obs, done):
        with torch.no_grad():
            next_actions = self.get_tgt_policy_actions(actor_target, next_obs)
            target_Q = critic_target.get_q_min(next_obs, next_actions)
            target_Q = reward + (1 - done) * (self.cfg.algo.gamma ** self.cfg.algo.nstep) * target_Q
        
        current_Q1, current_Q2 = critic.get_q1_q2(obs, action)
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
        grad_norm = self.optimizer_update(critic_optimizer, critic_loss)
        return critic_loss.item(), grad_norm
    
    def update_actor(self, actor, actor_optimizer, critic, obs, obs_dataset, action_dataset):
        critic.requires_grad_(False)
        action_pred = actor(obs)  # full batch for Q
        action_dataset_pred = actor(obs_dataset)  # offline only for BC-term
        with torch.no_grad():
            Q_dataset = critic.get_q1(obs_dataset, action_dataset)
            lmbda = self.cfg.algo.alpha / Q_dataset.abs().mean()
        
        Q_policy = critic.get_q1(obs, action_pred)
        bc_loss = F.mse_loss(action_dataset_pred, action_dataset)  # only for expert
        actor_loss = -lmbda * Q_policy.mean() + bc_loss
        grad_norm = self.optimizer_update(actor_optimizer, actor_loss)
        critic.requires_grad_(True)
        return actor_loss.item(), grad_norm
