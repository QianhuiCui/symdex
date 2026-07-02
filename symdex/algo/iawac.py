from copy import deepcopy
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from symdex.algo.ac_base import ActorCriticBase
from symdex.replay.nstep_replay import NStepReplay
from symdex.utils.torch_util import soft_update
from symdex.utils.common import handle_timeout, load_class_from_path
from symdex.algo.network import model_name_to_path
from symdex.utils.symmetry import load_symmetric_system, SymmetryManager
from symdex.utils.offline_buffer import OfflineBuffer

@dataclass
class AgentIAWAC(ActorCriticBase):
    def __post_init__(self):
        super().__post_init__()
        self.algo_multi_cfg = self.cfg.task.multi.AWAC
        self.single_obs_dim = list(self.cfg.task.multi.AWAC.single_agent_obs_dim)
        self.single_action_dim = int(self.cfg.task.multi.AWAC.single_agent_action_dim)
        self.obs_dim = self.cfg.task.multi.shared_obs_dim
        self.action_dim = self.single_action_dim * 2

        act_class = load_class_from_path(self.cfg.algo.act_class,
                                         model_name_to_path[self.cfg.algo.act_class])
        cri_class = load_class_from_path(self.cfg.algo.cri_class,
                                         model_name_to_path[self.cfg.algo.cri_class])
        if "Equivariant" in self.cfg.algo.act_class:
            self.G = load_symmetric_system(cfg=self.cfg.task.symmetry)
            self.actor = act_class(self.G, self.cfg.task.symmetry.AWAC.actor_input_fields[0], self.cfg.task.symmetry.AWAC.actor_output_fields[0], self.single_obs_dim[0], self.single_action_dim).to(self.cfg.device)
            self.actor_left = act_class(self.G, self.cfg.task.symmetry.AWAC.actor_input_fields[1], self.cfg.task.symmetry.AWAC.actor_output_fields[1], self.single_obs_dim[1], self.single_action_dim).to(self.cfg.device)
        else:
            self.actor = act_class(self.single_obs_dim[0], self.single_action_dim).to(self.cfg.device)
            self.actor_left = act_class(self.single_obs_dim[1], self.single_action_dim).to(self.cfg.device)
        if "Equivariant" in self.cfg.algo.cri_class:
            self.G = load_symmetric_system(cfg=self.cfg.task.symmetry)
            self.critic = cri_class(self.G, self.cfg.task.symmetry.AWAC.critic_input_fields[0] + ['Q_js'], self.cfg.task.symmetry.AWAC.critic_output_fields[0], self.single_obs_dim[0], self.single_action_dim).to(self.cfg.device)
            self.critic_left = cri_class(self.G, self.cfg.task.symmetry.AWAC.critic_input_fields[1] + ['Q_js'], self.cfg.task.symmetry.AWAC.critic_output_fields[1], self.single_obs_dim[1], self.single_action_dim).to(self.cfg.device)
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

        self.n_step_buffer = NStepReplay(self.obs_dim,
                                         self.action_dim,
                                         self.cfg.num_envs,
                                         self.cfg.algo.nstep,
                                         device=self.device,
                                         gamma=self.cfg.algo.gamma,
                                         left_agent=True)
        self.symmetry_manager = SymmetryManager(self.cfg.task.multi.AWAC, self.cfg.task.symmetry.symmetric_envs)
        self.offline_buffer = OfflineBuffer(self.cfg)
        self.offline_buffer.load(self.cfg.algo.offline.data_dir, self.cfg.algo.offline.pattern)

    def get_actions(self, obs, cur_symmetry_tracker, sample=True):
        if self.cfg.algo.obs_norm:
            obs = self.obs_rms.normalize(obs)
        ob_right, ob_left = self.symmetry_manager.get_multi_agent_obs(obs, cur_symmetry_tracker)
        actions_right, _ = self.actor.get_actions(ob_right, sample=sample)
        actions_left, _ = self.actor.get_actions(ob_left, sample=sample)
        actions = self.symmetry_manager.get_execute_action(actions_right, actions_left, cur_symmetry_tracker)
        return actions.clamp(-1.0, 1.0)
    
    def explore_env(self, env, timesteps: int, random: bool = False, offline: bool = False):
        if offline:
            data = (
                self.offline_buffer.offline_states,
                self.offline_buffer.offline_actions,
                self.offline_buffer.offline_rewards,
                self.offline_buffer.offline_next_states,
                self.offline_buffer.offline_dones,
                self.offline_buffer.offline_rewards_left,
            )
            return data, self.offline_buffer.offline_states.shape[0]
        
        obs_dim = (self.obs_dim,) if isinstance(self.obs_dim, int) else self.obs_dim
        traj_states = torch.empty((self.cfg.num_envs, timesteps) + (*obs_dim,), device=self.device)
        traj_actions = torch.empty((self.cfg.num_envs, timesteps) + (self.action_dim), device=self.device)
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
            
            raw_next_obs, reward, done, info = env.step(action)
            next_obs = raw_next_obs["policy"] if isinstance(raw_next_obs, dict) else raw_next_obs
            reward_right, reward_left = self.symmetry_manager.get_multi_agent_rew(info['detailed_reward'], cur_symmetry_tracker)
            self.update_tracker(reward_right + reward_left, done, info)
            if self.algo.handle_timeout:
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
        critic_loss_list, critic_loss_left_list = list(), list()
        actor_loss_list, actor_loss_left_list = list(), list()
        critic_grad_list, critic_grad_left_list = list(), list()
        actor_grad_list,  actor_grad_left_list  = list(), list()
        for _ in range(self.cfg.algo.update_times):
            obs, action, reward_right, next_obs, done, reward_left = memory.sample_batch(self.cfg.algo.batch_size)
            if self.cfg.algo.obs_norm:
                obs = self.obs_rms.normalize(obs)
                next_obs = self.obs_rms.normalize(next_obs)
            obs_right, obs_left = self.symmetry_manager.get_multi_agent_obs(obs, None)
            next_obs_right, next_obs_left = self.symmetry_manager.get_multi_agent_obs(next_obs, None)
            action_right, action_left = action[:, :22], action[:, 22:]

            critic_loss, critic_grad_norm = self.update_critic(self.critic, self.critic_target, self.critic_optimizer, self.actor_target,
                                                               obs_right, action_right, reward_right, next_obs_right, done)
            critic_loss_list.append(critic_loss)
            critic_grad_list.append(critic_grad_norm.item() if critic_grad_norm is not None else 0.0)
            critic_loss_left, critic_grad_norm_left = self.update_critic(self.critic_left, self.critic_target_left, self.critic_optimizer_left, self.actor_target_left,
                                                               obs_left, action_left, reward_left, next_obs_left, done)
            critic_loss_left_list.append(critic_loss_left)
            critic_grad_left_list.append(critic_grad_norm_left.item() if critic_grad_norm_left is not None else 0.0)

            actor_loss, actor_grad_norm = self.update_actor(self.actor, self.actor_optimizer, self.critic, obs_right, action_right)
            actor_loss_list.append(actor_loss)
            actor_grad_list.append(actor_grad_norm.item() if actor_grad_norm is not None else 0.0) 
            actor_loss_left, actor_grad_norm_left = self.update_actor(self.actor_left, self.actor_optimizer_left, self.critic_left, obs_left, action_left)
            actor_loss_left_list.append(actor_loss_left)
            actor_grad_left_list.append(actor_grad_norm_left.item() if actor_grad_norm_left is not None else 0.0)

            soft_update(self.critic_target, self.critic, self.cfg.algo.tau)
            soft_update(self.critic_target_left, self.critic_left, self.cfg.algo.tau)
            if not self.cfg.algo.no_tgt_actor:
                soft_update(self.actor_target, self.actor, self.cfg.algo.tau)
                soft_update(self.actor_target_left, self.actor_left, self.cfg.algo.tau)
        
        log_info = {
            "critic/critic_loss": np.mean(critic_loss_list),
            "critic/critic_loss_left": np.mean(critic_loss_left_list),
            "actor/actor_loss": np.mean(actor_loss_list),
            "actor/actor_loss_left": np.mean(actor_loss_left_list),
            "grad_norm/critic_grad_norm": np.mean(critic_grad_list),
            "grad_norm/critic_grad_norm_left": np.mean(critic_grad_left_list),
            "grad_norm/actor_grad_norm": np.mean(actor_grad_list),
            "grad_norm/actor_grad_norm_left": np.mean(actor_grad_left_list),
            "train/return": self.return_tracker.mean(),
            "train/episode_length": self.step_tracker.mean(),
            "train/success_rate": self.success_tracker.mean(),
        }
        self.add_info_tracker_log(log_info)
        return log_info
    
    def update_critic(self, critic, critic_target, critic_optimizer, actor, obs, action, reward, next_obs, done):
        with torch.no_grad():
            next_actions, _, log_prob, _ = actor.get_actions_logprob_entropy(next_obs, sample=True)
            target_Q = critic_target.get_q_min(next_obs, next_actions) - self.cfg.algo.alpha * log_prob
            target_Q = reward + (1.0 - done) * (self.cfg.algo.gamma ** self.cfg.algo.nstep) * target_Q
        current_Q1, current_Q2 = critic.get_q1_q2(obs, action)
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
        grad_norm = self.optimizer_update(critic_optimizer, critic_loss)
        return critic_loss.item(), grad_norm
    
    def update_actor(self, actor, actor_optimizer, critic, obs, action):
        critic.requires_grad_(False)
        with torch.no_grad():
            pi = actor.get_actions(obs, sample=True)
            Q_value = critic.get_q_min(obs, action)
            V_value = critic.get_q_min(obs, pi)
            adv = Q_value - V_value
            weights = F.softmax(adv / self.cfg.algo.beta, dim=0) * len(adv)
        _, _, log_prob, _ = actor.logprob_entropy(obs, action)
        actor_loss = -(weights * log_prob.unsqueeze(-1)).mean()
        grad_norm = self.optimizer_update(actor_optimizer, actor_loss)
        return actor_loss.item(), grad_norm