import os
import glob
import h5py
import numpy as np
import torch
from omegaconf import DictConfig


class OfflineBuffer:
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self._use_vision = cfg.algo.offline.vision
        self._use_pc = cfg.algo.offline.pc

        self.offline_states = None
        self.offline_next_states = None
        self.offline_actions = None
        self.offline_rewards = None
        self.offline_rewards_left = None
        self.offline_dones = None

        self.offline_vision = None
        self.offline_next_vision = None
        self.offline_pc = None
        self.offline_next_pc = None

    def load(self, data_dir, pattern):
        file_paths = sorted(glob.glob(os.path.join(data_dir, pattern)))
        if not file_paths:
            raise ValueError(f"[OfflineBuffer] No h5 files in {data_dir} ({pattern})")
        
        states, next_states, actions, rewards, rewards_left, dones = [], [], [], [], [], []
        vision, next_vision, pc, next_pc = [], [], [], []
        for path in file_paths:
            with h5py.File(path, 'r') as f:
                for ep_key in sorted(k for k in f.keys() if k.startswith('episode_')):
                    dataset = f[ep_key]["offline_data"]
                    r = dataset["rewards_right"][:]
                    r_left = dataset["rewards_left"][:]
                    nobs = dataset["next_observations"]["policy"][:]
                    done = np.clip(dataset['terminals'][:].astype(np.float32) + dataset['timeouts'][:].astype(np.float32), 0.0, 1.0)

                    r, r_left, nobs, done = self._nstep_episode(r, r_left, nobs, done, self.cfg.algo.nstep, self.cfg.algo.gamma)
                    states.append(dataset["observations"]["policy"][:])
                    next_states.append(nobs)
                    actions.append(dataset["actions"][:])
                    rewards.append(r)
                    rewards_left.append(r_left)
                    dones.append(done)

                    if self._use_vision:
                        vision.append(dataset["observations"]["vision"][:])
                        next_vision.append(dataset["next_observations"]["vision"][:])
                    if self._use_pc:
                        pc.append(dataset["observations"]["point_cloud"][:])
                        next_pc.append(dataset["next_observations"]["point_cloud"][:])
        
        self.offline_states = torch.from_numpy(np.concatenate(states, axis=0).astype(np.float32)).to(self.cfg.device)
        self.offline_next_states = torch.from_numpy(np.concatenate(next_states, axis=0).astype(np.float32)).to(self.cfg.device)
        self.offline_actions = torch.from_numpy(np.concatenate(actions, axis=0).astype(np.float32)).to(self.cfg.device)
        self.offline_rewards = self.cfg.algo.reward_scale * torch.from_numpy(np.concatenate(rewards, axis=0).astype(np.float32)).unsqueeze(-1).to(self.cfg.device)
        self.offline_rewards_left = self.cfg.algo.reward_scale * torch.from_numpy(np.concatenate(rewards_left, axis=0).astype(np.float32)).unsqueeze(-1).to(self.cfg.device)
        self.offline_dones = torch.from_numpy(np.concatenate(dones, axis=0).astype(np.float32)).unsqueeze(-1).to(self.cfg.device)

        if self._use_vision:
            self.offline_vision = torch.from_numpy(np.concatenate(vision, axis=0).astype(np.float32)).to(self.cfg.device)
            self.offline_next_vision = torch.from_numpy(np.concatenate(next_vision, axis=0).astype(np.float32)).to(self.cfg.device)
        if self._use_pc:
            self.offline_pc = torch.from_numpy(np.concatenate(pc, axis=0).astype(np.float32)).to(self.cfg.device)
            self.offline_next_pc = torch.from_numpy(np.concatenate(next_pc, axis=0).astype(np.float32)).to(self.cfg.device)
        
        print(f"[OfflineBuffer] {self.offline_states.shape[0]} transitions on {self.cfg.device}, vision={self._use_vision}, pc={self._use_pc}")

    def sample_batch(self, batch_size):
        idx = torch.randint(0, self.offline_states.shape[0], (batch_size,), device=self.cfg.device)
        batch = (
            self.offline_states[idx],
            self.offline_actions[idx],
            self.offline_rewards[idx],
            self.offline_next_states[idx],
            self.offline_dones[idx],
            self.offline_rewards_left[idx],
        )
        if self._use_vision:
            batch += (self.offline_vision[idx], self.offline_next_vision[idx])
        if self._use_pc:
            batch += (self.offline_pc[idx], self.offline_next_pc[idx])
        return batch
    
    @staticmethod
    def _nstep_episode(r, r_left, nobs, done, nstep, gamma):
        steps = len(r)
        rewards = np.zeros(steps, dtype=np.float32)
        rewards_left = np.zeros(steps, dtype=np.float32)
        next_obs = nobs.copy()
        dones = done.copy()

        for step in range(steps):
            n = min(nstep, steps - step)
            terminated = False

            for k in range(n):
                rewards[step] += (gamma ** k) * r[step + k]
                rewards_left[step] += (gamma ** k) * r_left[step + k]
                if done[step + k] > 0.5:
                    next_obs[step] = next_obs[step + k]
                    dones[step] = 1.0
                    terminated = True
                    break
            
            if not terminated:
                next_obs[step] = next_obs[step + n - 1]
                dones[step] = done[step + n - 1]
        return rewards, rewards_left, next_obs, dones