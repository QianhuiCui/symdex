import os
import glob
import h5py
import numpy as np
import torch


class OfflineBuffer:
    def __init__(self, device: str, use_vision: bool = False, use_pc: bool = False):
        self.device = torch.device(device)
        self.use_vision = use_vision
        self.use_pc = use_pc

        self.state = None
        self.next_state = None
        self.vision = None
        self.next_vision = None
        self.pc = None
        self.next_pc = None

        self.actions = None
        self.rewards = None
        self.not_done = None

        self.size = 0
        self.state_dim = None
        self.action_dim = None
        self.vision_dim = None
        self.pc_dim = None
    
    def _load_single_h5(self, file_path: str):
        out = {
            "state": [],
            "next_state": [],
            "vision": [],
            "next_vision": [],
            "pc": [],
            "next_pc": [],
            "actions": [],
            "rewards": [],
            "not_done": []
        }

        with h5py.File(file_path, "r") as f:
            episode_keys = sorted([k for k in f.keys() if k.startswith("episode_")])

            for ep_key in episode_keys:
                base = f[f"{ep_key}/offline_data"]

                # low dim policy obs
                state = np.asarray(base["observations"]["policy"], dtype=np.float32)
                next_state = np.asarray(base["next_observations"]["policy"], dtype=np.float32)
                out["state"].append(state)
                out["next_state"].append(next_state)

                if self.use_vision:  # vision obs
                    vision = np.asarray(base["observations"]["vision"], dtype=np.uint8)
                    next_vision = np.asarray(base["next_observations"]["vision"], dtype=np.uint8)
                    out["vision"].append(vision)
                    out["next_vision"].append(next_vision)
                if self.use_pc:  # point cloud obs
                    pc = np.asarray(base["observations"]["point_cloud"], dtype=np.float32)
                    next_pc = np.asarray(base["next_observations"]["point_cloud"], dtype=np.float32)
                    out["pc"].append(pc)
                    out["next_pc"].append(next_pc)
                
                actions = np.asarray(base["actions"], dtype=np.float32)
                rewards = np.asarray(base["rewards"], dtype=np.float32).reshape(-1, 1)
                terminals = np.asarray(base["terminals"], dtype=np.float32).reshape(-1, 1)
                timeouts = np.asarray(base["timeouts"], dtype=np.float32).reshape(-1, 1)
                dones = np.clip(terminals + timeouts, 0.0, 1.0)
                not_dones = 1.0 - dones

                out["actions"].append(actions)
                out["rewards"].append(rewards)
                out["not_done"].append(not_dones)

                steps = state.shape[0]
                assert actions.shape[0] == rewards.shape[0] == not_dones.shape[0] == steps, \
                    f"Length mismatch in {file_path} / {ep_key}"
                if self.use_vision:
                    assert vision.shape[0] == next_vision.shape[0] == steps, \
                        f"Vision length mismatch in {file_path} / {ep_key}"
                if self.use_pc:
                    assert pc.shape[0] == next_pc.shape[0] == steps, \
                        f"Point cloud length mismatch in {file_path} / {ep_key}"
                
        for key in out:
            if len(out[key]) > 0:
                out[key] = np.concatenate(out[key], axis=0)
            else:
                out[key] = None
        return out
    
    def action_stats(self) -> dict:
        abs_actions = np.abs(self.actions)
        stats = {
            "shape": tuple(self.actions.shape),
            "min": float(self.actions.min()),
            "max": float(self.actions.max()),
            "mean": float(self.actions.mean()),
            "std": float(self.actions.std()),
            "abs_p95": float(np.quantile(abs_actions, 0.95)),
            "abs_p99": float(np.quantile(abs_actions, 0.99)),
            "abs_p999": float(np.quantile(abs_actions, 0.999)),
        }
        return stats
    
    def estimate_max_action(self, quantile: float = 0.99, min_value: float = 1e-6) -> float:
        abs_actions = np.abs(self.actions).reshape(-1)
        max_action = float(np.quantile(abs_actions, quantile))
        max_action = max(max_action, min_value)
        return max_action
    
    def load_from_dir(self, root_dir: str, pattern: str = "*.h5"):
        file_paths = sorted(glob.glob(os.path.join(root_dir, pattern)))
        if len(file_paths) == 0:
            raise ValueError(f"No files found in {root_dir} with pattern {pattern}")
        
        all_data = {
            "state": [],
            "next_state": [],
            "vision": [],
            "next_vision": [],
            "pc": [],
            "next_pc": [],
            "actions": [],
            "rewards": [],
            "not_done": []
        }

        for path in file_paths:
            data = self._load_single_h5(path)
            for key in all_data:
                if data[key] is not None:
                    all_data[key].append(data[key])
        
        for key in all_data:
            if len(all_data[key]) > 0:
                all_data[key] = np.concatenate(all_data[key], axis=0)
            else:
                all_data[key] = None
        
        self.state = all_data["state"]
        self.next_state = all_data["next_state"]
        self.vision = all_data["vision"]
        self.next_vision = all_data["next_vision"]
        self.pc = all_data["pc"]
        self.next_pc = all_data["next_pc"]
        self.actions = all_data["actions"]
        self.rewards = all_data["rewards"]
        self.not_done = all_data["not_done"]

        self.size = self.actions.shape[0]
        self.state_dim = self.state.shape[1]
        self.action_dim = self.actions.shape[1]
        if self.use_vision:
            self.vision_dim = self.vision.shape[1:]
        if self.use_pc:
            self.pc_dim = self.pc.shape[1:]
        
    def normalize_states(self, eps = 1e-3):
        mean = self.state.mean(axis=0, keepdims=True).astype(np.float32)
        std = (self.state.std(axis=0, keepdims=True) + eps).astype(np.float32)
        self.state = (self.state - mean) / std
        self.next_state = (self.next_state - mean) / std
        return mean, std
    
    def sample(self, batch_size):
        idx = np.random.randint(0, self.size, size=batch_size)
        batch = {
            "state": torch.as_tensor(self.state[idx], dtype=torch.float32, device=self.device),
            "next_state": torch.as_tensor(self.next_state[idx], dtype=torch.float32, device=self.device),
            "actions": torch.as_tensor(self.actions[idx], dtype=torch.float32, device=self.device),
            "rewards": torch.as_tensor(self.rewards[idx], dtype=torch.float32, device=self.device),
            "not_done": torch.as_tensor(self.not_done[idx], dtype=torch.float32, device=self.device)
        }

        if self.use_vision:
            vision = torch.as_tensor(self.vision[idx], dtype=torch.float32, device=self.device) / 255.0
            next_vision = torch.as_tensor(self.next_vision[idx], dtype=torch.float32, device=self.device) / 255.0
            vision = vision.permute(0, 1, 4, 2, 3)  # (B, V, H, W, C) -> (B, V, C, H, W)
            next_vision = next_vision.permute(0, 1, 4, 2, 3)
            batch["vision"] = vision
            batch["next_vision"] = next_vision
        if self.use_pc:
            pc = torch.as_tensor(self.pc[idx], dtype=torch.float32, device=self.device)
            next_pc = torch.as_tensor(self.next_pc[idx], dtype=torch.float32, device=self.device)
            batch["pc"] = pc
            batch["next_pc"] = next_pc
        
        return batch