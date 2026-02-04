import torch
import time
import h5py
import numpy as np

from isaaclab.utils.math import quat_from_matrix

from pathlib import Path
LOGGER_ROOT = Path(__file__).resolve().parents[1] 
SAVE_DIR = LOGGER_ROOT / "teleop_logs"  # / "policy"


class TrajectoryLogger:
    def __init__(self, save_dir: Path=SAVE_DIR, task_name=None):
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        self.file_path = save_dir / f"{task_name}_{timestamp}.h5"
        self.h5_file = h5py.File(str(self.file_path), "w")
        self.episode_idx = 0
        self._epi_meta = {}
        self._reset_buffers()
        print(f"[Logger] Initialized trajectory log file: {self.file_path}")

    def _reset_buffers(self):
        """Internal buffer for current trial data."""
        self._obs_policy = []
        self._obs_vision = []
        self._next_obs_policy = []
        self._next_obs_vision = []

        self._act = []
        self._rew = []
        self._terminals = []
        self._timeouts = []
        self._rew_terms = []

    def start_episode(self, *, language_instruction: str, rew_cfg_hash: str | None, rew_names: list[str] | None, rew_weights: np.ndarray | None):
        """episode metadata when starting a new episode."""
        self._epi_meta = {
            "language_instruction": str(language_instruction),
            "rew_cfg_hash": "" if rew_cfg_hash is None else str(rew_cfg_hash),
        }
        self._epi_meta["reward_names"] = [str(name) for name in rew_names] if rew_names is not None else []
        self._epi_meta["reward_weights"] = np.asarray(rew_weights, dtype=np.float32) if rew_weights is not None else np.array([], dtype=np.float32)

    def add_transition(self, *, observation, action, reward, next_observation, terminated: bool, truncated: bool, rew_terms):
        self._obs_policy.append(np.asarray(observation["policy"], dtype=np.float32))
        self._obs_vision.append(np.asarray(observation["vision"], dtype=np.uint8))
        self._next_obs_policy.append(np.asarray(next_observation["policy"], dtype=np.float32))
        self._next_obs_vision.append(np.asarray(next_observation["vision"], dtype=np.uint8))

        self._act.append(np.asarray(action, dtype=np.float32))
        self._rew.append(float(reward))
        self._terminals.append(np.uint8(1 if terminated else 0))
        self._timeouts.append(np.uint8(1 if truncated else 0))
        self._rew_terms.append(np.asarray(rew_terms, dtype=np.float32))
    
    def save_episode(self):
        steps = len(self._rew)
        if len(self._rew) == 0:
            return
        grp = self.h5_file.create_group(f"episode_{self.episode_idx:03d}")

        # metadata
        meta = grp.create_group("epi_meta")
        for k, v in self._epi_meta.items():
            if isinstance(v, (list, tuple)):
                meta.create_dataset(k, data=np.array(v, dtype='S'))
            elif isinstance(v, np.ndarray):
                meta.create_dataset(k, data=v)
            else:
                meta.attrs[k] = v
        
        # offline rl data
        offline_data = grp.create_group("offline_data")
        
        # observations group
        obs_grp = offline_data.create_group("observations")
        obs_grp.create_dataset("policy", data=np.stack(self._obs_policy, axis=0))
        obs_grp.create_dataset("vision",
                               data=np.stack(self._obs_vision, axis=0),
                               compression="gzip",
                               compression_opts=4,
                               chunks=True,)
        # next_observations group
        next_obs_grp = offline_data.create_group("next_observations")
        next_obs_grp.create_dataset("policy", data=np.stack(self._next_obs_policy, axis=0))
        next_obs_grp.create_dataset("vision",
                                     data=np.stack(self._next_obs_vision, axis=0),
                                     compression="gzip",
                                     compression_opts=4,
                                     chunks=True,)
        
        offline_data.create_dataset("actions", data=np.stack(self._act, axis=0))
        offline_data.create_dataset("rewards", data=np.asarray(self._rew, dtype=np.float32))
        offline_data.create_dataset("terminals", data=np.asarray(self._terminals, dtype=np.uint8))
        offline_data.create_dataset("timeouts", data=np.asarray(self._timeouts, dtype=np.uint8))
        if len(self._rew_terms) == steps:
            offline_data.create_dataset("reward_terms", data=np.stack(self._rew_terms, axis=0))

        self.h5_file.flush()
        print(f"[Logger] Saved episode_{self.episode_idx:03d} with {len(self._rew)} steps to {self.file_path}")

        self._epi_meta = {}
        self._reset_buffers()
        self.episode_idx += 1

    def close(self):
        if len(self._rew) > 0:
            self.save_episode()
        self.h5_file.close()
        print(f"[Logger] Closed after {self.episode_idx} episodes saved to {self.file_path}")


class SphereWriter:
    def __init__(self, device, dtype=torch.float32):
        self.R_buf = torch.empty((1, 3, 3), device=device, dtype=dtype)
        self.pos_buf = torch.empty((1, 3), device=device, dtype=dtype)
        self.pose_buf = torch.empty((1, 7), device=device, dtype=dtype)

    def write(self, env, key, H_np):
        self.R_buf[0].copy_(torch.from_numpy(H_np[:3, :3]).to(self.R_buf.device, dtype=self.R_buf.dtype))
        self.pos_buf[0].copy_(torch.from_numpy(H_np[:3, 3]).to(self.pos_buf.device, dtype=self.pos_buf.dtype))
        quat = quat_from_matrix(self.R_buf)  # (1,4)
        self.pose_buf[:, :3] = self.pos_buf
        self.pose_buf[:, 3:] = quat
        env.unwrapped.scene[key].write_root_pose_to_sim(self.pose_buf)