import os
import time
import h5py
import numpy as np
import tempfile
import imageio

from pathlib import Path
LOGGER_ROOT = Path(__file__).resolve().parents[1] 
SAVE_DIR = LOGGER_ROOT / "teleop_logs" / "policy"


class TrajectoryLogger:
    def __init__(self, save_dir: Path=SAVE_DIR, task_name=None, video_fps=30):
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        self.file_path = save_dir / f"{task_name}_{timestamp}.h5"
        self.h5_file = h5py.File(str(self.file_path), "w")
        self.episode_idx = 0
        self.video_fps = int(video_fps)

        self._video_writers = {}   # camera_id -> imageio writer
        self._video_temp = {}      # camera_id -> NamedTemporaryFile
        # self._pending_point_clouds = []  # list[np.ndarray], length = steps

        self._reset_buffers()
        print(f"[Logger] Initialized trajectory log file: {self.file_path}")

    def _reset_buffers(self):
        """Internal buffer for current trial data."""
        self.steps = []
        self.metadata = {}
        # self._pending_point_clouds = []
    
    def add_video_frame(self, camera_id: str, frame: np.ndarray):
        """Add a frame to the camera video. frame: HxWx3, uint8, HWC"""
        cam = str(camera_id)
        if cam not in self._video_writers:
            tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
            writer = imageio.get_writer(tmp.name, fps=self.video_fps, macro_block_size=1)
            self._video_temp[cam] = tmp
            self._video_writers[cam] = writer
        self._video_writers[cam].append_data(frame)
    
    # def add_point_cloud(self, pc: np.ndarray):
    #     """Add the point cloud (N x 3/6) of the current step and align it with the steps order."""
    #     self._pending_point_clouds.append(np.asarray(pc, dtype=np.float32))
    
    def _write_obj(self, group, key, value):
        if hasattr(value, "detach"):
            value = value.detach().cpu().numpy()
        elif hasattr(value, "cpu") and hasattr(value, "numpy"):
            value = value.cpu().numpy()

        if isinstance(value, dict):
            sub = group.create_group(str(key))
            for k, v in value.items():
                self._write_obj(sub, str(k), v)
            return

        if isinstance(value, np.ndarray):
            group.create_dataset(str(key), data=value)
            return

        if np.isscalar(value) or isinstance(value, (list, tuple)):
            group.create_dataset(str(key), data=np.array(value))
            return

        try:
            group.attrs[str(key)] = value
        except Exception:
            group.attrs[str(key)] = str(value)

    def add_step(self, *, action, action_dict, observation, reward, is_first=False, is_last=False, is_terminal=False, language_instruction="", discount=1.0):
        """Add a step to the current episode."""
        step = dict(
            action=np.array(action, dtype=np.float32),
            action_dict={k: np.array(v, dtype=np.float32) for k, v in action_dict.items()},
            observation=observation,
            reward=float(reward),
            is_first=bool(is_first),
            is_last=bool(is_last),
            is_terminal=bool(is_terminal),
            language_instruction=str(language_instruction),
            discount=float(discount),
        )
        self.steps.append(step)
    
    def _finalize_videos_to_h5(self, ep_group):
        """Turn off writers, serialise mp4 into bytes, and write to /videos/*."""
        if not self._video_writers:
            return
        vids_grp = ep_group.create_group("videos")
        for cam, writer in self._video_writers.items():
            writer.close()
        for cam, tmp in self._video_temp.items():
            tmp.seek(0)
            raw = np.frombuffer(tmp.read(), dtype=np.uint8)
            ds = vids_grp.create_dataset(cam, data=raw)
            ds.attrs["container"] = "mp4"
            ds.attrs["fps"] = self.video_fps
            tmp.close()
            try:
                os.remove(tmp.name)
            except Exception:
                pass
        self._video_writers.clear()
        self._video_temp.clear()

    def save_episode(self):
        if len(self.steps) == 0:
            return
        
        group = self.h5_file.create_group(f"episode_{self.episode_idx:03d}")
        grp_meta = group.create_group("episode_metadata")
        grp_meta.attrs["file_path"] = str(self.file_path)
        grp_meta.attrs["num_steps"] = len(self.steps)

        grp_steps = group.create_group("steps")
        for i, step in enumerate(self.steps):
            step_grp = grp_steps.create_group(f"step_{i:04d}")

            # action
            step_grp.create_dataset("action", data=np.asarray(step["action"], dtype=np.float32))
            act_grp = step_grp.create_group("action_dict")
            for k, v in step["action_dict"].items():
                act_grp.create_dataset(k, data=np.asarray(v, dtype=np.float32))
            
            # observation 
            obs_grp = step_grp.create_group("observation")
            for k, v in step["observation"].items():
                self._write_obj(obs_grp, k, v)
            # point cloud (if exists, write to separate dataset per step)
            # if i < len(self._pending_point_clouds):
            #     pc = self._pending_point_clouds[i]
            #     if isinstance(pc, np.ndarray):
            #         step_grp.create_dataset("point_cloud", data=pc)
            
            # step-level attrs
            step_grp.attrs["reward"] = float(step["reward"])
            step_grp.attrs["is_first"] = bool(step["is_first"])
            step_grp.attrs["is_last"] = bool(step["is_last"])
            step_grp.attrs["is_terminal"] = bool(step["is_terminal"])
            step_grp.attrs["language_instruction"] = str(step["language_instruction"])
            step_grp.attrs["discount"] = float(step["discount"])
        
        self._finalize_videos_to_h5(group)
        
        self.h5_file.flush()
        print(f"[Logger] Saved episode_{self.episode_idx:03d} with {len(self.steps)} steps.")
        self._reset_buffers()
        self.episode_idx += 1

    def close(self):
        if len(self.steps) > 0:
            self.save_episode()
        self.h5_file.close()
        print(f"[Logger] Closed after {self.episode_idx} episodes.")