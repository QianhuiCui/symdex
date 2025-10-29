import os
import time
import h5py
import numpy as np


class TrajectoryLogger:
    def __init__(self, save_dir="./symdex/teleop_logs", task_name=None):
        os.makedirs(save_dir, exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        self.file_path = os.path.join(save_dir, f"{task_name}_{timestamp}.h5")
        self.h5_file = h5py.File(self.file_path, "w")
        self.trial_idx = 0
        self._reset_buffers()
        print(f"[Logger] Initialized trajectory log file: {self.file_path}")

    def _reset_buffers(self):
        """Internal buffer for current trial data."""
        self.current_data = {
            "timestamp": [],
            "joint_state": [],
            "pose_left": [],
            "pose_right": []
        }

    def add_sample(self, t, q, left_pose, right_pose):
        self.current_data["timestamp"].append(t)
        self.current_data["joint_state"].append(q if q is not None else np.zeros(44))
        self.current_data["pose_left"].append(left_pose if left_pose is not None else np.eye(4))
        self.current_data["pose_right"].append(right_pose if right_pose is not None else np.eye(4))

    def start_new_trial(self):
        """Store current trial to disk and begin a new one."""
        if len(self.current_data["timestamp"]) == 0:
            return  # ignore empty trial

        group_name = f"trial_{self.trial_idx:03d}"
        grp = self.h5_file.create_group(group_name)
        for k, v in self.current_data.items():
            grp.create_dataset(k, data=np.array(v))
        self.h5_file.flush()
        print(f"[Logger] Saved {group_name} with {len(self.current_data['timestamp'])} samples.")

        self.trial_idx += 1
        self._reset_buffers()

    def close(self):
        """Flush remaining data and close file."""
        if len(self.current_data["timestamp"]) > 0:
            self.start_new_trial()
        self.h5_file.close()
        print(f"[Logger] Closed file after {self.trial_idx} trials.")