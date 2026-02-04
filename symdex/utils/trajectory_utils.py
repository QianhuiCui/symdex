import time
import torch
import numpy as np
import os
import hashlib
import yaml


def now_ms() -> int:
    return int(time.monotonic_ns() // 1_000_000)

def to_numpy(x):
    if hasattr(x, "detach"):
        return x.detach().cpu().numpy()
    if hasattr(x, "cpu") and hasattr(x, "numpy"):
        return x.cpu().numpy()
    return np.asarray(x)

def to_str(value):
    if isinstance(value, (list, tuple)):
        language_instruction = value[0] if len(value) > 0 else ""
    elif torch.is_tensor(value):
        language_instruction = str(value[0].item())
    else:
        language_instruction = str(value) if value is not None else ""
    return language_instruction

def as_flag(value):
    if value is None:
        return False
    if isinstance(value, torch.Tensor):
        return bool(value.any().item())
    if isinstance(value, np.ndarray):
        return bool(value.any())
    if isinstance(value, (list, tuple)):
        return any(map(bool, value))
    return bool(value)

def get_obs(obs_dict):
    obs = {}
    for key, value in obs_dict.items():
        if key == "critic" or "policy" in key:
            obs[key] = obs_dict[key][0].detach().cpu().numpy().astype(np.float32, copy=False)
        elif "vision" in key:
            img = obs_dict[key][0, 0].detach().cpu().numpy()
            img = np.moveaxis(img, 0, -1)  # -> [H, W, 3]
            # convert to uint8 HWC
            if img.dtype != np.uint8:
                mn = float(img.min())
                mx = float(img.max())
                if mn >= -1.0 and mx <= 1.0:
                    img = ((img + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
                elif mx <= 1.0:
                    img = (img * 255.0).clip(0, 255).astype(np.uint8)
                else:
                    img = img.clip(0, 255).astype(np.uint8)
            obs[key] = img
    return obs

def depth_to_gray(depth):
    try:
        if isinstance(depth, torch.Tensor):
            depth = depth.detach().cpu().numpy()
    except Exception:
        pass

    depth_np = np.asarray(depth)

    if depth_np.ndim == 5 and depth_np.shape[0] == 1 and depth_np.shape[1] == 1:
        depth_np = depth_np[0, 0]             # -> (H, W)
    elif depth_np.ndim == 4 and depth_np.shape[0] == 1:
        depth_np = depth_np[0]                # -> (H, W) or (1, H, W)
    elif depth_np.ndim == 3 and depth_np.shape[0] == 1:
        depth_np = depth_np[0]                # -> (H, W)
    
    if depth_np.ndim == 3 and depth_np.shape[2] == 1:
        depth_np = depth_np[:, :, 0]          # -> (H, W)
    
    if depth_np.ndim != 2:
        raise ValueError(f"Depth frame must be HxW after normalization, got shape {depth_np.shape}")
    
    depth_np = depth_np.astype(np.float32)
    d_min, d_max = float(np.nanmin(depth_np)), float(np.nanmax(depth_np))
    den = (d_max - d_min) if (d_max - d_min) > 1e-12 else 1.0
    d_u8 = ((depth_np - d_min) / den * 255.0).clip(0, 255).astype(np.uint8)
    return d_u8 

def get_policy_obs(obs):
    """
    Extract policy observations from VecEnvWrapper._process_obs
    return float 1D numpy array: shape [obs_dim].
    """
    # cam_enable=False: obs torch.Tensor [B, obs_dim] or [obs_dim]
    if torch.is_tensor(obs):
        policy = obs.detach().cpu().numpy()
        return policy[0] if policy.ndim == 2 else policy
    
    # cam_enable=True: obs dict with key "policy"
    policy = obs["policy"]
    policy = policy.detach().cpu().numpy()
    return policy[0] if policy.ndim == 2 else policy