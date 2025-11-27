import time
import torch
import numpy as np


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

def build_observation_dict(env, obs):
    """
    Observation from env.step() converted into dict
    """
    if isinstance(obs, dict):
        out = {}
        for k, v in obs.items():
            out[k] = to_numpy(v)
        return out

    om = env.unwrapped.observation_manager
    obs_np = to_numpy(obs)  # [B, D] or [D]
    if obs_np.ndim == 1:
        vec = obs_np
    else:
        assert obs_np.shape[0] == 1, "Only single parallel environment recording is supported"
        vec = obs_np[0]

    out = {}
    for group_name in om.active_groups:
        grp = om.get_group(group_name)
        if not getattr(grp, "concatenate_terms", False):
            continue

        gdict = {}
        offset = 0
        for term_name in grp.active_terms:
            term = grp.get_term(term_name)
            if hasattr(term, "flat_dim"):
                dim = int(term.flat_dim)
                shape = getattr(term, "shape", None)
            elif hasattr(term, "dim"):
                dim = int(term.dim)
                shape = getattr(term, "shape", None)
            elif hasattr(term, "shape"):
                shape = tuple(term.shape)
                dim = int(np.prod(shape))
            else:
                raise RuntimeError(f"Cannot determine term dimension: {group_name}/{term_name}")

            sl = slice(offset, offset + dim)
            arr = vec[sl]
            if shape is not None:
                try:
                    arr = arr.reshape(shape)
                except Exception:
                    pass
            gdict[term_name] = arr.astype(np.float32, copy=False)
            offset += dim

        out[group_name] = gdict

    return out

def rgb_to_HWC(frame):
    """
    Convert RGB frame to HWC uint8 format.
       input: - (1, H, W, 3)/ (1, 1, H, W, 3): remove batch dim
              - (H, W, 3): keep as is
              - (3, H, W): convert to (H, W, 3)
        if dtype is not uint8:
              - if max <= 1.0: assume [0, 1], convert to [0, 255]
              - else: clip to [0, 255]
    """
    try:
        if isinstance(frame, torch.Tensor):
            frame = frame.detach().cpu().numpy()
    except Exception:
        pass

    frame_np = np.asarray(frame)

    if frame_np.ndim == 5 and frame_np.shape[0] == 1 and frame_np.shape[1] == 1:
        frame_np = frame_np[0, 0]             # -> (H, W, 3)
    elif frame_np.ndim == 4 and frame_np.shape[0] == 1:
        frame_np = frame_np[0]                # -> (H, W, 3) or (3, H, W)
    
    if frame_np.ndim == 3 and frame_np.shape[0] in (1, 3) and frame_np.shape[-1] not in (1, 3):
        frame_np = np.moveaxis(frame_np, 0, -1)  # -> (H, W, C)

    if frame_np.ndim != 3 or frame_np.shape[2] != 3:
        raise ValueError(f"RGB frame must be HxWx3 after normalization, got shape {frame_np.shape}")

    if frame_np.dtype != np.uint8:
        fmax = frame_np.max() if frame_np.size else 1.0
        if fmax <= 1.0:
            frame_np = (frame_np * 255.0).clip(0, 255).astype(np.uint8)
        else:
            frame_np = frame_np.clip(0, 255).astype(np.uint8)

    return frame_np

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