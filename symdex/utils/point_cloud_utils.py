import numpy as np
import torch
import torch.nn.functional as F
from collections.abc import Sequence
import open3d as o3d
import warp as wp

import isaaclab.utils.math as math_utils
from isaaclab.utils.array import TensorData, convert_to_torch


def crop_and_pad(depth_cloud: torch.Tensor, crop_range):
    B, N, _ = depth_cloud.shape
    device = depth_cloud.device   # keep computations on the same device

    # 1. Boolean crop mask ----------------------------------------------------
    mask = (
        (depth_cloud[..., 0] > crop_range[0][0]) & (depth_cloud[..., 0] < crop_range[0][1]) &
        (depth_cloud[..., 1] > crop_range[1][0]) & (depth_cloud[..., 1] < crop_range[1][1]) &
        (depth_cloud[..., 2] > crop_range[2][0]) & (depth_cloud[..., 2] < crop_range[2][1])
    )                                              # shape (B, N)

    # 2. Move valid points to the front with one gather -----------------------
    #    Sort mask so 1-entries come first → get permutation indices
    _, perm_idx = torch.sort(mask.int(), dim=1, descending=True)      # (B, N)
    depth_sorted = torch.gather(
        depth_cloud, 1, perm_idx.unsqueeze(-1).expand(-1, -1, 3)
    )                                                                # (B, N, 3)

    # 3. Trim to the maximum number of valid points --------------------------
    valid_counts  = mask.sum(dim=1)            # how many valids in each batch – (B,)
    max_valid_pts = int(valid_counts.max())    # scalar

    # Slice – all batches now (B, max_valid_pts, 3)
    output_cloud  = depth_sorted[:, :max_valid_pts, :]

    # 4. Zero-out any leftover slots for the shorter batches ------------------
    rng          = torch.arange(max_valid_pts, device=device)         # (max_valid_pts,)
    keep_mask    = (rng.unsqueeze(0) < valid_counts.unsqueeze(1)).unsqueeze(-1)  # (B, max_valid_pts, 1)
    output_cloud = output_cloud * keep_mask                           # invalid rows → 0
    output_cloud = torch.nan_to_num(output_cloud, nan=0.0, posinf=0.0, neginf=0.0)  # TODO: don't know why this is needed, even output_cloud and keep_mask are not nan, sometimes output_cloud is nan
    return output_cloud, valid_counts

def crop_and_pad_multi(
    depth_cloud: torch.Tensor,                  # (B, N, 3)
    crop_ranges: Sequence[Sequence[Sequence[float]]],  # (R, 3, 2)
    inclusive: bool = False,                    # use >= and <= if True
    squash_nans: bool = True                    # replace NaNs after masking
):
    """
    Returns:
      cropped: (B, R, M, 3)  where M = max #valid points over (B,R), zero-padded
      counts:  (B, R)        valid counts per (batch, range)
    """
    assert depth_cloud.ndim == 3 and depth_cloud.size(-1) == 3
    B, N, _ = depth_cloud.shape
    device, dtype = depth_cloud.device, depth_cloud.dtype

    ranges = torch.as_tensor(crop_ranges, dtype=dtype, device=device)  # (R, 3, 2)
    assert ranges.ndim == 3 and ranges.shape[1:] == (3, 2), "crop_ranges must be (R,3,2)"

    R = ranges.size(0)
    x, y, z = depth_cloud.unbind(dim=-1)  # (B, N) each

    # Broadcast to (R, B, N)
    def cmp(lo, hi, v):
        lo = lo[:, None, None]
        hi = hi[:, None, None]
        if inclusive:
            return (v[None, ...] >= lo) & (v[None, ...] <= hi)
        else:
            return (v[None, ...] >  lo) & (v[None, ...] <  hi)

    x_ok = cmp(ranges[:, 0, 0], ranges[:, 0, 1], x)
    y_ok = cmp(ranges[:, 1, 0], ranges[:, 1, 1], y)
    z_ok = cmp(ranges[:, 2, 0], ranges[:, 2, 1], z)
    mask = x_ok & y_ok & z_ok                                 # (R, B, N)

    # Move valid points to the front via a single gather per range/batch
    scores = mask.int()                                       # (R, B, N)
    perm_idx = torch.argsort(scores, dim=-1, descending=True) # (R, B, N)

    cloud_rb = depth_cloud.unsqueeze(0).expand(R, -1, -1, -1) # (R, B, N, 3)
    cloud_sorted = torch.gather(
        cloud_rb, 2, perm_idx[..., None].expand(-1, -1, -1, 3)
    )                                                         # (R, B, N, 3)

    valid_counts = mask.sum(dim=-1)                           # (R, B)
    M = int(valid_counts.max().item()) if N > 0 else 0
    cropped = cloud_sorted[:, :, :M, :]                       # (R, B, M, 3)

    # Zero out the padded tail per (R,B)
    if M > 0:
        rng = torch.arange(M, device=device)                  # (M,)
        keep = (rng[None, None, :] < valid_counts[..., None]) # (R, B, M)
        cropped = cropped * keep[..., None]                   # (R, B, M, 3)

    if squash_nans:
        # Note: NaN * 0 = NaN, so if input had NaNs in discarded rows,
        # they survive the multiply. This line cleans them up.
        cropped = torch.nan_to_num(cropped, nan=0.0, posinf=0.0, neginf=0.0)

    # Return as (B, R, M, 3) and counts (B, R)
    return cropped.permute(1, 0, 2, 3).contiguous(), valid_counts.permute(1, 0).contiguous()

def random_point_sampling(points: torch.Tensor, K: int):
    """
    Uniformly samples K points from each point cloud in a batch using gather.

    Args:
        points: (B, N, 3)
        K: number of points to sample

    Returns:
        sampled: (B, K, 3)
    """
    B, N, _ = points.shape
    assert K <= N, "K must be <= number of points"
    # random sampling
    indices = torch.randperm(N)[:K]
    sampled_points = points[:, indices]
    return sampled_points

def farthest_point_sampling(points: torch.Tensor, K: int):
    """
    Batched Farthest Point Sampling (FPS)

    Args:
        points: (B, N, 3)
        K: number of points to sample

    Returns:
        sampled_xyz: (B, K, 3)
    """
    B, N, _ = points.shape
    device = points.device

    # Safety: replace any NaNs/Infs in input
    valid_mask = torch.all(~torch.isnan(points) & ~torch.isinf(points), dim=-1)
    fallback = torch.tensor([0., 0., 0.], device=device)
    points = torch.where(valid_mask.unsqueeze(-1), points, fallback)

    dists = torch.full((B, N), float('inf'), device=device)
    sampled_idx = torch.zeros((B, K), dtype=torch.long, device=device)
    
    # Start from a random valid point (one with non-zero norm)
    first = torch.randint(0, N, (B,), device=device)
    sampled_idx[:, 0] = first

    batch_indices = torch.arange(B, device=device)

    centroid = points[batch_indices, first].unsqueeze(1)  # (B, 1, 3)
    dist = ((points - centroid) ** 2).sum(-1)  # (B, N)
    dists = torch.minimum(dists, dist)

    for i in range(1, K):
        farthest = torch.argmax(dists, dim=1)
        sampled_idx[:, i] = farthest

        centroid = points[batch_indices, farthest].unsqueeze(1)
        dist = ((points - centroid) ** 2).sum(-1)
        dist = torch.nan_to_num(dist, nan=1e10, posinf=1e10, neginf=1e10)
        dists = torch.minimum(dists, dist)

    # Gather sampled points
    sampled_xyz = torch.gather(
        points,
        dim=1,
        index=sampled_idx.unsqueeze(-1).expand(-1, -1, 3)
    )

    return sampled_xyz

def shuffle_point_torch(point_cloud):
    B, N, C = point_cloud.shape
    indices = torch.randperm(N)
    return point_cloud[:, indices]

def _exact_frac_mask(B, N, frac, device):
        k = max(1, int(round(N * frac)))
        perm = torch.rand(B, N, device=device).argsort(dim=1)
        mask = torch.zeros(B, N, dtype=torch.bool, device=device)
        mask.scatter_(1, perm[:, :k], True)
        return mask  # (B, N)

def radial_noise(x, frac=0.10, rel_sigma=0.01, origin=None):
    B, N, _ = x.shape
    m = _exact_frac_mask(B, N, frac, x.device).unsqueeze(-1)
    c = x.new_zeros(B,1,3) if origin is None else origin.view(B,1,3)
    dir = F.normalize(x - c, dim=-1)
    delta = torch.randn(B, N, 1, device=x.device)  # scalar per point
    return x + dir * delta * m

def create_pointcloud_from_depth(
    intrinsic_matrix: np.ndarray | torch.Tensor | wp.array,
    depth: np.ndarray | torch.Tensor | wp.array,
    position: Sequence[float] | None = None,
    orientation: Sequence[float] | None = None,
    device: torch.device | str | None = None,
    crop_range: Sequence[Sequence[float]] | None = None,
    max_points: int | None = None,
    num_cams: int | None = None,
    downsample: str = "random",
    add_noise: bool = False,
) -> np.ndarray | torch.Tensor:
    r"""Creates pointcloud from input depth image and camera intrinsic matrix.

    This function creates a pointcloud from a depth image and camera intrinsic matrix. The pointcloud is
    computed using the following equation:

    .. math::
        p_{camera} = K^{-1} \times [u, v, 1]^T \times d

    where :math:`K` is the camera intrinsic matrix, :math:`u` and :math:`v` are the pixel coordinates and
    :math:`d` is the depth value at the pixel.

    Additionally, the pointcloud can be transformed from the camera frame to a target frame by providing
    the position ``t`` and orientation ``R`` of the camera in the target frame:

    .. math::
        p_{target} = R_{target} \times p_{camera} + t_{target}

    Args:
        intrinsic_matrix: A (3, 3) or (B, 3, 3) array providing camera's calibration matrix.
        depth: An array of shape (H, W) or (B, H, W) with values encoding the depth measurement.
        keep_invalid: Whether to keep invalid points in the cloud or not. Invalid points
            correspond to pixels with depth values 0.0 or NaN. Defaults to False.
        position: The position of the camera in a target frame. Defaults to None.
        orientation: The orientation (w, x, y, z) of the camera in a target frame. Defaults to None.
        device: The device for torch where the computation should be executed.
            Defaults to None, i.e. takes the device that matches the depth image.
        max_points: Maximum number of points to keep per batch. If None, keeps all valid points.
            Defaults to None.

    Returns:
        An array/tensor of shape (B, N, 3) comprising of 3D coordinates of points, where B is the batch size
        and N is the number of points per batch (either max_points if specified or number of valid points).
        The returned datatype is torch if input depth is of type torch.tensor or wp.array. Otherwise, a np.ndarray
        is returned.
    """
    # We use PyTorch here for matrix multiplication since it is compiled with Intel MKL while numpy
    # by default uses OpenBLAS. With PyTorch (CPU), we could process a depth image of size (480, 640)
    # in 0.0051 secs, while with numpy it took 0.0292 secs.

    # convert to numpy matrix
    is_numpy = isinstance(depth, np.ndarray)
    # decide device
    if device is None and is_numpy:
        device = torch.device("cpu")
    # convert depth to torch tensor
    depth = convert_to_torch(depth, dtype=torch.float32, device=device)
    # update the device with the device of the depth image
    # note: this is needed since warp does not provide the device directly
    device = depth.device
    # convert inputs to torch tensors
    intrinsic_matrix = convert_to_torch(intrinsic_matrix, dtype=torch.float32, device=device)
    if position is not None:
        position = convert_to_torch(position, dtype=torch.float32, device=device)
    if orientation is not None:
        orientation = convert_to_torch(orientation, dtype=torch.float32, device=device)

    # handle batched inputs
    if len(depth.shape) == 2:
        depth = depth.unsqueeze(0)  # Add batch dimension if not present
        if len(intrinsic_matrix.shape) == 2:
            intrinsic_matrix = intrinsic_matrix.unsqueeze(0)
    
    # compute pointcloud
    depth_cloud = math_utils.unproject_depth(depth, intrinsic_matrix)  # Shape: (B, H*W, 3)
    # transform the point cloud to the target frame
    depth_cloud = math_utils.transform_points(depth_cloud, position, orientation)
    # Mask for valid points (not NaN or inf)
    depth_cloud = torch.nan_to_num(depth_cloud, nan=float('inf'), posinf=float('inf'), neginf=float('inf'))
    if add_noise:
        # add Gaussian noise to the point cloud
        depth_cloud = depth_cloud + torch.randn_like(depth_cloud) * 0.001
        # add Gaussian noise to the point cloud
        # depth_cloud = radial_noise(depth_cloud, frac=0.10, rel_sigma=0.01, origin=None)
    # merge the point cloud from multiple cameras
    if num_cams is not None:
        B = depth_cloud.shape[0] // num_cams
        depth_cloud = depth_cloud.reshape(num_cams, B, -1, 3)
        depth_cloud = depth_cloud.permute(1, 0, 2, 3).reshape(B, -1, 3)
    # crop the point cloud given xyz range
    if crop_range is not None:
        depth_cloud, valid_counts = crop_and_pad(depth_cloud, crop_range)
    # downsample the point cloud
    if downsample == "FSP":
        depth_cloud = farthest_point_sampling(points=depth_cloud, K=max_points)
    elif downsample == "random":
        if max_points > depth_cloud.shape[1]:
            pad_points = torch.zeros((B, max_points - depth_cloud.shape[1], 3), device=depth_cloud.device)
            depth_cloud = torch.cat([depth_cloud, pad_points], dim=1)
        depth_cloud = random_point_sampling(points=depth_cloud, K=max_points)
    else:
        pass
    assert not torch.isnan(depth_cloud).any(), "pc contains nan"
    # shuffle the point cloud
    depth_cloud = shuffle_point_torch(depth_cloud)
    # return everything according to input type
    if is_numpy:
        return depth_cloud.detach().cpu().numpy()
    else:
        return depth_cloud