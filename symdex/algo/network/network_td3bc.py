import copy
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from symdex.algo.network.mlp import MLPNet, MLPBlock
from symdex.utils.torch_util import slice_tensor


class StateEncoder(nn.Module):
    def __init__(self, multi_cfg, out_dim=256):
        super().__init__()

        self.obs_idx = multi_cfg.single_agent_obs_idx
        self.obs_dim = list(multi_cfg.single_agent_obs_dim)
        assert len(self.obs_idx) == 2, "TD3BC expects two state branches: right and left."
        assert len(self.obs_dim) == 2, "TD3BC expects two obs dims: right and left."

        self.right_encoder = MLPNet(
            in_dim=self.obs_dim[0],
            out_dim=128,
            hidden_layers=[256, 128],
        )

        self.left_encoder = MLPNet(
            in_dim=self.obs_dim[1],
            out_dim=128,
            hidden_layers=[256, 128],
        )

        self.fusion = MLPNet(
            in_dim=256,
            out_dim=out_dim,
            hidden_layers=[512, 256],
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        state_right = slice_tensor(state, self.obs_idx[0])
        state_left = slice_tensor(state, self.obs_idx[1])

        if state_right.shape[-1] != self.obs_dim[0]:
            raise RuntimeError(
                f"state_right dim mismatch: got {state_right.shape[-1]}, expected {self.obs_dim[0]}"
            )
        if state_left.shape[-1] != self.obs_dim[1]:
            raise RuntimeError(
                f"state_left dim mismatch: got {state_left.shape[-1]}, expected {self.obs_dim[1]}"
            )

        z_right = self.right_encoder(state_right)
        z_left = self.left_encoder(state_left)
        z = torch.cat([z_right, z_left], dim=-1)
        z = self.fusion(z)
        return z


class PCBranch(nn.Module):
    def __init__(self, out_dim: int = 128, max_points: int = 512):
        super().__init__()
        self.max_points = max_points  # downsample

        # Per-point MLP: abs_xyz(3) + norm_xyz(3) = 6 input dims
        self.point_mlp = nn.Sequential(
            nn.Linear(6, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
        )

        # max_pool(128) + mean_pool(128) + geom: center(3)+radius(1)+cov(9) = 13
        self.agg = nn.Sequential(
            nn.Linear(128 * 2 + 13, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, out_dim),
            nn.LayerNorm(out_dim),
            nn.ReLU(),
        )

    def forward(self, xyz: torch.Tensor) -> torch.Tensor:
        # xyz: (B, N, 3)
        if xyz.ndim != 3 or xyz.shape[-1] != 3:
            raise RuntimeError(f"Expected xyz shape (B, N, 3), got {tuple(xyz.shape)}")

        xyz = torch.nan_to_num(xyz, nan=0.0, posinf=0.0, neginf=0.0)
        B, N, _ = xyz.shape

        if N > self.max_points:
            idx = torch.randperm(N, device=xyz.device)[:self.max_points]
            xyz = xyz[:, idx]

        valid = xyz.abs().sum(dim=-1) > 1e-8  # (B, N)
        no_valid = valid.sum(dim=1) == 0
        if no_valid.any():
            valid = valid.clone()
            valid[no_valid, 0] = True

        valid_f = valid.float().unsqueeze(-1)  # (B, N, 1)
        count = valid_f.sum(dim=1, keepdim=True).clamp_min(1.0)  # (B, 1, 1)

        center = (xyz * valid_f).sum(dim=1, keepdim=True) / count  # (B, 1, 3)
        centered = xyz - center  # (B, N, 3)
        radius = (centered.norm(dim=-1, keepdim=True) * valid_f).amax(dim=1, keepdim=True).clamp_min(1e-6)  # (B, 1, 1)
        norm_xyz = centered / radius  # (B, N, 3), local shape of object

        c_valid = centered * valid_f
        cov = torch.bmm(c_valid.transpose(1, 2), c_valid)
        cov = cov / count.squeeze(1).unsqueeze(-1).clamp_min(1.0)  # (B, 3, 3)

        x = torch.cat([xyz, norm_xyz], dim=-1)  # (B, N, 6)
        h = self.point_mlp(x)  # (B, N, 128), learned point features

        mask   = valid.unsqueeze(-1)  # (B, N, 1)
        h_max  = h.masked_fill(~mask, -1e9).max(dim=1).values  # (B, 128), masked max pooling
        h_mean = (h * valid_f).sum(dim=1) / count.squeeze(1)  # (B, 128), masked mean pooling

        geom = torch.cat([center.squeeze(1),
                          radius.squeeze(1),
                          cov.reshape(B, 9),], dim=-1)  # (B, 13)

        return self.agg(torch.cat([h_max, h_mean, geom], dim=-1))    # (B, out_dim)


class PCEncoder(nn.Module):
    """
    Dual-branch PointNet encoder for sim-to-real Actor.
      pc[..., :3]  -> right branch (object_1 side)
      pc[..., 3:6] -> left  branch (object_2 side)
    """

    def __init__(self, out_dim: int = 256, branch_dim: int = 128, max_points: int = 512):
        super().__init__()
        self.right_encoder = PCBranch(out_dim=branch_dim, max_points=max_points)
        self.left_encoder  = PCBranch(out_dim=branch_dim, max_points=max_points)

        self.fusion = MLPBlock(
            in_dim=branch_dim * 2,
            out_dim=out_dim,
            hidden_dims=[256],
            act=nn.ReLU,
            layer_norm=True,
            activate_last=True,
        )

    def forward(self, pc: torch.Tensor) -> torch.Tensor:
        # pc: (B, N, 6)
        if pc.ndim != 3 or pc.shape[-1] != 6:
            raise RuntimeError(f"Expected pc shape (B, N, 6), got {tuple(pc.shape)}")

        pc = torch.nan_to_num(pc, nan=0.0, posinf=0.0, neginf=0.0)
        z_right = self.right_encoder(pc[..., :3])
        z_left  = self.left_encoder(pc[..., 3:6])
        return self.fusion(torch.cat([z_right, z_left], dim=-1))


class MultiModalEncoder(nn.Module):
    def __init__(self, actor_cfg, state_out_dim=256, pc_out_dim=256, out_dim=256, pc_max_points=512):
        super().__init__()
        self.state_encoder = StateEncoder(actor_cfg, out_dim=state_out_dim)
        self.pc_encoder = PCEncoder(out_dim=pc_out_dim, max_points=pc_max_points)
        self.fusion = MLPBlock(
            in_dim=state_out_dim + pc_out_dim,
            out_dim=out_dim,
            hidden_dims=[512, 256],
            act=nn.ReLU,
            layer_norm=True,
            activate_last=True,
        )
    
    def forward(self, batch: dict) -> torch.Tensor:
        z_state = self.state_encoder(batch['state'])
        z_pc = self.pc_encoder(batch['pc'])
        return self.fusion(torch.cat([z_state, z_pc], dim=-1))
