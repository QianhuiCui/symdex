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


class EdgeConvBlock(nn.Module):
    def __init__(self, in_dim, out_dim, k=16):
        super().__init__()
        self.k = k
        self.net = nn.Sequential(
            nn.Conv2d(in_dim * 2, out_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(),
            nn.Conv2d(out_dim, out_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(),
        )
    
    def forward(self, x: torch.Tensor, xyz: torch.Tensor) -> torch.Tensor:
        # x: (B, C, N), xyz: (B, 3, N)
        B, C, N = x.shape
        k = min(self.k, N-1)
        if k <= 0:
            return x.new_zeros(B, self.net[-2].num_features, N)
        
        dist = torch.cdist(xyz.transpose(1, 2), xyz.transpose(1, 2))  # (B, N, N)
        idx = dist.topk(k=k + 1, dim=-1, largest=False).indices[:, :, 1:]

        idx_base = torch.arange(B, device=x.device).view(B, 1, 1) * N
        idx = (idx + idx_base).reshape(-1)

        x_t = x.transpose(1, 2).contiguous()
        neighbors = x_t.reshape(B * N, C)[idx].reshape(B, N, k, C)
        center = x_t.reshape(B, N, 1, C).expand(-1, -1, k, -1)

        edge = torch.cat([center, neighbors - center], dim=-1)  
        edge = edge.permute(0, 3, 1, 2).contiguous()  # (B, 2C, N, k)

        out = self.net(edge)
        return out.max(dim=-1).values


class GeometryPCBranch(nn.Module):
    def __init__(self, out_dim, k, max_points):
        super().__init__()
        self.max_points = max_points
        self.edge1 = EdgeConvBlock(3, 64, k)
        self.edge2 = EdgeConvBlock(64, 128, k)

        # h1_global: 64, h2_global: 128, geom: center(3) + radius(1) + cov(9)
        self.proj = MLPBlock(
            in_dim=64 + 128 + 13,
            out_dim=out_dim,
            hidden_dims=[256],
            act=nn.ReLU,
            layer_norm=True,
            activate_last=True,
        )

    def forward(self, pc_xyz: torch.Tensor) -> torch.Tensor:
        # pc_xyz: (B, N, 3)
        if pc_xyz.ndim != 3 or pc_xyz.shape[-1] != 3:
            raise RuntimeError(f"Expected pc_xyz shape (B, N, 3), got {tuple(pc_xyz.shape)}")
        pc_xyz = torch.nan_to_num(pc_xyz, nan=0.0, posinf=0.0, neginf=0.0)
        B, N, _ = pc_xyz.shape
        if N > self.max_points:
            idx = torch.randperm(N, device=pc_xyz.device)[:self.max_points]
            pc_xyz = pc_xyz[:, idx]
        
        valid = pc_xyz.abs().sum(dim=-1) > 1e-8
        no_valid = valid.sum(dim=1) == 0
        if no_valid.any():
            valid = valid.clone()
            valid[no_valid, 0] = True
        
        valid_f = valid.float().unsqueeze(-1)  # (B, N, 1)
        count = valid_f.sum(dim=1, keepdim=True).clamp_min(1.0)  # (B, 1, 1)
        center = (pc_xyz * valid_f).sum(dim=1, keepdim=True) / count  # (B, 1, 3)
        centered = pc_xyz - center
        radius = (centered.norm(dim=-1, keepdim=True) * valid_f).amax(dim=1, keepdim=True).clamp_min(1e-6)  # (B, 1, 1)

        norm_xyz = centered / radius
        x0 = norm_xyz.transpose(1, 2).contiguous()   # (B, 3, N)
        xyz = norm_xyz.transpose(1, 2).contiguous()  # (B, 3, N)

        h1 = self.edge1(x0, xyz)  # (B, 64, N)
        h2 = self.edge2(h1, xyz)  # (B, 128, N)

        mask = valid.unsqueeze(1)  # (B, 1, N)

        h1_global = h1.masked_fill(~mask, -1e9).max(dim=-1).values
        h2_global = h2.masked_fill(~mask, -1e9).max(dim=-1).values

        centered_valid = centered * valid_f
        cov = torch.bmm(centered_valid.transpose(1, 2), centered_valid)
        cov = cov / count.squeeze(1).unsqueeze(-1).clamp_min(1.0)
        cov_feat = cov.reshape(B, 9)

        geom = torch.cat([center.squeeze(1),  # (B, 3)
                          radius.squeeze(1),  # (B, 1)
                          cov_feat,           # (B, 9)
                         ],dim=-1,)

        return self.proj(torch.cat([h1_global, h2_global, geom], dim=-1))


class PCEncoder(nn.Module):
    def __init__(self, out_dim=256, branch_dim=128, k=16, max_points=2048):
        super().__init__()

        self.right_encoder = GeometryPCBranch(out_dim=branch_dim, k=k, max_points=max_points)
        self.left_encoder = GeometryPCBranch(out_dim=branch_dim, k=k, max_points=max_points)

        self.fusion = MLPBlock(
            in_dim=branch_dim * 2,
            out_dim=out_dim,
            hidden_dims=[256],
            act=nn.ReLU,
            layer_norm=True,
            activate_last=True,
        )

    def forward(self, pc: torch.Tensor) -> torch.Tensor:
        # pc: (B, N, 6), right xyz + left xyz
        if pc.ndim != 3 or pc.shape[-1] != 6:
            raise RuntimeError(f"Expected pc shape (B, N, 6), got {tuple(pc.shape)}")

        pc = torch.nan_to_num(pc, nan=0.0, posinf=0.0, neginf=0.0)

        pc_right = pc[..., :3]
        pc_left = pc[..., 3:6]

        z_right = self.right_encoder(pc_right)  # (B, branch_dim)
        z_left = self.left_encoder(pc_left)     # (B, branch_dim)

        z_pc = self.fusion(torch.cat([z_right, z_left], dim=-1))
        return z_pc  # (B, out_dim)
