from pathlib import Path
from types import SimpleNamespace

import hydra
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, Dataset

from symdex.utils.offline_buffer import OfflineBuffer


PICKOBJECT_TD3BC_MULTI_CFG = SimpleNamespace(
    single_agent_obs_dim=[93, 96],
    single_agent_obs_idx=[
        [[0, 68], [139, 161], [186, 189]],
        [[68, 136], [136, 139], [161, 186]],
    ],
)

LABEL_IDX = {
    "object_pos_1": (56, 59),
    "object_pos_2": (124, 127),
    "tote_pos": (136, 139),
    "waiting_pos": (183, 186),
    "target_pos": (186, 189),
}


def slice_tensor(tensor, indices):
    if len(indices) == 1:
        start, end = indices[0]
        return tensor[:, start:end]
    return torch.cat([tensor[:, start:end] for start, end in indices], dim=1)


class MLPNet(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_layers=None, use_batchnorm=False):
        super().__init__()
        if hidden_layers is None:
            hidden_layers = [512, 256, 128]
        dims = [in_dim, *hidden_layers, out_dim]
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                if use_batchnorm:
                    layers.append(nn.BatchNorm1d(dims[i + 1]))
                layers.append(nn.ELU())
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class MLPBlock(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dims, act, layer_norm=False, dropout=0.0, activate_last=False):
        super().__init__()
        dims = [in_dim, *hidden_dims, out_dim]
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if (i != len(dims) - 2) or activate_last:
                if layer_norm:
                    layers.append(nn.LayerNorm(dims[i + 1]))
                layers.append(act())
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class StateEncoder(nn.Module):
    def __init__(self, multi_cfg, out_dim=256):
        super().__init__()
        self.obs_idx = multi_cfg.single_agent_obs_idx
        self.obs_dim = list(multi_cfg.single_agent_obs_dim)

        self.right_encoder = MLPNet(in_dim=self.obs_dim[0], out_dim=128, hidden_layers=[256, 128])
        self.left_encoder = MLPNet(in_dim=self.obs_dim[1], out_dim=128, hidden_layers=[256, 128])
        self.fusion = MLPNet(in_dim=256, out_dim=out_dim, hidden_layers=[512, 256])

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        state_right = slice_tensor(state, self.obs_idx[0])
        state_left = slice_tensor(state, self.obs_idx[1])

        if state_right.shape[-1] != self.obs_dim[0]:
            raise RuntimeError(f"state_right dim mismatch: got {state_right.shape[-1]}, expected {self.obs_dim[0]}")
        if state_left.shape[-1] != self.obs_dim[1]:
            raise RuntimeError(f"state_left dim mismatch: got {state_left.shape[-1]}, expected {self.obs_dim[1]}")

        z_right = self.right_encoder(state_right)
        z_left = self.left_encoder(state_left)
        return self.fusion(torch.cat([z_right, z_left], dim=-1))


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
        B, C, N = x.shape
        k = min(self.k, N - 1)
        if k <= 0:
            return x.new_zeros(B, self.net[-2].num_features, N)

        dist = torch.cdist(xyz.transpose(1, 2), xyz.transpose(1, 2))
        idx = dist.topk(k=k + 1, dim=-1, largest=False).indices[:, :, 1:]

        idx_base = torch.arange(B, device=x.device).view(B, 1, 1) * N
        idx = (idx + idx_base).reshape(-1)

        x_t = x.transpose(1, 2).contiguous()
        neighbors = x_t.reshape(B * N, C)[idx].reshape(B, N, k, C)
        center = x_t.reshape(B, N, 1, C).expand(-1, -1, k, -1)

        edge = torch.cat([center, neighbors - center], dim=-1)
        edge = edge.permute(0, 3, 1, 2).contiguous()
        return self.net(edge).max(dim=-1).values


class GeometryPCBranch(nn.Module):
    def __init__(self, out_dim=128, k=16, max_points=2048):
        super().__init__()
        self.max_points = max_points
        self.edge1 = EdgeConvBlock(3, 64, k)
        self.edge2 = EdgeConvBlock(64, 128, k)
        self.proj = MLPBlock(
            in_dim=64 + 128 + 13,
            out_dim=out_dim,
            hidden_dims=[256],
            act=nn.ReLU,
            layer_norm=True,
            activate_last=True,
        )

    def forward(self, pc_xyz: torch.Tensor) -> torch.Tensor:
        if pc_xyz.ndim != 3 or pc_xyz.shape[-1] != 3:
            raise RuntimeError(f"Expected pc_xyz shape (B, N, 3), got {tuple(pc_xyz.shape)}")

        pc_xyz = torch.nan_to_num(pc_xyz, nan=0.0, posinf=0.0, neginf=0.0)
        B, N, _ = pc_xyz.shape
        if N > self.max_points:
            idx = torch.randperm(N, device=pc_xyz.device)[: self.max_points]
            pc_xyz = pc_xyz[:, idx]

        valid = pc_xyz.abs().sum(dim=-1) > 1e-8
        no_valid = valid.sum(dim=1) == 0
        if no_valid.any():
            valid = valid.clone()
            valid[no_valid, 0] = True

        valid_f = valid.float().unsqueeze(-1)
        count = valid_f.sum(dim=1, keepdim=True).clamp_min(1.0)
        center = (pc_xyz * valid_f).sum(dim=1, keepdim=True) / count
        centered = pc_xyz - center
        radius = (centered.norm(dim=-1, keepdim=True) * valid_f).amax(dim=1, keepdim=True).clamp_min(1e-6)

        norm_xyz = centered / radius
        x0 = norm_xyz.transpose(1, 2).contiguous()
        xyz = norm_xyz.transpose(1, 2).contiguous()

        h1 = self.edge1(x0, xyz)
        h2 = self.edge2(h1, xyz)

        mask = valid.unsqueeze(1)
        h1_global = h1.masked_fill(~mask, -1e9).max(dim=-1).values
        h2_global = h2.masked_fill(~mask, -1e9).max(dim=-1).values

        centered_valid = centered * valid_f
        cov = torch.bmm(centered_valid.transpose(1, 2), centered_valid)
        cov = cov / count.squeeze(1).unsqueeze(-1).clamp_min(1.0)
        cov_feat = cov.reshape(B, 9)

        geom = torch.cat([center.squeeze(1), radius.squeeze(1), cov_feat], dim=-1)
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
        if pc.ndim != 3 or pc.shape[-1] != 6:
            raise RuntimeError(f"Expected pc shape (B, N, 6), got {tuple(pc.shape)}")
        pc = torch.nan_to_num(pc, nan=0.0, posinf=0.0, neginf=0.0)
        z_right = self.right_encoder(pc[..., :3])
        z_left = self.left_encoder(pc[..., 3:6])
        return self.fusion(torch.cat([z_right, z_left], dim=-1))


class MLPHead(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        return self.net(x)


class StateActionProbe(nn.Module):
    def __init__(self, action_dim):
        super().__init__()
        self.state_encoder = StateEncoder(PICKOBJECT_TD3BC_MULTI_CFG, out_dim=256)
        self.head = MLPHead(256, action_dim)

    def forward(self, state, pc):
        return self.head(self.state_encoder(state))


class PCActionProbe(nn.Module):
    def __init__(self, action_dim, pc_k=16, pc_max_points=512):
        super().__init__()
        self.pc_encoder = PCEncoder(out_dim=256, k=pc_k, max_points=pc_max_points)
        self.head = MLPHead(256, action_dim)

    def forward(self, state, pc):
        return self.head(self.pc_encoder(pc))


class StatePCActionProbe(nn.Module):
    def __init__(self, action_dim, pc_k=16, pc_max_points=512):
        super().__init__()
        self.state_encoder = StateEncoder(PICKOBJECT_TD3BC_MULTI_CFG, out_dim=256)
        self.pc_encoder = PCEncoder(out_dim=256, k=pc_k, max_points=pc_max_points)
        self.head = MLPHead(512, action_dim)

    def forward(self, state, pc):
        z_state = self.state_encoder(state)
        z_pc = self.pc_encoder(pc)
        return self.head(torch.cat([z_state, z_pc], dim=-1))


class PCLabelProbe(nn.Module):
    def __init__(self, label_dim, pc_k=16, pc_max_points=512):
        super().__init__()
        self.pc_encoder = PCEncoder(out_dim=256, k=pc_k, max_points=pc_max_points)
        self.head = MLPHead(256, label_dim)

    def forward(self, state, pc):
        return self.head(self.pc_encoder(pc))


class StateLabelProbe(nn.Module):
    def __init__(self, label_dim):
        super().__init__()
        self.state_encoder = StateEncoder(PICKOBJECT_TD3BC_MULTI_CFG, out_dim=256)
        self.head = MLPHead(256, label_dim)

    def forward(self, state, pc):
        return self.head(self.state_encoder(state))


class PretrainDataset(Dataset):
    def __init__(self, state, pc, target):
        self.state = state
        self.pc = pc
        self.target = target

    def __len__(self):
        return self.state.shape[0]

    def __getitem__(self, idx):
        return self.state[idx], self.pc[idx], self.target[idx]


def normalize_by_train(x, train_idx, eps=1e-6):
    mean = x[train_idx].mean(dim=0, keepdim=True)
    std = x[train_idx].std(dim=0, keepdim=True).clamp_min(eps)
    return (x - mean) / std, mean, std


def make_indices(n, val_ratio, seed):
    gen = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=gen)
    n_val = int(n * val_ratio)
    val_idx = perm[:n_val]
    train_idx = perm[n_val:]
    return train_idx, val_idx


def train_probe(name, model, state, pc, target, train_idx, val_idx, device, epochs, batch_size, lr):
    state_norm, state_mean, state_std = normalize_by_train(state, train_idx)
    target_norm, target_mean, target_std = normalize_by_train(target, train_idx)

    dataset = PretrainDataset(state_norm, pc, target_norm)
    train_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=torch.utils.data.SubsetRandomSampler(train_idx.tolist()),
        pin_memory=False,
    )
    val_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=torch.utils.data.SubsetRandomSampler(val_idx.tolist()),
        pin_memory=False,
    )

    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    best_val = float("inf")
    best_state = None

    print(f"\n[Train] {name}")
    print("[Baseline] normalized zero-predictor val_mse is about 1.0")

    for epoch in range(1, epochs + 1):
        model.train()
        train_sum = 0.0
        train_count = 0

        for s, p, y in train_loader:
            s = s.to(device, non_blocking=True)
            p = p.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            pred = model(s, p)
            loss = F.mse_loss(pred, y)
            
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()

            train_sum += loss.item() * s.shape[0]
            train_count += s.shape[0]

        model.eval()
        val_sum = 0.0
        val_count = 0
        with torch.no_grad():
            for s, p, y in val_loader:
                s = s.to(device, non_blocking=True)
                p = p.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                loss = F.mse_loss(model(s, p), y)
                val_sum += loss.item() * s.shape[0]
                val_count += s.shape[0]

        train_loss = train_sum / max(train_count, 1)
        val_loss = val_sum / max(val_count, 1)

        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        if epoch == 1 or epoch == epochs or epoch % 20 == 0:
            print(f"epoch {epoch:04d} | train_mse={train_loss:.6f} | val_mse={val_loss:.6f}")

    model.load_state_dict(best_state)
    print(f"[Best] {name}: val_mse={best_val:.6f}")
    return model, {
        "name": name,
        "best_val_mse": float(best_val),
        "state_mean": state_mean,
        "state_std": state_std,
        "target_mean": target_mean,
        "target_std": target_std,
    }


@torch.no_grad()
def print_pc_stats(pc):
    print("\n[PC stats]")
    print("pc shape:", tuple(pc.shape))
    print("pc mean/std:", pc.mean().item(), pc.std(unbiased=False).item())
    print("pc min/max:", pc.min().item(), pc.max().item())
    print("nan:", torch.isnan(pc).sum().item())
    print("inf:", torch.isinf(pc).sum().item())

    for side_name, x in [("right", pc[..., :3]), ("left", pc[..., 3:6])]:
        valid = x.abs().sum(dim=-1) > 1e-8
        print(f"{side_name} valid_ratio:", valid.float().mean().item())
        print(f"{side_name} empty_ratio:", (valid.sum(dim=1) == 0).float().mean().item())


def clean_results(results):
    return {key: {"name": value["name"], "best_val_mse": value["best_val_mse"]} for key, value in results.items()}


@hydra.main(config_path="symdex/cfg", config_name="default", version_base=None)
def main(cfg: DictConfig):
    pretrain_cfg = cfg.algo.pretrain
    # if cfg.algo.name != "TD3BC":
    #     raise RuntimeError(f"Expected algo=td3_bc / TD3BC, got cfg.algo.name={cfg.algo.name}")
    # if cfg.env_name != "PickObjectEnv-v0":
    #     raise RuntimeError(f"Expected task=pickObject, got env_name={cfg.env_name}")

    if str(cfg.device).startswith("cuda") and not torch.cuda.is_available():
        device = torch.device("cpu")
        print("[Device] CUDA requested but unavailable; using CPU.")
    else:
        device = torch.device(cfg.device)
    print(f"[Device] model/batches: {device}")
    print("[Device] dataset tensors: CPU")

    buffer = OfflineBuffer(device="cpu", use_vision=False, use_pc=True)
    buffer.load_from_dir(cfg.algo.offline.data_dir, pattern=cfg.algo.offline.pattern)

    state = torch.as_tensor(buffer.state, dtype=torch.float32)
    pc = torch.as_tensor(buffer.pc, dtype=torch.float32)
    action = torch.as_tensor(buffer.actions, dtype=torch.float32)

    print("\n[Dataset]")
    print("state:", tuple(state.shape))
    print("pc:", tuple(pc.shape))
    print("action:", tuple(action.shape))

    if state.shape[-1] != 189:
        raise RuntimeError(f"Expected PickObject TD3BC state dim 189, got {state.shape[-1]}")
    if pc.ndim != 3 or pc.shape[-1] != 6:
        raise RuntimeError(f"Expected pc shape (B, N, 6), got {tuple(pc.shape)}")

    print_pc_stats(pc)

    train_idx, val_idx = make_indices(
        n=state.shape[0],
        val_ratio=float(pretrain_cfg.val_ratio),
        seed=int(pretrain_cfg.seed),
    )
    action_dim = action.shape[-1]
    pc_k = int(pretrain_cfg.get("pc_k", 16))
    pc_max_points = int(pretrain_cfg.get("pc_max_points", 512))
    if pc.shape[1] > pc_max_points:
        idx = torch.randperm(pc.shape[1])[:pc_max_points]
        pc = pc[:, idx, :]
        print(f"[PC] pre-subsampled to {pc_max_points} pts → pc shape: {tuple(pc.shape)}")
    results = {}
    state_action_model = None
    pc_action_model = None
    state_pc_action_model = None

    if pretrain_cfg.train_state_action:
        state_action_model, info = train_probe(
            name="state_encoder -> action",
            model=StateActionProbe(action_dim),
            state=state,
            pc=pc,
            target=action,
            train_idx=train_idx,
            val_idx=val_idx,
            device=device,
            epochs=int(pretrain_cfg.epochs),
            batch_size=int(pretrain_cfg.batch_size),
            lr=float(pretrain_cfg.lr),
        )
        results["state_action"] = info

    if pretrain_cfg.train_pc_action:
        pc_action_model, info = train_probe(
            name="pc_encoder -> action",
            model=PCActionProbe(action_dim, pc_k=pc_k, pc_max_points=pc_max_points),
            state=state,
            pc=pc,
            target=action,
            train_idx=train_idx,
            val_idx=val_idx,
            device=device,
            epochs=int(pretrain_cfg.epochs),
            batch_size=int(pretrain_cfg.batch_size),
            lr=float(pretrain_cfg.lr),
        )
        results["pc_action"] = info

    if pretrain_cfg.train_state_pc_action:
        state_pc_action_model, info = train_probe(
            name="state_encoder + pc_encoder -> action",
            model=StatePCActionProbe(action_dim, pc_k=pc_k, pc_max_points=pc_max_points),
            state=state,
            pc=pc,
            target=action,
            train_idx=train_idx,
            val_idx=val_idx,
            device=device,
            epochs=int(pretrain_cfg.epochs),
            batch_size=int(pretrain_cfg.batch_size),
            lr=float(pretrain_cfg.lr),
        )
        results["state_pc_action"] = info

    for label_name in list(pretrain_cfg.labels):
        if label_name not in LABEL_IDX:
            raise RuntimeError(f"Unknown pretrain label: {label_name}. Valid labels: {sorted(LABEL_IDX)}")
        start, end = LABEL_IDX[label_name]
        label = state[:, start:end]

        if pretrain_cfg.train_pc_labels:
            _, info = train_probe(
                name=f"pc_encoder -> {label_name}",
                model=PCLabelProbe(label_dim=end - start, pc_k=pc_k, pc_max_points=pc_max_points),
                state=state,
                pc=pc,
                target=label,
                train_idx=train_idx,
                val_idx=val_idx,
                device=device,
                epochs=int(pretrain_cfg.epochs),
                batch_size=int(pretrain_cfg.batch_size),
                lr=float(pretrain_cfg.lr),
            )
            results[f"pc_{label_name}"] = info

        if pretrain_cfg.train_state_labels:
            _, info = train_probe(
                name=f"state_encoder -> {label_name}",
                model=StateLabelProbe(label_dim=end - start),
                state=state,
                pc=pc,
                target=label,
                train_idx=train_idx,
                val_idx=val_idx,
                device=device,
                epochs=int(pretrain_cfg.epochs),
                batch_size=int(pretrain_cfg.batch_size),
                lr=float(pretrain_cfg.lr),
            )
            results[f"state_{label_name}"] = info

    print("\n[Summary: lower val_mse is better]")
    for key, value in results.items():
        print(f"{key}: {value['best_val_mse']:.6f}")

    print("\n[Usefulness check]")
    print("StateEncoder useful if state_action val_mse << 1.0")
    print("PCEncoder useful if pc_action or pc_object_pos val_mse << 1.0")
    print("PC helps TD3BC if state_pc_action val_mse < state_action val_mse")

    if pretrain_cfg.save_checkpoint:
        if state_pc_action_model is not None:
            state_encoder = state_pc_action_model.state_encoder
            pc_encoder = state_pc_action_model.pc_encoder
            main_task = "state_encoder + pc_encoder -> action"
        elif state_action_model is not None and pc_action_model is not None:
            state_encoder = state_action_model.state_encoder
            pc_encoder = pc_action_model.pc_encoder
            main_task = "separate state_action and pc_action"
        else:
            raise RuntimeError("Cannot save encoder checkpoint without trained state and pc encoders.")

        ckpt_path = Path(pretrain_cfg.checkpoint_path)
        if not ckpt_path.is_absolute():
            ckpt_path = Path.cwd() / ckpt_path
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)

        torch.save(
            {
                "state_encoder": state_encoder.state_dict(),
                "pc_encoder": pc_encoder.state_dict(),
                "state_encoder_from_state_action": (
                    state_action_model.state_encoder.state_dict() if state_action_model is not None else None
                ),
                "pc_encoder_from_pc_action": (
                    pc_action_model.pc_encoder.state_dict() if pc_action_model is not None else None
                ),
                "results": clean_results(results),
                "multi_cfg": {
                    "single_agent_obs_dim": PICKOBJECT_TD3BC_MULTI_CFG.single_agent_obs_dim,
                    "single_agent_obs_idx": PICKOBJECT_TD3BC_MULTI_CFG.single_agent_obs_idx,
                },
                "encoder_config": {
                    "state_out_dim": 256,
                    "pc_out_dim": 256,
                    "pc_branch_dim": 128,
                    "pc_k": pc_k,
                    "pc_max_points": pc_max_points,
                },
                "state_dim": int(state.shape[-1]),
                "pc_shape": tuple(pc.shape[1:]),
                "action_dim": int(action_dim),
                "main_pretrain_task": main_task,
                "cfg": OmegaConf.to_container(cfg, resolve=True),
            },
            ckpt_path,
        )
        print(f"\n[Checkpoint] encoder-only checkpoint saved to: {ckpt_path}")


if __name__ == "__main__":
    main()
