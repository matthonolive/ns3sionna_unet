import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from einops import rearrange

from mlink.antenna import AntennaGrid, AntennaDatabase
from mlink.feature import build_feature_tensor
from mlink.geometry import generate_wall_map, walls_to_mesh
from mlink.scene import Scene

from torch.utils.tensorboard import SummaryWriter

os.environ.setdefault("MPLBACKEND", "Agg")  # must be set before pyplot import

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt


from torchvision.utils import make_grid


def save_loss_plot(train_losses, val_losses, out_path: Path):
    plt.figure()
    plt.plot(train_losses, label="train")
    plt.plot(val_losses, label="val")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def tensor_to_vis01(t: torch.Tensor) -> torch.Tensor:
    """
    Convert a (1,H,W) or (H,W) tensor to [0,1] for visualization.
    Uses robust scaling via percentiles to avoid outliers dominating.
    """
    if t.dim() == 3 and t.shape[0] == 1:
        t = t[0]
    t = t.detach().float().cpu()
    lo = torch.quantile(t.flatten(), 0.02)
    hi = torch.quantile(t.flatten(), 0.98)
    t = (t - lo) / (hi - lo + 1e-6)
    return t.clamp(0, 1)


@torch.no_grad()
def log_prediction_panel(writer: SummaryWriter, model, batch, stats, device, global_step: int, tag="pred"):
    """
    Logs a small panel: cost | z_rel | true_rss | pred_rss | abs_err
    Assumes your training inputs are normalized; we de-normalize rss for viewing.
    """
    x, y = batch
    x = x.to(device)
    y = y.to(device)

    pred = model(x)

    # de-normalize rss for visualization: y = y_norm * y_std + y_mean
    y_mean = stats["y_mean"].to(device)  # shape (1,1,1)
    y_std  = stats["y_std"].to(device)   # shape (1,1,1)

    y_dn = y * y_std + y_mean
    p_dn = pred * y_std + y_mean

    # pick first item in batch
    cost = x[0, 0:1]          # (1,H,W)
    zrel = x[0, 1:2]          # (1,H,W)  <-- channel 1 is z_rel inside height_cond
    true = y_dn[0, 0:1]
    prd  = p_dn[0, 0:1]
    err  = (prd - true).abs()

    imgs = torch.stack([
        tensor_to_vis01(cost),
        tensor_to_vis01(zrel),
        tensor_to_vis01(true),
        tensor_to_vis01(prd),
        tensor_to_vis01(err),
    ], dim=0)  # (5,H,W)

    grid = make_grid(imgs.unsqueeze(1), nrow=5)  # (C,H,W) where C=1 here
    writer.add_image(tag, grid, global_step)


@dataclass
class CFG:
    out_dir: str = "runs/height_cond"
    frequency: float = 2e9

    img_hw: tuple[int, int] = (64, 64)

    # dataset size
    num_scenes: int = 128
    train_frac: float = 0.9

    # receiver slices per scene (K in rx_grid.shape = (K,H,W))
    K_slices: int = 4
    z_step: float = 1.0              # meters between slices (before scaling)
    z_margin: float = 0.5            # keep slices away from floor/ceiling

    # room heights (before scaling)
    floor_h: float = 0.0
    ceil_min: float = 8.0
    ceil_max: float = 20.0

    # keep consistent with your test.py, but you can set to 1.0 if you want
    scale: float = 0.625

    # tx grid like test.py (25 transmitters)
    tx_origin_xy: tuple[float, float] = (1.75, 1.75)
    tx_z: float = 2.4
    tx_spacing_xy: float = 12.0
    tx_shape: tuple[int, int, int] = (1, 5, 5)

    # training
    batch_size: int = 16
    num_workers: int = 2
    lr: float = 2e-4
    epochs: int = 20
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    amp: bool = True

    # model I/O
    save_state_path: str = "model_state.pt"
    save_jit_path: str = "model.pt"
    save_stats_path: str = "norm_stats.npz"

cfg = CFG()


# ----------------------------
# Utilities
# ----------------------------

def default_material_db(freq: float) -> pl.DataFrame:
    return pl.DataFrame(
        data={
            "id": [0],
            "frequency": [freq],
            "permittivity": [4.0],
            "permeability": [1.0],
            "conductivity": [0.01],
            "transmission_loss_vertical": [10.0],
            "transmission_loss_horizontal": [20.0],
            "reflection_loss": [9.0],
            "diffraction_loss_min": [8.0],
            "diffraction_loss_max": [15.0],
            "diffraction_loss": [5.0],
            "name": ["0"],
            "thickness": [0.1],
        }
    )


def make_scene(rng: np.random.Generator) -> Scene:
    H, W = cfg.img_hw

    ceiling_h = float(rng.uniform(cfg.ceil_min, cfg.ceil_max))
    mesh = walls_to_mesh(
        generate_wall_map(
            (H, W),
            min_wall_length=8,
            min_door_length=4,
            max_partitions=24,
            rng=rng,
        ),
        floor_height=cfg.floor_h,
        ceiling_height=ceiling_h,
    ).apply_scale(cfg.scale)

    # pick z_start so that K slices fit inside [floor+margin, ceil-margin]
    usable = max(ceiling_h - cfg.floor_h - 2 * cfg.z_margin, 1e-3)
    total_span = (cfg.K_slices - 1) * cfg.z_step
    if total_span > usable:
        z_step = usable / max(cfg.K_slices - 1, 1)
    else:
        z_step = cfg.z_step

    z_start = cfg.floor_h + cfg.z_margin
    z_end = (ceiling_h - cfg.z_margin) - total_span
    if z_end < z_start:
        z0 = z_start
    else:
        z0 = float(rng.uniform(z_start, z_end))

    rx_grid = AntennaGrid(
        origin=cfg.scale * np.asarray([0.0, 0.0, z0], dtype=np.float32),
        deltas=cfg.scale
        * np.asarray(
            [[1.0, 0.0, 0.0],
             [0.0, 1.0, 0.0],
             [0.0, 0.0, z_step]],
            dtype=np.float32,
        ),
        shape=(cfg.K_slices, H, W),
    )

    tx_grid = AntennaGrid(
        origin=cfg.scale * np.asarray([cfg.tx_origin_xy[0], cfg.tx_origin_xy[1], cfg.tx_z], dtype=np.float32),
        deltas=cfg.scale
        * np.asarray(
            [[cfg.tx_spacing_xy, 0.0, 0.0],
             [0.0, cfg.tx_spacing_xy, 0.0],
             [0.0, 0.0, 1.0]],
            dtype=np.float32,
        ),
        shape=cfg.tx_shape,
    )

    antenna_db = AntennaDatabase.from_grid(tx_grid, rx_grid)
    mat_db = default_material_db(cfg.frequency)

    face2material = {k: 0 for k in range(mesh.faces.shape[0])}
    return Scene(mesh=mesh, material_database=mat_db, face2material=face2material, antenna_database=antenna_db)


def build_xy_dataset(n_scenes: int, seed: int = 0) -> tuple[np.ndarray, dict]:
    """
    Returns a float32 array of shape:
      (N, C_total, H, W)
    where requested=["cost","height_cond","rss"] => C_total = 1 + 3 + 1 = 5
    """
    rng = np.random.default_rng(seed)

    tmp_scene = make_scene(rng)
    num_tx = tmp_scene.antenna_database.tx_coords.shape[0]
    K = tmp_scene.antenna_database.rx_grid.shape[0]
    H, W = tmp_scene.antenna_database.rx_grid.shape[1:]

    requested = ["cost", "height_cond", "rss"]
    C_total = 5

    total_samples = n_scenes * num_tx * K
    data = np.empty((total_samples, C_total, H, W), dtype=np.float32)

    idx = 0
    for s in range(n_scenes):
        scene = make_scene(rng)

        ft = build_feature_tensor(scene, cfg.frequency, requested=requested)
        # expected: (num_tx, C_total, K, H, W)
        ft = rearrange(ft, "tx c k h w -> (tx k) c h w").astype(np.float32)

        n = ft.shape[0]
        data[idx : idx + n] = ft
        idx += n

        print(f"[{s+1}/{n_scenes}] wrote {n} samples (running total {idx})")

    meta = {"requested": requested, "C_total": C_total, "H": H, "W": W, "num_tx": num_tx, "K": K}
    return data, meta


class TensorDataset(Dataset):
    def __init__(self, data: np.ndarray, stats: dict | None = None):
        self.data = data
        self.stats = stats

    def __len__(self) -> int:
        return self.data.shape[0]

    def __getitem__(self, i: int):
        # x: (4,H,W), y: (1,H,W)
        x = torch.from_numpy(self.data[i, 0:4].copy()).float()
        y = torch.from_numpy(self.data[i, 4:5].copy()).float()

        if self.stats is not None:
            # stats are shaped to broadcast with (C,H,W) directly
            x = (x - self.stats["x_mean"]) / self.stats["x_std"]
            y = (y - self.stats["y_mean"]) / self.stats["y_std"]

        return x, y


def compute_norm_stats(data: np.ndarray, max_samples: int = 2048, seed: int = 0) -> dict:
    """
    IMPORTANT: returns stats shaped for (C,H,W) tensors:
      x_mean/x_std: (4,1,1)
      y_mean/y_std: (1,1,1)
    so normalization does NOT introduce an extra dimension.
    """
    rng = np.random.default_rng(seed)
    N = data.shape[0]
    take = min(max_samples, N)
    idx = rng.choice(N, size=take, replace=False)

    x = data[idx, 0:4]  # (take,4,H,W)
    y = data[idx, 4:5]  # (take,1,H,W)

    xt = torch.from_numpy(x)
    yt = torch.from_numpy(y)

    x_mean = xt.mean(dim=(0, 2, 3), keepdim=False).view(4, 1, 1)
    x_std  = xt.std(dim=(0, 2, 3), keepdim=False).clamp_min(1e-6).view(4, 1, 1)

    y_mean = yt.mean(dim=(0, 2, 3), keepdim=False).view(1, 1, 1)
    y_std  = yt.std(dim=(0, 2, 3), keepdim=False).clamp_min(1e-6).view(1, 1, 1)

    return {"x_mean": x_mean.float(), "x_std": x_std.float(), "y_mean": y_mean.float(), "y_std": y_std.float()}


# ----------------------------
# Model (small U-Net)
# ----------------------------

class TinyUNet(nn.Module):
    def __init__(self, in_ch: int = 4, out_ch: int = 1, base: int = 32):
        super().__init__()
        self.enc1 = nn.Sequential(nn.Conv2d(in_ch, base, 3, padding=1), nn.ReLU(),
                                  nn.Conv2d(base, base, 3, padding=1), nn.ReLU())
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = nn.Sequential(nn.Conv2d(base, base*2, 3, padding=1), nn.ReLU(),
                                  nn.Conv2d(base*2, base*2, 3, padding=1), nn.ReLU())
        self.pool2 = nn.MaxPool2d(2)

        self.mid = nn.Sequential(nn.Conv2d(base*2, base*4, 3, padding=1), nn.ReLU(),
                                 nn.Conv2d(base*4, base*4, 3, padding=1), nn.ReLU())

        self.up2 = nn.ConvTranspose2d(base*4, base*2, 2, stride=2)
        self.dec2 = nn.Sequential(nn.Conv2d(base*4, base*2, 3, padding=1), nn.ReLU(),
                                  nn.Conv2d(base*2, base*2, 3, padding=1), nn.ReLU())

        self.up1 = nn.ConvTranspose2d(base*2, base, 2, stride=2)
        self.dec1 = nn.Sequential(nn.Conv2d(base*2, base, 3, padding=1), nn.ReLU(),
                                  nn.Conv2d(base, base, 3, padding=1), nn.ReLU())

        self.out = nn.Conv2d(base, out_ch, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        m = self.mid(self.pool2(e2))

        d2 = self.up2(m)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))

        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))

        return self.out(d1)


# ----------------------------
# Train + export
# ----------------------------

def main():
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(cfg.device, "enabled")  # FIX: was CFG.device

    writer = SummaryWriter(log_dir=str(out_dir / "tb"))

    print("Building dataset...")
    dataset_file = out_dir / "dataset.npy"
    if dataset_file.exists():
        print("Loading cached dataset:", dataset_file)
        data = np.load(dataset_file, mmap_mode="r")  # memory-mapped read
        meta = {}
    else:
        print("Building dataset...")
        data, meta = build_xy_dataset(cfg.num_scenes, seed=0)
        np.save(dataset_file, data)

    stats = compute_norm_stats(data)
    np.savez(
        out_dir / cfg.save_stats_path,
        x_mean=stats["x_mean"].numpy(),
        x_std=stats["x_std"].numpy(),
        y_mean=stats["y_mean"].numpy(),
        y_std=stats["y_std"].numpy(),
        meta=np.array([str(meta)], dtype=object),
    )

    # split
    N = data.shape[0]
    n_train = int(cfg.train_frac * N)
    perm = np.random.default_rng(0).permutation(N)
    train_idx = perm[:n_train]
    val_idx = perm[n_train:]

    train_ds = TensorDataset(data[train_idx], stats=stats)
    val_ds = TensorDataset(data[val_idx], stats=stats)

    train_dl = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers)
    val_dl = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)

    # Optional sanity check (uncomment once):
    # x0, y0 = train_ds[0]
    # print("sample shapes:", x0.shape, y0.shape)  # should be torch.Size([4,64,64]) torch.Size([1,64,64])
    # xb, yb = next(iter(train_dl))
    # print("batch shapes:", xb.shape, yb.shape)  # should be [B,4,64,64] and [B,1,64,64]

    model = TinyUNet(in_ch=4, out_ch=1).to(cfg.device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    loss_fn = nn.L1Loss()

    amp_enabled = cfg.amp and cfg.device.startswith("cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    train_losses = []
    val_losses = []
    global_step = 0

    for epoch in range(cfg.epochs):
        # -------------------------
        # Train
        # -------------------------
        model.train()
        tr_loss_sum = 0.0

        for x, y in train_dl:
            x = x.to(cfg.device)
            y = y.to(cfg.device)

            opt.zero_grad(set_to_none=True)

            with torch.amp.autocast(device_type="cuda", enabled=amp_enabled):
                pred = model(x)          # x is (B,4,H,W) now
                loss = loss_fn(pred, y)  # y is (B,1,H,W)

            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            loss_item = float(loss.detach().cpu())
            writer.add_scalar("loss/train_step", loss_item, global_step)

            tr_loss_sum += loss_item
            global_step += 1

        tr_loss = tr_loss_sum / max(len(train_dl), 1)

        # -------------------------
        # Validate
        # -------------------------
        model.eval()
        va_loss_sum = 0.0
        first_val_batch = None

        with torch.no_grad():
            for x, y in val_dl:
                if first_val_batch is None:
                    first_val_batch = (x.clone(), y.clone())

                x = x.to(cfg.device)
                y = y.to(cfg.device)

                pred = model(x)
                va_loss_sum += float(loss_fn(pred, y).detach().cpu())

        va_loss = va_loss_sum / max(len(val_dl), 1)

        # -------------------------
        # Epoch-level logging
        # -------------------------
        train_losses.append(tr_loss)
        val_losses.append(va_loss)

        writer.add_scalar("loss/train_epoch", tr_loss, epoch)
        writer.add_scalar("loss/val_epoch", va_loss, epoch)

        if (epoch % 2) == 0 and first_val_batch is not None:
            log_prediction_panel(
                writer=writer,
                model=model,
                batch=first_val_batch,
                stats=stats,
                device=cfg.device,
                global_step=epoch,
                tag="panel/cost_zrel_true_pred_err",
            )

        save_loss_plot(train_losses, val_losses, out_dir / "loss_curve.png")

        print(f"epoch {epoch+1:03d} | train L1 {tr_loss:.4f} | val L1 {va_loss:.4f}")

    # save state_dict
    torch.save(model.state_dict(), out_dir / cfg.save_state_path)

    # export TorchScript model.pt
    model.eval()
    example = torch.randn(1, 4, cfg.img_hw[0], cfg.img_hw[1], device=cfg.device)
    traced = torch.jit.trace(model, example)
    traced.save(str(out_dir / cfg.save_jit_path))

    print("Saved:")
    print(" -", out_dir / cfg.save_state_path)
    print(" -", out_dir / cfg.save_jit_path)
    print(" -", out_dir / cfg.save_stats_path)

    writer.close()


if __name__ == "__main__":
    main()
