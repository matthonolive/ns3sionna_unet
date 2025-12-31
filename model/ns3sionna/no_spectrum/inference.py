import os
os.environ.setdefault("MPLBACKEND", "Agg")  # avoid tkinter crashes in scripts
import matplotlib
matplotlib.use("Agg")

from pathlib import Path
import numpy as np
import polars as pl
import torch
import matplotlib.pyplot as plt
from einops import rearrange

from mlink.antenna import AntennaGrid, AntennaDatabase
from mlink.feature import build_feature_tensor
from mlink.geometry import generate_wall_map, walls_to_mesh
from mlink.scene import Scene


# ----------------------------
# Paths
# ----------------------------
RUN_DIR = Path("runs/height_cond")
MODEL_PT = RUN_DIR / "model.pt"           # TorchScript model
STATS_NPZ = RUN_DIR / "norm_stats.npz"    # normalization stats
OUT_PNG = RUN_DIR / "inference_4x4.png"


# ----------------------------
# Scene config (match training)
# ----------------------------
FREQ = 2e9
IMG_HW = (64, 64)
SCALE = 0.625

K_SLICES = 4
Z_STEP = 1.0
Z_MARGIN = 0.5

FLOOR_H = 0.0
CEIL_MIN = 8.0
CEIL_MAX = 20.0

TX_ORIGIN_XY = (1.75, 1.75)
TX_Z = 2.4
TX_SPACING_XY = 12.0
TX_SHAPE = (1, 5, 5)  # 25 TX, all same height


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


def make_scene_with_walls(seed: int = 0):
    rng = np.random.default_rng(seed)
    H, W = IMG_HW

    # 2D wall map for plotting
    walls_2d = generate_wall_map(
        (H, W),
        min_wall_length=8,
        min_door_length=4,
        max_partitions=24,
        rng=rng,
    )

    ceiling_h = float(rng.uniform(CEIL_MIN, CEIL_MAX))
    mesh = walls_to_mesh(
        walls_2d,
        floor_height=FLOOR_H,
        ceiling_height=ceiling_h,
    ).apply_scale(SCALE)

    # choose z0 such that K slices fit within [floor+margin, ceil-margin]
    usable = max(ceiling_h - FLOOR_H - 2 * Z_MARGIN, 1e-3)
    total_span = (K_SLICES - 1) * Z_STEP
    if total_span > usable:
        z_step = usable / max(K_SLICES - 1, 1)
    else:
        z_step = Z_STEP

    z_start = FLOOR_H + Z_MARGIN
    z_end = (ceiling_h - Z_MARGIN) - total_span
    z0 = z_start if z_end < z_start else float(rng.uniform(z_start, z_end))

    rx_grid = AntennaGrid(
        origin=SCALE * np.asarray([0.0, 0.0, z0], dtype=np.float32),
        deltas=SCALE * np.asarray(
            [[1.0, 0.0, 0.0],
             [0.0, 1.0, 0.0],
             [0.0, 0.0, z_step]],
            dtype=np.float32,
        ),
        shape=(K_SLICES, H, W),
    )

    tx_grid = AntennaGrid(
        origin=SCALE * np.asarray([TX_ORIGIN_XY[0], TX_ORIGIN_XY[1], TX_Z], dtype=np.float32),
        deltas=SCALE * np.asarray(
            [[TX_SPACING_XY, 0.0, 0.0],
             [0.0, TX_SPACING_XY, 0.0],
             [0.0, 0.0, 1.0]],
            dtype=np.float32,
        ),
        shape=TX_SHAPE,
    )

    antenna_db = AntennaDatabase.from_grid(tx_grid, rx_grid)
    mat_db = default_material_db(FREQ)
    face2material = {k: 0 for k in range(mesh.faces.shape[0])}

    scene = Scene(mesh=mesh, material_database=mat_db, face2material=face2material, antenna_database=antenna_db)
    return scene, walls_2d


def load_stats(npz_path: Path):
    d = np.load(npz_path, allow_pickle=True)
    x_mean = torch.from_numpy(d["x_mean"]).float()  # expect (4,1,1)
    x_std  = torch.from_numpy(d["x_std"]).float()
    y_mean = torch.from_numpy(d["y_mean"]).float()  # expect (1,1,1)
    y_std  = torch.from_numpy(d["y_std"]).float()
    return {"x_mean": x_mean, "x_std": x_std, "y_mean": y_mean, "y_std": y_std}


@torch.no_grad()
def predict(scene: Scene, model, stats, device: str = "cuda"):
    """
    Returns dict with:
      walls: (H,W) from generator
      cost: (K,H,W) for a chosen TX
      rss_gt: (K,H,W)
      rss_pred: (K,H,W)
      z_slices: list of slice heights (meters, scaled coords)
    """
    model = model.to(device).eval()
    x_mean = stats["x_mean"].to(device)
    x_std  = stats["x_std"].to(device)
    y_mean = stats["y_mean"].to(device)
    y_std  = stats["y_std"].to(device)

    # Build features (cost + height_cond + rss)
    ft = build_feature_tensor(scene, FREQ, requested=["cost", "height_cond", "rss"]).astype(np.float32)
    # expected: (tx, 5, K, H, W)
    num_tx = ft.shape[0]
    K = ft.shape[2]

    # Choose a TX to visualize (center TX of 5x5 grid)
    tx_idx = num_tx // 2

    # Extract GT maps for that TX
    cost_gt = ft[tx_idx, 0, :, :, :]      # (K,H,W)
    rss_gt  = ft[tx_idx, 4, :, :, :]      # (K,H,W)

    # Prepare model input for ALL tx+slice, then select the one we want after reshaping
    x_all = ft[:, 0:4, :, :, :]  # (tx,4,K,H,W)
    x_all = rearrange(x_all, "tx c k h w -> (tx k) c h w")  # (tx*K,4,H,W)
    x = torch.from_numpy(x_all).to(device)

    # normalize
    x = (x - x_mean) / x_std

    # predict normalized rss
    y_hat = model(x)  # (tx*K,1,H,W)

    # de-normalize
    y_hat = y_hat * y_std + y_mean  # (tx*K,1,H,W)

    # reshape back to (tx,K,H,W)
    y_hat = rearrange(y_hat, "(tx k) 1 h w -> tx k h w", tx=num_tx, k=K)
    rss_pred = y_hat[tx_idx].detach().cpu().numpy()  # (K,H,W)

    # slice heights (scaled coords) for labels
    rx_grid = scene.antenna_database.rx_grid
    z_slices = []
    for k in range(K):
        z = float((rx_grid.origin + k * rx_grid.deltas[2])[2])
        z_slices.append(z)

    return {
        "tx_idx": tx_idx,
        "cost": cost_gt,
        "rss_gt": rss_gt,
        "rss_pred": rss_pred,
        "z_slices": z_slices,
    }


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)

    if not MODEL_PT.exists():
        raise FileNotFoundError(f"Missing {MODEL_PT}")
    if not STATS_NPZ.exists():
        raise FileNotFoundError(f"Missing {STATS_NPZ}")

    stats = load_stats(STATS_NPZ)
    model = torch.jit.load(str(MODEL_PT), map_location=device)

    scene, walls_2d = make_scene_with_walls(seed=123)

    out = predict(scene, model, stats, device=device)

    K = min(4, out["cost"].shape[0])  # plot first 4 slices
    fig, axes = plt.subplots(4, 4, figsize=(16, 16), constrained_layout=True)

    col_titles = ["Walls (2D)", "Cost map", "GT RSS (dBm)", "Pred RSS (dBm)"]
    for c, t in enumerate(col_titles):
        axes[0, c].set_title(t)

    for r in range(4):
        for c in range(4):
            axes[r, c].axis("off")

    for r in range(K):
        z = out["z_slices"][r]
        # left-side row label
        axes[r, 0].text(
            0.02, 0.98, f"slice {r}\nz={z:.2f}",
            transform=axes[r, 0].transAxes,
            va="top", ha="left", fontsize=10,
            bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"),
        )

        axes[r, 0].imshow(walls_2d, origin="lower")
        axes[r, 1].imshow(out["cost"][r], origin="lower")
        axes[r, 2].imshow(out["rss_gt"][r], origin="lower")
        axes[r, 3].imshow(out["rss_pred"][r], origin="lower")

    fig.suptitle(f"Inference check (TX idx={out['tx_idx']})", fontsize=14)
    fig.savefig(OUT_PNG, dpi=150)
    plt.close(fig)

    print("Saved:", OUT_PNG)


if __name__ == "__main__":
    main()
