#!/usr/bin/env python3
import sionna.rt  # must be first for your setup

import os
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import argparse
import json
from dataclasses import replace as dc_replace
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter, generic_filter

import torch
import tensorflow as tf
from sionna.rt import load_scene

# mlink
from mlink.antenna import AntennaGrid, AntennaDatabase
from mlink.feature import build_feature_tensor
from mlink.scene import Scene as MlinkScene
from mlink.channel_tdl import RtCfg, subcarrier_frequencies_centered, compute_tdl_batch


C0 = 299_792_458.0


# ----------------------------
# small helpers
# ----------------------------

def enable_tf_memory_growth():
    try:
        gpus = tf.config.list_physical_devices("GPU")
        for g in gpus:
            try:
                tf.config.experimental.set_memory_growth(g, True)
            except Exception:
                pass
    except Exception:
        pass


def fspl_db(d_m: np.ndarray, fc_hz: float) -> np.ndarray:
    d = np.maximum(np.asarray(d_m, dtype=np.float32), 1e-6)
    lam = C0 / float(fc_hz)
    return (20.0 * np.log10(4.0 * np.pi * d / lam)).astype(np.float32)


def to_c11(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a)
    if a.ndim == 1:
        a = a[:, None, None]
    return a.astype(np.float32)


def tau_from_target(tau_tgt: np.ndarray, tau_target: str, tau_log_eps_ns: float) -> np.ndarray:
    tau_tgt = np.asarray(tau_tgt, dtype=np.float32)
    if tau_target == "raw":
        return np.maximum(tau_tgt, 0.0)
    if tau_target == "log10":
        return np.maximum((10.0 ** tau_tgt) - float(tau_log_eps_ns), 0.0)
    raise ValueError(f"Unsupported tau_target={tau_target}")


def masked_gaussian_2d(img: np.ndarray, mask: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0:
        return img.astype(np.float32)
    m = mask.astype(np.float32)
    num = gaussian_filter(img * m, sigma=sigma, mode="nearest")
    den = gaussian_filter(m, sigma=sigma, mode="nearest")
    out = np.zeros_like(img, dtype=np.float32)
    good = den > 1e-6
    out[good] = (num[good] / den[good]).astype(np.float32)
    return out


def masked_median_2d(img: np.ndarray, mask: np.ndarray, size: int = 3) -> np.ndarray:
    if size <= 1:
        return img.astype(np.float32)

    work = img.astype(np.float32).copy()
    work[~mask] = np.nan

    def nanmed(w):
        return np.nanmedian(w)

    out = generic_filter(work, nanmed, size=size, mode="nearest")
    out = out.astype(np.float32)
    out[~np.isfinite(out)] = 0.0
    return out


def smooth_map_stack(
    x_map: np.ndarray,
    wb_map: np.ndarray,
    no_path_wb_db: float,
    smooth_kind: str = "median",
    smooth_median_size: int = 3,
    smooth_gauss_sigma: float = 1.0,
) -> np.ndarray:
    K, H, W = x_map.shape
    out = np.zeros_like(x_map, dtype=np.float32)
    for k in range(K):
        mask = wb_map[k] < no_path_wb_db
        if smooth_kind == "none":
            out[k] = x_map[k]
        elif smooth_kind == "gaussian":
            out[k] = masked_gaussian_2d(x_map[k], mask, smooth_gauss_sigma)
        elif smooth_kind == "median":
            out[k] = masked_median_2d(x_map[k], mask, smooth_median_size)
        else:
            raise ValueError("smooth_kind must be one of: none, gaussian, median")
    return out


def pick_frequency_hz(args, model_meta, placements, scene_info) -> float:
    if args.frequency_hz is not None:
        return float(args.frequency_hz)
    if model_meta is not None and "frequency_hz" in model_meta:
        return float(model_meta["frequency_hz"])
    if placements is not None and "frequency_hz" in placements:
        return float(placements["frequency_hz"])
    if scene_info is not None and "frequency_hz" in scene_info:
        return float(scene_info["frequency_hz"])
    return 5.21e9


def load_json_if_exists(path: Path):
    if path.exists():
        return json.loads(path.read_text())
    return None


def load_tx_xyz(scene_dir: Path, args, placements, bbox) -> np.ndarray:
    if args.tx_xyz is not None:
        vals = [float(x) for x in args.tx_xyz.split(",")]
        if len(vals) != 3:
            raise ValueError("--tx-xyz must be formatted as x,y,z")
        return np.asarray(vals, dtype=np.float32)

    if placements is not None and "tx_xyz" in placements:
        return np.asarray(placements["tx_xyz"], dtype=np.float32)

    # fallback: scene center
    x = 0.5 * (float(bbox.min.x) + float(bbox.max.x))
    y = 0.5 * (float(bbox.min.y) + float(bbox.max.y))
    z = 0.5 * (float(bbox.min.z) + float(bbox.max.z))
    return np.asarray([x, y, z], dtype=np.float32)


def build_rx_grid_and_coords(
    bbox,
    H: int,
    W: int,
    K: int,
    scale_m: float,
    z_step_cells: float,
    z_margin_m: float,
    origin_xy_mode: str = "bbox_min",
):
    z_step_m = float(scale_m * z_step_cells)

    if origin_xy_mode == "zero":
        x0 = 0.0
        y0 = 0.0
    else:
        x0 = float(bbox.min.x)
        y0 = float(bbox.min.y)

    z_min = float(bbox.min.z)
    z_max = float(bbox.max.z)
    total_span = (K - 1) * z_step_m
    z0 = z_min + z_margin_m
    if z0 + total_span > (z_max - z_margin_m):
        z0 = max(z_min, (z_max - z_margin_m) - total_span)

    origin = np.array([x0, y0, z0], dtype=np.float32)

    rx_grid = AntennaGrid(
        origin=origin,
        deltas=np.asarray(
            [
                [scale_m, 0.0, 0.0],
                [0.0, scale_m, 0.0],
                [0.0, 0.0, z_step_m],
            ],
            dtype=np.float32,
        ),
        shape=(K, H, W),
    )

    # keep this exactly aligned with your ns3 UNet attach_scene logic
    xs = x0 + scale_m * np.arange(W, dtype=np.float32)
    ys = y0 + scale_m * np.arange(H, dtype=np.float32)
    zs = z0 + z_step_m * np.arange(K, dtype=np.float32)
    Z, Y, X = np.meshgrid(zs, ys, xs, indexing="ij")
    rx_coords = np.stack([X, Y, Z], axis=-1).reshape(-1, 3).astype(np.float32)

    return rx_grid, rx_coords, origin, z_step_m


def build_scene_for_grid(base_scene: MlinkScene, tx_xyz: np.ndarray, rx_grid: AntennaGrid, rx_coords: np.ndarray) -> MlinkScene:
    adb = AntennaDatabase(
        tx_coords=np.asarray(tx_xyz, dtype=np.float32).reshape(1, 3),
        rx_coords=rx_coords,
        tx_grid=None,
        rx_grid=rx_grid,
    )
    return dc_replace(base_scene, antenna_database=adb)


def infer_feature_counts(scene_for_grid, freq_hz: float, dataset_features: list[str]) -> dict[str, int]:
    counts = {}
    for feat in dataset_features:
        x = build_feature_tensor(scene_for_grid, freq_hz, requested=[feat]).astype(np.float32)
        counts[feat] = int(x.shape[1])
    return counts


def build_keep_idx(dataset_features, model_features, K, feat_counts):
    offsets = {}
    off = 0
    for f in dataset_features:
        offsets[f] = (off, off + feat_counts[f])
        off += feat_counts[f]

    keep_in_slice = []
    for f in model_features:
        a, b = offsets[f]
        keep_in_slice.extend(range(a, b))

    keep = []
    for k in range(K):
        base = k * off
        keep.extend([base + i for i in keep_in_slice])

    return np.asarray(keep, dtype=np.int64)


def compute_rt_maps(
    base_scene: MlinkScene,
    scene_for_grid: MlinkScene,
    tx_xyz: np.ndarray,
    rx_coords: np.ndarray,
    K: int,
    H: int,
    W: int,
    freq_hz: float,
    fft_size: int,
    subcarrier_spacing_hz: float,
    rx_batch: int,
    no_path_wb_db: float,
    smooth_kind: str,
    smooth_median_size: int,
    smooth_gauss_sigma: float,
):
    if hasattr(base_scene, "to_sionna_geometry"):
        si_geom = base_scene.to_sionna_geometry(freq_hz)
    else:
        si_geom = base_scene.to_sionna(freq_hz)

    freqs = subcarrier_frequencies_centered(fft_size, subcarrier_spacing_hz)
    rt_cfg = RtCfg(
        max_depth=10,
        samples_per_src=1_000_000,
        diffuse_reflection=True,
        diffraction=True,
        edge_diffraction=True,
        diffraction_lit_region=True,
    )

    P = rx_coords.shape[0]
    wb_all = np.zeros((P,), dtype=np.float32)
    ex_all = np.zeros((P,), dtype=np.float32)
    tau_all = np.zeros((P,), dtype=np.float32)

    for i0 in range(0, P, rx_batch):
        i1 = min(i0 + rx_batch, P)
        wb_db, ex_s, _taps, tau_rms_s = compute_tdl_batch(
            si_scene=si_geom,
            tx_xyz=np.asarray(tx_xyz, dtype=np.float32),
            rx_xyz=rx_coords[i0:i1],
            frequencies_hz=freqs,
            L_taps=int(fft_size),
            rt=rt_cfg,
            return_tau_rms=True,
        )

        wb_all[i0:i1] = wb_db.astype(np.float32)
        ex_all[i0:i1] = (ex_s * 1e9).astype(np.float32)

        good = wb_db < no_path_wb_db
        if np.any(good):
            idx_g = np.nonzero(good)[0]
            tau_all[i0 + idx_g] = (tau_rms_s[good] * 1e9).astype(np.float32)

        print(f"[RT] {i1:6d}/{P:6d} receivers done", flush=True)

    wb_map = wb_all.reshape(K, H, W)
    ex_map = ex_all.reshape(K, H, W)
    tau_map = tau_all.reshape(K, H, W)

    ex_sm = smooth_map_stack(
        ex_map, wb_map, no_path_wb_db=no_path_wb_db,
        smooth_kind=smooth_kind,
        smooth_median_size=smooth_median_size,
        smooth_gauss_sigma=smooth_gauss_sigma,
    )
    tau_sm = smooth_map_stack(
        tau_map, wb_map, no_path_wb_db=no_path_wb_db,
        smooth_kind=smooth_kind,
        smooth_median_size=smooth_median_size,
        smooth_gauss_sigma=smooth_gauss_sigma,
    )

    valid = wb_map < no_path_wb_db
    ex_sm[~valid] = np.nan
    tau_sm[~valid] = np.nan
    wb_plot = wb_map.copy()
    wb_plot[~valid] = np.nan

    return wb_plot, tau_sm, ex_sm


def compute_unet_maps(
    base_scene: MlinkScene,
    scene_for_grid: MlinkScene,
    model_dir: Path,
    freq_hz: float,
    tx_xyz: np.ndarray,
    rx_coords: np.ndarray,
    K: int,
    H: int,
    W: int,
    device: str,
    no_path_wb_db: float,
    delta_idx: int,
    excess_idx: int,
    tau_idx: int,
):
    meta = json.loads((model_dir / "meta.json").read_text())
    stats = np.load(model_dir / "norm_stats.npz")
    model = torch.jit.load(str(model_dir / "model.pt"), map_location=device).eval()

    dataset_features = meta.get(
        "dataset_features",
        ["binary_walls", "electrical_distance", "cost", "height_cond"],
    )
    model_features = meta.get("model_features", dataset_features)

    x = build_feature_tensor(scene_for_grid, freq_hz, requested=dataset_features).astype(np.float32)
    c_in = int(x.shape[1])
    x_stack = x.transpose(0, 2, 1, 3, 4).reshape(1, K * c_in, H, W)
    x_chw = x_stack[0]

    x_mean = to_c11(stats["x_mean"])
    x_std = np.maximum(to_c11(stats["x_std"]), 1e-6)
    y_mean = to_c11(stats["y_mean"])
    y_std = np.maximum(to_c11(stats["y_std"]), 1e-6)

    keep_idx = None
    if "keep_idx" in stats.files:
        keep_idx = np.asarray(stats["keep_idx"], dtype=np.int64)
        if keep_idx.size == 0:
            keep_idx = None

    if keep_idx is None and x_chw.shape[0] != x_mean.shape[0]:
        # try reconstructing keep_idx from meta if needed
        feat_counts = infer_feature_counts(scene_for_grid, freq_hz, dataset_features)
        keep_idx = build_keep_idx(dataset_features, model_features, K, feat_counts)

    if keep_idx is not None:
        x_chw = x_chw[keep_idx, :, :]

    if x_chw.shape[0] != x_mean.shape[0]:
        raise RuntimeError(
            f"UNet input channels mismatch: built {x_chw.shape[0]}, "
            f"but norm_stats expects {x_mean.shape[0]}"
        )

    x_t = torch.from_numpy(x_chw).to(device)
    x_mean_t = torch.from_numpy(x_mean).to(device)
    x_std_t = torch.from_numpy(x_std).to(device)
    y_mean_t = torch.from_numpy(y_mean).to(device)
    y_std_t = torch.from_numpy(y_std).to(device)

    with torch.no_grad():
        pred_n = model(((x_t - x_mean_t) / x_std_t).unsqueeze(0)).squeeze(0)
        pred = pred_n * y_std_t + y_mean_t

    pred = pred.detach().float().cpu().numpy()

    y_ch = int(pred.shape[0] // K)
    if y_ch < 3:
        raise RuntimeError(f"Expected at least 3 output channels per slice, got y_ch={y_ch}")

    pred = pred.reshape(K, y_ch, H, W)

    tau_target = meta.get("tau_target", "raw")
    tau_log_eps_ns = float(meta.get("tau_log_eps_ns", 1e-3))

    delta_map = pred[:, delta_idx].astype(np.float32)
    excess_map = pred[:, excess_idx].astype(np.float32)
    tau_map = tau_from_target(pred[:, tau_idx], tau_target=tau_target, tau_log_eps_ns=tau_log_eps_ns)

    d_m = np.linalg.norm(rx_coords - np.asarray(tx_xyz, dtype=np.float32)[None, :], axis=1).reshape(K, H, W).astype(np.float32)
    wb_map = fspl_db(d_m, freq_hz) + delta_map

    valid = wb_map < no_path_wb_db
    wb_plot = wb_map.copy()
    wb_plot[~valid] = np.nan
    excess_plot = excess_map.copy()
    tau_plot = tau_map.copy()
    excess_plot[~valid] = np.nan
    tau_plot[~valid] = np.nan

    return wb_plot, tau_plot.astype(np.float32), excess_plot.astype(np.float32), meta


def compute_cost231_map(scene_for_grid: MlinkScene, freq_hz: float, no_path_wb_db: float) -> np.ndarray:
    cost_feat = build_feature_tensor(scene_for_grid, freq_hz, requested=["cost"]).astype(np.float32)
    # feature "cost" is negative path loss; convert to positive loss in dB
    wb_map = -cost_feat[0, 0]
    wb_map = wb_map.astype(np.float32)
    wb_map[wb_map >= no_path_wb_db] = np.nan
    return wb_map


def compute_friis_map(tx_xyz: np.ndarray, rx_coords: np.ndarray, K: int, H: int, W: int, freq_hz: float) -> np.ndarray:
    d_m = np.linalg.norm(rx_coords - np.asarray(tx_xyz, dtype=np.float32)[None, :], axis=1)
    return fspl_db(d_m, freq_hz).reshape(K, H, W).astype(np.float32)


def choose_slice_k(args, tx_xyz: np.ndarray, origin: np.ndarray, z_step_m: float, K: int) -> int:
    if args.slice_k is not None:
        k = int(args.slice_k)
        return max(0, min(K - 1, k))

    if args.slice_mode == "mid":
        return K // 2

    # default: nearest slice to TX height
    kf = (float(tx_xyz[2]) - float(origin[2])) / float(z_step_m)
    k = int(np.rint(kf))
    return max(0, min(K - 1, k))


def draw_panel(ax, data2d, title, extent, walls2d=None, tx_xy=None, sta_xy=None, vmin=None, vmax=None, cmap="viridis"):
    im = ax.imshow(
        data2d,
        origin="lower",
        extent=extent,
        interpolation="nearest",
        aspect="equal",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
    )

    if walls2d is not None:
        # walls_2d.npy uses first index along x; transpose to plot in x-horizontal/y-vertical convention
        ax.contour(
            walls2d.T.astype(np.float32),
            levels=[0.5],
            colors="k",
            linewidths=0.6,
            origin="lower",
            extent=extent,
        )

    if sta_xy is not None and len(sta_xy) > 0:
        sta_xy = np.asarray(sta_xy, dtype=np.float32)
        ax.scatter(sta_xy[:, 0], sta_xy[:, 1], s=10, marker="x", c="white", linewidths=0.7)

    if tx_xy is not None:
        ax.scatter([tx_xy[0]], [tx_xy[1]], s=40, marker="*", c="red", edgecolors="white", linewidths=0.8)

    ax.set_title(title, fontsize=10)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    return im


def finite_minmax(*arrays, pad_frac=0.02):
    vals = []
    for arr in arrays:
        arr = np.asarray(arr, dtype=np.float32)
        good = np.isfinite(arr)
        if np.any(good):
            vals.append(arr[good])
    if not vals:
        return None, None
    vals = np.concatenate(vals)
    vmin = float(np.min(vals))
    vmax = float(np.max(vals))
    if vmax <= vmin:
        return vmin, vmax
    pad = pad_frac * (vmax - vmin)
    return vmin - pad, vmax + pad


def main():
    enable_tf_memory_growth()

    ap = argparse.ArgumentParser(description="Plot RT / UNet / COST231 / Friis maps for one scene")
    ap.add_argument("--scene-dir", type=str, required=True, help="Directory containing scene.xml, placements.json, walls_2d.npy, ...")
    ap.add_argument("--model-dir", type=str, required=True, help="Directory containing meta.json, model.pt, norm_stats.npz")
    ap.add_argument("--out-png", type=str, default="compare_maps.png")
    ap.add_argument("--out-npz", type=str, default="compare_maps.npz")
    ap.add_argument("--device", type=str, default=("cuda" if torch.cuda.is_available() else "cpu"))

    ap.add_argument("--frequency-hz", type=float, default=None, help="Override frequency in Hz")
    ap.add_argument("--scale-m", type=float, default=None, help="Override grid spacing in meters")
    ap.add_argument("--H", type=int, default=None)
    ap.add_argument("--W", type=int, default=None)
    ap.add_argument("--K", type=int, default=None)

    ap.add_argument("--z-step-cells", type=float, default=1.0)
    ap.add_argument("--z-margin-cells", type=float, default=0.5)
    ap.add_argument("--origin-xy-mode", type=str, default="bbox_min", choices=["bbox_min", "zero"])

    ap.add_argument("--fft-size", type=int, default=None)
    ap.add_argument("--subcarrier-spacing-hz", type=float, default=None)
    ap.add_argument("--rx-batch", type=int, default=256)
    ap.add_argument("--no-path-wb-db", type=float, default=199.5)

    ap.add_argument("--smooth-kind", type=str, default="median", choices=["none", "gaussian", "median"])
    ap.add_argument("--smooth-median-size", type=int, default=3)
    ap.add_argument("--smooth-gauss-sigma", type=float, default=1.0)

    ap.add_argument("--slice-mode", type=str, default="tx", choices=["tx", "mid"])
    ap.add_argument("--slice-k", type=int, default=None)

    ap.add_argument("--delta-idx", type=int, default=0)
    ap.add_argument("--excess-idx", type=int, default=1)
    ap.add_argument("--tau-idx", type=int, default=2)

    ap.add_argument("--tx-xyz", type=str, default=None, help="Optional override: x,y,z")
    args = ap.parse_args()

    scene_dir = Path(args.scene_dir).resolve()
    model_dir = Path(args.model_dir).resolve()

    if not (scene_dir / "scene.xml").exists():
        raise FileNotFoundError(f"Could not find {scene_dir / 'scene.xml'}")
    if not (model_dir / "meta.json").exists():
        raise FileNotFoundError(f"Could not find {model_dir / 'meta.json'}")
    if not (model_dir / "model.pt").exists():
        raise FileNotFoundError(f"Could not find {model_dir / 'model.pt'}")
    if not (model_dir / "norm_stats.npz").exists():
        raise FileNotFoundError(f"Could not find {model_dir / 'norm_stats.npz'}")

    placements = load_json_if_exists(scene_dir / "placements.json")
    scene_info = load_json_if_exists(scene_dir / "scene_info.json")
    model_meta = json.loads((model_dir / "meta.json").read_text())

    walls_2d = None
    if (scene_dir / "walls_2d.npy").exists():
        walls_2d = np.load(scene_dir / "walls_2d.npy")

    freq_hz = pick_frequency_hz(args, model_meta, placements, scene_info)

    # grid sizes
    H = int(args.H if args.H is not None else (walls_2d.shape[0] if walls_2d is not None else model_meta.get("H", 64)))
    W = int(args.W if args.W is not None else (walls_2d.shape[1] if walls_2d is not None else model_meta.get("W", 64)))
    K = int(args.K if args.K is not None else model_meta.get("K", 4))
    scale_m = float(args.scale_m if args.scale_m is not None else model_meta.get("scale_m", (placements.get("cell_size_m", 0.625) if placements else 0.625)))
    z_margin_m = float(args.z_margin_cells) * scale_m

    fft_size = int(args.fft_size if args.fft_size is not None else model_meta.get("fft_size", 3072))
    subcarrier_spacing_hz = float(
        args.subcarrier_spacing_hz
        if args.subcarrier_spacing_hz is not None
        else model_meta.get("subcarrier_spacing_hz", 78_125.0)
    )

    print(f"scene_dir              : {scene_dir}")
    print(f"model_dir              : {model_dir}")
    print(f"frequency_hz           : {freq_hz}")
    print(f"grid (K,H,W)           : {(K, H, W)}")
    print(f"scale_m                : {scale_m}")
    print(f"fft_size               : {fft_size}")
    print(f"subcarrier_spacing_hz  : {subcarrier_spacing_hz}")
    print(f"device                 : {args.device}")

    si_scene = load_scene(str(scene_dir / "scene.xml"))
    si_scene.frequency = float(freq_hz)
    bbox = si_scene.mi_scene.bbox()

    tx_xyz = load_tx_xyz(scene_dir, args, placements, bbox)
    sta_xyz = np.asarray(placements["sta_xyz"], dtype=np.float32) if placements and "sta_xyz" in placements else None

    base_scene = MlinkScene.from_sionna(si_scene)
    base_scene.sionna_scene = si_scene  # preserve original radio materials

    rx_grid, rx_coords, origin, z_step_m = build_rx_grid_and_coords(
        bbox=bbox,
        H=H,
        W=W,
        K=K,
        scale_m=scale_m,
        z_step_cells=args.z_step_cells,
        z_margin_m=z_margin_m,
        origin_xy_mode=args.origin_xy_mode,
    )

    scene_for_grid = build_scene_for_grid(base_scene, tx_xyz, rx_grid, rx_coords)

    # RT
    print("\n=== Computing full RT maps ===")
    rt_wb, rt_tau, rt_ex = compute_rt_maps(
        base_scene=base_scene,
        scene_for_grid=scene_for_grid,
        tx_xyz=tx_xyz,
        rx_coords=rx_coords,
        K=K,
        H=H,
        W=W,
        freq_hz=freq_hz,
        fft_size=fft_size,
        subcarrier_spacing_hz=subcarrier_spacing_hz,
        rx_batch=args.rx_batch,
        no_path_wb_db=args.no_path_wb_db,
        smooth_kind=args.smooth_kind,
        smooth_median_size=args.smooth_median_size,
        smooth_gauss_sigma=args.smooth_gauss_sigma,
    )

    # UNet
    print("\n=== Computing full UNet maps ===")
    unet_wb, unet_tau, unet_ex, _unet_meta = compute_unet_maps(
        base_scene=base_scene,
        scene_for_grid=scene_for_grid,
        model_dir=model_dir,
        freq_hz=freq_hz,
        tx_xyz=tx_xyz,
        rx_coords=rx_coords,
        K=K,
        H=H,
        W=W,
        device=args.device,
        no_path_wb_db=args.no_path_wb_db,
        delta_idx=args.delta_idx,
        excess_idx=args.excess_idx,
        tau_idx=args.tau_idx,
    )

    # COST231 and Friis
    print("\n=== Computing COST231 / Friis maps ===")
    cost_wb = compute_cost231_map(scene_for_grid, freq_hz=freq_hz, no_path_wb_db=args.no_path_wb_db)
    friis_wb = compute_friis_map(tx_xyz, rx_coords, K=K, H=H, W=W, freq_hz=freq_hz)

    k = choose_slice_k(args, tx_xyz=tx_xyz, origin=origin, z_step_m=z_step_m, K=K)
    z_slice = float(origin[2] + k * z_step_m)

    print(f"\nUsing slice k={k} at z={z_slice:.3f} m")

    x0 = float(origin[0])
    y0 = float(origin[1])
    x1 = x0 + W * scale_m
    y1 = y0 + H * scale_m
    extent = (x0, x1, y0, y1)

    # consistent color limits
    path_vmin, path_vmax = finite_minmax(rt_wb[k], unet_wb[k], cost_wb[k], friis_wb[k])
    tau_vmin, tau_vmax = finite_minmax(rt_tau[k], unet_tau[k])
    ex_vmin, ex_vmax = finite_minmax(rt_ex[k], unet_ex[k])

    fig, axes = plt.subplots(2, 4, figsize=(20, 10), constrained_layout=True)
    axes = axes.ravel()

    ims = []
    ims.append(draw_panel(
        axes[0], rt_wb[k], "RT path loss [dB]",
        extent, walls2d=walls_2d, tx_xy=tx_xyz[:2], sta_xy=sta_xyz[:, :2] if sta_xyz is not None else None,
        vmin=path_vmin, vmax=path_vmax, cmap="viridis"
    ))
    ims.append(draw_panel(
        axes[1], rt_tau[k], "RT RMS delay spread label [ns]",
        extent, walls2d=walls_2d, tx_xy=tx_xyz[:2], sta_xy=sta_xyz[:, :2] if sta_xyz is not None else None,
        vmin=tau_vmin, vmax=tau_vmax, cmap="magma"
    ))
    ims.append(draw_panel(
        axes[2], rt_ex[k], "RT excess delay label [ns]",
        extent, walls2d=walls_2d, tx_xy=tx_xyz[:2], sta_xy=sta_xyz[:, :2] if sta_xyz is not None else None,
        vmin=ex_vmin, vmax=ex_vmax, cmap="plasma"
    ))
    ims.append(draw_panel(
        axes[3], unet_wb[k], "UNet path loss [dB]",
        extent, walls2d=walls_2d, tx_xy=tx_xyz[:2], sta_xy=sta_xyz[:, :2] if sta_xyz is not None else None,
        vmin=path_vmin, vmax=path_vmax, cmap="viridis"
    ))
    ims.append(draw_panel(
        axes[4], unet_tau[k], "UNet RMS delay spread [ns]",
        extent, walls2d=walls_2d, tx_xy=tx_xyz[:2], sta_xy=sta_xyz[:, :2] if sta_xyz is not None else None,
        vmin=tau_vmin, vmax=tau_vmax, cmap="magma"
    ))
    ims.append(draw_panel(
        axes[5], unet_ex[k], "UNet excess delay [ns]",
        extent, walls2d=walls_2d, tx_xy=tx_xyz[:2], sta_xy=sta_xyz[:, :2] if sta_xyz is not None else None,
        vmin=ex_vmin, vmax=ex_vmax, cmap="plasma"
    ))
    ims.append(draw_panel(
        axes[6], cost_wb[k], "COST231 path loss [dB]",
        extent, walls2d=walls_2d, tx_xy=tx_xyz[:2], sta_xy=sta_xyz[:, :2] if sta_xyz is not None else None,
        vmin=path_vmin, vmax=path_vmax, cmap="viridis"
    ))
    ims.append(draw_panel(
        axes[7], friis_wb[k], "Friis path loss [dB]",
        extent, walls2d=walls_2d, tx_xy=tx_xyz[:2], sta_xy=sta_xyz[:, :2] if sta_xyz is not None else None,
        vmin=path_vmin, vmax=path_vmax, cmap="viridis"
    ))

    for ax, im in zip(axes, ims):
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(
        f"{scene_dir.name} | k={k}, z={z_slice:.3f} m | tx=({tx_xyz[0]:.2f}, {tx_xyz[1]:.2f}, {tx_xyz[2]:.2f}) m",
        fontsize=12,
    )

    out_png = Path(args.out_png)
    if not out_png.is_absolute():
        out_png = scene_dir / out_png
    fig.savefig(out_png, dpi=200)
    plt.close(fig)

    out_npz = Path(args.out_npz)
    if not out_npz.is_absolute():
        out_npz = scene_dir / out_npz
    np.savez_compressed(
        out_npz,
        rt_wb=rt_wb,
        rt_tau=rt_tau,
        rt_ex=rt_ex,
        unet_wb=unet_wb,
        unet_tau=unet_tau,
        unet_ex=unet_ex,
        cost_wb=cost_wb,
        friis_wb=friis_wb,
        tx_xyz=tx_xyz,
        sta_xyz=sta_xyz,
        walls_2d=walls_2d,
        slice_k=np.asarray([k], dtype=np.int32),
        z_slice_m=np.asarray([z_slice], dtype=np.float32),
        frequency_hz=np.asarray([freq_hz], dtype=np.float64),
        scale_m=np.asarray([scale_m], dtype=np.float32),
    )

    print(f"\nSaved figure to: {out_png}")
    print(f"Saved maps   to: {out_npz}")


if __name__ == "__main__":
    main()