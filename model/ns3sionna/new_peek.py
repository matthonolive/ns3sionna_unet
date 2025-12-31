"""Peek at a memmap run (x.dat/y.dat) and optionally a TorchScript model.

Works with both:
  - 2.5D stacked tensors: x (N, K*c_in, H, W), y (N, K*y_ch, H, W)
  - non-stacked tensors:  x (N, c_in,   H, W), y (N, y_ch,   H, W)

It will also export the *binary_walls* input channel as an image.

Example:
  python new_peek.py --run runs/merged --sample 12 --k 3
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt


def _read_meta(run: Path) -> dict:
    meta_path = run / "meta.json"
    if not meta_path.exists():
        return {}
    return json.loads(meta_path.read_text())


def _infer_n_from_file(
    path: Path,
    dtype: np.dtype,
    H: int,
    W: int,
    candidates_C: list[int],
    preferred_N: Optional[int] = None,
    prefer_C: Optional[int] = None,
) -> Tuple[int, int]:
    """Infer (N, C) for a (N,C,H,W) memmap file given H,W and candidate C values."""
    dt = np.dtype(dtype)
    n_elem = path.stat().st_size // dt.itemsize
    best = None  # (score, N, C)

    for C in candidates_C:
        if C <= 0:
            continue
        denom = int(C) * int(H) * int(W)
        if denom <= 0:
            continue
        if n_elem % denom != 0:
            continue
        N = n_elem // denom
        # scoring: match preferred_N strongly; then prefer_C if set
        score = 0
        if preferred_N is not None:
            score += abs(int(N) - int(preferred_N)) * 1000
        if prefer_C is not None:
            score += abs(int(C) - int(prefer_C))
        # tie-breaker: prefer larger N (more samples) only if score equal
        cand = (score, -int(N), int(C), int(N))
        if best is None or cand < best:
            best = cand

    if best is None:
        raise ValueError(
            f"Could not infer shape for {path}. "
            f"Tried C in {candidates_C} with H={H}, W={W}, dtype={dt}."
        )
    _, _, C, N = best
    return int(N), int(C)


def _memmap_4d(path: Path, dtype: np.dtype, shape: Tuple[int, int, int, int]) -> np.memmap:
    return np.memmap(path, dtype=dtype, mode="r", shape=shape)


def _pick_first_written_sample(y_mm: np.memmap, max_scan: int = 500) -> int:
    """Heuristic: return first index whose y isn't all-zeros/all-sentinel."""
    n = int(min(max_scan, y_mm.shape[0]))
    # Look at channel 0 only (usually wb for slice 0)
    for i in range(n):
        a = np.asarray(y_mm[i, 0], dtype=np.float32)
        if np.any(np.isfinite(a)) and (np.nanstd(a) > 1e-6):
            return i
    return 0


def _save_img(arr: np.ndarray, out: Path, title: str, vmin=None, vmax=None, cmap="viridis"):
    plt.figure(figsize=(5, 4), dpi=160)
    plt.imshow(arr, origin="upper", vmin=vmin, vmax=vmax, cmap=cmap)
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.title(title)
    plt.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out)
    plt.close()


def _maybe_load_stats(run: Path):
    stats_path = run / "norm_stats.npz"
    if not stats_path.exists():
        return None
    d = np.load(stats_path)
    # stored as (C,1,1)
    return {
        "x_mean": d["x_mean"].astype(np.float32),
        "x_std": d["x_std"].astype(np.float32),
        "y_mean": d["y_mean"].astype(np.float32),
        "y_std": d["y_std"].astype(np.float32),
        "keep_idx": d["keep_idx"].astype(np.int64),
    }


def _maybe_load_torchscript(model_path: Path, device: str = "cpu"):
    if not model_path.exists():
        return None
    import torch

    m = torch.jit.load(str(model_path), map_location=device)
    m.eval()
    return m


def main(
    run_dir: str,
    sample: Optional[int] = None,
    k: int = 0,
    walls_idx: Optional[int] = None,
    out_dir: Optional[str] = None,
    model_path: Optional[str] = None,
    device: str = "cpu",
    assume_stacked: str = "auto",  # auto|true|false
):
    run = Path(run_dir)
    if not run.exists():
        raise FileNotFoundError(run)

    x_path = run / "x.dat"
    y_path = run / "y.dat"
    if not x_path.exists() or not y_path.exists():
        raise FileNotFoundError(f"Missing x.dat/y.dat in {run}")

    meta = _read_meta(run)
    H = int(meta.get("H", 64))
    W = int(meta.get("W", 64))
    K = int(meta.get("K", 1))
    c_in = meta.get("c_in", None)
    y_ch = int(meta.get("y_ch", 3))
    total_meta = meta.get("total_samples", None)
    total_meta = int(total_meta) if total_meta is not None else None

    x_dtype = np.dtype(meta.get("x_dtype", "float32"))
    y_dtype = np.dtype(meta.get("y_dtype", "float16"))

    # Decide stacked vs not
    if assume_stacked not in {"auto", "true", "false"}:
        raise ValueError("assume_stacked must be auto|true|false")

    # Candidate channel counts
    if c_in is None:
        # if meta missing c_in, try common small values
        c_in_cands = [4, 3, 5, 6, 8, 1]
    else:
        c_in_cands = [int(c_in)]

    yC_cands = [y_ch]
    xC_cands = c_in_cands.copy()

    if K > 1:
        yC_cands += [y_ch * K]
        xC_cands += [c * K for c in c_in_cands]

    prefer_yC = None
    prefer_xC = None
    if assume_stacked == "true" and K > 1:
        prefer_yC = y_ch * K
        if c_in is not None:
            prefer_xC = int(c_in) * K
    elif assume_stacked == "false":
        prefer_yC = y_ch
        if c_in is not None:
            prefer_xC = int(c_in)

    N_y, C_y = _infer_n_from_file(y_path, y_dtype, H, W, yC_cands, preferred_N=total_meta, prefer_C=prefer_yC)
    N_x, C_x = _infer_n_from_file(x_path, x_dtype, H, W, xC_cands, preferred_N=N_y, prefer_C=prefer_xC)

    N = min(N_x, N_y)
    if N != N_x or N != N_y:
        print(f"[warn] N mismatch: from y={N_y}, from x={N_x}. Using N={N}.")

    y_mm = _memmap_4d(y_path, y_dtype, (N, C_y, H, W))
    x_mm = _memmap_4d(x_path, x_dtype, (N, C_x, H, W))

    is_y_stacked = (C_y == y_ch * K) if K > 1 else (C_y != y_ch)
    is_x_stacked = (c_in is not None and C_x == int(c_in) * K) if K > 1 else (C_x != (int(c_in) if c_in is not None else C_x))

    if sample is None:
        sample = _pick_first_written_sample(y_mm)

    sample = int(np.clip(sample, 0, N - 1))
    k = int(np.clip(k, 0, max(K - 1, 0)))

    out = Path(out_dir) if out_dir is not None else (run / "peek")
    out.mkdir(parents=True, exist_ok=True)

    # Figure per-slice channel counts
    if is_y_stacked and K > 1:
        y_per_slice = y_ch
        y_off = k * y_per_slice
    else:
        y_per_slice = y_ch
        y_off = 0

    if c_in is None:
        # infer c_in from C_x and K if possible
        if K > 1 and (C_x % K == 0):
            c_in_eff = C_x // K
        else:
            c_in_eff = C_x
    else:
        c_in_eff = int(c_in)

    if is_x_stacked and K > 1:
        x_per_slice = c_in_eff
        x_off = k * x_per_slice
    else:
        x_per_slice = c_in_eff
        x_off = 0

    # Determine which per-slice channel is binary_walls
    feat_names = meta.get("requested_features") or meta.get("x_channels")
    if isinstance(feat_names, list) and "binary_walls" in feat_names:
        walls_idx_eff = int(feat_names.index("binary_walls"))
    else:
        walls_idx_eff = int(walls_idx) if walls_idx is not None else 0
        if walls_idx is None:
            print(
                "[warn] meta does not include requested_features/x_channels; "
                "assuming binary_walls is per-slice channel 0. "
                "(override with --walls-idx)"
            )

    # -------------------------
    # Export LABEL maps
    # -------------------------
    y_samp = np.asarray(y_mm[sample], dtype=np.float32)  # (C_y,H,W)
    wb_lab = y_samp[y_off + 0]
    ex_lab = y_samp[y_off + 1] if (y_ch >= 2) else None
    tau_lab = y_samp[y_off + 2] if (y_ch >= 3) else None

    _save_img(wb_lab, out / f"s{sample:05d}_k{k}_wb_label.png", f"WB label (s={sample}, k={k})")
    if ex_lab is not None:
        _save_img(ex_lab, out / f"s{sample:05d}_k{k}_ex_label.png", f"Excess delay label (ns) (s={sample}, k={k})")
    if tau_lab is not None:
        _save_img(tau_lab, out / f"s{sample:05d}_k{k}_tauRMS_label.png", f"Tau RMS label (ns) (s={sample}, k={k})")

    # -------------------------
    # Export INPUT binary_walls
    # -------------------------
    x_samp = np.asarray(x_mm[sample], dtype=np.float32)  # (C_x,H,W)
    walls = x_samp[x_off + walls_idx_eff]
    walls_bin = (walls > 0.5).astype(np.float32)

    _save_img(walls, out / f"s{sample:05d}_k{k}_walls_raw.png", f"binary_walls raw (s={sample}, k={k})", vmin=0.0, vmax=1.0, cmap="gray")
    _save_img(walls_bin, out / f"s{sample:05d}_k{k}_walls_bin.png", f"binary_walls bin (s={sample}, k={k})", vmin=0.0, vmax=1.0, cmap="gray")

    # -------------------------
    # Optional: model prediction
    # -------------------------
    stats = _maybe_load_stats(run)
    if model_path is None:
        # default: prefer run/model.pt
        cand = run / "model.pt"
        model_path = str(cand) if cand.exists() else None

    model = _maybe_load_torchscript(Path(model_path), device=device) if model_path is not None else None

    if model is not None and stats is not None:
        import torch

        x_t = torch.from_numpy(x_samp.astype(np.float32)).unsqueeze(0)  # (1,C,H,W)
        x_mean = torch.from_numpy(stats["x_mean"]).to(x_t.dtype)
        x_std = torch.from_numpy(stats["x_std"]).to(x_t.dtype)
        y_mean = torch.from_numpy(stats["y_mean"]).to(x_t.dtype)
        y_std = torch.from_numpy(stats["y_std"]).to(x_t.dtype)
        keep_idx = stats.get("keep_idx", None)

        # Normalize exactly like training
        if keep_idx is not None:
            keep_idx = torch.from_numpy(keep_idx).long().to(x_t.device)
            x_t = x_t[:, keep_idx]
            
        x_n = (x_t - x_mean) / torch.clamp(x_std, min=1e-6)
        x_n = x_n.to(device)

        with torch.no_grad():
            pred_n = model(x_n)  # (1,C_y,H,W)
        pred_p = (pred_n.cpu() * y_std + y_mean).squeeze(0).numpy()  # (C_y,H,W)

        wb_pr = pred_p[y_off + 0]
        _save_img(wb_pr, out / f"s{sample:05d}_k{k}_wb_pred.png", f"WB pred (s={sample}, k={k})")
        _save_img(wb_pr - wb_lab, out / f"s{sample:05d}_k{k}_wb_err.png", f"WB pred - label (dB) (s={sample}, k={k})")

        if ex_lab is not None:
            ex_pr = pred_p[y_off + 1]
            _save_img(ex_pr, out / f"s{sample:05d}_k{k}_ex_pred.png", f"Excess delay pred (ns) (s={sample}, k={k})")
            _save_img(ex_pr - ex_lab, out / f"s{sample:05d}_k{k}_ex_err.png", f"Excess delay pred - label (ns) (s={sample}, k={k})")

        if tau_lab is not None:
            tau_pr = pred_p[y_off + 2]
            _save_img(tau_pr, out / f"s{sample:05d}_k{k}_tauRMS_pred.png", f"Tau RMS pred (ns) (s={sample}, k={k})")
            _save_img(tau_pr - tau_lab, out / f"s{sample:05d}_k{k}_tauRMS_err.png", f"Tau RMS pred - label (ns) (s={sample}, k={k})")

        # Quick masked WB metrics (optional, for the chosen slice only)
        no_path = float(meta.get("no_path_wb_db", meta.get("no_path_wb", 199.5)))
        m = (wb_lab < no_path).astype(np.float32)
        if m.sum() > 0:
            mae = (np.abs(wb_pr - wb_lab) * m).sum() / m.sum()
            g_pr = np.power(10.0, -wb_pr / 10.0)
            g_lb = np.power(10.0, -wb_lab / 10.0)
            nmse = ((g_pr - g_lb) ** 2 * m).sum() / np.maximum(((g_lb ** 2) * m).sum(), 1e-12)
            nmse_db = 10.0 * np.log10(np.maximum(nmse, 1e-12))
            print(f"WB slice metrics (s={sample}, k={k}): MAE={mae:.3f} dB, NMSE={nmse:.4f} ({nmse_db:.2f} dB)")

    print("\nWrote outputs to:", out)
    print(f"Inferred shapes: N={N}, x=(N,{C_x},{H},{W}), y=(N,{C_y},{H},{W})")
    print(f"Assumptions: K={K}, y_ch={y_ch}, c_in={c_in_eff}, y_stacked={is_y_stacked}, x_stacked={is_x_stacked}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="Run directory containing x.dat/y.dat/meta.json")
    ap.add_argument("--sample", type=int, default=None, help="Sample index (default: auto-pick)")
    ap.add_argument("--k", type=int, default=0, help="Slice index k (default: 0)")
    ap.add_argument("--walls-idx", type=int, default=None, help="Per-slice channel index of binary_walls (default: 0)")
    ap.add_argument("--out", default=None, help="Output directory (default: <run>/peek)")
    ap.add_argument("--model", default=None, help="TorchScript model path (default: <run>/model.pt if present)")
    ap.add_argument("--device", default="cpu", help="Torch device for inference (cpu, cuda:0, ...) ")
    ap.add_argument("--assume-stacked", default="auto", choices=["auto", "true", "false"], help="Force stacked/unstacked interpretation")
    args = ap.parse_args()

    main(
        run_dir=args.run,
        sample=args.sample,
        k=args.k,
        walls_idx=args.walls_idx,
        out_dir=args.out,
        model_path=args.model,
        device=args.device,
        assume_stacked=args.assume_stacked,
    )
