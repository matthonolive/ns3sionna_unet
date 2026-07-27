#!/usr/bin/env python3
"""
validate_cfr.py
==================================================================
CFR-level validation: Sionna RT vs UNet vs COST-231 vs log-distance,
per link, WITHOUT ns-3 (runs the propagators directly).

Run from model/ns3sionna/ in the SERVER venv (it imports the propagator
classes from ns3unet_spectrum.py, which needs sionna/torch/tf):

  python validate_cfr.py \
      --scene_dir worldbuilding/valsuite/v02_single_wall \
      --unet_run /abs/path/runs/residual_cost --device cuda:0

What it measures and why
------------------------------------------------------------------
The synthetic CFR is a STOCHASTIC realization (exponential PDP + Rician
K from a LOS ray test), so per-subcarrier phase can never match RT.
The meaningful comparison is statistical, per link:

  wb_db          wideband loss (also cross-checks the ns-3 pipeline)
  tau_rms        RT ground truth from the CIR, the UNet tau head, and a
                 CFR-derived estimate computed with the SAME estimator
                 for every model (IFFT -> PDP -> power-weighted std)
  Bc50           50% coherence bandwidth from the frequency
                 autocorrelation (frequency-selectivity scale)
  ripple         std / peak-to-peak of 20log10|H| across subcarriers
  K_est          moment-based Rician K estimate from |H|^2

Synthetic models are averaged over --n_realizations seeds; RT is
deterministic. Outputs: <out_dir>/cfr_summary.csv, per-link overlay
plots (|H(f)| + PDP), and a printed table.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np

# ns3unet_spectrum imports sionna.rt first (required ordering) and pulls
# the full server environment; run this in the server venv.
from ns3unet_spectrum import (UNetTdlPropagator, Cost231Propagator,
                              LogDistancePropagator)
from sionna.rt import load_scene, Transmitter, Receiver, PlanarArray, PathSolver

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

C0 = 299792458.0


# ------------------------------------------------------------------
# metrics (identical estimator for every model)
# ------------------------------------------------------------------

def norm_cfr(H):
    p = float(np.mean(np.abs(H) ** 2))
    return H / math.sqrt(max(p, 1e-30))


def pdp_from_cfr(H, scs_hz):
    """H sampled on the ASCENDING centered frequency grid ->
    (t_ns, p) power-delay profile over the first half of the delay axis."""
    N = len(H)
    taps = np.fft.ifft(np.fft.ifftshift(H))
    p = np.abs(taps) ** 2
    Ts_ns = 1e9 / (N * scs_hz)
    half = N // 2                      # discard the wrap-around half
    t_ns = np.arange(half) * Ts_ns
    return t_ns, p[:half]


def tau_rms_from_cfr(H, scs_hz, floor_db=30.0):
    t_ns, p = pdp_from_cfr(H, scs_hz)
    pk = float(p.max())
    if pk <= 0:
        return 0.0
    m = p >= pk * 10.0 ** (-floor_db / 10.0)
    w = p[m] / p[m].sum()
    t = t_ns[m]
    mu = float(np.sum(w * t))
    mu2 = float(np.sum(w * t * t))
    return math.sqrt(max(mu2 - mu * mu, 0.0))


def coherence_bw_khz(H, scs_hz, level=0.5):
    N = len(H)
    for k in range(1, N // 2):
        r = np.vdot(H[:-k], H[k:]) / max(np.linalg.norm(H[:-k]) * np.linalg.norm(H[k:]), 1e-30)
        if abs(r) < level:
            return k * scs_hz / 1e3
    return (N // 2) * scs_hz / 1e3


def ripple_db(H):
    mag = 20.0 * np.log10(np.abs(H) + 1e-12)
    return float(mag.std()), float(mag.max() - mag.min())


def k_factor_est_db(H):
    """Moment-based Rician K estimate from per-subcarrier power."""
    P = np.abs(H) ** 2
    mu = float(P.mean())
    if mu <= 0:
        return float("nan")
    gamma = float(P.var()) / (mu * mu)
    if gamma >= 1.0:
        return -np.inf                 # Rayleigh-like or worse
    s = math.sqrt(1.0 - gamma)
    if s >= 1.0:
        return np.inf
    K = s / (1.0 - s)
    return 10.0 * math.log10(max(K, 1e-12))


def tau_rms_from_cir(a, tau):
    a = np.squeeze(np.asarray(a)).reshape(-1)
    t = np.squeeze(np.asarray(tau)).reshape(-1)
    m = np.isfinite(t) & (t >= 0) & np.isfinite(a)
    if not np.any(m):
        return 0.0
    p = np.abs(a[m]) ** 2
    if p.sum() <= 0:
        return 0.0
    w = p / p.sum()
    tt = t[m] * 1e9
    mu = float(np.sum(w * tt))
    mu2 = float(np.sum(w * tt * tt))
    return math.sqrt(max(mu2 - mu * mu, 0.0))


def metrics_of(H, scs_hz):
    Hn = norm_cfr(H)
    rs, rpp = ripple_db(Hn)
    return {
        "tau_cfr_ns": tau_rms_from_cfr(Hn, scs_hz),
        "Bc50_kHz": coherence_bw_khz(Hn, scs_hz),
        "ripple_std_db": rs,
        "ripple_pp_db": rpp,
        "K_est_db": k_factor_est_db(Hn),
    }


def avg_metrics(ms):
    keys = ms[0].keys()
    out = {}
    for k in keys:
        v = np.asarray([m[k] for m in ms], dtype=np.float64)
        v = v[np.isfinite(v)]
        out[k] = float(np.mean(v)) if v.size else float("nan")
    return out


# ------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene_dir", type=str, required=True)
    ap.add_argument("--unet_run", type=str, required=True)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--fft_size", type=int, default=3072)
    ap.add_argument("--subcarrier_spacing_hz", type=float, default=78125.0)
    ap.add_argument("--fc_hz", type=float, default=None,
                    help="default: placements.json frequency_hz")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--n_realizations", type=int, default=8,
                    help="synthetic-CFR seeds averaged per link")
    ap.add_argument("--rt_fast", action="store_true")
    ap.add_argument("--unet_cov_thresh", type=float, default=0.5)
    ap.add_argument("--cost231_tau_rms_ns", type=float, default=5.0)
    ap.add_argument("--logdist_exponent", type=float, default=3.0)
    ap.add_argument("--k_los_db", type=float, default=6.0)
    ap.add_argument("--k_nlos_db", type=float, default=2.0)
    ap.add_argument("--logdist_ref_m", type=float, default=1.0)
    ap.add_argument("--max_links", type=int, default=64)
    ap.add_argument("--out_dir", type=str, default=None,
                    help="default: <scene_dir>/cfr_validation")
    args = ap.parse_args()

    scene_dir = Path(args.scene_dir)
    out_dir = Path(args.out_dir) if args.out_dir else scene_dir / "cfr_validation"
    out_dir.mkdir(parents=True, exist_ok=True)

    placements = json.loads((scene_dir / "placements.json").read_text())
    fc = float(args.fc_hz if args.fc_hz is not None else placements["frequency_hz"])
    N = int(args.fft_size)
    scs = float(args.subcarrier_spacing_hz)
    tx_xyz = np.asarray(placements["tx_xyz"], dtype=np.float32)
    sta_xyz = np.asarray(placements["sta_xyz"], dtype=np.float32)[: args.max_links]

    # ascending centered grid -> propagators auto-set fft_shift=True, so
    # every H below (RT and synthetic) lives on the SAME frequency axis
    freqs = fc + (np.arange(N) - N // 2) * scs

    print(f"[cfr] scene={scene_dir.name} fc={fc/1e9:.3f} GHz N={N} "
          f"scs={scs/1e3:.1f} kHz links={len(sta_xyz)}")

    # ---------------- RT ground truth ----------------
    scene = load_scene(str(scene_dir / "scene.xml"))
    scene.frequency = fc
    bbox = scene.mi_scene.bbox()
    scene.tx_array = PlanarArray(num_rows=1, num_cols=1, vertical_spacing=0.5,
                                 horizontal_spacing=0.5, pattern="iso",
                                 polarization="V")
    scene.rx_array = scene.tx_array

    scene.add(Transmitter(name="tx", position=tx_xyz.tolist(),
                          orientation=[0, -180, 0]))
    for j, p in enumerate(sta_xyz):
        scene.add(Receiver(name=f"rx{j}", position=p.tolist(),
                           orientation=[0, -180, 0]))

    rt = dict(max_depth=3 if args.rt_fast else 10, samples_per_src=10 ** 6,
              los=True, specular_reflection=True,
              diffuse_reflection=not args.rt_fast, refraction=True,
              synthetic_array=False, diffraction=not args.rt_fast,
              edge_diffraction=not args.rt_fast,
              diffraction_lit_region=not args.rt_fast)
    print(f"[cfr] ray tracing ({'fast' if args.rt_fast else 'realistic'}) ...")
    paths = PathSolver()(scene=scene, **rt)
    a, tau = paths.cir(sampling_frequency=1e9, normalize_delays=False,
                       out_type="numpy")
    h_rt = paths.cfr(frequencies=freqs, sampling_frequency=1.0,
                     num_time_steps=1, normalize_delays=True,
                     normalize=False, out_type="numpy")

    # ---------------- surrogates ----------------
    kk = dict(k_los_db=args.k_los_db, k_nlos_db=args.k_nlos_db)
    unet = UNetTdlPropagator(run_dir=args.unet_run, device=args.device,
                             cov_thresh=args.unet_cov_thresh,
                             scale_m=0.625, z_step_cells=1.0,
                             z_margin_m=0.625 * 0.5, origin_xy_mode="bbox_min",
                             **kk)
    cost = Cost231Propagator(scale_m=0.625, z_step_cells=1.0,
                             z_margin_m=0.625 * 0.5, origin_xy_mode="bbox_min",
                             default_tau_rms_ns=args.cost231_tau_rms_ns, **kk)
    logd = LogDistancePropagator(exponent=args.logdist_exponent,
                                 ref_dist_m=args.logdist_ref_m,
                                 default_tau_rms_ns=args.cost231_tau_rms_ns,
                                 **kk)
    for p in (unet, cost, logd):
        p.attach_scene(scene, bbox, fc, N, scs, freqs)
        p.predict_for_tx(tx_xyz)

    models = {"unet": unet, "cost231": cost, "logdist": logd}

    # ---------------- per-link comparison ----------------
    rows = []
    print(f"\n{'link':>5} {'model':>8} {'wb dB':>8} {'wbErr':>7} {'tauCIR':>7} "
          f"{'tauHead':>8} {'tauCFR':>7} {'Bc50kHz':>8} {'rip dB':>7} {'K dB':>6}")
    for j in range(len(sta_xyz)):
        rx = sta_xyz[j]
        d_m = float(np.linalg.norm(tx_xyz - rx))

        h_j = np.squeeze(h_rt[j])
        pwr = float(np.mean(np.abs(h_j) ** 2))
        tau_cir = tau_rms_from_cir(a[j], tau[j])
        if not np.isfinite(pwr) or pwr <= 0:
            wb_rt, H_rt_n, rt_m = 199.5, None, None
        else:
            wb_rt = float(-10.0 * np.log10(pwr))
            H_rt_n = norm_cfr(h_j.astype(np.complex64))
            rt_m = metrics_of(H_rt_n, scs)

        def emit(model, wb, tau_head, m, no_path=False):
            r = {"link": j, "model": model, "d_m": round(d_m, 3),
                 "no_path": int(no_path),
                 "wb_db": round(wb, 3),
                 "wb_err_db": round(wb - wb_rt, 3) if model != "rt" else 0.0,
                 "tau_rt_cir_ns": round(tau_cir, 3),
                 "tau_head_ns": (round(tau_head, 3) if tau_head is not None else "")}
            r.update({k: (round(v, 3) if np.isfinite(v) else v)
                      for k, v in (m or {}).items()})
            rows.append(r)
            th = f"{tau_head:8.2f}" if tau_head is not None else "       -"
            mm = m or {k: float("nan") for k in
                       ("tau_cfr_ns", "Bc50_kHz", "ripple_std_db", "K_est_db")}
            print(f"{j:>5} {model:>8} {wb:>8.2f} "
                  f"{(wb - wb_rt) if model != 'rt' else 0.0:>+7.2f} "
                  f"{tau_cir:>7.2f} {th} {mm['tau_cfr_ns']:>7.2f} "
                  f"{mm['Bc50_kHz']:>8.0f} {mm['ripple_std_db']:>7.2f} "
                  f"{mm['K_est_db']:>6.1f}")

        emit("rt", wb_rt, None, rt_m)

        plot_H = {"rt": H_rt_n}
        for name, prop in models.items():
            wb, tau_h, _ = prop.sample_heads(rx)
            no_path = wb >= (prop.no_path_wb - 1e-3)
            if no_path:
                emit(name, float(prop.no_path_wb), 0.0, None, no_path=True)
                continue
            ms, first = [], None
            for r_i in range(args.n_realizations):
                seed = (args.seed * 1315423911) ^ (j * 2654435761) ^ (r_i * 97531)
                H = prop.synthesize_cfr(tau_rms_ns=tau_h,
                                        mi_scene=scene.mi_scene,
                                        tx_xyz=tx_xyz, rx_xyz=rx,
                                        seed=seed, synthetic=True)
                if first is None:
                    first = norm_cfr(H)
                ms.append(metrics_of(H, scs))
            emit(name, float(wb), float(tau_h), avg_metrics(ms))
            plot_H[name] = first

        # per-link overlay plot: |H(f)| + PDP
        if H_rt_n is not None:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
            off_mhz = (freqs - fc) / 1e6
            for name, H in plot_H.items():
                if H is None:
                    continue
                ax1.plot(off_mhz, 20 * np.log10(np.abs(H) + 1e-12),
                         lw=0.7, alpha=0.85, label=name)
                t_ns, p = pdp_from_cfr(H, scs)
                pdb = 10 * np.log10(p / max(p.max(), 1e-30) + 1e-12)
                ax2.plot(t_ns, pdb, lw=0.8, alpha=0.85, label=name)
            ax1.set_xlabel("freq offset (MHz)"); ax1.set_ylabel("|H| (dB)")
            ax1.set_title(f"link {j} (d={d_m:.1f} m)  CFR magnitude")
            ax2.set_xlabel("delay (ns)"); ax2.set_ylabel("PDP (dB rel peak)")
            ax2.set_xlim(0, max(6 * max(tau_cir, 5.0), 60))
            ax2.set_ylim(-45, 3); ax2.set_title("power-delay profile")
            for ax in (ax1, ax2):
                ax.legend(fontsize=8); ax.grid(alpha=0.3)
            fig.tight_layout()
            fig.savefig(out_dir / f"link_{j:02d}.png", dpi=130)
            plt.close(fig)

    out_csv = out_dir / "cfr_summary.csv"
    with open(out_csv, "w", newline="") as f:
        keys = list(rows[0].keys())
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"\n[done] {out_csv} + per-link plots in {out_dir}")
    print("reading guide: tau_head vs tau_rt_cir scores the tau head; "
          "tau_cfr/Bc50/ripple vs RT scores the SYNTHESIS model given tau; "
          "K_est vs RT scores the heuristic LOS/NLOS K-factor.")


if __name__ == "__main__":
    main()