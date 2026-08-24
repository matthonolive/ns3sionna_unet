import sionna.rt # must be the first import as otherwise Python crashes

import os
import argparse
import numpy as np
import time
import GPUtil
import zmq
import gc

import csv 
import bisect

from common import message_pb2
from common.message_debug import *

from collections import deque, OrderedDict
import warnings

import tensorflow as tf

try:
    gpus = tf.config.list_physical_devices("GPU")
    for g in gpus:
        tf.config.experimental.set_memory_growth(g, True)
except Exception:
    pass


import mitsuba as mi
import drjit as _dr  # allocator hygiene (flush_malloc_cache); aliased to avoid clashes
import math
from millify import millify

from ns3sionna_utils import subcarrier_frequencies, compute_coherence_time, SECOND, MILLISECOND, coherence_from_velocities, \
    MAX_COHERENCE_TIME

import sionna
from sionna.rt import load_scene, Camera, Transmitter, Receiver, PlanarArray, PathSolver

import json
from pathlib import Path
import torch

from dataclasses import replace

from mlink.antenna import AntennaGrid, AntennaDatabase
from mlink.feature import build_feature_tensor, REGISTRY, Specification
from trimesh.intersections import mesh_plane
from mlink.geometry import generate_wall_map, walls_to_mesh
from mlink.scene import Scene as MlinkScene
from mlink.channel_tdl import RtCfg, subcarrier_frequencies_centered, compute_tdl_batch

# import mobility models
from mobility import *

from time import perf_counter


# ==================================================================
# Feature patch: bounds-safe binary_walls (required for tiling scenes)
# ==================================================================
# Identical to the rasterizer used for the offpatch training store
# (train_delta_tau.py / offpatch_finetune.py). The stock mlink feature
# indexes rasterized wall segments into the RX grid without clipping;
# when a 64x64 patch sits inside a larger scene, wall segments extend
# beyond the patch and the indices overflow (IndexError). This version
# DROPS out-of-bounds points (walls outside the patch don't appear --
# matching what the model saw in training) and is a behavioral no-op
# for in-patch geometry, so it is registered globally.
def binary_walls_offpatch(scene, frequency: float) -> np.ndarray:
    mesh = scene.mesh
    z_normal = np.asarray([0.0, 0.0, 1.0], dtype=np.float32)

    rx_grid = scene.antenna_database.rx_grid
    if rx_grid is None:
        raise Exception("Receivers must be initialized with a grid!")
    K, H, W = rx_grid.shape

    def rasterize_line(line_xyz: np.ndarray) -> np.ndarray:
        src = np.asarray(rx_grid.xyz2ijk(line_xyz[0, :]), dtype=np.float32)
        dst = np.asarray(rx_grid.xyz2ijk(line_xyz[1, :]), dtype=np.float32)
        n = int(max(np.max(np.abs(dst - src)), 1)) + 1
        pts = np.linspace(src, dst, num=n, endpoint=True).astype(np.int32)

        keep = (
            (pts[:, 0] >= 0) & (pts[:, 0] < H) &
            (pts[:, 1] >= 0) & (pts[:, 1] < W)
        )
        pts = pts[keep]
        if pts.size == 0:
            return np.empty((0, 2), dtype=np.int32)
        return pts[:, :2]

    wall_tensor_lst = []
    for k in range(K):
        plane_origin = rx_grid.origin + k * rx_grid.deltas[2]
        lines = mesh_plane(
            mesh,
            plane_normal=z_normal,
            plane_origin=plane_origin,
            return_faces=False,
        )

        walls = np.zeros((H, W), dtype=np.float32)
        if lines is not None and len(lines) > 0:
            for line in lines:
                ij = rasterize_line(np.asarray(line))
                if ij.shape[0] > 0:
                    walls[ij[:, 0], ij[:, 1]] = 1.0

        wall_tensor_lst.append(walls)

    wall_tensor = np.stack(wall_tensor_lst, axis=0).astype(np.float32)  # (K,H,W)
    wall_maps = wall_tensor[None, None, :, :, :]
    wall_maps = np.repeat(wall_maps, repeats=scene.antenna_database.tx_coords.shape[0], axis=0)
    return wall_maps


REGISTRY["binary_walls"] = Specification(
    name="binary_walls",
    requires=(),
    fn=binary_walls_offpatch,
)


class UNetTdlPropagator:
    """Residual-over-cost U-Net propagator (train_delta_tau.py checkpoints).

    Model contract (must match training exactly):
        head layout (3 per slice): [r, tau, coverage_logit]
        r    = pred[:,0] * r_std + r_mean            (clipped to +-r_clip_db)
        r    = 0 where num_obstructions < nobs_los_thresh       (LOS gate)
        wb   = wb_cost + r,  wb_cost = -cost feature (cost is stored as
               NEGATIVE path loss in the feature tensor)
        tau  = inv_tau_target( pred[:,1] * tau_std + tau_mean ), >= 0
        cov  = sigmoid(pred[:,2])   -- raw logit, NOT affine-unnormalized

    Differences vs. the legacy (delta-over-Friis) propagator:
      * sample_heads() returns the FULL wideband loss (dB). Callers must
        NOT add FSPL to it.
      * excess-delay head no longer exists; excess_ns is always None.
      * Cached maps are (K, 3, H, W) = [wb_db, tau_ns, cov_prob], physical
        units, CONTINUOUS wb (no sentinel written into the map). No-path
        is decided from the interpolated coverage probability AFTER
        trilinear interpolation, so sentinel values never bleed into
        neighboring finite samples.
      * cost and num_obstructions are REQUIRED input features.
      * Legacy kwargs (y_wb_idx / y_excess_idx / y_tau_rms_idx) are
        accepted and ignored so old call sites keep working.
    """

    # model output channel layout (matches train_delta_tau.py)
    P_R, P_TAU, P_COV = 0, 1, 2

    def __init__(
        self,
        run_dir: str,
        device: str = "cuda",
        # MUST match training grid params
        scale_m: float = 0.625,
        K: int = 4,
        H: int = 64,
        W: int = 64,
        z_step_cells: float = 1.0,
        z_margin_m: float = 0.3125,  # e.g., 0.5 cell * 0.625m
        origin_xy_mode: str = "bbox_min",  # "bbox_min" or "zero"
        dataset_features=None,
        no_path_wb: float = 199.5,
        fft_shift: bool = False,
        cov_thresh: float = 0.5,           # no-path threshold on coverage head
        tx_cache_size: int = 256,          # LRU size for per-TX/(TX,patch) map cache
        k_los_db: float = 6.0,             # Rician K for CFR synthesis (LOS)
        k_nlos_db: float = 2.0,            # Rician K for CFR synthesis (NLOS)
        # ---- legacy kwargs: accepted and ignored ----
        y_wb_idx: int = None,
        y_excess_idx: int = None,
        y_tau_rms_idx: int = None,
    ):
        if any(v is not None for v in (y_wb_idx, y_excess_idx, y_tau_rms_idx)):
            warnings.warn(
                "[UNet] y_wb_idx / y_excess_idx / y_tau_rms_idx are ignored: "
                "the residual-over-cost model has a fixed head layout "
                "[r, tau, coverage_logit].")

        run = Path(run_dir)
        self.meta = json.loads((run / "meta.json").read_text())
        stats = np.load(run / "norm_stats.npz")

        # ---- refuse to run a checkpoint this class doesn't understand ----
        param = self.meta.get("parametrization", None)
        if param != "residual_over_cost":
            raise RuntimeError(
                f"UNetTdlPropagator (residual version) got parametrization="
                f"{param!r}. This class only supports 'residual_over_cost' "
                f"checkpoints from train_delta_tau.py. For legacy "
                f"delta-over-Friis checkpoints, use the previous version of "
                f"this file.")

        self.device = torch.device(device)
        self.model = torch.jit.load(str(run / "model.pt"), map_location=self.device).eval()

        # ---- grid sizes (use meta if present, else fall back) ----
        self.H = int(self.meta.get("H", H))
        self.W = int(self.meta.get("W", W))
        self.K = int(self.meta.get("K", K))
        self.scale_m = float(self.meta.get("scale_m", scale_m))
        self.z_step_m = float(self.scale_m * z_step_cells)
        self.z_margin_m = float(z_margin_m)
        self.origin_xy_mode = origin_xy_mode

        # ---- head layout ----
        self.y_ch = int(self.meta.get("y_ch", 3))
        if self.y_ch != 3:
            raise RuntimeError(
                f"meta.json reports y_ch={self.y_ch}, expected 3 "
                f"([r, tau, coverage_logit]) for residual_over_cost.")

        # ---- features (must match what you trained with) ----
        if dataset_features is None:
            dataset_features = self.meta.get("dataset_features", [
                "binary_walls", "electrical_distance", "cost",
                "num_obstructions", "height_cond"])
        self.dataset_features = list(dataset_features)
        for req in ("cost", "num_obstructions"):
            if req not in self.dataset_features:
                raise RuntimeError(
                    f"Feature '{req}' is required to reconstruct "
                    f"wb = wb_cost + r with the LOS gate, but is missing "
                    f"from dataset_features={self.dataset_features}.")

        # ---- physics / thresholds ----
        self.no_path_wb = float(self.meta.get("no_path_wb_db", no_path_wb))
        self.cov_thresh = float(cov_thresh)
        self.los_gate_residual = bool(self.meta.get("los_gate_residual", True))
        self.nobs_los_thresh = float(self.meta.get("nobs_los_thresh", 0.5))
        self.r_clip_db = float(self.meta.get("r_clip_db", 60.0))
        self.tau_target = str(self.meta.get("tau_target", "raw"))
        self.tau_log_eps_ns = float(self.meta.get("tau_log_eps_ns", 1e-3))
        self.fft_shift = bool(fft_shift)

        # ---- normalization stats ----
        def to_c11(a):
            a = np.asarray(a)
            if a.ndim == 1:
                a = a[:, None, None]
            return a

        for key in ("x_mean", "x_std", "r_mean", "r_std", "tau_mean", "tau_std"):
            if key not in stats.files:
                raise RuntimeError(
                    f"norm_stats.npz is missing '{key}'. This does not look "
                    f"like a residual_over_cost checkpoint "
                    f"(found keys: {list(stats.files)}).")

        x_mean = to_c11(stats["x_mean"])
        x_std = to_c11(stats["x_std"])
        self.C_model = int(x_mean.shape[0])
        self.Y_model = self.K * self.y_ch

        self.x_mean = torch.from_numpy(x_mean).float().to(self.device)
        self.x_std = torch.from_numpy(x_std).float().to(self.device).clamp_min(1e-6)
        self.r_mean = float(stats["r_mean"])
        self.r_std = max(float(stats["r_std"]), 1e-6)
        self.tau_mean = float(stats["tau_mean"])
        self.tau_std = max(float(stats["tau_std"]), 1e-6)

        self.keep_idx = None
        if "keep_idx" in stats.files:
            ki = stats["keep_idx"]
            if ki is not None and np.size(ki) > 0:
                self.keep_idx = np.asarray(ki, dtype=np.int64)

        # ---- set later by attach_scene ----
        self._base_scene = None
        self._rx_grid = None
        self._rx_coords = None
        self._origin = None
        self._needs_tiling = False
        self._bbox_min_x = 0.0
        self._bbox_min_y = 0.0

        # per-feature channel counts (lazy, cached at first prediction)
        self._feat_counts = None
        self._cost_off = None
        self._nobs_off = None

        # CFR synthesis K-factors (calibrated against RT; see validate_cfr)
        self.k_los_db = float(k_los_db)
        self.k_nlos_db = float(k_nlos_db)

        # per-TX map cache. Keys are the TX position QUANTIZED to half a
        # grid cell: features live on a scale_m grid, so sub-cell TX motion
        # produces effectively identical inputs -- rebuilding for every
        # millimeter of walker movement is what made mobile scenes slow.
        # Reuse introduces at most a quarter-cell (~scale_m/4) TX position
        # approximation, far below the grid resolution. A small LRU (not a
        # single slot) prevents alternating TX nodes from evicting each
        # other every request.
        self.tx_cache_size = int(tx_cache_size)
        self._tx_cache = OrderedDict()     # qkey -> (K,3,H,W) maps
        self._patch_cache = OrderedDict()  # (qkey, patch_key) -> (origin, maps)
        self._cache_hits = 0
        self._cache_misses = 0

        # currently active state
        self._cached_tx_key = None         # quantized key of current TX
        self._cached_tx_pos = None
        self._cached_maps = None           # (K, 3, H, W): [wb_db, tau_ns, cov]

    def attach_scene(self, sionna_scene, bbox, fc_hz: float, fft_size: int, subcarrier_spacing_hz: float, frequencies_hz: np.ndarray):
        self.fc_hz = float(fc_hz)
        self.fft_size = int(fft_size)
        self.subcarrier_spacing_hz = float(subcarrier_spacing_hz)
        self.frequencies = np.asarray(frequencies_hz, dtype=np.float64)

        self._base_scene = MlinkScene.from_sionna(sionna_scene)

        print("\n[MATDBG] extracted material_database:")
        print(self._base_scene.material_database)

        mi_scene = getattr(sionna_scene, "mi_scene", None)
        if mi_scene is None:
            mi_scene = getattr(sionna_scene, "_scene", None)

        for i, shape in enumerate(mi_scene.shapes()):
            params = mi.traverse(shape)
            keys = sorted([k for k in params.keys() if "bsdf" in k])

            def scalar(key):
                if key not in params:
                    return None
                return float(np.asarray(params[key]).ravel()[0])

            print(
                f"[MATDBG] shape {i}: "
                f"id={shape.id() if hasattr(shape, 'id') else '<no-id>'} "
                f"has_eta={'bsdf.eta_r' in params} "
                f"has_sigma={'bsdf.sigma' in params} "
                f"eta_r={scalar('bsdf.eta_r')} "
                f"sigma={scalar('bsdf.sigma')} "
                f"d={scalar('bsdf.d')} "
                f"keys={keys}"
            )

        off = self.frequencies - self.fc_hz
        if (off[0] < 0) and (off[-1] > 0) and np.all(np.diff(off) > 0):
            self.fft_shift = True
        else:
            self.fft_shift = False

        # origin
        if self.origin_xy_mode == "zero":
            x0 = 0.0; y0 = 0.0
        else:
            x0 = float(bbox.min.x); y0 = float(bbox.min.y)

        z_min = float(bbox.min.z); z_max = float(bbox.max.z)
        total_span = (self.K - 1) * self.z_step_m
        z0 = z_min + self.z_margin_m
        if z0 + total_span > (z_max - self.z_margin_m):
            z0 = max(z_min, (z_max - self.z_margin_m) - total_span)

        self._origin = np.array([x0, y0, z0], dtype=np.float32)

        self._rx_grid = AntennaGrid(
            origin=self._origin.astype(np.float32),
            deltas=np.asarray(
                [[self.scale_m, 0.0, 0.0],
                 [0.0, self.scale_m, 0.0],
                 [0.0, 0.0, self.z_step_m]], dtype=np.float32),
            shape=(self.K, self.H, self.W),
        )

        # coords (K*H*W,3)
        xs = x0 + self.scale_m * np.arange(self.W, dtype=np.float32)
        ys = y0 + self.scale_m * np.arange(self.H, dtype=np.float32)
        zs = z0 + self.z_step_m * np.arange(self.K, dtype=np.float32)
        Z, Y, X = np.meshgrid(zs, ys, xs, indexing="ij")
        self._rx_coords = np.stack([X, Y, Z], axis=-1).reshape(-1, 3).astype(np.float32)

        # detect whether the scene is larger than one patch
        self._bbox_min_x = float(bbox.min.x)
        self._bbox_min_y = float(bbox.min.y)
        scene_cells_x = int(np.ceil((float(bbox.max.x) - self._bbox_min_x) / self.scale_m))
        scene_cells_y = int(np.ceil((float(bbox.max.y) - self._bbox_min_y) / self.scale_m))
        self._needs_tiling = (scene_cells_x > self.W) or (scene_cells_y > self.H)

        if self._needs_tiling:
            print(f"[UNet] Scene is {scene_cells_x}x{scene_cells_y} cells, patch is {self.W}x{self.H} -> tiling enabled")
        else:
            print(f"[UNet] Scene is {scene_cells_x}x{scene_cells_y} cells, fits in one patch -> tiling disabled")

        # feature offsets depend only on the feature list; recompute lazily
        self._feat_counts = None
        self._cost_off = None
        self._nobs_off = None

        # new scene -> old maps invalid
        self._tx_cache.clear()
        self._patch_cache.clear()
        self._cached_tx_key = None
        self._cached_maps = None

    def _tx_qkey(self, p):
        """TX position quantized to half a grid cell."""
        q = self.scale_m * 0.5
        v = np.asarray(p, dtype=np.float64).reshape(3)
        return (int(round(v[0] / q)), int(round(v[1] / q)), int(round(v[2] / q)))

    def _cache_stat(self, hit: bool):
        if hit:
            self._cache_hits += 1
        else:
            self._cache_misses += 1
        n = self._cache_hits + self._cache_misses
        if n % 200 == 0:
            print(f"[UNet] tx-cache: hits={self._cache_hits} "
                  f"misses={self._cache_misses} "
                  f"({100.0 * self._cache_hits / n:.1f}% hit)", flush=True)

    # --------------------------------------------------------------
    # feature-offset bookkeeping (cost / num_obstructions extraction)
    # --------------------------------------------------------------
    def _ensure_feat_offsets(self, scene):
        """Compute per-feature channel counts once (mirrors training's
        infer_feature_channel_counts / feature_slice_offset). One-time
        cost at the first prediction; cached afterwards."""
        if self._cost_off is not None and self._nobs_off is not None:
            return
        counts = {}
        for f in self.dataset_features:
            xf = build_feature_tensor(scene, self.fc_hz, requested=[f]).astype(np.float32)
            counts[f] = int(xf.shape[1])
        self._feat_counts = counts

        off = 0
        offsets = {}
        for f in self.dataset_features:
            offsets[f] = off
            off += counts[f]
        self._cost_off = offsets["cost"]
        self._nobs_off = offsets["num_obstructions"]
        print(f"[UNet] feature channel counts: {counts} "
              f"(cost_off={self._cost_off}, nobs_off={self._nobs_off})")

    # --------------------------------------------------------------
    # forward + physical reconstruction
    # --------------------------------------------------------------
    def _forward_normalized(self, x_chw: torch.Tensor) -> torch.Tensor:
        """x_chw: (C_model, H, W) RAW features -> NORMALIZED model output
        (K*3, H, W). No affine un-normalization here: the heads are mixed
        (normalized r, normalized tau target, raw logit)."""
        x_n = (x_chw - self.x_mean) / self.x_std
        with torch.no_grad():
            pred_n = self.model(x_n.unsqueeze(0)).squeeze(0)  # (Y_model,H,W)
        return pred_n

    def _tau_from_target(self, tau_t: np.ndarray) -> np.ndarray:
        if self.tau_target == "log10":
            tau = np.power(10.0, tau_t) - self.tau_log_eps_ns
        else:
            tau = tau_t
        return np.maximum(tau, 0.0)

    def _reconstruct_maps(self, pred_n: torch.Tensor, x_raw: np.ndarray) -> np.ndarray:
        """pred_n: (K*3, H, W) normalized model output.
        x_raw:  (1, c_in, K, H, W) RAW (un-normalized) feature tensor.
        Returns (K, 3, H, W) physical maps [wb_db, tau_ns, cov_prob].
        wb is CONTINUOUS (no sentinel); no-path is applied after interp."""
        p = pred_n.detach().float().cpu().numpy().reshape(
            self.K, self.y_ch, self.H, self.W)

        cost = x_raw[0, self._cost_off]          # (K, H, W), negative PL
        nobs = x_raw[0, self._nobs_off]          # (K, H, W)
        wb_cost = -cost.astype(np.float32)

        # residual head
        r = p[:, self.P_R] * self.r_std + self.r_mean
        r = np.clip(r, -self.r_clip_db, self.r_clip_db)
        if self.los_gate_residual:
            r[nobs < self.nobs_los_thresh] = 0.0

        wb = (wb_cost + r).astype(np.float32)

        # tau head
        tau = self._tau_from_target(
            p[:, self.P_TAU] * self.tau_std + self.tau_mean).astype(np.float32)

        # coverage head: logit -> probability (numerically stable sigmoid)
        z = p[:, self.P_COV]
        cov = np.where(z >= 0.0,
                       1.0 / (1.0 + np.exp(-np.abs(z))),
                       np.exp(-np.abs(z)) / (1.0 + np.exp(-np.abs(z)))).astype(np.float32)
        # LOS always has a path: mirror the training-time guarantee so the
        # coverage head can never drop a true-LOS link
        if self.los_gate_residual:
            cov[nobs < self.nobs_los_thresh] = 1.0

        return np.stack([wb, tau, cov], axis=1)   # (K, 3, H, W)

    def _predict_maps(self, scene) -> np.ndarray:
        """Common path for full-scene and patch inference."""
        self._ensure_feat_offsets(scene)

        # x: (1, c_in, K, H, W)
        x = build_feature_tensor(scene, self.fc_hz, requested=self.dataset_features).astype(np.float32)
        c_in = x.shape[1]

        # stack: (1,c_in,K,H,W) -> (1,K*c_in,H,W)
        x_stack = x.transpose(0, 2, 1, 3, 4).reshape(1, self.K * c_in, self.H, self.W)
        x_chw = x_stack[0]  # (K*c_in,H,W)

        if self.keep_idx is not None:
            if self.keep_idx.size != self.C_model:
                raise RuntimeError(
                    f"keep_idx size={self.keep_idx.size} but model expects C_model={self.C_model}. "
                    "Your norm_stats/model are inconsistent.")
            x_chw = x_chw[self.keep_idx, :, :]

        if x_chw.shape[0] != self.C_model:
            raise RuntimeError(
                f"UNet input channel mismatch: built {x_chw.shape[0]} channels "
                f"(K={self.K}, c_in={c_in}) but model expects C_model={self.C_model} "
                f"(from norm_stats.x_mean).")

        x_t = torch.from_numpy(x_chw).to(self.device)
        pred_n = self._forward_normalized(x_t)  # (Y_model,H,W) normalized

        if int(pred_n.shape[0]) != self.Y_model:
            raise RuntimeError(
                f"UNet output channel mismatch: model outputs {int(pred_n.shape[0])}, "
                f"expected K*y_ch={self.Y_model}.")

        return self._reconstruct_maps(pred_n, x)

    # --------------------------------------------------------------
    # per-TX prediction (single-patch mode) and tiling
    # --------------------------------------------------------------
    def predict_for_tx(self, tx_pos_xyz: np.ndarray):
        assert self._base_scene is not None, "call attach_scene() first"

        tx_pos_xyz = np.asarray(tx_pos_xyz, dtype=np.float32).reshape(3)
        qkey = self._tx_qkey(tx_pos_xyz)
        self._cached_tx_pos = tx_pos_xyz.copy()
        self._cached_tx_key = qkey

        if self._needs_tiling:
            # tiling mode defers inference to sample_heads; the patch
            # cache is keyed by (qkey, patch) so no invalidation is needed
            return

        maps = self._tx_cache.get(qkey)
        if maps is not None:
            self._tx_cache.move_to_end(qkey)
            self._cached_maps = maps
            self._cache_stat(hit=True)
            return
        self._cache_stat(hit=False)

        adb = AntennaDatabase(tx_pos_xyz.reshape(1, 3), self._rx_coords, None, self._rx_grid)
        scene = replace(self._base_scene, antenna_database=adb)

        maps = self._predict_maps(scene)
        self._tx_cache[qkey] = maps
        while len(self._tx_cache) > self.tx_cache_size:
            self._tx_cache.popitem(last=False)
        self._cached_maps = maps

        # #DEBUG DUMP
        # self._dump_wb_debug_png(
        #     maps=self._cached_maps,
        #     tx_pos_xyz=tx_pos_xyz,
        #     origin_xyz=self._origin,
        #     out_dir="debug_wb_unet",
        #     prefix="unet_full",
        # )

    def _patch_key_for(self, rx_xyz):
        """Snap an RX position to a patch-grid cell (stride = half patch)."""
        stride = (self.W // 2) * self.scale_m  # 32 cells in meters
        ix = int(np.floor((rx_xyz[0] - self._bbox_min_x) / stride))
        iy = int(np.floor((rx_xyz[1] - self._bbox_min_y) / stride))
        return (ix, iy)

    def _patch_origin_for(self, patch_key, rx_xyz):
        """Compute the world-space origin of a patch, ensuring RX is inside."""
        stride = (self.W // 2) * self.scale_m
        # start from the grid-snapped position
        x0 = self._bbox_min_x + patch_key[0] * stride
        y0 = self._bbox_min_y + patch_key[1] * stride
        # clamp so patch doesn't extend past the original origin on the low end
        x0 = max(x0, self._bbox_min_x)
        y0 = max(y0, self._bbox_min_y)
        return np.array([x0, y0, self._origin[2]], dtype=np.float32)

    def _run_patch(self, patch_key, tx_pos, rx_xyz):
        """Build features for a 64x64 patch and run inference."""
        from dataclasses import replace as dc_replace

        patch_origin = self._patch_origin_for(patch_key, rx_xyz)

        rx_grid = AntennaGrid(
            origin=patch_origin,
            deltas=np.asarray([
                [self.scale_m, 0.0, 0.0],
                [0.0, self.scale_m, 0.0],
                [0.0, 0.0, self.z_step_m],
            ], dtype=np.float32),
            shape=(self.K, self.H, self.W),
        )

        # build RX coords for this patch
        xs = patch_origin[0] + self.scale_m * np.arange(self.W, dtype=np.float32)
        ys = patch_origin[1] + self.scale_m * np.arange(self.H, dtype=np.float32)
        zs = patch_origin[2] + self.z_step_m * np.arange(self.K, dtype=np.float32)
        Z, Y, X = np.meshgrid(zs, ys, xs, indexing="ij")
        rx_coords = np.stack([X, Y, Z], axis=-1).reshape(-1, 3).astype(np.float32)

        adb = AntennaDatabase(tx_pos.reshape(1, 3), rx_coords, None, rx_grid)
        patch_scene = dc_replace(self._base_scene, antenna_database=adb)

        maps = self._predict_maps(patch_scene)
        qkey = self._tx_qkey(tx_pos)
        self._patch_cache[(qkey, patch_key)] = (patch_origin, maps)
        while len(self._patch_cache) > self.tx_cache_size:
            self._patch_cache.popitem(last=False)

    # --------------------------------------------------------------
    # sampling
    # --------------------------------------------------------------
    def _trilerp(self, vol_khw: np.ndarray, kf: float, yf: float, xf: float) -> float:
        K, H, W = vol_khw.shape
        kf = float(np.clip(kf, 0.0, K - 1.0))
        yf = float(np.clip(yf, 0.0, H - 1.0))
        xf = float(np.clip(xf, 0.0, W - 1.0))

        k0 = int(np.floor(kf)); k1 = min(k0 + 1, K - 1); wk = kf - k0
        y0 = int(np.floor(yf)); y1 = min(y0 + 1, H - 1); wy = yf - y0
        x0 = int(np.floor(xf)); x1 = min(x0 + 1, W - 1); wx = xf - x0

        c000 = vol_khw[k0, y0, x0]; c001 = vol_khw[k0, y0, x1]
        c010 = vol_khw[k0, y1, x0]; c011 = vol_khw[k0, y1, x1]
        c100 = vol_khw[k1, y0, x0]; c101 = vol_khw[k1, y0, x1]
        c110 = vol_khw[k1, y1, x0]; c111 = vol_khw[k1, y1, x1]

        c00 = c000 * (1 - wx) + c001 * wx
        c01 = c010 * (1 - wx) + c011 * wx
        c10 = c100 * (1 - wx) + c101 * wx
        c11 = c110 * (1 - wx) + c111 * wx

        c0 = c00 * (1 - wy) + c01 * wy
        c1 = c10 * (1 - wy) + c11 * wy

        return float(c0 * (1 - wk) + c1 * wk)

    def sample_heads(self, rx_pos_xyz: np.ndarray):
        """Returns (wb_db, tau_rms_ns, excess_ns).

        CONTRACT CHANGE vs. the legacy class:
          * wb_db is the FULL wideband loss (wb_cost + gated residual),
            already including all wall losses. Do NOT add FSPL to it.
          * If the interpolated coverage probability is below cov_thresh,
            wb_db == self.no_path_wb (sentinel) and tau == 0.
          * excess_ns is always None (the head no longer exists)."""
        rx = np.asarray(rx_pos_xyz, dtype=np.float32).reshape(3)
        if not self._needs_tiling:
            # Original path: single full-scene grid
            assert self._cached_maps is not None, "call predict_for_tx() first"
            x0, y0, z0 = self._origin.tolist()
            xf = (rx[0] - x0) / self.scale_m
            yf = (rx[1] - y0) / self.scale_m
            kf = (rx[2] - z0) / self.z_step_m
            maps = self._cached_maps
        else:
            # Tiling path: LRU keyed by (quantized TX, patch)
            patch_key = self._patch_key_for(rx)
            ck = (self._cached_tx_key, patch_key)
            entry = self._patch_cache.get(ck)
            if entry is None:
                self._cache_stat(hit=False)
                self._run_patch(patch_key, self._cached_tx_pos, rx)
                entry = self._patch_cache[ck]
            else:
                self._patch_cache.move_to_end(ck)
                self._cache_stat(hit=True)
            patch_origin, maps = entry

            x0, y0, z0 = patch_origin.tolist()
            xf = (rx[0] - x0) / self.scale_m
            yf = (rx[1] - y0) / self.scale_m
            kf = (rx[2] - z0) / self.z_step_m

        if (xf < 0 or xf > self.W-1 or yf < 0 or yf > self.H-1 or kf < 0 or kf > self.K-1):
            print(f"[warn] RX outside grid -> clamping: xf={xf:.2f}, yf={yf:.2f}, kf={kf:.2f}, rx={rx}")

        wb = self._trilerp(maps[:, 0, :, :], kf, yf, xf)
        tau = self._trilerp(maps[:, 1, :, :], kf, yf, xf)
        cov = self._trilerp(maps[:, 2, :, :], kf, yf, xf)

        if cov < self.cov_thresh:
            return float(self.no_path_wb), 0.0, None
        return float(wb), float(max(tau, 0.0)), None

    def _dump_wb_debug_png(
        self,
        maps: np.ndarray,
        tx_pos_xyz: np.ndarray,
        origin_xyz: np.ndarray,
        walls_khw: np.ndarray = None,
        out_dir: str = "debug_wb_unet",
        prefix: str = "unet",
    ):
        """
        maps:      (K, 3, H, W) physical [wb_db, tau_ns, cov_prob]
        tx_pos_xyz:(3,)
        origin_xyz:(3,)
        walls_khw: optional (K, H, W) binary wall map on same grid
        """
        import os
        os.makedirs(out_dir, exist_ok=True)

        import matplotlib
        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt

        tx = np.asarray(tx_pos_xyz, dtype=np.float32).reshape(3)
        x0, y0, z0 = [float(v) for v in origin_xyz]

        # wb map is already the full loss -- no Friis reconstruction needed
        wb_db = maps[:, 0, :, :]
        cov = maps[:, 2, :, :]

        wb_plot = wb_db.copy()
        wb_plot[cov < self.cov_thresh] = np.nan

        for k in range(self.K):
            plt.figure(figsize=(6, 5))
            im = plt.imshow(wb_plot[k], origin="lower")
            plt.colorbar(im, label="Wideband loss (dB)")

            if walls_khw is not None:
                plt.contour(
                    walls_khw[k],
                    levels=[0.5],
                    colors="white",
                    linewidths=0.8,
                    origin="lower",
                )

            plt.scatter(
                [(tx[0] - x0) / self.scale_m],
                [(tx[1] - y0) / self.scale_m],
                c="red",
                s=30,
                marker="x",
                label="TX",
            )
            plt.legend(loc="upper right")
            plt.title(
                f"{prefix} WB loss slice k={k}  "
                f"TX=({tx[0]:.2f},{tx[1]:.2f},{tx[2]:.2f})"
            )
            plt.tight_layout()
            plt.savefig(
                os.path.join(
                    out_dir,
                    f"{prefix}_tx_{tx[0]:.2f}_{tx[1]:.2f}_{tx[2]:.2f}_k{k}.png",
                ),
                dpi=150,
            )
            plt.close()


    def synthesize_cfr(self, tau_rms_ns: float, mi_scene, tx_xyz, rx_xyz, seed: int, synthetic = False) -> np.ndarray:


        if not synthetic:
            H = np.ones(self.fft_size, dtype=np.complex64)
            if self.fft_shift:
                H = np.fft.fftshift(H)
            return H

        def is_los_mi(mi_scene, tx_xyz, rx_xyz, eps=1e-2):
            tx = np.asarray(tx_xyz, dtype=np.float32)
            rx = np.asarray(rx_xyz, dtype=np.float32)
            dvec = rx - tx
            dist = float(np.linalg.norm(dvec))
            if dist < 1e-6:
                return True
            direction = dvec / dist

            # Nudge the origin forward a bit to avoid self-intersection / boundary precision issues
            o = mi.Point3f(tx + eps * direction)
            d = mi.Vector3f(direction)

            ray = mi.Ray3f(o, d)
            ray.maxt = mi.Float(max(dist - 2*eps, 0.0))

            si = mi_scene.ray_intersect(ray, mi.RayFlags.Minimal, False, True)
            return not bool(si.is_valid()) 

        los = is_los_mi(mi_scene, tx_xyz, rx_xyz)

        K_db = self.k_los_db if los else self.k_nlos_db
        
        N = self.fft_size
        df = self.subcarrier_spacing_hz
        Ts = 1.0 / (N * df)

        tau_rms = max(float(tau_rms_ns), 1e-3) * 1e-9

        #Rician K-factor
        K_lin = 10.0**(float(K_db) / 10.0)
        if K_lin <= 0.0:
            K_lin = 0.0

        #Power split 
        if K_lin > 0.0: 
            p_spec = K_lin / (K_lin + 1.0)
            p_diff = 1.0 / (K_lin + 1.0) 

            tau_d = tau_rms * (K_lin + 1.0) / np.sqrt(2.0 * K_lin + 1.0)

        else:
            p_spec = 0.0
            p_diff = 1.0
            tau_d = tau_rms

        
        L = int(np.clip(np.ceil(6.0 * tau_d / Ts), 1, N))
        t = np.arange(L, dtype=np.float64) * Ts
        p = np.exp(-t / max(tau_d, 1e-12))
        p = p / (p.sum() + 1e-12)

        rng = np.random.default_rng(seed)

        w = (rng.standard_normal(L) + 1j * rng.standard_normal(L)) * np.sqrt(0.5)
        taps = w * np.sqrt(p_diff * p)

        if p_spec > 0.0:
            phi = rng.uniform(0.0, 2.0 * np.pi)
            taps[0] += np.sqrt(p_spec) * np.exp(1j * phi)

        H = np.fft.fft(taps, n=N).astype(np.complex64)
        H = H / np.sqrt(np.mean(np.abs(H) ** 2) + 1e-12)
        if self.fft_shift:
            H = np.fft.fftshift(H)
        return H
    

def _read_mobility_trace_csv(path: str):
    """
    Reads a CSV with at least: t_s, node, x, y, z
    Returns dict: node_id -> (t_ns_sorted, pos_sorted (N,3), vel_sorted (N,3))
    Uses piecewise-linear interpolation; velocity is segment slope.
    """
    with open(path, "r", newline="") as f:
        while True:
            hdr_line = f.readline()
            if hdr_line == "":
                raise RuntimeError(f"mobility trace '{path}' has no header")
            hdr_line = hdr_line.strip()
            if hdr_line and not hdr_line.startswith("#"):
                break

        header = [h.strip().lower() for h in hdr_line.split(",")]
        col = {name: i for i, name in enumerate(header)}

        def need(name):
            if name not in col:
                raise RuntimeError(f"mobility trace '{path}' missing column '{name}' (have {header})")
            return col[name]

        it_t = need("t_s"); it_n = need("node")
        it_x = need("x");   it_y = need("y"); it_z = need("z")

        reader = csv.reader(f)
        raw = {}  # node -> list of (t_ns, pos)
        for row in reader:
            if not row:
                continue
            if row[0].strip().startswith("#"):
                continue
            if len(row) <= max(it_t, it_n, it_x, it_y, it_z):
                continue

            t_s = float(row[it_t].strip())
            n   = int(float(row[it_n].strip()))
            x   = float(row[it_x].strip())
            y   = float(row[it_y].strip())
            z   = float(row[it_z].strip())

            t_ns = int(round(t_s * 1e9))
            raw.setdefault(n, []).append((t_ns, np.array([x, y, z], dtype=np.float32)))

    trace = {}
    for n, items in raw.items():
        items.sort(key=lambda p: p[0])

        # de-dup by time
        t_list = []
        p_list = []
        last_t = None
        for t, p in items:
            if last_t is not None and t == last_t:
                continue
            t_list.append(t)
            p_list.append(p)
            last_t = t

        t_ns = np.asarray(t_list, dtype=np.int64)
        pos  = np.asarray(p_list, dtype=np.float32)

        vel = np.zeros_like(pos, dtype=np.float32)
        if len(t_ns) >= 2:
            dt = (t_ns[1:] - t_ns[:-1]).astype(np.float64) / 1e9
            dt = np.maximum(dt, 1e-12)
            seg_v = ((pos[1:] - pos[:-1]) / dt[:, None]).astype(np.float32, copy=False)

            # --- your existing "avoid zero-velocity segments" fix ---
            v_eps = 1e-3
            speed = np.linalg.norm(seg_v, axis=1)
            last = None
            for i in range(seg_v.shape[0]):
                if speed[i] < v_eps:
                    if last is not None:
                        seg_v[i] = last
                else:
                    last = seg_v[i].copy()

            speed2 = np.linalg.norm(seg_v, axis=1)
            nz = np.where(speed2 >= v_eps)[0]
            if nz.size > 0:
                first = nz[0]
                if first > 0:
                    seg_v[:first] = seg_v[first]
            # -----------------------------------------------

            vel[:-1] = seg_v
            vel[-1]  = seg_v[-1]

        # >>> THIS WAS MISSING <<<
        trace[n] = (t_ns, pos, vel)

    if not trace:
        raise RuntimeError(f"mobility trace '{path}' parsed to 0 nodes (check header/columns)")

    return trace



class TraceMobility:
    """
    Mobility model driven by an external time->position trace.
    Compatible with server calls:
      - .pos, .velocity
      - .get_pos_at(t_ns), .get_velo_at(t_ns)
      - .update_pos(t_ns, pos, velo, ...) (no-op-ish, keeps latest)
    """

    def __init__(self, node_id: int, t_ns: np.ndarray, pos: np.ndarray, vel: np.ndarray):
        self.node_id = int(node_id)
        self.t_ns = np.asarray(t_ns, dtype=np.int64)
        self.pos_arr = np.asarray(pos, dtype=np.float32)
        self.vel_arr = np.asarray(vel, dtype=np.float32)

        # current state
        self.pos = self.pos_arr[0].copy()
        self.velocity = self.vel_arr[0].copy()

        # optional history (kept for debugging; not required)
        self.pos_history = {int(self.t_ns[0]): self.pos.copy()}

    def _interp(self, t_ns: int):
        t_ns = int(t_ns)

        # clamp outside range
        if t_ns <= int(self.t_ns[0]):
            return self.pos_arr[0], self.vel_arr[0]
        if t_ns >= int(self.t_ns[-1]):
            return self.pos_arr[-1], self.vel_arr[-1]

        i = bisect.bisect_right(self.t_ns.tolist(), t_ns) - 1
        i = max(0, min(i, len(self.t_ns) - 2))

        t0 = int(self.t_ns[i]); t1 = int(self.t_ns[i + 1])
        p0 = self.pos_arr[i];    p1 = self.pos_arr[i + 1]

        # linear interpolation
        a = (t_ns - t0) / max(1.0, float(t1 - t0))
        p = (1.0 - a) * p0 + a * p1

        # piecewise-constant vel = segment slope
        v = self.vel_arr[i]
        return p.astype(np.float32), v.astype(np.float32)

    def set_time(self, t_ns: int):
        p, v = self._interp(t_ns)
        self.pos = p
        self.velocity = v
        self.pos_history[int(t_ns)] = p.copy()

    def get_pos_at(self, t_ns: int):
        p, _ = self._interp(t_ns)
        return p

    def get_velo_at(self, t_ns: int):
        _, v = self._interp(t_ns)
        return v

    def update_pos(self, t_ns, next_pos, velocity, *_args):
        # if some legacy code calls this, accept it, but keep trace as truth.
        self.pos = np.asarray(next_pos, dtype=np.float32)
        self.velocity = np.asarray(velocity, dtype=np.float32)
        self.pos_history[int(t_ns)] = self.pos.copy()

    def check_set_new_velocity(self, *_args, **_kwargs):
        # trace-driven: no random updates
        return


class Cost231Propagator:
    """COST-231 multi-wall model. Same interface as UNetTdlPropagator but no neural network."""

    def __init__(self, scale_m=0.625, K=4, H=64, W=64, z_step_cells=1.0,
                 z_margin_m=0.3125, origin_xy_mode="bbox_min",
                 dataset_features=None, no_path_wb=199.5, fft_shift=False,
                 default_tau_rms_ns=5.0, k_los_db=6.0, k_nlos_db=2.0):
        self.scale_m = float(scale_m)
        self.H, self.W, self.K = int(H), int(W), int(K)
        # minimum grid dims; attach_scene() enlarges H/W to cover the scene
        self._base_H, self._base_W = int(H), int(W)
        self.z_step_m = float(scale_m * z_step_cells)
        self.z_margin_m = float(z_margin_m)
        self.origin_xy_mode = origin_xy_mode
        self.no_path_wb = float(no_path_wb)
        self.fft_shift = bool(fft_shift)
        self.default_tau_rms_ns = float(default_tau_rms_ns)
        self.k_los_db = float(k_los_db)
        self.k_nlos_db = float(k_nlos_db)

        if dataset_features is None:
            dataset_features = ["cost"]
        self.dataset_features = list(dataset_features)

        self._base_scene = None
        self._rx_grid = None
        self._rx_coords = None
        self._origin = None

        self._cached_tx_key = None
        self._cached_cost_map = None  # (K, H, W) total path loss in dB (negative)

    def attach_scene(self, sionna_scene, bbox, fc_hz, fft_size, subcarrier_spacing_hz, frequencies_hz):
        self.fc_hz = float(fc_hz)
        self.fft_size = int(fft_size)
        self.subcarrier_spacing_hz = float(subcarrier_spacing_hz)
        self.frequencies = np.asarray(frequencies_hz, dtype=np.float64)

        self._base_scene = MlinkScene.from_sionna(sionna_scene)
        print("\n[MATDBG] extracted material_database:")
        print(self._base_scene.material_database)

        mi_scene = getattr(sionna_scene, "mi_scene", None)
        if mi_scene is None:
            mi_scene = getattr(sionna_scene, "_scene", None)

        for i, shape in enumerate(mi_scene.shapes()):
            params = mi.traverse(shape)
            keys = sorted([k for k in params.keys() if "bsdf" in k])

            def scalar(key):
                if key not in params:
                    return None
                return float(np.asarray(params[key]).ravel()[0])

            print(
                f"[MATDBG] shape {i}: "
                f"id={shape.id() if hasattr(shape, 'id') else '<no-id>'} "
                f"has_eta={'bsdf.eta_r' in params} "
                f"has_sigma={'bsdf.sigma' in params} "
                f"eta_r={scalar('bsdf.eta_r')} "
                f"sigma={scalar('bsdf.sigma')} "
                f"d={scalar('bsdf.d')} "
                f"keys={keys}"
            )

        off = self.frequencies - self.fc_hz
        if (off[0] < 0) and (off[-1] > 0) and np.all(np.diff(off) > 0):
            self.fft_shift = True
        else:
            self.fft_shift = False

        if self.origin_xy_mode == "zero":
            x0 = 0.0; y0 = 0.0
        else:
            x0 = float(bbox.min.x); y0 = float(bbox.min.y)

        z_min = float(bbox.min.z); z_max = float(bbox.max.z)
        total_span = (self.K - 1) * self.z_step_m
        z0 = z_min + self.z_margin_m
        if z0 + total_span > (z_max - self.z_margin_m):
            z0 = max(z_min, (z_max - self.z_margin_m) - total_span)

        self._origin = np.array([x0, y0, z0], dtype=np.float32)

        # --- size the grid to cover the whole scene ------------------------
        # FIX (tier_large LOS-gate defect): the stock 64x64 grid spans only
        # 40 m at 0.625 m/cell, and _trilerp CLAMPS out-of-range coordinates.
        # On scenes larger than one patch (tier_medium/large) every RX beyond
        # the grid was silently pinned to the grid edge, so the cost231 leg
        # reported wrong wb losses over most of the scene -- observed in the
        # campaigns as unet != cost231 (r != 0) at zero-wall links. Unlike
        # the U-Net, the analytic map has no trained patch size, so we simply
        # enlarge the grid to the scene footprint (cost scales linearly in
        # cells: one ray bundle per cell).
        span_x = float(bbox.max.x) - x0
        span_y = float(bbox.max.y) - y0
        W_needed = int(np.ceil(span_x / self.scale_m)) + 1
        H_needed = int(np.ceil(span_y / self.scale_m)) + 1
        self.W = int(np.clip(W_needed, self._base_W, 1024))
        self.H = int(np.clip(H_needed, self._base_H, 1024))
        print(f"[COST231] grid sized to scene: {self.H}x{self.W} cells "
              f"({self.H * self.scale_m:.1f} x {self.W * self.scale_m:.1f} m)")

        # new scene -> old map invalid (latent stale-cache bug when the
        # server is reused across sim inits and a TX position recurs)
        self._cached_tx_key = None
        self._cached_cost_map = None

        self._rx_grid = AntennaGrid(
            origin=self._origin,
            deltas=np.asarray([
                [self.scale_m, 0, 0],
                [0, self.scale_m, 0],
                [0, 0, self.z_step_m],
            ], dtype=np.float32),
            shape=(self.K, self.H, self.W),
        )

        xs = x0 + self.scale_m * np.arange(self.W, dtype=np.float32)
        ys = y0 + self.scale_m * np.arange(self.H, dtype=np.float32)
        zs = z0 + self.z_step_m * np.arange(self.K, dtype=np.float32)
        Z, Y, X = np.meshgrid(zs, ys, xs, indexing="ij")
        self._rx_coords = np.stack([X, Y, Z], axis=-1).reshape(-1, 3).astype(np.float32)

    def predict_for_tx(self, tx_pos_xyz):
        assert self._base_scene is not None, "call attach_scene() first"
        tx_pos_xyz = np.asarray(tx_pos_xyz, dtype=np.float32).reshape(3)
        key = tuple(np.round(tx_pos_xyz, 3).tolist())
        if key == self._cached_tx_key and self._cached_cost_map is not None:
            return

        adb = AntennaDatabase(tx_pos_xyz.reshape(1, 3), self._rx_coords, None, self._rx_grid)
        scene = replace(self._base_scene, antenna_database=adb)

        # cost feature: (1, 1, K, H, W) — total path loss (negative dB)
        x = build_feature_tensor(scene, self.fc_hz, requested=["cost"]).astype(np.float32)
        # shape is (num_tx=1, c_in=1, K, H, W)
        self._cached_cost_map = x[0, 0]  # (K, H, W)
        self._cached_tx_key = key
        self._cached_tx_pos = tx_pos_xyz.copy()


        #  # --- DEBUG DUMP START ---
        # import os
        # import matplotlib.pyplot as plt

        # os.makedirs("debug_cost231", exist_ok=True)

        # # Optional: also get walls for overlay
        # walls = build_feature_tensor(scene, self.fc_hz, requested=["binary_walls"]).astype(np.float32)[0, 0]

        # for k in range(self._cached_cost_map.shape[0]):
        #     loss_db = -self._cached_cost_map[k]   # convert from negative path-loss map to positive dB

        #     plt.figure(figsize=(6, 5))
        #     plt.imshow(loss_db, origin="lower")
        #     plt.colorbar(label="COST231 path loss (dB)")

        #     # overlay walls as contours
        #     plt.contour(walls[k], levels=[0.5], linewidths=0.8)

        #     plt.title(f"COST231 slice k={k} TX=({key[0]:.2f}, {key[1]:.2f}, {key[2]:.2f})")
        #     plt.tight_layout()
        #     plt.savefig(f"debug_cost231/cost231_tx_{key[0]:.2f}_{key[1]:.2f}_{key[2]:.2f}_k{k}.png", dpi=150)
        #     plt.close()
        # # --- DEBUG DUMP END ---

    def _trilerp(self, vol_khw, kf, yf, xf):
        K, H, W = vol_khw.shape
        kf = float(np.clip(kf, 0, K - 1))
        yf = float(np.clip(yf, 0, H - 1))
        xf = float(np.clip(xf, 0, W - 1))

        k0 = int(np.floor(kf)); k1 = min(k0 + 1, K - 1); wk = kf - k0
        y0 = int(np.floor(yf)); y1 = min(y0 + 1, H - 1); wy = yf - y0
        x0 = int(np.floor(xf)); x1 = min(x0 + 1, W - 1); wx = xf - x0

        c000 = vol_khw[k0, y0, x0]; c001 = vol_khw[k0, y0, x1]
        c010 = vol_khw[k0, y1, x0]; c011 = vol_khw[k0, y1, x1]
        c100 = vol_khw[k1, y0, x0]; c101 = vol_khw[k1, y0, x1]
        c110 = vol_khw[k1, y1, x0]; c111 = vol_khw[k1, y1, x1]

        c00 = c000 * (1 - wx) + c001 * wx
        c01 = c010 * (1 - wx) + c011 * wx
        c10 = c100 * (1 - wx) + c101 * wx
        c11 = c110 * (1 - wx) + c111 * wx

        c0 = c00 * (1 - wy) + c01 * wy
        c1 = c10 * (1 - wy) + c11 * wy

        return float(c0 * (1 - wk) + c1 * wk)

    def sample_heads(self, rx_pos_xyz):
        assert self._cached_cost_map is not None, "call predict_for_tx() first"
        rx = np.asarray(rx_pos_xyz, dtype=np.float32).reshape(3)
        x0, y0, z0 = self._origin.tolist()
        xf = (rx[0] - x0) / self.scale_m
        yf = (rx[1] - y0) / self.scale_m
        kf = (rx[2] - z0) / self.z_step_m

        # cost feature is total path loss (negative dB, e.g. -85)
        # wb_loss = -cost_value
        cost_val = self._trilerp(self._cached_cost_map, kf, yf, xf)
        wb_db = -cost_val

        # COST-231 has no delay information — use a fixed default
        tau_rms_ns = self.default_tau_rms_ns
        excess_ns = 0.0

        return float(wb_db), float(tau_rms_ns), float(excess_ns)

    def synthesize_cfr(self, tau_rms_ns: float, mi_scene, tx_xyz, rx_xyz, seed: int, synthetic = False) -> np.ndarray:

        if not synthetic:
            H = np.ones(self.fft_size, dtype=np.complex64)
            if self.fft_shift:
                H = np.fft.fftshift(H)
            return H

        def is_los_mi(mi_scene, tx_xyz, rx_xyz, eps=1e-2):
            tx = np.asarray(tx_xyz, dtype=np.float32)
            rx = np.asarray(rx_xyz, dtype=np.float32)
            dvec = rx - tx
            dist = float(np.linalg.norm(dvec))
            if dist < 1e-6:
                return True
            direction = dvec / dist

            # Nudge the origin forward a bit to avoid self-intersection / boundary precision issues
            o = mi.Point3f(tx + eps * direction)
            d = mi.Vector3f(direction)

            ray = mi.Ray3f(o, d)
            ray.maxt = mi.Float(max(dist - 2*eps, 0.0))

            si = mi_scene.ray_intersect(ray, mi.RayFlags.Minimal, False, True)
            return not bool(si.is_valid()) 

        los = is_los_mi(mi_scene, tx_xyz, rx_xyz)

        K_db = self.k_los_db if los else self.k_nlos_db
        
        N = self.fft_size
        df = self.subcarrier_spacing_hz
        Ts = 1.0 / (N * df)

        tau_rms = max(float(tau_rms_ns), 1e-3) * 1e-9

        #Rician K-factor
        K_lin = 10.0**(float(K_db) / 10.0)
        if K_lin <= 0.0:
            K_lin = 0.0

        #Power split 
        if K_lin > 0.0: 
            p_spec = K_lin / (K_lin + 1.0)
            p_diff = 1.0 / (K_lin + 1.0) 

            tau_d = tau_rms * (K_lin + 1.0) / np.sqrt(2.0 * K_lin + 1.0)

        else:
            p_spec = 0.0
            p_diff = 1.0
            tau_d = tau_rms

        
        L = int(np.clip(np.ceil(6.0 * tau_d / Ts), 1, N))
        t = np.arange(L, dtype=np.float64) * Ts
        p = np.exp(-t / max(tau_d, 1e-12))
        p = p / (p.sum() + 1e-12)

        rng = np.random.default_rng(seed)

        w = (rng.standard_normal(L) + 1j * rng.standard_normal(L)) * np.sqrt(0.5)
        taps = w * np.sqrt(p_diff * p)

        if p_spec > 0.0:
            phi = rng.uniform(0.0, 2.0 * np.pi)
            taps[0] += np.sqrt(p_spec) * np.exp(1j * phi)

        H = np.fft.fft(taps, n=N).astype(np.complex64)
        H = H / np.sqrt(np.mean(np.abs(H) ** 2) + 1e-12)
        if self.fft_shift:
            H = np.fft.fftshift(H)
        return H

class LogDistancePropagator:
    """Log-distance path-loss baseline. Same interface as UNetTdlPropagator
    but purely analytic and geometry-blind (no walls):

        PL(d) = PL(d0) + 10 * n * log10(d / d0)

    with PL(d0) = FSPL(d0, fc). n = 2.0 recovers Friis; typical indoor
    NLOS values are 3.0-4.0. Like COST-231, it has no delay information,
    so tau_rms is a fixed default and excess delay is 0."""

    def __init__(self, exponent=3.0, ref_dist_m=1.0, no_path_wb=199.5,
                 fft_shift=False, default_tau_rms_ns=5.0,
                 k_los_db=6.0, k_nlos_db=2.0):
        self.exponent = float(exponent)
        self.ref_dist_m = max(float(ref_dist_m), 1e-3)
        self.no_path_wb = float(no_path_wb)
        self.fft_shift = bool(fft_shift)
        self.default_tau_rms_ns = float(default_tau_rms_ns)
        self.k_los_db = float(k_los_db)
        self.k_nlos_db = float(k_nlos_db)
        self._cached_tx_pos = None

    def attach_scene(self, sionna_scene, bbox, fc_hz, fft_size,
                     subcarrier_spacing_hz, frequencies_hz):
        self.fc_hz = float(fc_hz)
        self.fft_size = int(fft_size)
        self.subcarrier_spacing_hz = float(subcarrier_spacing_hz)
        self.frequencies = np.asarray(frequencies_hz, dtype=np.float64)

        off = self.frequencies - self.fc_hz
        if (off[0] < 0) and (off[-1] > 0) and np.all(np.diff(off) > 0):
            self.fft_shift = True
        else:
            self.fft_shift = False

        lam = 299792458.0 / self.fc_hz
        self._pl0_db = 20.0 * np.log10(4.0 * np.pi * self.ref_dist_m / lam)
        print(f"[LogDist] n={self.exponent}, d0={self.ref_dist_m} m, "
              f"PL(d0)={self._pl0_db:.2f} dB")

    def predict_for_tx(self, tx_pos_xyz):
        self._cached_tx_pos = np.asarray(tx_pos_xyz, dtype=np.float32).reshape(3)

    def sample_heads(self, rx_pos_xyz):
        assert self._cached_tx_pos is not None, "call predict_for_tx() first"
        rx = np.asarray(rx_pos_xyz, dtype=np.float32).reshape(3)
        d = max(float(np.linalg.norm(rx - self._cached_tx_pos)), self.ref_dist_m)
        wb_db = self._pl0_db + 10.0 * self.exponent * np.log10(d / self.ref_dist_m)
        return float(wb_db), float(self.default_tau_rms_ns), 0.0

    def synthesize_cfr(self, tau_rms_ns: float, mi_scene, tx_xyz, rx_xyz,
                       seed: int, synthetic=False) -> np.ndarray:
        # identical synthesis model to the other surrogates so CFR
        # differences reflect only tau/wb, not the generator
        if not synthetic:
            H = np.ones(self.fft_size, dtype=np.complex64)
            if self.fft_shift:
                H = np.fft.fftshift(H)
            return H

        def is_los_mi(mi_scene, tx_xyz, rx_xyz, eps=1e-2):
            tx = np.asarray(tx_xyz, dtype=np.float32)
            rx = np.asarray(rx_xyz, dtype=np.float32)
            dvec = rx - tx
            dist = float(np.linalg.norm(dvec))
            if dist < 1e-6:
                return True
            direction = dvec / dist
            o = mi.Point3f(tx + eps * direction)
            d = mi.Vector3f(direction)
            ray = mi.Ray3f(o, d)
            ray.maxt = mi.Float(max(dist - 2 * eps, 0.0))
            si = mi_scene.ray_intersect(ray, mi.RayFlags.Minimal, False, True)
            return not bool(si.is_valid())

        los = is_los_mi(mi_scene, tx_xyz, rx_xyz) if mi_scene is not None else True
        K_db = self.k_los_db if los else self.k_nlos_db

        N = self.fft_size
        df = self.subcarrier_spacing_hz
        Ts = 1.0 / (N * df)
        tau_rms = max(float(tau_rms_ns), 1e-3) * 1e-9

        K_lin = 10.0 ** (float(K_db) / 10.0)
        if K_lin > 0.0:
            p_spec = K_lin / (K_lin + 1.0)
            p_diff = 1.0 / (K_lin + 1.0)
            tau_d = tau_rms * (K_lin + 1.0) / np.sqrt(2.0 * K_lin + 1.0)
        else:
            p_spec = 0.0
            p_diff = 1.0
            tau_d = tau_rms

        L = int(np.clip(np.ceil(6.0 * tau_d / Ts), 1, N))
        t = np.arange(L, dtype=np.float64) * Ts
        p = np.exp(-t / max(tau_d, 1e-12))
        p = p / (p.sum() + 1e-12)

        rng = np.random.default_rng(seed)
        w = (rng.standard_normal(L) + 1j * rng.standard_normal(L)) * np.sqrt(0.5)
        taps = w * np.sqrt(p_diff * p)
        if p_spec > 0.0:
            phi = rng.uniform(0.0, 2.0 * np.pi)
            taps[0] += np.sqrt(p_spec) * np.exp(1j * phi)

        H = np.fft.fft(taps, n=N).astype(np.complex64)
        H = H / np.sqrt(np.mean(np.abs(H) ** 2) + 1e-12)
        if self.fft_shift:
            H = np.fft.fftshift(H)
        return H


class SionnaEnv:

    # just compute the given single point-to-point channel
    MODE_P2P        = 1
    # extend the given P2P channel to include all receivers to take broadcast nature of wireless into account
    MODE_P2MP       = 2
    # compute also future channels; used only if the transmitter is static (receivers are mobile)
    MODE_P2MP_LAH   = 3

    """
    This class represents the Sionna component of ns3sionna. It represents the environment where the node
    placement, mobility is controlled from the client component of ns3sionna. For IPC ZMQ is used.

    author: Pilz, Zubow
    """
    def __init__(self, model_folder='./models/', rt_fast=False, default_mode=MODE_P2P, rt_max_parallel_links=256, est_csi=True, 
                 use_unet=False, unet_run="residual_cost_v2", unet_device="cuda", unet_no_path_wb=199.5, unet_y_wb_idx=0, unet_y_tau_rms_idx=2, unet_y_excess_idx=-1,
                 unet_cov_thresh=0.5, unet_tx_cache=256,
                 VERBOSE=True,
                 CHECKS_ENABLED=True,
                 mobility_trace_in: str = "",
                 use_cost231=False, cost231_tau_rms_ns=5.0,
                 use_logdist=False, logdist_exponent=3.0, logdist_ref_m=1.0,
                 logdist_tau_rms_ns=5.0,
                 synthetic_cfr=False,
                 cfr_k_los_db=6.0, cfr_k_nlos_db=2.0,
                 rt_samples_per_src=None, rt_flush_every=8):
        self.model_folder = model_folder
        self.rt_fast = rt_fast
        if rt_fast:
            self.rt_max_depth = 3  # very small
            self.rt_samples_per_src = 10 ** 6
            self.rt_los = True  # compute and include the direct Line-of-Sight path when it exists
            self.rt_specular_reflection = True  # Can rays bounce off surfaces?
            self.rt_diffuse_reflection = False
            self.rt_refraction = True  # Can rays pass through materials?
            self.rt_synthetic_array = False  # Set True for fast simulation using one ray trace for whole array; per-element effects computed analytically
            self.rt_diffraction = False  # costly
            self.rt_edge_diffraction = False  # rays that bend around edges
            self.rt_diffraction_lit_region = False  # higher physical accuracy; for mmWave or THz channels
        else: # realistic but slow
            self.rt_max_depth = 10  # sufficient even for rich multipath
            # RAISED 1e6 -> 1e7 to match the v2 training-label convention.
            # samples_per_src is shared across every receiver placed in the
            # solve (up to rt_max_parallel_links=256 in LAH mode); at 1e6 the
            # per-receiver diffuse/higher-order path capture is diluted,
            # systematically darkening deep-NLOS ground truth (the same
            # mechanism behind the +8.8 dB label bias). Convergence backed by
            # rt_batch_probe.py (solo/crowd x 1e6/1e7).
            self.rt_samples_per_src = 10 ** 6
            self.rt_max_num_paths_per_src = 10**7
            self.rt_los = True  # compute and include the direct Line-of-Sight path when it exists
            self.rt_specular_reflection = True  # Can rays bounce off surfaces?
            self.rt_diffuse_reflection = True
            self.rt_refraction = True  # Can rays pass through materials?
            self.rt_synthetic_array = False  # Set True for fast simulation using one ray trace for whole array; per-element effects computed analytically
            self.rt_diffraction = True  # costly
            self.rt_edge_diffraction = True  # rays that bend around edges
            self.rt_diffraction_lit_region = True  # higher physical accuracy; for mmWave or THz channels

        # explicit override (A/B probes, convergence sweeps)
        if rt_samples_per_src is not None:
            self.rt_samples_per_src = int(rt_samples_per_src)

        # Dr.Jit allocator hygiene: the malloc cache grows by a few MB per
        # PathSolver call and is never trimmed. Over hundreds of CSI requests
        # this eats the VRAM headroom on an 8 GB WDDM GPU and triggers paging
        # stalls (~35x slowdown) -- same failure mode as the label-generation
        # pipeline, fixed the same way: flush every N solves, INSIDE the
        # serving loop, not per-scene.
        self._rt_flush_every = max(1, int(rt_flush_every))
        self._rt_solves_since_flush = 0

        # default mode
        self.default_mode = default_mode

        # maximum number of parallel computations (needed to fit GPU)
        self.rt_max_parallel_links = rt_max_parallel_links

        # estimate small-scale fading
        self.est_csi = est_csi

        self.use_unet = use_unet
        self.unet_run = unet_run
        self.unet_device = unet_device
        self.unet_no_path_wb = unet_no_path_wb
        # DEPRECATED: head indices are fixed by the residual-over-cost model
        # ([r, tau, coverage_logit]); these are kept only for CLI compatibility
        # and are NOT passed to the propagator anymore.
        self.unet_y_wb_idx = unet_y_wb_idx
        self.unet_y_tau_rms_idx = unet_y_tau_rms_idx
        self.unet_y_excess_idx = unet_y_excess_idx
        self.unet_cov_thresh = float(unet_cov_thresh)
        self.unet_tx_cache = int(unet_tx_cache)
        self._unet = None

        if not self.use_unet:
            print(f'Init ns3sionna with rt_fast={rt_fast}, est_csi={est_csi}, '
                  f'rt_samples_per_src={self.rt_samples_per_src:.0e}, '
                  f'rt_flush_every={self._rt_flush_every}')

        # cost231
        self.use_cost231 = use_cost231
        self.cost231_tau_rms_ns = cost231_tau_rms_ns
        self._cost231 = None

        # log-distance baseline
        self.use_logdist = use_logdist
        self.logdist_exponent = float(logdist_exponent)
        self.logdist_ref_m = float(logdist_ref_m)
        self.logdist_tau_rms_ns = float(logdist_tau_rms_ns)
        self._logdist = None

        # send synthetic (tau-shaped Rician) CFRs instead of flat ones;
        # needed for CFR-level validation against RT
        self.synthetic_cfr = bool(synthetic_cfr)
        self.cfr_k_los_db = float(cfr_k_los_db)
        self.cfr_k_nlos_db = float(cfr_k_nlos_db)

        self.VERBOSE = VERBOSE
        self.CHECKS_ENABLED = CHECKS_ENABLED

        # check GPU support
        self.gpus = tf.config.list_physical_devices("GPU")

        if len(self.gpus) > 0:
            print("GPU support detected:", self.gpus)
        else:
            print("Using CPU backend")

        self.disp_r = 10
        # storing information about every node under simulation
        self.node_info = {}
        # all node which are currently placed on the scene
        self.placed_radio_node_names = []

        self._tim = {
            "rt":   {"n": 0, "t": 0.0},
            "unet": {"n": 0, "t": 0.0},
            "rt_lah":   {"n": 0, "t": 0.0},
            "unet_lah": {"n": 0, "t": 0.0},
        }
        self._tim_print_every = 20

        self.mobility_trace_in = mobility_trace_in
        self._trace = None
        self._use_trace = False


    def init_simulation_env(self, sim_init_msg):
        '''
        Initializes the Sionna environment
        :param sim_init_msg: the received ZMQ message
        :return: (success, error_msg)
        '''
        if self.VERBOSE:
            print_sim_init(sim_init_msg)

        # Load the sionna scene
        filepath = os.path.join(self.model_folder, sim_init_msg.scene_fname)
        try:
            self.scene = load_scene(filepath)
        except Exception as e:
            return False, "Failed to load scene file in: " + filepath + ", error: " + str(e)

        self.bbox = self.scene.mi_scene.bbox()

        if self.VERBOSE:
            # show some stats about the scene
            dx = self.bbox.max.x - self.bbox.min.x
            dy = self.bbox.max.y - self.bbox.min.y
            dz = self.bbox.max.z - self.bbox.min.z
            print(f'Scenario with dx={dx:.2f}, dy={dy:.2f}, dz={dz:.2f}')

        # set mode/submode if valid
        if sim_init_msg.mode > -1:
            self.mode = sim_init_msg.mode
        else:
            self.mode = self.default_mode # default mode

        # todo: use submode
        if sim_init_msg.sub_mode > -1:
            self.sub_mode = sim_init_msg.sub_mode
        else:
            self.sub_mode = self.rt_max_parallel_links

        self.time_evo_model = sim_init_msg.time_evo_model

        # Set scene parameters
        self.scene.frequency = sim_init_msg.frequency * 1e6
        self.scene.bandwidth = sim_init_msg.channel_bw * 1e6 # max channel bandwidth

        self.fc = sim_init_msg.frequency * 1e6
        self.fft_size = sim_init_msg.fft_size  # max FFT size
        # todo: min Tc is not used
        self.min_coherence_time_ms = sim_init_msg.min_coherence_time_ms # min Tc
        self.subcarrier_spacing = sim_init_msg.subcarrier_spacing # in Hz

        print(f'Operating in mode: {self.mode}, sub_mode: {self.sub_mode}, time_evo_model: {self.time_evo_model}'
              f', fc: {sim_init_msg.frequency} MHz, B: {sim_init_msg.channel_bw} MHz, FFT size: {self.fft_size}')

        # Subcarrier frequencies
        self.frequencies = subcarrier_frequencies(num_subcarriers=self.fft_size, subcarrier_spacing=self.subcarrier_spacing)

        # Set the random seed for reproducibility
        np.random.seed(sim_init_msg.seed)
        tf.random.set_seed(sim_init_msg.seed)
        self.my_seed = sim_init_msg.seed

        # --- Optional deterministic mobility from trace ---
        if getattr(self, "mobility_trace_in", ""):
            print(f"[TRACE] Loading mobility trace: {self.mobility_trace_in}")
            self._trace = _read_mobility_trace_csv(self.mobility_trace_in)
            self._use_trace = True
        else:
            self._trace = None
            self._use_trace = False

        ##SONIC
        if self.use_unet:
            if self._unet is None:
                self._unet = UNetTdlPropagator(
                    run_dir=self.unet_run,
                    device=self.unet_device,
                    no_path_wb=self.unet_no_path_wb,
                    cov_thresh=self.unet_cov_thresh,
                    tx_cache_size=self.unet_tx_cache,
                    k_los_db=self.cfr_k_los_db,
                    k_nlos_db=self.cfr_k_nlos_db,
                    # IMPORTANT: set these to match train_delta_tau CFG
                    scale_m=0.625,
                    z_step_cells=1.0,
                    z_margin_m=0.625 * 0.5,   # if training used z_margin=0.5 cells
                    origin_xy_mode="bbox_min", # or "zero" if your scenes are 0-based
                )
            self._unet.attach_scene(self.scene, self.bbox, self.fc, self.fft_size, self.subcarrier_spacing, self.frequencies)

        elif self.use_cost231:
            if self._cost231 is None:
                self._cost231 = Cost231Propagator(
                    scale_m=0.625,
                    z_step_cells=1.0,
                    z_margin_m=0.625 * 0.5,
                    origin_xy_mode="bbox_min",
                    default_tau_rms_ns=self.cost231_tau_rms_ns,
                    k_los_db=self.cfr_k_los_db,
                    k_nlos_db=self.cfr_k_nlos_db,
                )
            self._cost231.attach_scene(self.scene, self.bbox, self.fc, self.fft_size, self.subcarrier_spacing, self.frequencies)

        elif self.use_logdist:
            if self._logdist is None:
                self._logdist = LogDistancePropagator(
                    exponent=self.logdist_exponent,
                    ref_dist_m=self.logdist_ref_m,
                    default_tau_rms_ns=self.logdist_tau_rms_ns,
                    k_los_db=self.cfr_k_los_db,
                    k_nlos_db=self.cfr_k_nlos_db,
                )
            self._logdist.attach_scene(self.scene, self.bbox, self.fc, self.fft_size, self.subcarrier_spacing, self.frequencies)

        # configure mobility models
        self._init_mobility(sim_init_msg)

        # for mode 3 if only constant speed model supported
        if self.mode == SionnaEnv.MODE_P2MP_LAH:
            speed_arr = []
            for node_id in list(self.node_info.keys()):
                if isinstance(self.node_info[node_id], RandomWalkMobility):
                    if self.node_info[node_id].speed != RandomWalkMobility.SPEED_CONSTANT:
                        warnings.warn(f"Only constant speed model is supported when using mode P2MP(LAH); switching to mode P2P.", UserWarning)
                        self.mode = SionnaEnv.MODE_P2MP
                        break
                    else:
                        speed_arr.append(self.node_info[node_id].speed_params[0])
            # compute coherence time assuming worst case: fastest nodes move away from each other
            speed_arr.sort(reverse=True)
            if len(speed_arr) >= 2:
                self.chan_coh_time_mode3 = compute_coherence_time(speed_arr[0] + speed_arr[1], self.fc, model='rappaport2')
            elif len(speed_arr) == 1:
                self.chan_coh_time_mode3 = compute_coherence_time(speed_arr[0], self.fc, model='rappaport2')
            else:
                self.chan_coh_time_mode3 = MAX_COHERENCE_TIME

            print(f'Running mode=3 w/ Tc: {self.chan_coh_time_mode3/1e6}ms')

        # Configure antenna array for all transmitters/receivers
        self.scene.tx_array = PlanarArray(num_rows=1, num_cols=1, vertical_spacing=0.5, horizontal_spacing=0.5,
                                     pattern="iso", polarization="V")

        self.scene.rx_array = PlanarArray(num_rows=1, num_cols=1, vertical_spacing=0.5, horizontal_spacing=0.5,
                                     pattern="iso", polarization="V")

        # set current sim time to 0ns
        self.sim_time = 0

        return True, "OK"


    def _rt_after_solve(self, n_solves: int = 1):
        """Call after each PathSolver solve. Flushes the Dr.Jit malloc cache
        every self._rt_flush_every solves so cached allocations can't
        accumulate across CSI requests (WDDM paging -> stalls)."""
        self._rt_solves_since_flush += int(n_solves)
        if self._rt_solves_since_flush >= self._rt_flush_every:
            try:
                _dr.flush_malloc_cache()
            except Exception as e:
                print(f"[RT] dr.flush_malloc_cache() failed: {e}", flush=True)
            self._rt_solves_since_flush = 0

    def _tim_add(self, key: str, dt_s: float):
        d = self._tim[key]
        d["n"] += 1
        d["t"] += dt_s
        if (d["n"] % self._tim_print_every) == 0:
            avg_ms = 1e3 * d["t"] / max(1, d["n"])
            print(f"[TIM] {key}: n={d['n']} avg={avg_ms:.2f} ms", flush=True)

    def compute_cfr(self, csi_req, reply_wrapper):

        tx_node_id = csi_req.tx_node
        rx_node_id = csi_req.rx_node

        if self.mode == SionnaEnv.MODE_P2MP_LAH:
            if isinstance(self.node_info[tx_node_id], RandomWalkMobility) and isinstance(self.node_info[rx_node_id], ConstantMobility):
                # Exploit channel reciprocity - swap mobile TX with static RX
                csi_req.tx_node = rx_node_id
                csi_req.rx_node = tx_node_id

        # check if mode 3 can be used
        if self.mode == SionnaEnv.MODE_P2MP_LAH and isinstance(self.node_info[csi_req.tx_node], ConstantMobility):
            # mode=3 is feasible if TX is fixed
            return self.compute_cfr_with_lookahead(csi_req, reply_wrapper)
        else:
            req_mode = self.mode
            # mode=1/2 or if TX node is mobile
            if self.mode == SionnaEnv.MODE_P2MP_LAH:
                req_mode = SionnaEnv.MODE_P2MP
                print(f'Fallback to mode={req_mode} as TX node is mobile')

            return self.compute_cfr_classic(csi_req, reply_wrapper, req_mode)


    def compute_cfr_with_lookahead(self, csi_req, reply_wrapper):
        '''
        Compute the requested CFR
        :param csi_req: received CSI request (ZMQ)
        :param reply_wrapper: the response
        '''

        if self.VERBOSE:
            print_csi_request(csi_req)

        tx_node_id = csi_req.tx_node
        rx_node_id = csi_req.rx_node # this rx node must be included in result set
        req_sim_time = csi_req.time # we need CFR at that point in time [ns]

        assert self.time_evo_model == 'position'

        # execute mobility

        # update position of all nodes
        nodes_to_update = list(self.node_info.keys())

        # compute look-ahead
        look_ahead = math.floor(self.sub_mode / (len(nodes_to_update) - 1))
        if look_ahead <= 0:
            # behave like classic P2MP at the requested time
            return self.compute_cfr_classic(csi_req, reply_wrapper, SionnaEnv.MODE_P2MP)

        print(f'compute CFR to #RX={len(nodes_to_update) - 1} with LAH={look_ahead}')

        # sim future node positions
        lah_time_vec = []
        for lah_i in range(look_ahead):
            # Advance all relevant nodes to the *absolute* time req_sim_time
            # (this function should internally compute dt = req_sim_time - self.sim_time,
            #  update positions/velocities, and set self.sim_time = req_sim_time)
            self._advance_nodes_to_time(req_sim_time, nodes_to_update)

            # Now we are "at" req_sim_time
            lah_time_vec.append(req_sim_time)

            # Compute Tc at this time using the updated node states
            csi_tc_arr = []
            for node_id in nodes_to_update:
                if node_id == tx_node_id:
                    continue

                tc = coherence_from_velocities(
                    self.node_info[node_id].velocity,
                    self.node_info[tx_node_id].velocity,
                    self.fc,
                    pos_tx=self.node_info[node_id].pos,
                    pos_rx=self.node_info[tx_node_id].pos,
                )
                csi_tc_arr.append(tc)

            Tc_p2mp = int(np.min(np.asarray(csi_tc_arr))) if len(csi_tc_arr) else MAX_COHERENCE_TIME

            # Next lookahead time
            req_sim_time = req_sim_time + Tc_p2mp



        # place TX and RX nodes together with their future positions
        rx_nodes = [nid for nid in nodes_to_update if nid != tx_node_id]

        
        if self.use_unet:

            return self._compute_cfr_with_lookahead_unet(lah_time_vec=lah_time_vec,
                                                         tx_node_id=tx_node_id,
                                                         rx_nodes=rx_nodes,
                                                         reply_wrapper=reply_wrapper)
        
        elif self.use_cost231:

            return self._compute_cfr_with_lookahead_surrogate(
                lah_time_vec=lah_time_vec,
                tx_node_id=tx_node_id,
                rx_nodes=rx_nodes,
                reply_wrapper=reply_wrapper,
                propagator=self._cost231,
                no_path_wb=self._cost231.no_path_wb,
                is_delta=False,
                label="COST231-LAH",
            )

        elif self.use_logdist:

            return self._compute_cfr_with_lookahead_surrogate(
                lah_time_vec=lah_time_vec,
                tx_node_id=tx_node_id,
                rx_nodes=rx_nodes,
                reply_wrapper=reply_wrapper,
                propagator=self._logdist,
                no_path_wb=self._logdist.no_path_wb,
                is_delta=False,
                label="LOGDIST-LAH",
            )

        self._place_tx_rx_nodes_with_lah(lah_time_vec, tx_node_id, rx_nodes)

        # create pathsolver; todo: check reuse
        p_solver  = PathSolver()

        # Compute propagation paths
        paths = p_solver(scene=self.scene,
                         max_depth=self.rt_max_depth,
                         samples_per_src=self.rt_samples_per_src,
                         max_num_paths_per_src=self.rt_max_num_paths_per_src,
                         los=self.rt_los,
                         specular_reflection=self.rt_specular_reflection,  # Can rays bounce off surfaces?
                         diffuse_reflection=self.rt_diffuse_reflection,
                         refraction=self.rt_refraction,  # Can rays pass through materials?
                         synthetic_array=self.rt_synthetic_array,
                         diffraction=self.rt_diffraction,  # costly
                         edge_diffraction=self.rt_edge_diffraction,  # rays that bend around edges
                         diffraction_lit_region=self.rt_diffraction_lit_region)  # higher physical accuracy

        # AZU: sampling_frequency is only used if num_time_steps > 1
        # a: shape [num_rx, num_rx_ant, num_tx, num_tx_ant, num_paths, num_time_steps],
        a, tau = paths.cir(sampling_frequency=1e9, normalize_delays=False, out_type="numpy")

        # shape: [num_rx, num_rx_ant, num_tx, num_tx_ant, num_ofdm_symbols, num_subcarriers]
        h_raw = paths.cfr(frequencies=self.frequencies,
                  sampling_frequency=1.0,  # not used
                  num_time_steps=1,
                  normalize_delays=True,
                  # If set to True, path delays are normalized such that the first path between any pair of
                  # antennas of a transmitter and receiver arrives at tau=0
                  normalize=False,  # Normalize energy
                  out_type="numpy")

        # RT outputs are numpy now; allow the allocator cache to be trimmed
        self._rt_after_solve()

        # max single transmitter
        assert h_raw.shape[2] == 1
        num_computed_lnks = h_raw.shape[0]

        # Create ZMQ response
        chan_response = reply_wrapper.channel_state_response

        Tc_p2mp_lah = []
        for lah_time_idx, lah_time in enumerate(lah_time_vec): # iterate over time
            csi = chan_response.csi.add()

            csi.start_time = lah_time
            # tx node is fixed
            tx_pos = self.node_info[tx_node_id].pos
            csi.tx_node.id = tx_node_id
            csi.tx_node.position.x = tx_pos[0]
            csi.tx_node.position.y = tx_pos[1]
            csi.tx_node.position.z = tx_pos[2]

            # get receiver(s)
            csi_tc_arr = []
            for curr_rx_id, curr_rx_node in enumerate(rx_nodes):

                rx_id = lah_time_idx * len(rx_nodes) + curr_rx_id

                lnk_tau = np.squeeze(tau[rx_id, :, :, :, :])
                lnk_delay = int(round(np.min(lnk_tau[lnk_tau >= 0] * 1e9), 0))

                h = np.squeeze(h_raw[rx_id, :, :, :, :, ])

                # see Parseval's theorem
                lnk_loss = float(-10 * np.log10(np.mean(np.abs(h) ** 2)))

                # for frequency-selective channel
                power = np.mean(np.abs(h) ** 2)  # shape [batch_size, 1, 1, 1]

                h_normalized = h / np.sqrt(power)

                # plausibility test
                if self.CHECKS_ENABLED:
                    power_normalized = np.mean(np.abs(h_normalized) ** 2)
                    assert math.isclose(power_normalized, 1.0, rel_tol=1e-3)   # Should be close to 1

                if self.VERBOSE:
                    print(f'{lah_time/1e9}s: {tx_node_id}->{curr_rx_node} lnk_delay = {lnk_delay}ns, wb_loss = {lnk_loss:.3f}dB, CFR shape: {h_normalized.shape}')

                rx_node_info = csi.rx_nodes.add()
                rx_pos = self.node_info[curr_rx_node].get_pos_at(lah_time)
                rx_node_info.id = curr_rx_node
                rx_node_info.position.x = rx_pos[0]
                rx_node_info.position.y = rx_pos[1]
                rx_node_info.position.z = rx_pos[2]
                rx_node_info.delay = lnk_delay
                rx_node_info.wb_loss = lnk_loss

                if self.est_csi:
                    rx_node_info.frequencies.extend(self.frequencies.tolist())
                    rx_node_info.csi_imag.extend(np.imag(h_normalized).tolist())
                    rx_node_info.csi_real.extend(np.real(h_normalized).tolist())

                tc = coherence_from_velocities(self.node_info[curr_rx_node].get_velo_at(lah_time),
                                                self.node_info[tx_node_id].velocity, self.fc,
                                                pos_tx=self.node_info[curr_rx_node].get_pos_at(lah_time),
                                                pos_rx=self.node_info[tx_node_id].pos)

                rx_node_info.end_time2 = csi.start_time + tc
                csi_tc_arr.append(tc)

            # take the worst case Tc from all RX nodes
            Tc_p2mp = int(np.min(np.asarray(csi_tc_arr)))
            Tc_p2mp_lah.append(Tc_p2mp)
            csi.end_time = csi.start_time + Tc_p2mp - 1 # -1ns to have non-overlapping intervals

        print(f'{self.sim_time / 1e9}s: Computed CSI with Tc: {np.round(np.asarray(Tc_p2mp_lah) / 1e6,2)}ms, #links: {num_computed_lnks}')

        return num_computed_lnks


    def _compute_cfr_with_lookahead_unet(self, lah_time_vec: list, tx_node_id: int, rx_nodes: list, reply_wrapper):
        """Compute LAH CSI using the UNet surrogate (no ray tracing).

        Mirrors the response format of the RT-based LAH path:
        - One CSI sample per time in `lah_time_vec`
        - Each sample contains all RX nodes
        - CFR is normalized (mean |H|^2 ~= 1) when `self.est_csi` is enabled

        NOTE (residual-over-cost model): sample_heads() returns the FULL
        wideband loss (wb_cost + gated residual, or the no-path sentinel
        when the coverage head says there is no path). Do NOT add FSPL.
        """

        # Precompute lists to avoid huge Python overhead in inner loops
        freqs_list = self.frequencies.tolist()
        zeros_f_list = np.zeros((self.fft_size,), dtype=np.float32).tolist()

        # TX must be fixed in MODE_P2MP_LAH
        tx_pos = np.array(self.node_info[tx_node_id].pos, dtype=np.float32)
        self._unet.predict_for_tx(tx_pos)

        chan_response = reply_wrapper.channel_state_response

        Tc_p2mp_lah = []
        num_links = 0

        for lah_time in lah_time_vec:
            csi = chan_response.csi.add()
            csi.start_time = int(lah_time)

            # TX info (fixed)
            csi.tx_node.id = int(tx_node_id)
            csi.tx_node.position.x = float(tx_pos[0])
            csi.tx_node.position.y = float(tx_pos[1])
            csi.tx_node.position.z = float(tx_pos[2])

            csi_tc_arr = []
            for curr_rx_node in rx_nodes:
                rx_pos = np.array(self.node_info[curr_rx_node].get_pos_at(lah_time), dtype=np.float32)

                # FULL wideband loss directly from the surrogate (no FSPL added)
                wb_db, tau_rms_ns, excess_ns = self._unet.sample_heads(rx_pos)
                wb_db = float(wb_db)
                d_m = float(np.linalg.norm(tx_pos - rx_pos))

                base_ns = d_m / 299792458.0 * 1e9
                ex = 0.0 if excess_ns is None else float(max(0.0, excess_ns))
                delay_ns = base_ns + ex

                no_path = wb_db >= (self._unet.no_path_wb - 1e-3)

                rx_node_info = csi.rx_nodes.add()
                rx_node_info.id = int(curr_rx_node)
                rx_node_info.position.x = float(rx_pos[0])
                rx_node_info.position.y = float(rx_pos[1])
                rx_node_info.position.z = float(rx_pos[2])

                if no_path:
                    rx_node_info.delay = 0
                    rx_node_info.wb_loss = float(self._unet.no_path_wb)
                    if self.est_csi:
                        rx_node_info.frequencies.extend(freqs_list)
                        rx_node_info.csi_imag.extend(zeros_f_list)
                        rx_node_info.csi_real.extend(zeros_f_list)
                else:
                    rx_node_info.delay = int(round(delay_ns))
                    rx_node_info.wb_loss = float(wb_db)

                    if self.est_csi:
                        seed = (
                            (int(self.my_seed) * 1315423911)
                            ^ (int(tx_node_id) * 2654435761)
                            ^ (int(curr_rx_node) * 97531)
                            ^ (int(lah_time) & 0xFFFFFFFF)
                        )
                        h_norm = self._unet.synthesize_cfr(tau_rms_ns=tau_rms_ns, mi_scene=self.scene.mi_scene, tx_xyz=tx_pos, rx_xyz=rx_pos, seed=seed, synthetic=self.synthetic_cfr)
                        # mag_db = 20*np.log10(np.abs(h_norm) + 1e-12)
                        # print("ripple_pp_dB=", float(mag_db.max() - mag_db.min()),
                        #     "ripple_std_dB=", float(mag_db.std()))

                        if self.CHECKS_ENABLED:
                            p = float(np.mean(np.abs(h_norm) ** 2))
                            assert abs(p - 1.0) < 1e-2

                        rx_node_info.frequencies.extend(freqs_list)
                        rx_node_info.csi_imag.extend(np.imag(h_norm).tolist())
                        rx_node_info.csi_real.extend(np.real(h_norm).tolist())

                tc = coherence_from_velocities(
                    self.node_info[curr_rx_node].get_velo_at(lah_time),
                    self.node_info[tx_node_id].velocity,
                    self.fc,
                    pos_tx=self.node_info[curr_rx_node].get_pos_at(lah_time),
                    pos_rx=self.node_info[tx_node_id].pos,
                )
                rx_node_info.end_time2 = int(csi.start_time + tc)
                csi_tc_arr.append(tc)
                num_links += 1

            Tc_p2mp = int(np.min(np.asarray(csi_tc_arr)))
            Tc_p2mp_lah.append(Tc_p2mp)
            csi.end_time = int(csi.start_time + Tc_p2mp - 1)  # non-overlapping intervals

        print(
            f"{self.sim_time / 1e9}s: Computed CSI with Tc: "
            f"{np.round(np.asarray(Tc_p2mp_lah) / 1e6, 2)}ms, #links: {num_links} (UNet-LAH)"
        )
        return num_links
    
    def _compute_cfr_with_lookahead_surrogate(self, lah_time_vec, tx_node_id, rx_nodes, reply_wrapper,
                                               propagator, no_path_wb, is_delta, label):
        """Generic LAH path for any propagator with predict_for_tx/sample_heads/synthesize_cfr interface.

        is_delta=True is only valid for LEGACY delta-over-Friis propagators.
        The residual-over-cost UNet and COST-231 both return the full
        wideband loss (is_delta=False)."""

        def fspl_db(d_m, fc_hz):
            d_m = max(float(d_m), 1e-6)
            lam = 299792458.0 / float(fc_hz)
            return 20.0 * np.log10(4.0 * np.pi * d_m / lam)

        freqs_list = self.frequencies.tolist()
        zeros_f_list = np.zeros((self.fft_size,), dtype=np.float32).tolist()

        tx_pos = np.array(self.node_info[tx_node_id].pos, dtype=np.float32)
        propagator.predict_for_tx(tx_pos)

        chan_response = reply_wrapper.channel_state_response
        Tc_p2mp_lah = []
        num_links = 0

        for lah_time in lah_time_vec:
            csi = chan_response.csi.add()
            csi.start_time = int(lah_time)
            csi.tx_node.id = int(tx_node_id)
            csi.tx_node.position.x = float(tx_pos[0])
            csi.tx_node.position.y = float(tx_pos[1])
            csi.tx_node.position.z = float(tx_pos[2])

            csi_tc_arr = []
            for curr_rx_node in rx_nodes:
                rx_pos = np.array(self.node_info[curr_rx_node].get_pos_at(lah_time), dtype=np.float32)

                head_0, tau_rms_ns, excess_ns = propagator.sample_heads(rx_pos)
                d_m = float(np.linalg.norm(tx_pos - rx_pos))

                if is_delta:
                    wb_db = float(fspl_db(d_m, self.fc) + float(head_0))
                else:
                    wb_db = float(head_0)

                base_ns = d_m / 299792458.0 * 1e9
                ex = 0.0 if excess_ns is None else float(max(0.0, excess_ns))

                no_path = wb_db >= (no_path_wb - 1e-3)

                rx_node_info = csi.rx_nodes.add()
                rx_node_info.id = int(curr_rx_node)
                rx_node_info.position.x = float(rx_pos[0])
                rx_node_info.position.y = float(rx_pos[1])
                rx_node_info.position.z = float(rx_pos[2])

                if no_path:
                    rx_node_info.delay = 0
                    rx_node_info.wb_loss = float(no_path_wb)
                    if self.est_csi:
                        rx_node_info.frequencies.extend(freqs_list)
                        rx_node_info.csi_imag.extend(zeros_f_list)
                        rx_node_info.csi_real.extend(zeros_f_list)
                else:
                    rx_node_info.delay = int(round(base_ns + ex))
                    rx_node_info.wb_loss = float(wb_db)
                    if self.est_csi:
                        seed = (
                            (int(self.my_seed) * 1315423911)
                            ^ (int(tx_node_id) * 2654435761)
                            ^ (int(curr_rx_node) * 97531)
                            ^ (int(lah_time) & 0xFFFFFFFF)
                        )
                        h_norm = propagator.synthesize_cfr(
                            tau_rms_ns=tau_rms_ns, mi_scene=self.scene.mi_scene,
                            tx_xyz=tx_pos, rx_xyz=rx_pos, seed=seed,
                            synthetic=self.synthetic_cfr,
                        )
                        rx_node_info.frequencies.extend(freqs_list)
                        rx_node_info.csi_imag.extend(np.imag(h_norm).tolist())
                        rx_node_info.csi_real.extend(np.real(h_norm).tolist())

                tc = coherence_from_velocities(
                    self.node_info[curr_rx_node].get_velo_at(lah_time),
                    self.node_info[tx_node_id].velocity, self.fc,
                    pos_tx=self.node_info[curr_rx_node].get_pos_at(lah_time),
                    pos_rx=self.node_info[tx_node_id].pos,
                )
                rx_node_info.end_time2 = int(csi.start_time + tc)
                csi_tc_arr.append(tc)
                num_links += 1

            Tc_p2mp = int(np.min(np.asarray(csi_tc_arr)))
            Tc_p2mp_lah.append(Tc_p2mp)
            csi.end_time = int(csi.start_time + Tc_p2mp - 1)

        print(f"{self.sim_time / 1e9}s: Computed CSI with Tc: "
              f"{np.round(np.asarray(Tc_p2mp_lah) / 1e6, 2)}ms, #links: {num_links} ({label})")
        return num_links


    def _place_tx_rx_nodes_with_lah(self, lah_time_vec: list, tx_node: int, rx_nodes: list):
        '''
        Place the given nodes together with their lookahead positions in the scenario
        :param lah_time_vec: time vector containing the lah time vector
        :param tx_node: the transmitting node
        :param rx_nodes: the receiver nodes
        '''

        # remove old tx and rx nodes
        for placed_node in self.placed_radio_node_names:
            self.scene.remove(placed_node)
        self.placed_radio_node_names.clear()

        # only supported if TX is fixed
        fixed_tx_node = isinstance(self.node_info[tx_node], ConstantMobility)
        assert fixed_tx_node

        # Create transmitter
        tx_pos = self.node_info[tx_node].pos
        tx_node_name = "tx"
        tx = Transmitter(name=tx_node_name, position=tx_pos, orientation=[0, -180, 0], display_radius=self.disp_r)
        self.scene.add(tx)
        self.placed_radio_node_names.append(tx_node_name)

        for lah_time_idx, lah_time in enumerate(lah_time_vec):
            # Create a receiver(s)
            for rx_node_i in rx_nodes:
                rx_node_name = "rx" + str(rx_node_i) + "." + str(lah_time_idx)
                rx_pos = self.node_info[rx_node_i].get_pos_at(lah_time)
                rx = Receiver(name=rx_node_name, position=rx_pos, orientation=[0, -180, 0], display_radius=self.disp_r)
                self.scene.add(rx)
                self.placed_radio_node_names.append(rx_node_name)


    def compute_cfr_classic(self, csi_req, reply_wrapper, req_mode):
        '''
        Compute the requested CFR
        :param csi_req: received CSI request (ZMQ)
        :param reply_wrapper: the response
        '''

        if self.VERBOSE:
            print_csi_request(csi_req)

        tx_node_id = csi_req.tx_node
        rx_node_id = csi_req.rx_node # this rx node must be included in result set
        req_sim_time = csi_req.time # we need CFR at that point in time [ns]

        if self.time_evo_model == 'doppler':
            #(lnk_delay, lnk_loss, h_normalized) = self.compute_cfr_via_doppler()
            pass
        else: # position=based
            (rx_nodes, lnk_delay, lnk_loss, h_normalized) = self._compute_cfr_via_position(req_sim_time, tx_node_id, rx_node_id, req_mode)

        # Create ZMQ response
        chan_response = reply_wrapper.channel_state_response
        csi = chan_response.csi.add()

        csi.start_time = self.sim_time
        # tx node info
        tx_pos = self.node_info[tx_node_id].pos
        csi.tx_node.id = tx_node_id
        csi.tx_node.position.x = tx_pos[0]
        csi.tx_node.position.y = tx_pos[1]
        csi.tx_node.position.z = tx_pos[2]

        csi_tc_arr = []
        # for all rx nodes
        for idx, comp_rx_node_id in enumerate(rx_nodes):
            rx_node_info = csi.rx_nodes.add()
            rx_pos = self.node_info[comp_rx_node_id].pos
            rx_node_info.id = comp_rx_node_id
            rx_node_info.position.x = rx_pos[0]
            rx_node_info.position.y = rx_pos[1]
            rx_node_info.position.z = rx_pos[2]
            rx_node_info.delay = lnk_delay[idx]
            rx_node_info.wb_loss = lnk_loss[idx]

            if self.est_csi:
                rx_node_info.frequencies.extend(self.frequencies.tolist())
                rx_node_info.csi_imag.extend(np.imag(h_normalized[idx]).tolist())
                rx_node_info.csi_real.extend(np.real(h_normalized[idx]).tolist())

            # compute coherence time: with direction vectors you can compute the radial (projected) relative
            # speed directly and from that the Doppler and coherence time.
            tc = coherence_from_velocities(self.node_info[comp_rx_node_id].velocity,
                                            self.node_info[tx_node_id].velocity, self.fc,
                                            pos_tx=self.node_info[comp_rx_node_id].pos,
                                            pos_rx=self.node_info[tx_node_id].pos)
            rx_node_info.end_time2 = csi.start_time + tc
            csi_tc_arr.append(tc)

        # take the worst case Tc from all RX nodes
        Tc_p2mp = int(np.min(np.asarray(csi_tc_arr)))

        print(f'{self.sim_time / 1e9}s: Computed CSI with Tc: {round(Tc_p2mp / 1e6,2)}ms, #links: {len(rx_nodes)}')

        csi.end_time = self.sim_time + Tc_p2mp

        #print(f"[DBG] fft_size={self.fft_size} len(freqs)={len(self.frequencies)} "f"len(csi_real)={len(rx_node_info.csi_real)}")
        return len(rx_nodes)


    def _walk(self, node_id, dt):
        """
        Move the given node to the given time interval.
        :param node_id: node_id of the node to move
        :param dt: time interval in ns
        """
        init_dt = dt

        pos = self.node_info[node_id].pos
        velocity = self.node_info[node_id].velocity

        speed = np.linalg.norm(velocity)
        if speed == 0:
            return

        # Check if the next position is inside the borders
        # Calculate direction vector and travel distance
        direction = velocity / speed
        distance = np.linalg.norm(velocity * dt / 1e9) # convert dt into sec

        while True:
            # Create a ray
            ray = mi.Ray3f(mi.Point3f(pos), mi.Vector3f(direction))
            ray.maxt = mi.Float(distance)

            # Calculate the intersection of the ray with the scene
            si = self.scene.mi_scene.ray_intersect(ray, mi.RayFlags.Minimal, False, True)

            if si.is_valid():
                # If the ray hits an object, calculate the reflection

                # The intersection point (position) is set back by one centimeter to prevent cases
                # where the intersection point is found behind a wall
                t_np = si.t.numpy()
                t = float(t_np[0]) - 0.01
                pos = pos + t * direction

                # The reflected direction is calculated in the z plane
                n = np.squeeze(si.n.numpy())
                n = [n[1], -n[0], 0.0]
                n = n / np.linalg.norm(n)

                mob_theta = self.node_info[node_id].get_next_direction_angle()
                direction = - (direction - 2 * (np.dot(direction, n) + mob_theta) * n)

                velocity = direction * speed

                # make sure we do not change speed
                velocity = (velocity / np.linalg.norm(velocity)) * speed

                distance -= t
                dt -= (t / speed) * 1e9

            else:
                # If the ray does not hit an object, calculate the next position
                break

        next_pos = pos + (velocity * dt / 1e9)

        # update node pos & velocity
        self.node_info[node_id].update_pos(self.sim_time + init_dt, next_pos, velocity, False)
        # check if new velocity must be set
        self.node_info[node_id].check_set_new_velocity(self.sim_time + init_dt, distance)


    def _place_tx_rx_node(self, tx_node: int, rx_nodes: list):
        '''
        Place the given nodes in the scenario
        :param tx_node: the transmitting node
        :param rx_nodes: the receiver nodes
        '''

        # remove old tx and rx nodes
        for placed_node in self.placed_radio_node_names:
            self.scene.remove(placed_node)
        self.placed_radio_node_names.clear()

        # Create transmitter
        tx_pos = self.node_info[tx_node].pos
        tx_node_name = "tx"
        tx = Transmitter(name=tx_node_name, position=tx_pos, orientation=[0, -180, 0], display_radius=self.disp_r)
        self.scene.add(tx)
        self.placed_radio_node_names.append(tx_node_name)

        # Create a receiver(s)
        for rx_node_i in rx_nodes:
            rx_node_name = "rx" + str(rx_node_i)
            rx_pos = self.node_info[rx_node_i].pos
            rx = Receiver(name=rx_node_name, position=rx_pos, orientation=[0, -180, 0], display_radius=self.disp_r)
            self.scene.add(rx)
            self.placed_radio_node_names.append(rx_node_name)

    def _dump_rt_debug_png(
        self,
        tx_pos_xyz,
        out_dir="debug_wb_rt",
        prefix="rt_full",
        batch_size=512,
        scale_m=0.625,
        K=4,
        H=64,
        W=64,
        z_step_cells=1.0,
        z_margin_m=0.3125,
        origin_xy_mode="bbox_min",
    ):
        """
        Ray-trace a full dense KxHxW receiver grid and dump WB-loss PNGs.
        This is for debugging only: it can be expensive.
        """
        import os
        os.makedirs(out_dir, exist_ok=True)

        import matplotlib
        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt

        tx_pos_xyz = np.asarray(tx_pos_xyz, dtype=np.float32).reshape(3)
        z_step_m = float(scale_m * z_step_cells)

        # --- build grid origin exactly like UNet attach_scene() ---
        if origin_xy_mode == "zero":
            x0 = 0.0
            y0 = 0.0
        else:
            x0 = float(self.bbox.min.x)
            y0 = float(self.bbox.min.y)

        z_min = float(self.bbox.min.z)
        z_max = float(self.bbox.max.z)
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

        xs = x0 + scale_m * np.arange(W, dtype=np.float32)
        ys = y0 + scale_m * np.arange(H, dtype=np.float32)
        zs = z0 + z_step_m * np.arange(K, dtype=np.float32)
        Z, Y, X = np.meshgrid(zs, ys, xs, indexing="ij")
        rx_coords = np.stack([X, Y, Z], axis=-1).reshape(-1, 3).astype(np.float32)

        # --- build matching wall map using mlink feature pipeline ---
        walls_khw = None
        try:
            adb = AntennaDatabase(tx_pos_xyz.reshape(1, 3), rx_coords, None, rx_grid)
            dbg_scene = replace(MlinkScene.from_sionna(self.scene), antenna_database=adb)
            wall_feat = build_feature_tensor(dbg_scene, self.fc, requested=["binary_walls"]).astype(np.float32)
            walls_khw = wall_feat[0, 0]   # (K,H,W)
        except Exception as e:
            print(f"[RTDBG] failed to build binary_walls overlay: {e}")

        # --- ray trace in batches ---
        wb_flat = np.full((rx_coords.shape[0],), self.unet_no_path_wb, dtype=np.float32)

        p_solver = PathSolver()

        for start in range(0, rx_coords.shape[0], batch_size):
            stop = min(start + batch_size, rx_coords.shape[0])
            chunk = rx_coords[start:stop]

            temp_names = []
            try:
                tx_name = "_dbg_tx"
                tx = Transmitter(
                    name=tx_name,
                    position=tx_pos_xyz,
                    orientation=[0, -180, 0],
                    display_radius=0.1,
                )
                self.scene.add(tx)
                temp_names.append(tx_name)

                for j, pos in enumerate(chunk):
                    rx_name = f"_dbg_rx_{j}"
                    rx = Receiver(
                        name=rx_name,
                        position=pos,
                        orientation=[0, -180, 0],
                        display_radius=0.1,
                    )
                    self.scene.add(rx)
                    temp_names.append(rx_name)

                paths = p_solver(
                    scene=self.scene,
                    max_depth=self.rt_max_depth,
                    samples_per_src=self.rt_samples_per_src,
                    max_num_paths_per_src=self.rt_max_num_paths_per_src,
                    los=self.rt_los,
                    specular_reflection=self.rt_specular_reflection,
                    diffuse_reflection=self.rt_diffuse_reflection,
                    refraction=self.rt_refraction,
                    synthetic_array=self.rt_synthetic_array,
                    diffraction=self.rt_diffraction,
                    edge_diffraction=self.rt_edge_diffraction,
                    diffraction_lit_region=self.rt_diffraction_lit_region,
                )

                a, tau = paths.cir(
                    sampling_frequency=1e9,
                    normalize_delays=False,
                    out_type="numpy",
                )

                h_raw = paths.cfr(
                    frequencies=self.frequencies,
                    sampling_frequency=1.0,
                    num_time_steps=1,
                    normalize_delays=True,
                    normalize=False,
                    out_type="numpy",
                )

                for j in range(stop - start):
                    tau_j = np.squeeze(tau[j, :, :, :, :])
                    valid = np.isfinite(tau_j) & (tau_j >= 0)

                    if not np.any(valid):
                        continue

                    h_j = np.squeeze(h_raw[j, :, :, :, :, :])
                    pwr = float(np.mean(np.abs(h_j) ** 2))
                    if np.isfinite(pwr) and pwr > 0:
                        wb_flat[start + j] = float(-10.0 * np.log10(pwr))

                print(f"[RTDBG] finished batch {start}:{stop} / {rx_coords.shape[0]}")
                self._rt_after_solve()

            finally:
                for name in temp_names:
                    try:
                        self.scene.remove(name)
                    except Exception:
                        pass

        wb_khw = wb_flat.reshape(K, H, W)

        # --- dump plots ---
        txx = (tx_pos_xyz[0] - x0) / scale_m
        txy = (tx_pos_xyz[1] - y0) / scale_m

        for k in range(K):
            plt.figure(figsize=(6, 5))
            im = plt.imshow(wb_khw[k], origin="lower")
            plt.colorbar(im, label="Ray-traced wideband loss (dB)")

            if walls_khw is not None:
                plt.contour(
                    walls_khw[k],
                    levels=[0.5],
                    colors="white",
                    linewidths=0.8,
                    origin="lower",
                )

            plt.scatter([txx], [txy], c="red", s=30, marker="x", label="TX")
            plt.legend(loc="upper right")
            plt.title(
                f"{prefix} WB loss slice k={k} "
                f"TX=({tx_pos_xyz[0]:.2f},{tx_pos_xyz[1]:.2f},{tx_pos_xyz[2]:.2f})"
            )
            plt.tight_layout()
            plt.savefig(
                os.path.join(
                    out_dir,
                    f"{prefix}_tx_{tx_pos_xyz[0]:.2f}_{tx_pos_xyz[1]:.2f}_{tx_pos_xyz[2]:.2f}_k{k}.png",
                ),
                dpi=150,
            )
            plt.close()

        print(f"[RTDBG] saved RT debug maps to: {out_dir}")

    def _compute_cfr_via_position(self, req_sim_time, tx_node, rx_node, req_mode):
        '''
        Compute the link propagation delay, wideband loss and normalized CFR
        :param req_sim_time: current simulation time
        :param tx_node: the transmitter node id
        :param rx_node: the receiver node id
        :return: (list(rx_node), list(link propagation delay), list(wideband loss), list(normalized CFR))
        '''

        def fspl_db(d_m: float, fc_hz: float) -> float:
            d_m = max(float(d_m), 1e-6)
            lam = 299792458.0 / float(fc_hz)
            return 20.0 * np.log10(4.0 * np.pi * d_m / lam)
        
        t_req0 = perf_counter()

        # # execute mobility
        # dt = req_sim_time - self.sim_time

        # estimate the node we need to update their position
        if req_mode == SionnaEnv.MODE_P2P:
            nodes_to_update = [tx_node, rx_node] # only TX and RX
        else:
            # both P2MP and P2MP_LAH
            nodes_to_update = list(self.node_info.keys())

        # for node_id in nodes_to_update:
        #     self._walk(node_id, dt)

        # # update time
        # self.sim_time = req_sim_time

        self._advance_nodes_to_time(req_sim_time, nodes_to_update)

        # place TX and RX
        rx_nodes = nodes_to_update
        rx_nodes.remove(tx_node)

        ###Helper for checking stats on reciprocity###

        def _tau_rms_ns_from_cir(a_link, tau_link):
            '''
            Docstring for _tau_rms_ns_from_cir
            
            :param a_link: arrays for one link
            :param tau_link: tau in seconds

            returns tau_rms in ns
            '''

            a_link = np.asarray(a_link)
            tau_link = np.asarray(tau_link)

            a_flat = np.squeeze(a_link).reshape(-1)
            tau_flat = np.squeeze(tau_link).reshape(-1)

            m = np.isfinite(tau_flat) & (tau_flat >= 0) & np.isfinite(a_flat)
            if not np.any(m):
                return 0.0
            
            p = np.abs(a_flat[m])**2 
            ps = float(np.sum(p))
            if ps <= 0.0:
                return 0.0
            w = p / ps
            t = tau_flat[m]
            mu = float(np.sum(w*t))
            mu2 = float(np.sum(w*t**2))
            var = max(0.0, mu2 - mu**2 )
            return math.sqrt(var) * 1e9  # in ns

        if self.use_unet:
            t0 = perf_counter()
            tx_pos = np.array(self.node_info[tx_node].pos, dtype=np.float32)

            t_pred0 = perf_counter()
            self._unet.predict_for_tx(tx_pos)
            t_pred = perf_counter() - t_pred0

            lnk_delay_arr = []
            lnk_loss_arr = []
            h_normalized_arr = []

            for curr_rx_node in rx_nodes:

                rx_pos = np.array(self.node_info[curr_rx_node].pos, dtype=np.float32)

                # RESIDUAL-OVER-COST MODEL: sample_heads returns the FULL
                # wideband loss (wb_cost + gated residual). Do NOT add FSPL.
                wb_db, tau_rms_ns, excess_ns = self._unet.sample_heads(rx_pos)
                wb_db = float(wb_db)
                d_m = float(np.linalg.norm(tx_pos - rx_pos))

                # no-path guard (sentinel is emitted by the coverage head)
                if wb_db >= (self._unet.no_path_wb - 1e-3):
                    lnk_loss_arr.append(float(self._unet.no_path_wb))
                    lnk_delay_arr.append(0)
                    h_normalized_arr.append(np.zeros((self.fft_size,), dtype=np.complex64))
                    continue

                ##Friis for comparison (debug print only; NOT added to wb)###
                friis_loss_db = fspl_db(d_m, self.fc)
                print(f"[FRIIS] tx={tx_node} rx={curr_rx_node} d={d_m:.3f} m fspl={friis_loss_db:.2f} dB")

                base_ns = d_m / 299792458.0 * 1e9
                ex = 0.0 if excess_ns is None else float(max(0.0, excess_ns))
                lnk_delay_arr.append(int(round(base_ns + ex)))

                # wb loss direct from model
                lnk_loss_arr.append(float(wb_db))

                # normalized CFR from G(tau_rms)
                seed = (int(self.my_seed) * 1315423911) ^ (int(tx_node) * 2654435761) ^ (int(curr_rx_node) * 97531) ^ (int(self.sim_time) & 0xffffffff)
                print(f"Wideband loss: rx={curr_rx_node} wb={wb_db:.1f}dB")

                #Testing physical laws
                delay_ns = base_ns + ex
                delay_int_ns = int(round(delay_ns))

                print(
                    f"[CFR] tx={tx_node} rx={curr_rx_node} "
                    f"d={d_m:.3f} m "
                    f"delay={delay_int_ns} ns (raw={delay_ns:.3f}; base={base_ns:.3f}+ex={ex:.3f}) "
                    f"tau_rms={tau_rms_ns:.4f} ns"
                )
                
                h_norm = self._unet.synthesize_cfr(tau_rms_ns=tau_rms_ns, mi_scene=self.scene.mi_scene, tx_xyz=tx_pos, rx_xyz=rx_pos, seed=seed, synthetic=self.synthetic_cfr)

                # sanity: mean |H|^2 ~ 1
                if self.CHECKS_ENABLED:
                    p = float(np.mean(np.abs(h_norm) ** 2))
                    assert abs(p - 1.0) < 1e-2

                h_normalized_arr.append(h_norm)

            t_total = perf_counter() - t0
            self._tim_add("unet", t_total)
            print(f"[TIM] unet breakdown: predict_for_tx={1e3*t_pred:.2f} ms total={1e3*t_total:.2f} ms", flush=True)

            return rx_nodes, lnk_delay_arr, lnk_loss_arr, h_normalized_arr
        
        elif self.use_cost231:
            t0 = perf_counter()
            tx_pos = np.array(self.node_info[tx_node].pos, dtype=np.float32)

            self._cost231.predict_for_tx(tx_pos)

            lnk_delay_arr = []
            lnk_loss_arr = []
            h_normalized_arr = []

            for curr_rx_node in rx_nodes:
                rx_pos = np.array(self.node_info[curr_rx_node].pos, dtype=np.float32)

                wb_db, tau_rms_ns, excess_ns = self._cost231.sample_heads(rx_pos)
                d_m = float(np.linalg.norm(tx_pos - rx_pos))

                if wb_db >= (self._cost231.no_path_wb - 1e-3):
                    lnk_loss_arr.append(float(self._cost231.no_path_wb))
                    lnk_delay_arr.append(0)
                    h_normalized_arr.append(np.zeros((self.fft_size,), dtype=np.complex64))
                    continue

                base_ns = d_m / 299792458.0 * 1e9
                ex = float(max(0.0, excess_ns))
                lnk_delay_arr.append(int(round(base_ns + ex)))
                lnk_loss_arr.append(float(wb_db))

                if self.est_csi:
                    seed = (int(self.my_seed) * 1315423911) ^ (int(tx_node) * 2654435761) ^ (int(curr_rx_node) * 97531) ^ (int(self.sim_time) & 0xffffffff)
                    h_norm = self._cost231.synthesize_cfr(tau_rms_ns=tau_rms_ns, mi_scene=self.scene.mi_scene, tx_xyz=tx_pos, rx_xyz=rx_pos, seed=seed, synthetic=self.synthetic_cfr)
                    h_normalized_arr.append(h_norm)
                else:
                    h_normalized_arr.append(np.zeros((self.fft_size,), dtype=np.complex64))

                friis_loss_db = fspl_db(d_m, self.fc)
                print(f"[FRIIS] tx={tx_node} rx={curr_rx_node} d={d_m:.3f} m fspl={friis_loss_db:.2f} dB")
                print(f"Wideband loss: rx={curr_rx_node} wb={wb_db:.1f}dB")
                print(
                    f"[COST231] tx={tx_node} rx={curr_rx_node} "
                    f"d={d_m:.3f} m "
                    f"delay={int(round(base_ns + ex))} ns (raw={base_ns + ex:.3f}; base={base_ns:.3f}+ex={ex:.3f}) "
                    f"tau_rms={tau_rms_ns:.4f} ns"
                )

            t_total = perf_counter() - t0
            print(f"[TIM] cost231 total={1e3*t_total:.2f} ms", flush=True)

            return rx_nodes, lnk_delay_arr, lnk_loss_arr, h_normalized_arr

        elif self.use_logdist:
            t0 = perf_counter()
            tx_pos = np.array(self.node_info[tx_node].pos, dtype=np.float32)

            self._logdist.predict_for_tx(tx_pos)

            lnk_delay_arr = []
            lnk_loss_arr = []
            h_normalized_arr = []

            for curr_rx_node in rx_nodes:
                rx_pos = np.array(self.node_info[curr_rx_node].pos, dtype=np.float32)

                wb_db, tau_rms_ns, excess_ns = self._logdist.sample_heads(rx_pos)
                d_m = float(np.linalg.norm(tx_pos - rx_pos))

                base_ns = d_m / 299792458.0 * 1e9
                lnk_delay_arr.append(int(round(base_ns)))
                lnk_loss_arr.append(float(wb_db))

                if self.est_csi:
                    seed = (int(self.my_seed) * 1315423911) ^ (int(tx_node) * 2654435761) ^ (int(curr_rx_node) * 97531) ^ (int(self.sim_time) & 0xffffffff)
                    h_norm = self._logdist.synthesize_cfr(tau_rms_ns=tau_rms_ns, mi_scene=self.scene.mi_scene, tx_xyz=tx_pos, rx_xyz=rx_pos, seed=seed, synthetic=self.synthetic_cfr)
                    h_normalized_arr.append(h_norm)
                else:
                    h_normalized_arr.append(np.zeros((self.fft_size,), dtype=np.complex64))

                print(f"[LOGDIST] tx={tx_node} rx={curr_rx_node} d={d_m:.3f} m wb={wb_db:.1f} dB delay={int(round(base_ns))} ns")

            t_total = perf_counter() - t0
            print(f"[TIM] logdist total={1e3*t_total:.2f} ms", flush=True)

            return rx_nodes, lnk_delay_arr, lnk_loss_arr, h_normalized_arr

        t0 = perf_counter()

        t_place0 = perf_counter()
        self._place_tx_rx_node(tx_node, rx_nodes)
        t_place = perf_counter() - t_place0

        # create pathsolver; todo: check reuse
        p_solver  = PathSolver()
        

        #Deterministic seed based on link and time if you feel so inclined 

        a = int(min(tx_node, rx_nodes[0]))
        b = int(max(tx_node, rx_nodes[0]))
        rt_seed = (int(self.my_seed) * 1315423911) ^ (a * 2654435761) ^ (b * 97531)

        # Compute propagation paths
        t_paths0 = perf_counter()
        paths = p_solver(scene=self.scene,
                         max_depth=self.rt_max_depth,
                         samples_per_src=self.rt_samples_per_src,
                         max_num_paths_per_src=self.rt_max_num_paths_per_src,
                         los=self.rt_los,
                         specular_reflection=self.rt_specular_reflection,  # Can rays bounce off surfaces?
                         diffuse_reflection=self.rt_diffuse_reflection,
                         refraction=self.rt_refraction,  # Can rays pass through materials?
                         synthetic_array=self.rt_synthetic_array,
                         diffraction=self.rt_diffraction,  # costly
                         edge_diffraction=self.rt_edge_diffraction,  # rays that bend around edges
                         diffraction_lit_region=self.rt_diffraction_lit_region) # higher physical accuracy
        t_paths = perf_counter() - t_paths0


        # AZU: sampling_frequency is only used if num_time_steps > 1
        # a: shape [num_rx, num_rx_ant, num_tx, num_tx_ant, num_paths, num_time_steps],
        t_cfr0 = perf_counter()
        a, tau = paths.cir(sampling_frequency=1e9, normalize_delays=False, out_type="numpy")
        t_cfr = perf_counter() - t_cfr0

        # shape: [num_rx, num_rx_ant, num_tx, num_tx_ant, num_ofdm_symbols, num_subcarriers]
        h_raw = paths.cfr(frequencies=self.frequencies,
                  sampling_frequency=1.0,  # not used
                  num_time_steps=1,
                  normalize_delays=True,
                  # If set to True, path delays are normalized such that the first path between any pair of
                  # antennas of a transmitter and receiver arrives at tau=0
                  normalize=False,  # Normalize energy
                  out_type="numpy")

        # RT outputs are numpy now; allow the allocator cache to be trimmed
        self._rt_after_solve()

        lnk_delay_arr = []
        lnk_loss_arr = []
        h_normalized_arr = []

        for rx_id, curr_rx_node in enumerate(rx_nodes):
            
            tx_pos = np.array(self.node_info[tx_node].pos, dtype=np.float64)
            rx_pos = np.array(self.node_info[curr_rx_node].pos, dtype=np.float64)
            d_m = float(np.linalg.norm(tx_pos - rx_pos))

            # ###DEBUGGING###
            # if not hasattr(self, "_rt_debug_dumped"):
            #     self._rt_debug_dumped = set()

            # tx_key = tuple(np.round(tx_pos.astype(np.float32), 3).tolist())
            # if tx_key not in self._rt_debug_dumped:
            #     self._dump_rt_debug_png(
            #         tx_pos_xyz=tx_pos.astype(np.float32),
            #         out_dir="debug_wb_rt",
            #         prefix="rt_full",
            #         batch_size=256,   # try 256 first
            #         scale_m=0.625,
            #         K=4,
            #         H=64,
            #         W=64,
            #         z_step_cells=1.0,
            #         z_margin_m=0.625 * 0.5,
            #         origin_xy_mode="bbox_min",
            #     )
            #     self._rt_debug_dumped.add(tx_key)
            # ###DEBUG DUMP#

            ##Friis for comparison###
            friis_loss_db = fspl_db(d_m, self.fc)
            print(f"[FRIIS] tx={tx_node} rx={curr_rx_node} d={d_m:.3f} m fspl={friis_loss_db:.2f} dB")

            base_ns = d_m / 299792458.0 * 1e9


            lnk_tau = np.squeeze(tau[rx_id, :, :, :, :])

            valid = lnk_tau[np.isfinite(lnk_tau) & (lnk_tau >= 0)]

            if valid.size == 0:
                # No path found (common with --rt_fast in blocked scenes)
                if self.VERBOSE:
                    print(f"[RT] NO-PATH tx={tx_node} rx={curr_rx_node} d={d_m:.3f}m (rt_fast={self.rt_fast})")
                lnk_delay_arr.append(int(round(base_ns)))  # or 0 if you prefer
                lnk_loss_arr.append(float(self.unet_no_path_wb))  # 199.5 dB sentinel
                h_normalized_arr.append(np.zeros((self.fft_size,), dtype=np.complex64))
                continue

            lnk_delay = int(round(valid.min() * 1e9, 0))

            ### Reciprocity stats ###
            ex_ns = float(max(0.0, lnk_delay - base_ns))
            a_link = np.squeeze(a[rx_id, :, :, :, :, :])
            tau_link = np.squeeze(tau[rx_id, :, :, :, :])
            tau_rms_ns = _tau_rms_ns_from_cir(a_link, tau_link)
            

            h = np.squeeze(h_raw[rx_id, :, :, :, :, ])

            # see Parseval's theorem
            lnk_loss = float(-10 * np.log10(np.mean(np.abs(h) ** 2)))

            print(
                f"[RT] tx={tx_node} rx={curr_rx_node} "
                f"d={d_m:.3f}m delay={lnk_delay}ns (base={base_ns:.3f}+ex={ex_ns:.3f}) "
                f"wb={lnk_loss:.3f}dB tau_rms={tau_rms_ns:.4f}ns"
            )

            # for frequency-selective channel
            power = np.mean(np.abs(h) ** 2)  # shape [batch_size, 1, 1, 1]

            if (not np.isfinite(power)) or power <= 0:
                lnk_delay_arr.append(int(round(base_ns)))
                lnk_loss_arr.append(float(self.unet_no_path_wb))
                h_normalized_arr.append(np.zeros((self.fft_size,), dtype=np.complex64))
                continue

            h_normalized = h / np.sqrt(power)

            # plausibility test
            if self.CHECKS_ENABLED:
                power_normalized = np.mean(np.abs(h_normalized) ** 2)
                assert math.isclose(power_normalized, 1.0, rel_tol=1e-3)   # Should be close to 1

            if self.VERBOSE:
                print(f'{self.sim_time/1e9}s: lnk_delay = {lnk_delay}ns, wb_loss = {lnk_loss:.3f}dB, CFR shape: {h_normalized.shape}')

            lnk_delay_arr.append(lnk_delay)
            lnk_loss_arr.append(lnk_loss)
            h_normalized_arr.append(h_normalized)

        t_total = perf_counter() - t0

        self._tim_add("rt", t_total)
        print(f"[TIM] rt breakdown: place={1e3*t_place:.2f} ms "
            f"paths={1e3*t_paths:.2f} ms cir={1e3*t_cfr:.2f} ms cfr={1e3*t_cfr:.2f} ms "
            f"total={1e3*t_total:.2f} ms", flush=True)

        return rx_nodes, lnk_delay_arr, lnk_loss_arr, h_normalized_arr


    def _get_mobility_history(self, node_id):
        ts = sorted(self.node_info[node_id].pos_history.keys())
        pos = [self.node_info[node_id].pos_history[t] for t in ts]
        return ts, pos


    def _init_mobility(self, sim_init_msg):

        # If trace mode: ignore sim_init_msg mobility models and drive from CSV.
        if getattr(self, "_use_trace", False):
            for node_info in sim_init_msg.nodes:
                nid = int(node_info.id)

                if nid in self._trace:
                    t_ns, pos, vel = self._trace[nid]
                    self.node_info[nid] = TraceMobility(nid, t_ns, pos, vel)
                else:
                    # fallback: use whatever init msg says (constant) if node missing in trace
                    if node_info.HasField("constant_position_model"):
                        pos = node_info.constant_position_model.position
                        self.node_info[nid] = ConstantMobility(nid, [pos.x, pos.y, pos.z])
                    elif node_info.HasField("random_walk_model"):
                        pos = node_info.random_walk_model.position
                        self.node_info[nid] = ConstantMobility(nid, [pos.x, pos.y, pos.z])
                    else:
                        self.node_info[nid] = ConstantMobility(nid, [0.0, 0.0, 0.0])

            # initialize server time to trace start if you want (optional)
            self.sim_time = 0
            for nid, m in self.node_info.items():
                if isinstance(m, TraceMobility):
                    # snap to t=0 (or earliest trace time)
                    m.set_time(0)

            print(f"[TRACE] Enabled trace mobility for {sum(isinstance(m, TraceMobility) for m in self.node_info.values())} nodes")
            return
        
        # Store information about each node: ID, mobility model
        for node_info in sim_init_msg.nodes:
            if (node_info.HasField("constant_position_model")):
                # fixed position; no mobility
                pos = node_info.constant_position_model.position
                self.node_info[node_info.id] = ConstantMobility(node_info.id, [pos.x, pos.y, pos.z])
            elif (node_info.HasField("random_walk_model")):
                # mobile scenario
                random_walk_model = node_info.random_walk_model
                pos = random_walk_model.position

                mode = None
                if random_walk_model.HasField("wall_value"):
                    mode = RandomWalkMobility.MODE_WALL
                    mode_params = random_walk_model.wall_value
                elif random_walk_model.HasField("time_value"):
                    mode = RandomWalkMobility.MODE_TIME
                    mode_params = random_walk_model.time_value
                elif random_walk_model.HasField("distance_value"):
                    mode = RandomWalkMobility.MODE_DISTANCE
                    mode_params = random_walk_model.distance_value

                speed = None
                if random_walk_model.speed.HasField("uniform"):
                    speed = RandomWalkMobility.SPEED_UNIFORM
                    speed_params = (random_walk_model.speed.uniform.min, random_walk_model.speed.uniform.max)
                elif random_walk_model.speed.HasField("constant"):
                    speed = RandomWalkMobility.SPEED_CONSTANT
                    speed_params = (random_walk_model.speed.constant.value,)
                elif random_walk_model.speed.HasField("normal"):
                    speed = RandomWalkMobility.SPEED_NORMAL
                    speed_params = (random_walk_model.speed.normal.mean, random_walk_model.speed.normal.variance)

                direction = None
                if random_walk_model.direction.HasField("uniform"):
                    direction = RandomWalkMobility.DIRECTION_UNIFORM
                    direction_params = (random_walk_model.direction.uniform.min,
                                        random_walk_model.direction.uniform.max)
                elif random_walk_model.direction.HasField("constant"):
                    direction = RandomWalkMobility.DIRECTION_CONSTANT
                    direction_params = (random_walk_model.direction.constant.value,)
                elif random_walk_model.direction.HasField("normal"):
                    direction = RandomWalkMobility.DIRECTION_NORMAL
                    direction_params = (random_walk_model.direction.normal.mean,
                                        random_walk_model.direction.normal.variance)

                self.node_info[node_info.id] = RandomWalkMobility(node_info.id, [pos.x, pos.y, pos.z],
                                                                  mode, mode_params, speed, speed_params,
                                                                  direction, direction_params)

    def _advance_nodes_to_time(self, req_sim_time: int, nodes_to_update: list):
        if getattr(self, "_use_trace", False):
            for nid in nodes_to_update:
                m = self.node_info[nid]
                if isinstance(m, TraceMobility):
                    m.set_time(req_sim_time)
            self.sim_time = req_sim_time
            return

        # original behavior
        dt = req_sim_time - self.sim_time
        for nid in nodes_to_update:
            self._walk(nid, dt)
        self.sim_time = req_sim_time



    def run(self):
        '''
        Handles communication with the ns3 simulator using ZMQ socket
        '''

        context = zmq.Context()
        socket = zmq.Socket(context, zmq.REP)
        socket.bind("tcp://*:5555")

        print("Sionna server socket ready ...")

        last_call_times = deque(maxlen=10)
        total_num_csi_samples = 0

        do_terminate = False
        while not do_terminate:
            # Receive message from ns3 simulator
            #t0 = time.time()
            ns3_msg_str = socket.recv()
            #print(f"[ZMQ] recv after {time.time()-t0:.3f}s", flush=True)

            # Deserialize received message
            ns3_msg = message_pb2.Wrapper()
            ns3_msg.ParseFromString(ns3_msg_str)

            # Prepare reply message
            resp_msg = message_pb2.Wrapper()

            # Fill the reply message
            if ns3_msg.HasField("sim_init_msg"):
                # handle SimInitMessage & send ACK
                successful, error_msg = self.init_simulation_env(ns3_msg.sim_init_msg)
                resp_msg.sim_ack.no_error = successful
                resp_msg.sim_ack.error_msg = error_msg
                resp_msg.sim_ack.SetInParent()

                if successful:
                    print("Sionna server init sucessful ...")
                else:
                    print("Sionna server init failed ...")
                    do_terminate = True

            elif ns3_msg.HasField("channel_state_request"):
                # handle ChannelStateRequest by sending ChannelStateResponse
                start_time = time.time()
                num_csi_req = self.compute_cfr(ns3_msg.channel_state_request, resp_msg)
                total_num_csi_samples += num_csi_req
                last_call_times.append(time.time() - start_time)

                if total_num_csi_samples % 1000 == 0: # every 1k make printout
                    print(f'Total no. computed CSI samples: {millify(total_num_csi_samples)}')

                if self.VERBOSE:
                    avg_call_time = sum(last_call_times) / len(last_call_times)
                    print("t=%.9fs: average event processing time: %.2f sec"
                          % (ns3_msg.channel_state_request.time/1e9, avg_call_time))
                    # show GPU load
                    if len(self.gpus) > 0:
                        GPUtil.showUtilization()

            elif ns3_msg.HasField("sim_close_request"):
                #print("[ZMQ] got sim_close_request", flush=True)
                do_terminate = True
                resp_msg.sim_ack.SetInParent()

            # Serialize and send the reply message
            #t_send = time.time()
            socket.send(resp_msg.SerializeToString())
            #print(f"[ZMQ] sent reply in {time.time()-t_send:.3f}s", flush=True)
            #payload = resp_msg.SerializeToString()
            #print(f"[ZMQ] send size={len(payload)/1e6:.2f} MB", flush=True)

        print("[ZMQ] closing socket...", flush=True)
        t_close = time.time()
        socket.close()
        print(f"[ZMQ] socket closed in {time.time()-t_close:.3f}s", flush=True)
        print("Computed no. CSI samples: %d" % total_num_csi_samples)
        print("Sionna server socket closed.")


    def release(self):
        # delete / release the scene before loading a new one
        del self.scene
        gc.collect()  # force garbage collection

    


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_folder", type=str, default='models/', help="The folder containing the XML files of the scenes")
    parser.add_argument("--single_run", help="Whether not to terminate after single run", action='store_true')
    parser.add_argument("--default_mode", type=int, default=SionnaEnv.MODE_P2MP, help="Which mode to use if not set by ns3")
    parser.add_argument("--rt_fast", help="Use simplified raytracing for faster computations", action='store_true')
    parser.add_argument("--rt_max_parallel_links", type=int, default=32, help="Max no. of link simulated at once; depends on GPU memory")
    parser.add_argument("--est_csi", action="store_true", help="Send CSI vectors (needed for spectrum model)")
    parser.add_argument("--verbose", help="Whether to run in verbose mode", action='store_true')

    parser.add_argument("--use_unet", action="store_true", help="Use U-Net surrogate instead of Sionna ray tracing")
    parser.add_argument("--unet_run", type=str, default="residual_cost_v2", help="Path to run dir containing model.pt/meta.json/norm_stats.npz")
    parser.add_argument("--unet_device", type=str, default="cuda", help="cpu|cuda|cuda:0")
    parser.add_argument("--unet_no_path_wb", type=float, default=199.5, help="No-path sentinel wb_loss (dB)")
    parser.add_argument("--unet_cov_thresh", type=float, default=0.5, help="Coverage-probability threshold below which a link is declared no-path")
    parser.add_argument("--unet_tx_cache", type=int, default=256, help="LRU entries for cached UNet maps (per TX, or per (TX,patch) when tiling); ~200 KB each. Size it >= n_static_tx * patches_per_scene")
    parser.add_argument("--unet_y_wb_idx", type=int, default=0, help="DEPRECATED (ignored): residual model has fixed head layout [r, tau, coverage_logit]")
    parser.add_argument("--unet_y_tau_rms_idx", type=int, default=2, help="DEPRECATED (ignored): residual model has fixed head layout")
    parser.add_argument("--unet_y_excess_idx", type=int, default=-1, help="DEPRECATED (ignored): residual model has no excess-delay head")
    parser.add_argument("--mobility_trace_in", type=str, default="",
                    help="CSV with t_s,node,x,y,z,... used to drive mobility deterministically")
    parser.add_argument("--use_cost231", action="store_true", help="Use COST-231 multi-wall model (no RT, no UNet)")
    parser.add_argument("--cost231_tau_rms_ns", type=float, default=5.0, help="Default tau_rms (ns) for COST-231 CFR synthesis")
    parser.add_argument("--use_logdist", action="store_true", help="Use log-distance path-loss baseline (no RT, no UNet, no walls)")
    parser.add_argument("--logdist_exponent", type=float, default=3.0, help="Log-distance path-loss exponent n (2.0 = Friis)")
    parser.add_argument("--logdist_ref_m", type=float, default=1.0, help="Log-distance reference distance d0 (m)")
    parser.add_argument("--logdist_tau_rms_ns", type=float, default=5.0, help="Default tau_rms (ns) for log-distance CFR synthesis")
    parser.add_argument("--synthetic_cfr", action="store_true", help="Send synthetic tau-shaped Rician CFRs from surrogates instead of flat CFRs (for CFR-level validation)")
    parser.add_argument("--cfr_k_los_db", type=float, default=6.0, help="Rician K (dB) for synthetic CFRs on LOS links (RT-calibrated default)")
    parser.add_argument("--cfr_k_nlos_db", type=float, default=2.0, help="Rician K (dB) for synthetic CFRs on NLOS links (RT-calibrated default)")
    parser.add_argument("--rt_samples_per_src", type=float, default=None,
                    help="Override RT ray budget per source (default: 1e7 realistic, 1e6 with --rt_fast). Shared across all receivers in a solve.")
    parser.add_argument("--rt_flush_every", type=int, default=8,
                    help="Flush the Dr.Jit malloc cache every N RT solves (allocator hygiene on WDDM GPUs)")
    args = parser.parse_args()

    print("ns3sionna v1.0")
    while True:
        print("Using config: model_folder=%s, single_run=%s, mode=%d, rt_fast=%s, rt_max_parallel_links=%d, est_csi=%r"
              % (args.model_folder, args.single_run, args.default_mode, args.rt_fast, args.rt_max_parallel_links, args.est_csi))
        print("Waiting for new job ...")
        env = SionnaEnv(args.model_folder, args.rt_fast, args.default_mode, args.rt_max_parallel_links,
                        args.est_csi, args.use_unet, args.unet_run, args.unet_device, args.unet_no_path_wb,
                        args.unet_y_wb_idx,  args.unet_y_tau_rms_idx, args.unet_y_excess_idx,
                        unet_cov_thresh=args.unet_cov_thresh,
                        unet_tx_cache=args.unet_tx_cache,
                        VERBOSE=args.verbose,
                        mobility_trace_in=args.mobility_trace_in,
                        use_cost231=args.use_cost231,
                        cost231_tau_rms_ns=args.cost231_tau_rms_ns,
                        use_logdist=args.use_logdist,
                        logdist_exponent=args.logdist_exponent,
                        logdist_ref_m=args.logdist_ref_m,
                        logdist_tau_rms_ns=args.logdist_tau_rms_ns,
                        synthetic_cfr=args.synthetic_cfr,
                        cfr_k_los_db=args.cfr_k_los_db,
                        cfr_k_nlos_db=args.cfr_k_nlos_db,
                        rt_samples_per_src=args.rt_samples_per_src,
                        rt_flush_every=args.rt_flush_every)
        env.run()

        if args.single_run:
            break