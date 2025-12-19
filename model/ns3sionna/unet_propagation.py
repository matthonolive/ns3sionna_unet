### LOSS FUNCTIONS ###

# unet_propagation.py
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch

from mlink.antenna import AntennaGrid, AntennaDatabase
from mlink.feature import build_feature_tensor
from mlink.scene import Scene as MlinkScene
from mlink.constants import FREE_SPACE_CONSTS


# ----------------------------
# Config
# ----------------------------

@dataclass(frozen=True)
class GridCfg:
    H: int = 64
    W: int = 64
    cell_size_m: float = 0.625
    origin_policy: str = "fit_points"         # "fit_points" or "center_points"
    origin_margin_cells: int = 2              # only used for fit_points


@dataclass(frozen=True)
class HeightCfg:
    policy: str = "per_receiver_bucket"       # "fixed" | "per_receiver_bucket"
    bucket_m: float = 0.25
    clamp_margin_m: float = 0.1


@dataclass(frozen=True)
class RuntimeCfg:
    device: str = "cpu"
    features: Tuple[str, ...] = ("cost", "height_cond")
    grid: GridCfg = GridCfg()
    height: HeightCfg = HeightCfg()


@dataclass(frozen=True)
class CalibrationCfg:
    train_frequency_hz: float = 2e9
    train_tx_power_dbm: float = 44.0
    runtime_tx_power_dbm: float = 44.0
    frequency_correction: str = "friis"       # "none" or "friis"


@dataclass(frozen=True)
class LinkCfg:
    delay_policy: str = "distance_over_c"     # currently only supported
    loss_policy: str = "tx_minus_rss"         # currently only supported


@dataclass(frozen=True)
class UNetModelCfg:
    version: int
    name: str
    torchscript_model: Path
    norm_stats_npz: Path
    runtime: RuntimeCfg
    calibration: CalibrationCfg
    link: LinkCfg

    @staticmethod
    def load(path: str | Path) -> "UNetModelCfg":
        p = Path(path)
        data = json.loads(p.read_text())

        base = p.parent

        def rpath(x: str) -> Path:
            q = Path(x)
            return q if q.is_absolute() else (base / q)

        grid = data.get("runtime", {}).get("grid", {})
        height = data.get("runtime", {}).get("height", {})

        runtime = RuntimeCfg(
            device=data.get("runtime", {}).get("device", "cpu"),
            features=tuple(data.get("runtime", {}).get("features", ["cost", "height_cond"])),
            grid=GridCfg(
                H=int(grid.get("H", 64)),
                W=int(grid.get("W", 64)),
                cell_size_m=float(grid.get("cell_size_m", 0.625)),
                origin_policy=str(grid.get("origin_policy", "fit_points")),
                origin_margin_cells=int(grid.get("origin_margin_cells", 2)),
            ),
            height=HeightCfg(
                policy=str(height.get("policy", "per_receiver_bucket")),
                bucket_m=float(height.get("bucket_m", 0.25)),
                clamp_margin_m=float(height.get("clamp_margin_m", 0.1)),
            ),
        )

        cal = data.get("calibration", {})
        calibration = CalibrationCfg(
            train_frequency_hz=float(cal.get("train_frequency_hz", 2e9)),
            train_tx_power_dbm=float(cal.get("train_tx_power_dbm", 44.0)),
            runtime_tx_power_dbm=float(cal.get("runtime_tx_power_dbm", cal.get("train_tx_power_dbm", 44.0))),
            frequency_correction=str(cal.get("frequency_correction", "friis")),
        )

        link = data.get("link", {})
        link_cfg = LinkCfg(
            delay_policy=str(link.get("delay_policy", "distance_over_c")),
            loss_policy=str(link.get("loss_policy", "tx_minus_rss")),
        )

        return UNetModelCfg(
            version=int(data.get("version", 1)),
            name=str(data.get("name", "unet_rss")),
            torchscript_model=rpath(data["paths"]["torchscript_model"]),
            norm_stats_npz=rpath(data["paths"]["norm_stats_npz"]),
            runtime=runtime,
            calibration=calibration,
            link=link_cfg,
        )


# ----------------------------
# Predictor
# ----------------------------

class UNetPropagator:
    """
    Predict RSS maps and return per-link delay + wideband loss (no CSI).
    """

    def __init__(self, cfg: UNetModelCfg):
        self.cfg = cfg
        self.device = torch.device(cfg.runtime.device)

        # Load TorchScript model
        self.model = torch.jit.load(str(cfg.torchscript_model), map_location=self.device).eval()

        # Load normalization stats saved during training (x_mean/x_std/y_mean/y_std)
        d = np.load(str(cfg.norm_stats_npz), allow_pickle=True)
        self.x_mean = torch.from_numpy(d["x_mean"]).float().to(self.device)  # (4,1,1)
        self.x_std  = torch.from_numpy(d["x_std"]).float().to(self.device)
        self.y_mean = torch.from_numpy(d["y_mean"]).float().to(self.device)  # (1,1,1)
        self.y_std  = torch.from_numpy(d["y_std"]).float().to(self.device)

        # Scene cache
        self._base_scene: MlinkScene | None = None
        self._sionna_scene = None

    def set_scene(self, sionna_scene) -> None:
        """
        Call once after ns3sionna loads a scene.
        """
        self._sionna_scene = sionna_scene
        base = MlinkScene.from_sionna(sionna_scene)
        base.sionna_scene = sionna_scene  # reuse underlying mi_scene for ray_features
        self._base_scene = base

    # ----- grid helpers -----

    def _choose_origin_xy(self, tx_pos: np.ndarray, rx_pos: np.ndarray) -> Tuple[float, float]:
        """
        Choose x0,y0 for the 64x64 window.
        """
        H, W = self.cfg.runtime.grid.H, self.cfg.runtime.grid.W
        cell = self.cfg.runtime.grid.cell_size_m
        span_x = (H - 1) * cell
        span_y = (W - 1) * cell

        pts = np.vstack([tx_pos[None, :], rx_pos])
        xmin, ymin = float(pts[:, 0].min()), float(pts[:, 1].min())
        xmax, ymax = float(pts[:, 0].max()), float(pts[:, 1].max())

        if self.cfg.runtime.grid.origin_policy == "fit_points":
            # try to fit bbox into window with margin
            margin = self.cfg.runtime.grid.origin_margin_cells * cell
            need_x = (xmax - xmin) + 2 * margin
            need_y = (ymax - ymin) + 2 * margin
            if need_x <= span_x and need_y <= span_y:
                x0 = xmin - margin
                y0 = ymin - margin
                return x0, y0

        # fallback: center window on bbox center
        cx = 0.5 * (xmin + xmax)
        cy = 0.5 * (ymin + ymax)
        x0 = cx - 0.5 * span_x
        y0 = cy - 0.5 * span_y
        return x0, y0

    def _clamp_z(self, z: float) -> float:
        assert self._base_scene is not None
        zmin = float(self._base_scene.mesh.bounds[0, 2])
        zmax = float(self._base_scene.mesh.bounds[1, 2])
        m = self.cfg.runtime.height.clamp_margin_m
        if zmax - zmin < 2 * m:
            return float(np.clip(z, zmin, zmax))
        return float(np.clip(z, zmin + m, zmax - m))

    def _make_scene_for_slice(self, tx_pos: np.ndarray, rx_positions: np.ndarray, z_slice: float) -> Tuple[MlinkScene, AntennaGrid]:
        """
        Create an mlink.Scene with tx coords and a single-slice rx_grid at z_slice.
        """
        assert self._base_scene is not None

        H, W = self.cfg.runtime.grid.H, self.cfg.runtime.grid.W
        cell = self.cfg.runtime.grid.cell_size_m

        x0, y0 = self._choose_origin_xy(tx_pos, rx_positions)

        z0 = self._clamp_z(float(z_slice))

        rx_grid = AntennaGrid(
            origin=np.asarray([x0, y0, z0], dtype=np.float32),
            deltas=np.asarray(
                [[cell, 0.0, 0.0],
                 [0.0, cell, 0.0],
                 [0.0, 0.0, 1.0]],  # K=1 so dz doesn't matter
                dtype=np.float32,
            ),
            shape=(1, H, W),
        )

        tx_grid = AntennaGrid(
            origin=tx_pos.astype(np.float32),
            deltas=np.asarray([[1.0, 0.0, 0.0],
                               [0.0, 1.0, 0.0],
                               [0.0, 0.0, 1.0]], dtype=np.float32),
            shape=(1, 1, 1),
        )

        antenna_db = AntennaDatabase.from_grid(tx_grid, rx_grid)

        sc = MlinkScene(
            mesh=self._base_scene.mesh,
            material_database=self._base_scene.material_database,
            face2material=self._base_scene.face2material,
            antenna_database=antenna_db,
            sionna_scene=self._sionna_scene,
        )
        return sc, rx_grid

    # ----- sampling -----

    @staticmethod
    def _bilinear_sample(img_hw: np.ndarray, i_f: float, j_f: float) -> float:
        """
        img_hw: (H,W), i_f/j_f are continuous indices where i is x-axis index, j is y-axis index.
        """
        H, W = img_hw.shape
        i0 = int(np.floor(i_f))
        j0 = int(np.floor(j_f))
        i1 = i0 + 1
        j1 = j0 + 1

        # clamp
        i0c = max(0, min(H - 1, i0))
        i1c = max(0, min(H - 1, i1))
        j0c = max(0, min(W - 1, j0))
        j1c = max(0, min(W - 1, j1))

        di = i_f - i0
        dj = j_f - j0

        v00 = img_hw[i0c, j0c]
        v10 = img_hw[i1c, j0c]
        v01 = img_hw[i0c, j1c]
        v11 = img_hw[i1c, j1c]

        return float((1 - di) * (1 - dj) * v00 +
                     di * (1 - dj) * v10 +
                     (1 - di) * dj * v01 +
                     di * dj * v11)

    def _world_to_ij_float(self, rx_grid: AntennaGrid, xyz: np.ndarray) -> Tuple[float, float]:
        # continuous (i,j,k) in grid coordinates
        frac = rx_grid.inv @ (xyz - rx_grid.origin)
        i_f, j_f = float(frac[0]), float(frac[1])
        return i_f, j_f

    # ----- inference -----

    @torch.no_grad()
    def _predict_rss_map_dbm(self, sc: MlinkScene, fc_hz: float) -> np.ndarray:
        """
        Returns (H,W) RSS map in dBm for the single TX and single z-slice.
        """
        ft = build_feature_tensor(sc, fc_hz, requested=list(self.cfg.runtime.features)).astype(np.float32)
        # ft: (tx=1, C=4, K=1, H, W) -> take [0,:,0]
        x_np = ft[0, :, 0, :, :]  # (4,H,W)

        x = torch.from_numpy(x_np[None, :, :, :]).to(self.device)  # (1,4,H,W)
        x = (x - self.x_mean) / self.x_std

        y_hat = self.model(x)  # (1,1,H,W) normalized
        y_hat = y_hat * self.y_std + self.y_mean  # de-norm
        rss = y_hat[0, 0].detach().cpu().numpy()  # (H,W) dBm @ train_tx_power_dbm

        # Calibrate to runtime tx power and (optional) frequency correction
        cal = self.cfg.calibration
        rss = rss + (cal.runtime_tx_power_dbm - cal.train_tx_power_dbm)

        if cal.frequency_correction == "friis":
            if abs(fc_hz - cal.train_frequency_hz) / cal.train_frequency_hz > 1e-9:
                rss = rss - 20.0 * np.log10(fc_hz / cal.train_frequency_hz)

        return rss.astype(np.float32)

    def predict_links(self, fc_hz: float, tx_pos: np.ndarray, rx_positions: np.ndarray) -> Tuple[List[int], List[float]]:
        """
        Returns:
          delays_ns: List[int] length N
          wb_loss_db: List[float] length N
        """
        assert self._base_scene is not None, "Call set_scene(scene) before predict_links()."
        tx_pos = np.asarray(tx_pos, dtype=np.float32)
        rx_positions = np.asarray(rx_positions, dtype=np.float32)

        N = rx_positions.shape[0]
        delays_ns = np.zeros((N,), dtype=np.int64)
        wb_loss_db = np.zeros((N,), dtype=np.float32)

        # delay: distance/c
        if self.cfg.link.delay_policy != "distance_over_c":
            raise ValueError(f"Unsupported delay_policy={self.cfg.link.delay_policy}")
        d = np.linalg.norm(rx_positions - tx_pos[None, :], axis=1)
        delays_ns[:] = np.rint((d / FREE_SPACE_CONSTS.c) * 1e9).astype(np.int64)

        # loss from rss
        if self.cfg.link.loss_policy != "tx_minus_rss":
            raise ValueError(f"Unsupported loss_policy={self.cfg.link.loss_policy}")

        height_cfg = self.cfg.runtime.height
        if height_cfg.policy == "fixed":
            z_keys = np.zeros((N,), dtype=np.int64)
            z_vals = np.full((N,), self._clamp_z(float(rx_positions[:, 2].mean())), dtype=np.float32)
            key_to_z = {0: float(z_vals[0])}
        elif height_cfg.policy == "per_receiver_bucket":
            b = max(float(height_cfg.bucket_m), 1e-6)
            z_bucket = np.rint(rx_positions[:, 2] / b).astype(np.int64)
            z_keys = z_bucket
            key_to_z: Dict[int, float] = {}
            for k in np.unique(z_bucket):
                z_mean = float(np.mean(rx_positions[z_bucket == k, 2]))
                key_to_z[int(k)] = self._clamp_z(z_mean)
        else:
            raise ValueError(f"Unsupported height.policy={height_cfg.policy}")

        # For each z bucket: build one slice map, sample all receivers in that bucket
        for k, z_slice in key_to_z.items():
            idxs = np.where(z_keys == k)[0]
            sc, rx_grid = self._make_scene_for_slice(tx_pos, rx_positions[idxs], z_slice)
            rss_map = self._predict_rss_map_dbm(sc, float(fc_hz))

            for ii in idxs:
                i_f, j_f = self._world_to_ij_float(rx_grid, rx_positions[ii])
                rss_dbm = self._bilinear_sample(rss_map, i_f, j_f)
                wb_loss_db[ii] = float(self.cfg.calibration.runtime_tx_power_dbm - rss_dbm)

        return delays_ns.tolist(), wb_loss_db.tolist()
