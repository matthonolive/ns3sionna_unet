import sionna.rt # must be the first import as otherwise Python crashes

import os
import argparse
import numpy as np
import time
import GPUtil
import zmq
import gc

from common import message_pb2
from common.message_debug import *

from collections import deque
import warnings

import tensorflow as tf

try:
    gpus = tf.config.list_physical_devices("GPU")
    for g in gpus:
        tf.config.experimental.set_memory_growth(g, True)
except Exception:
    pass


import mitsuba as mi
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
from mlink.feature import build_feature_tensor
from mlink.geometry import generate_wall_map, walls_to_mesh
from mlink.scene import Scene as MlinkScene
from mlink.channel_tdl import RtCfg, subcarrier_frequencies_centered, compute_tdl_batch

# import mobility models
from mobility import *

class UNetTdlPropagator:
    def __init__(
        self,
        run_dir: str,
        device: str = "cuda",
        # MUST match training_tdl.py grid params
        scale_m: float = 0.625,
        K: int = 4,
        H: int = 64,
        W: int = 64,
        z_step_cells: float = 1.0,
        z_margin_m: float = 0.3125,  # e.g., 0.5 cell * 0.625m
        origin_xy_mode: str = "bbox_min",  # "bbox_min" or "zero"
        dataset_features=None,
        y_wb_idx: int = 0,
        y_excess_idx: int = 1,
        y_tau_rms_idx: int = 2,
        no_path_wb: float = 199.5,
        fft_shift: bool = False,
    ):
        run = Path(run_dir)
        self.meta = json.loads((run / "meta.json").read_text())
        stats = np.load(run / "norm_stats.npz")

        self.device = torch.device(device)
        self.model = torch.jit.load(str(run / "model.pt"), map_location=self.device).eval()

        # grid sizes (use meta if present, else fall back)
        self.H = int(self.meta.get("H", H))
        self.W = int(self.meta.get("W", W))
        self.K = int(self.meta.get("K", K))
        self.scale_m = float(self.meta.get("scale_m", scale_m))
        self.z_step_m = float(self.scale_m * z_step_cells)
        self.z_margin_m = float(z_margin_m)
        self.origin_xy_mode = origin_xy_mode

        # features (must match what you trained with)
        if dataset_features is None:
            dataset_features = self.meta.get("dataset_features", ["binary_walls", "electrical_distance", "cost", "height_cond"])
        self.dataset_features = list(dataset_features)

        # heads
        self.y_wb_idx = int(y_wb_idx)
        self.y_excess_idx = int(y_excess_idx)
        self.y_tau_rms_idx = int(y_tau_rms_idx)
        self.no_path_wb = float(no_path_wb)
        self.fft_shift = bool(fft_shift)

        # --- Load stats; coerce to (C,1,1) and (Y,1,1) ---
        x_mean = stats["x_mean"]
        x_std  = stats["x_std"]
        y_mean = stats["y_mean"]
        y_std  = stats["y_std"]

        def to_c11(a):
            a = np.asarray(a)
            if a.ndim == 1:
                a = a[:, None, None]
            return a

        x_mean = to_c11(x_mean)
        x_std  = to_c11(x_std)
        y_mean = to_c11(y_mean)
        y_std  = to_c11(y_std)

        self.C_model = int(x_mean.shape[0])
        self.Y_model = int(y_mean.shape[0])

        self.x_mean = torch.from_numpy(x_mean).float().to(self.device)
        self.x_std  = torch.from_numpy(x_std).float().to(self.device).clamp_min(1e-6)
        self.y_mean = torch.from_numpy(y_mean).float().to(self.device)
        self.y_std  = torch.from_numpy(y_std).float().to(self.device).clamp_min(1e-6)

        self.keep_idx = None
        if "keep_idx" in stats.files:
            ki = stats["keep_idx"]
            if ki is not None and np.size(ki) > 0:
                self.keep_idx = np.asarray(ki, dtype=np.int64)

        # set later
        self._base_scene = None
        self._rx_grid = None
        self._rx_coords = None
        self._origin = None

        # cache per TX
        self._cached_tx_key = None
        self._cached_maps = None  # (K,y_ch,H,W) float32

    def attach_scene(self, sionna_scene, bbox, fc_hz: float, fft_size: int, subcarrier_spacing_hz: float, frequencies_hz: np.ndarray):
        self.fc_hz = float(fc_hz)
        self.fft_size = int(fft_size)
        self.subcarrier_spacing_hz = float(subcarrier_spacing_hz)
        self.frequencies = np.asarray(frequencies_hz, dtype=np.float64)

        self._base_scene = MlinkScene.from_sionna(sionna_scene)

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

    def _forward(self, x_chw: torch.Tensor) -> torch.Tensor:
        # x_chw: (C_model,H,W)
        x_n = (x_chw - self.x_mean) / self.x_std
        with torch.no_grad():
            pred_n = self.model(x_n.unsqueeze(0)).squeeze(0)  # (Y_model,H,W)
        pred = pred_n * self.y_std + self.y_mean
        return pred

    def predict_for_tx(self, tx_pos_xyz: np.ndarray):
        assert self._base_scene is not None, "call attach_scene() first"

        tx_pos_xyz = np.asarray(tx_pos_xyz, dtype=np.float32).reshape(3)
        key = tuple(np.round(tx_pos_xyz, 3).tolist())
        if key == self._cached_tx_key and self._cached_maps is not None:
            return

        adb = AntennaDatabase(tx_pos_xyz.reshape(1, 3), self._rx_coords, None, self._rx_grid)
        scene = replace(self._base_scene, antenna_database=adb)

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
                    "Your norm_stats/model are inconsistent."
                )
            x_chw = x_chw[self.keep_idx, :, :]

        if x_chw.shape[0] != self.C_model:
            if self.keep_idx is not None:
                raise RuntimeError(
                    f"Internal inconsistency: keep_idx size={self.keep_idx.size} "
                    f"but resulting x_chw has {x_chw.shape[0]} channels, expected {self.C_model}. "
                    "This suggests keep_idx is malformed."
                )
            else:
                raise RuntimeError(
                    f"UNet input channel mismatch: built K*c_in={x_chw.shape[0]} channels "
                    f"(K={self.K}, c_in={c_in}) but model expects C_model={self.C_model} "
                    f"(from norm_stats.x_mean). "
                    f"This usually means your training used channel selection/reordering, but meta.json doesn't record it."
                )

        x_t = torch.from_numpy(x_chw).to(self.device)
        pred = self._forward(x_t)  # (Y_model,H,W)

        # output reshape: expect Y_model == K*y_ch
        y_ch = 3  # wb, excess, tau_rms
        if self.Y_model != self.K * y_ch:
            raise RuntimeError(
                f"UNet output channel mismatch: model outputs Y_model={self.Y_model}, expected K*y_ch={self.K*y_ch}. "
                f"Either y_ch isn't 3 for this run, or the model isn't slice-stacked."
            )

        maps = pred.view(self.K, y_ch, self.H, self.W).detach().float().cpu().numpy()
        self._cached_maps = maps
        self._cached_tx_key = key

    # --- sampling + CFR synthesis stay the same as before ---
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
        assert self._cached_maps is not None, "call predict_for_tx() first"

        rx = np.asarray(rx_pos_xyz, dtype=np.float32).reshape(3)
        x0, y0, z0 = self._origin.tolist()
        xf = (rx[0] - x0) / self.scale_m
        yf = (rx[1] - y0) / self.scale_m
        kf = (rx[2] - z0) / self.z_step_m

        maps = self._cached_maps  # (K,3,H,W)

        if (xf < 0 or xf > self.W-1 or yf < 0 or yf > self.H-1 or kf < 0 or kf > self.K-1):
            print(f"[warn] RX outside grid -> clamping: xf={xf:.2f}, yf={yf:.2f}, kf={kf:.2f}, rx={rx}")
        wb = self._trilerp(maps[:, self.y_wb_idx, :, :], kf, yf, xf)
        tau = self._trilerp(maps[:, self.y_tau_rms_idx, :, :], kf, yf, xf)
        ex = self._trilerp(maps[:, self.y_excess_idx, :, :], kf, yf, xf) if self.y_excess_idx >= 0 else None
        return float(wb), float(tau), (None if ex is None else float(ex))

    def synthesize_cfr(self, tau_rms_ns: float, seed: int) -> np.ndarray:
        
        N = self.fft_size
        df = self.subcarrier_spacing_hz
        Ts = 1.0 / (N * df)

        tau = max(float(tau_rms_ns), 1e-3) * 1e-9
        L = int(np.clip(np.ceil(6.0 * tau / Ts), 1, N))
        t = np.arange(L, dtype=np.float64) * Ts
        p = np.exp(-t / max(tau, 1e-12))
        p = p / (p.sum() + 1e-12)

        rng = np.random.default_rng(seed)
        taps = (rng.standard_normal(L) + 1j * rng.standard_normal(L)) * np.sqrt(0.5 * p)

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
                 use_unet=False, unet_run="unet", unet_device="cuda", unet_no_path_wb=199.5, unet_y_wb_idx=0, unet_y_tau_rms_idx=2, unet_y_excess_idx=-1,
                 VERBOSE=True,
                 CHECKS_ENABLED=True):
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
            self.rt_max_depth = 5  # sufficient even for rich multipath
            self.rt_samples_per_src = 10 ** 6  # 10 ** 6
            self.rt_los = True  # compute and include the direct Line-of-Sight path when it exists
            self.rt_specular_reflection = True  # Can rays bounce off surfaces?
            self.rt_diffuse_reflection = True
            self.rt_refraction = True  # Can rays pass through materials?
            self.rt_synthetic_array = False  # Set True for fast simulation using one ray trace for whole array; per-element effects computed analytically
            self.rt_diffraction = True  # costly
            self.rt_edge_diffraction = True  # rays that bend around edges
            self.rt_diffraction_lit_region = True  # higher physical accuracy; for mmWave or THz channels

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
        self.unet_y_wb_idx = unet_y_wb_idx
        self.unet_y_tau_rms_idx = unet_y_tau_rms_idx
        self.unet_y_excess_idx = unet_y_excess_idx
        self._unet = None

        if not self.use_unet:
            print(f'Init ns3sionna with rt_fast={rt_fast}, est_csi={est_csi}')

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

        ##SONIC
        if self.use_unet:
            if self._unet is None:
                self._unet = UNetTdlPropagator(
                    run_dir=self.unet_run,
                    device=self.unet_device,
                    no_path_wb=self.unet_no_path_wb,
                    y_wb_idx=self.unet_y_wb_idx,
                    y_tau_rms_idx=self.unet_y_tau_rms_idx,
                    y_excess_idx=self.unet_y_excess_idx,
                    # IMPORTANT: set these to match training_tdl CFG
                    scale_m=0.625,
                    z_step_cells=1.0,
                    z_margin_m=0.625 * 0.5,   # if training used z_margin=0.5 cells
                    origin_xy_mode="bbox_min", # or "zero" if your scenes are 0-based
                )
            self._unet.attach_scene(self.scene, self.bbox, self.fc, self.fft_size, self.subcarrier_spacing, self.frequencies)

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
                                     pattern="tr38901", polarization="V")

        self.scene.rx_array = PlanarArray(num_rows=1, num_cols=1, vertical_spacing=0.5, horizontal_spacing=0.5,
                                     pattern="dipole", polarization="V")

        # set current sim time to 0ns
        self.sim_time = 0

        return True, "OK"


    def compute_cfr(self, csi_req, reply_wrapper):

        tx_node_id = csi_req.tx_node
        rx_node_id = csi_req.rx_node

        if self.mode == SionnaEnv.MODE_P2MP_LAH:
            if isinstance(self.node_info[tx_node_id], RandomWalkMobility) and isinstance(self.node_info[rx_node_id], ConstantMobility):
                # Exploit channel reciprocity - swap mobile TX with static RX
                csi_req.tx_node = rx_node_id
                csi_req.rx_node = tx_node_id

        # check if mode 3 can be used
        if self.mode == SionnaEnv.MODE_P2MP_LAH and isinstance(self.node_info[tx_node_id], ConstantMobility):
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
            # move in time
            dt = req_sim_time - self.sim_time

            csi_tc_arr = []
            for node_id in nodes_to_update:
                # perform walk
                self._walk(node_id, dt)
                if node_id != tx_node_id:
                    tc = coherence_from_velocities(self.node_info[node_id].velocity,
                                                    self.node_info[tx_node_id].velocity, self.fc,
                                                    pos_tx=self.node_info[node_id].pos,
                                                    pos_rx=self.node_info[tx_node_id].pos)
                    csi_tc_arr.append(tc)

            # take the worst case Tc from all RX nodes
            Tc_p2mp = int(np.min(np.asarray(csi_tc_arr)))

            # update time
            self.sim_time = req_sim_time
            lah_time_vec.append(req_sim_time)
            # new req time is old + Tc
            req_sim_time = req_sim_time + Tc_p2mp


        # place TX and RX nodes together with their future positions
        rx_nodes = nodes_to_update
        rx_nodes.remove(tx_node_id)

        self._place_tx_rx_nodes_with_lah(lah_time_vec, tx_node_id, rx_nodes)

        # create pathsolver; todo: check reuse
        p_solver  = PathSolver()

        # Compute propagation paths
        paths = p_solver(scene=self.scene,
                         max_depth=self.rt_max_depth,
                         samples_per_src=self.rt_samples_per_src,
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
            rx_node_info.end_time2 = tc
            csi_tc_arr.append(tc)

        # take the worst case Tc from all RX nodes
        Tc_p2mp = int(np.min(np.asarray(csi_tc_arr)))

        print(f'{self.sim_time / 1e9}s: Computed CSI with Tc: {round(Tc_p2mp / 1e6,2)}ms, #links: {len(rx_nodes)}')

        csi.end_time = self.sim_time + Tc_p2mp
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


    def _compute_cfr_via_position(self, req_sim_time, tx_node, rx_node, req_mode):
        '''
        Compute the link propagation delay, wideband loss and normalized CFR
        :param req_sim_time: current simulation time
        :param tx_node: the transmitter node id
        :param rx_node: the receiver node id
        :return: (list(rx_node), list(link propagation delay), list(wideband loss), list(normalized CFR))
        '''

        # execute mobility
        dt = req_sim_time - self.sim_time

        # estimate the node we need to update their position
        if req_mode == SionnaEnv.MODE_P2P:
            nodes_to_update = [tx_node, rx_node] # only TX and RX
        else:
            # both P2MP and P2MP_LAH
            nodes_to_update = list(self.node_info.keys())

        for node_id in nodes_to_update:
            self._walk(node_id, dt)

        # update time
        self.sim_time = req_sim_time

        # place TX and RX
        rx_nodes = nodes_to_update
        rx_nodes.remove(tx_node)

        if self.use_unet:
            tx_pos = np.array(self.node_info[tx_node].pos, dtype=np.float32)
            self._unet.predict_for_tx(tx_pos)

            lnk_delay_arr = []
            lnk_loss_arr = []
            h_normalized_arr = []

            for curr_rx_node in rx_nodes:
                rx_pos = np.array(self.node_info[curr_rx_node].pos, dtype=np.float32)

                wb_db, tau_rms_ns, excess_ns = self._unet.sample_heads(rx_pos)

                # no-path guard
                if wb_db >= (self.unet_no_path_wb - 1e-3):
                    lnk_loss_arr.append(float(self.unet_no_path_wb))
                    lnk_delay_arr.append(0)
                    h_normalized_arr.append(np.zeros((self.fft_size,), dtype=np.complex64))
                    continue

                # delay: d/c + excess
                d_m = float(np.linalg.norm(tx_pos - rx_pos))
                base_ns = d_m / 299792458.0 * 1e9
                ex = 0.0 if excess_ns is None else float(max(0.0, excess_ns))
                lnk_delay_arr.append(int(round(base_ns + ex)))

                # wb loss direct from model
                lnk_loss_arr.append(float(wb_db))

                # normalized CFR from G(tau_rms)
                seed = (int(self.my_seed) * 1315423911) ^ (int(tx_node) * 2654435761) ^ (int(curr_rx_node) * 97531) ^ (int(self.sim_time) & 0xffffffff)
                print(f"rx={curr_rx_node} wb={wb_db:.1f}dB")

                #Testing physical laws
                delay_ns = base_ns + ex
                delay_int_ns = int(round(delay_ns))

                print(
                    f"[CFR] tx={tx_node} rx={curr_rx_node} "
                    f"d={d_m:.3f} m "
                    f"delay={delay_int_ns} ns (raw={delay_ns:.3f}; base={base_ns:.3f}+ex={ex:.3f}) "
                    f"tau_rms={tau_rms_ns:.4f} ns"
                )
                
                h_norm = self._unet.synthesize_cfr(tau_rms_ns=tau_rms_ns, seed=seed)

                # sanity: mean |H|^2 ~ 1
                if self.CHECKS_ENABLED:
                    p = float(np.mean(np.abs(h_norm) ** 2))
                    assert abs(p - 1.0) < 1e-2

                h_normalized_arr.append(h_norm)

            return rx_nodes, lnk_delay_arr, lnk_loss_arr, h_normalized_arr

        self._place_tx_rx_node(tx_node, rx_nodes)

        # create pathsolver; todo: check reuse
        p_solver  = PathSolver()

        # Compute propagation paths
        paths = p_solver(scene=self.scene,
                         max_depth=self.rt_max_depth,
                         samples_per_src=self.rt_samples_per_src,
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

        lnk_delay_arr = []
        lnk_loss_arr = []
        h_normalized_arr = []

        for rx_id, curr_rx_node in enumerate(rx_nodes):
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
                print(f'{self.sim_time/1e9}s: lnk_delay = {lnk_delay}ns, wb_loss = {lnk_loss:.3f}dB, CFR shape: {h_normalized.shape}')

            lnk_delay_arr.append(lnk_delay)
            lnk_loss_arr.append(lnk_loss)
            h_normalized_arr.append(h_normalized)

        return rx_nodes, lnk_delay_arr, lnk_loss_arr, h_normalized_arr


    def _get_mobility_history(self, node_id):
        ts = sorted(self.node_info[node_id].pos_history.keys())
        pos = [self.node_info[node_id].pos_history[t] for t in ts]
        return ts, pos


    def _init_mobility(self, sim_init_msg):
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
            ns3_msg_str = socket.recv()

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
                do_terminate = True
                resp_msg.sim_ack.SetInParent()

            # Serialize and send the reply message
            socket.send(resp_msg.SerializeToString())

        socket.close()
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
    parser.add_argument("--rt_max_parallel_links", type=int, default=256, help="Max no. of link simulated at once; depends on GPU memory")
    parser.add_argument("--est_csi", action="store_true", help="Send CSI vectors (needed for spectrum model)")
    parser.add_argument("--verbose", help="Whether to run in verbose mode", action='store_true')

    parser.add_argument("--use_unet", action="store_true", help="Use U-Net surrogate instead of Sionna ray tracing")
    parser.add_argument("--unet_run", type=str, default="unet", help="Path to run dir containing model.pt/meta.json/norm_stats.npz")
    parser.add_argument("--unet_device", type=str, default="cuda", help="cpu|cuda|cuda:0")
    parser.add_argument("--unet_no_path_wb", type=float, default=199.5, help="No-path sentinel wb_loss (dB)")
    parser.add_argument("--unet_y_wb_idx", type=int, default=0, help="Which output channel is wb_loss")
    parser.add_argument("--unet_y_tau_rms_idx", type=int, default=2, help="Which output channel is tau_rms (ns)")
    parser.add_argument("--unet_y_excess_idx", type=int, default=-1, help="Optional channel index for excess delay (ns), -1 disables")
    args = parser.parse_args()

    print("ns3sionna v1.0")
    while True:
        print("Using config: model_folder=%s, single_run=%s, mode=%d, rt_fast=%s, rt_max_parallel_links=%d, est_csi=%r"
              % (args.model_folder, args.single_run, args.default_mode, args.rt_fast, args.rt_max_parallel_links, args.est_csi))
        print("Waiting for new job ...")
        env = SionnaEnv(args.model_folder, args.rt_fast, args.default_mode, args.rt_max_parallel_links,
                        args.est_csi, args.use_unet, args.unet_run, args.unet_device, args.unet_no_path_wb,
                        args.unet_y_wb_idx,  args.unet_y_tau_rms_idx, args.unet_y_excess_idx,
                        VERBOSE=args.verbose)
        env.run()

        if args.single_run:
            break

