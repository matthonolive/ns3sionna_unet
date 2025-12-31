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
import mitsuba as mi
import math
from millify import millify

import torch

from ns3sionna_utils import subcarrier_frequencies, compute_coherence_time, SECOND, MILLISECOND, coherence_from_velocities, \
    MAX_COHERENCE_TIME

os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")

import sionna
from sionna.rt import load_scene



# import mobility models
from mobility import *

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
    def __init__(self,
             model_folder: str = "./models/",
             unet_config: str = "./unet_model_config.json",
             default_mode: int | None = None,
             VERBOSE: bool = True,
             CHECKS_ENABLED: bool = True, 
             emit_cfr: bool = False):
        
        self.model_folder = model_folder 
        self.VERBOSE = VERBOSE 
        self.CHECKS_ENABLED = CHECKS_ENABLED 
        self.emit_cfr = emit_cfr
        
        self.default_mode = self.MODE_P2P if default_mode is None else default_mode 


        print("torch.cuda.is_available =", torch.cuda.is_available())
        if torch.cuda.is_available():
            print("torch device name =", torch.cuda.get_device_name(0))
        from unet_propagation import UNetModelCfg, UNetPropagator

        self.unet_config_path = unet_config 
        self.unet_cfg = UNetModelCfg.load(unet_config)
        self.propagator = UNetPropagator(self.unet_cfg) 

        print("UNet device =", self.propagator.device)

        print(f"Init ns3sionna (UNet mode): {self.unet_cfg.name}")
        print(f"  config  : {self.unet_config_path}")
        print(f"  model   : {self.unet_cfg.torchscript_model}")
        print(f"  stats   : {self.unet_cfg.norm_stats_npz}")
        print(f"  device  : {self.unet_cfg.runtime.device}")
        print(f"  grid    : {self.unet_cfg.runtime.grid.H}x{self.unet_cfg.runtime.grid.W}, cell={self.unet_cfg.runtime.grid.cell_size_m}m")

        self.node_info = {}


    def init_simulation_env(self, sim_init_msg):
        if self.VERBOSE:
            print_sim_init(sim_init_msg)

        # Load the sionna scene (XML)
        filepath = os.path.join(self.model_folder, sim_init_msg.scene_fname)
        try:
            self.scene = load_scene(filepath)
        except Exception as e:
            return False, f"Failed to load scene file in: {filepath}, error: {e}"

        self.bbox = self.scene.mi_scene.bbox()

        if self.VERBOSE:
            dx = self.bbox.max.x - self.bbox.min.x
            dy = self.bbox.max.y - self.bbox.min.y
            dz = self.bbox.max.z - self.bbox.min.z
            print(f"Scenario with dx={dx:.2f}, dy={dy:.2f}, dz={dz:.2f}")

        # mode/submode
        self.mode = sim_init_msg.mode if sim_init_msg.mode > -1 else self.default_mode
        self.sub_mode = sim_init_msg.sub_mode if sim_init_msg.sub_mode > -1 else 0  # you can repurpose this later
        self.time_evo_model = sim_init_msg.time_evo_model

        # Operating params (still needed for delay/coherence calculations)
        self.fc = sim_init_msg.frequency * 1e6
        self.scene.frequency = self.fc
        self.scene.bandwidth = sim_init_msg.channel_bw * 1e6

        # Attach scene to UNet propagator (THIS is the “feeding the unet the scene” step)
        self.propagator.set_scene(self.scene)

        # FFT/OFDM params: only keep if still used elsewhere
        self.fft_size = sim_init_msg.fft_size
        self.subcarrier_spacing = sim_init_msg.subcarrier_spacing
        self.min_coherence_time_ms = sim_init_msg.min_coherence_time_ms

        if self.emit_cfr:
            self.frequencies = subcarrier_frequencies(
                num_subcarriers=self.fft_size,
                subcarrier_spacing=self.subcarrier_spacing
            )
        else:
            self.frequencies = None

        print(
            f"Operating in mode: {self.mode}, sub_mode: {self.sub_mode}, time_evo_model: {self.time_evo_model}, "
            f"fc: {sim_init_msg.frequency} MHz, B: {sim_init_msg.channel_bw} MHz"
        )

        # Seeds (you can remove TF entirely if it’s now unused elsewhere)
        np.random.seed(sim_init_msg.seed)
        self.my_seed = sim_init_msg.seed

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
        Compute the requested CFR with look-ahead (LAH), but using the UNet propagator.
        Produces only per-link delay + wideband loss (no CSI / no ray tracing).

        :param csi_req: received CSI request (ZMQ)
        :param reply_wrapper: the response
        :return: num_computed_lnks
        '''
        if self.VERBOSE:
            print_csi_request(csi_req)

        tx_node_id = csi_req.tx_node
        rx_node_id = csi_req.rx_node  # must be included in result set
        req_sim_time = csi_req.time   # CFR at that point in time [ns]

        assert self.time_evo_model == 'position'

        # --- mobility / look-ahead time construction ---
        nodes_to_update = list(self.node_info.keys())

        # guard: if we have <=1 node, nothing to compute
        if tx_node_id not in self.node_info:
            raise KeyError(f"Unknown tx_node_id={tx_node_id}")
        if len(nodes_to_update) <= 1:
            return 0

        denom = (len(nodes_to_update) - 1)
        look_ahead = math.floor(self.sub_mode / denom) if denom > 0 else 1
        if look_ahead < 1:
            look_ahead = 1  # fallback: at least compute one snapshot

        print(f'compute CFR to #RX={len(nodes_to_update) - 1} with LAH={look_ahead}')

        # We will snapshot positions/velocities at each LAH time so we don’t depend on get_pos_at/get_velo_at.
        lah_time_vec = []
        pos_snap = {}   # time_ns -> {node_id: np.array([x,y,z])}
        vel_snap = {}   # time_ns -> {node_id: np.array([vx,vy,vz]) or whatever your velocity type is}

        curr_time = req_sim_time
        for _ in range(look_ahead):
            dt = curr_time - self.sim_time

            # walk all nodes by dt
            for node_id in nodes_to_update:
                self._walk(node_id, dt)

            # snapshot after walking to curr_time
            pos_snap[curr_time] = {}
            vel_snap[curr_time] = {}
            for node_id in nodes_to_update:
                # node_info[...] should have .pos and .velocity (as used in your original code)
                pos_snap[curr_time][node_id] = np.array(self.node_info[node_id].pos, dtype=np.float32)
                vel_snap[curr_time][node_id] = self.node_info[node_id].velocity

            # compute Tc across all RX (worst-case = min)
            csi_tc_arr = []
            tx_pos = pos_snap[curr_time][tx_node_id]
            tx_vel = vel_snap[curr_time][tx_node_id]
            for node_id in nodes_to_update:
                if node_id == tx_node_id:
                    continue
                rx_pos = pos_snap[curr_time][node_id]
                rx_vel = vel_snap[curr_time][node_id]
                tc = coherence_from_velocities(rx_vel, tx_vel, self.fc,
                                            pos_tx=rx_pos,
                                            pos_rx=tx_pos)
                csi_tc_arr.append(tc)

            Tc_p2mp = int(np.min(np.asarray(csi_tc_arr))) if len(csi_tc_arr) else MAX_COHERENCE_TIME

            # update sim time to curr_time
            self.sim_time = curr_time
            lah_time_vec.append(curr_time)

            # next look-ahead time is + Tc
            curr_time = curr_time + Tc_p2mp

        # --- receiver set (must include rx_node_id) ---
        rx_nodes = [n for n in nodes_to_update if n != tx_node_id]
        if rx_node_id != tx_node_id and rx_node_id not in rx_nodes and rx_node_id in self.node_info:
            rx_nodes.append(rx_node_id)

        # --- build response using UNet propagator ---
        chan_response = reply_wrapper.channel_state_response
        Tc_p2mp_lah = []
        num_computed_lnks = 0

        for lah_time in lah_time_vec:
            csi = chan_response.csi.add()
            csi.start_time = lah_time

            # TX at this time
            tx_pos = pos_snap[lah_time][tx_node_id]
            csi.tx_node.id = tx_node_id
            csi.tx_node.position.x = float(tx_pos[0])
            csi.tx_node.position.y = float(tx_pos[1])
            csi.tx_node.position.z = float(tx_pos[2])

            # RX positions at this time (in consistent order with rx_nodes)
            rx_positions = np.asarray([pos_snap[lah_time][rid] for rid in rx_nodes], dtype=np.float32)

            # UNet prediction: delay + wideband loss per RX
            lnk_delay_arr, lnk_loss_arr = self.propagator.predict_links(self.fc, tx_pos, rx_positions)

            # Per-RX fields + per-link coherence time (still useful for LAH interval end markers)
            csi_tc_arr = []
            tx_vel = vel_snap[lah_time][tx_node_id]

            for i, rid in enumerate(rx_nodes):
                rx_node_info = csi.rx_nodes.add()
                rx_pos = rx_positions[i]

                rx_node_info.id = rid
                rx_node_info.position.x = float(rx_pos[0])
                rx_node_info.position.y = float(rx_pos[1])
                rx_node_info.position.z = float(rx_pos[2])

                rx_node_info.delay = int(lnk_delay_arr[i])
                rx_node_info.wb_loss = float(lnk_loss_arr[i])

                if self.emit_cfr:
                    dn = int(lnk_delay_arr[i])
                    tau_s = dn * 1e-9
                    f = np.asarray(self.frequencies, dtype=np.float64)
                    h = np.exp(-1j * 2.0 * np.pi * f * tau_s).astype(np.complex64)

                    if hasattr(rx_node_info, "frequencies"):
                        rx_node_info.frequencies.extend([int(float(x)) for x in self.frequencies])
                    if hasattr(rx_node_info, "csi_real"):
                        rx_node_info.csi_real.extend(np.real(h).astype(np.float32).tolist())
                    if hasattr(rx_node_info, "csi_imag"):
                        rx_node_info.csi_imag.extend(np.imag(h).astype(np.float32).tolist())


                # coherence time (keep same call convention as your original)
                rx_vel = vel_snap[lah_time][rid]
                tc = coherence_from_velocities(rx_vel, tx_vel, self.fc,
                                            pos_tx=rx_pos,
                                            pos_rx=tx_pos)
                tc = int(tc)
                rx_node_info.end_time2 = csi.start_time + tc
                csi_tc_arr.append(tc)

                if self.VERBOSE:
                    print(f'{lah_time/1e9}s: {tx_node_id}->{rid} '
                        f'lnk_delay={int(lnk_delay_arr[i])}ns, wb_loss={float(lnk_loss_arr[i]):.3f}dB')

            # worst-case Tc for this snapshot (min over RX)
            Tc_p2mp = int(np.min(np.asarray(csi_tc_arr))) if len(csi_tc_arr) else MAX_COHERENCE_TIME
            Tc_p2mp_lah.append(Tc_p2mp)
            csi.end_time = csi.start_time + Tc_p2mp - 1  # -1ns to keep non-overlapping intervals

            num_computed_lnks += len(rx_nodes)

        print(f'{self.sim_time / 1e9}s: Computed (UNet) LAH with Tc: '
            f'{np.round(np.asarray(Tc_p2mp_lah) / 1e6, 2)}ms, #links: {num_computed_lnks}')

        return num_computed_lnks

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

            if self.emit_cfr:
                h = h_normalized[idx]  # complex vector length fft_size

                # (optional but safe) include frequencies if the proto supports it
                if hasattr(rx_node_info, "frequencies"):
                    rx_node_info.frequencies.extend([int(float(x)) for x in self.frequencies])

                # CSI arrays (these fields exist in the original ns3sionna proto)
                if hasattr(rx_node_info, "csi_real"):
                    rx_node_info.csi_real.extend(np.real(h).astype(np.float32).tolist())
                if hasattr(rx_node_info, "csi_imag"):
                    rx_node_info.csi_imag.extend(np.imag(h).astype(np.float32).tolist())

            # compute coherence time: with direction vectors you can compute the radial (projected) relative
            # speed directly and from that the Doppler and coherence time.
            tc = coherence_from_velocities(self.node_info[comp_rx_node_id].velocity,
                                            self.node_info[tx_node_id].velocity, self.fc,
                                            pos_tx=self.node_info[comp_rx_node_id].pos,
                                            pos_rx=self.node_info[tx_node_id].pos)
            rx_node_info.end_time2 = csi.start_time + int(tc)
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

        rx_nodes = nodes_to_update
        rx_nodes.remove(tx_node)

        tx_pos = np.asarray(self.node_info[tx_node].pos, dtype=np.float32)
        rx_positions = np.asarray([self.node_info[r].pos for r in rx_nodes], dtype=np.float32)

        lnk_delay_arr, lnk_loss_arr = self.propagator.predict_links(self.fc, tx_pos, rx_positions)

        if self.emit_cfr:
            f = np.asarray(self.frequencies, dtype=np.float64)  # (Nsub,)
            h_normalized_arr = []
            for dn in lnk_delay_arr:
                tau_s = float(dn) * 1e-9
                h = np.exp(-1j * 2.0 * np.pi * f * tau_s).astype(np.complex64)  # (Nsub,)
                h_normalized_arr.append(h)
        else:
            h_normalized_arr = [None] * len(rx_nodes)

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
                    # if len(self.gpus) > 0:
                    #     GPUtil.showUtilization()

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
    parser.add_argument("--est_csi", help="Whether to estimate complex CSI per OFDM subcarrier", type=bool, default=True)
    parser.add_argument("--verbose", help="Whether to run in verbose mode", action='store_true')
    parser.add_argument("--unet_config", type=str, required=True,
                    help="Path to UNet model config JSON (see unet_model_config.json)")
    parser.add_argument("--emit_cfr", action="store_true", help="Include CFR/CSI arrays in replies")
    parser.add_argument("--no_emit_cfr", action="store_false", dest="emit_cfr")
    parser.set_defaults(emit_cfr=False)
    args = parser.parse_args()

    print("ns3sionna v1.0")
    while True:
        print("Using config: model_folder=%s, single_run=%s, mode=%d, rt_fast=%s, rt_max_parallel_links=%d, est_csi=%r"
              % (args.model_folder, args.single_run, args.default_mode, args.rt_fast, args.rt_max_parallel_links, args.est_csi))
        print("Waiting for new job ...")
        env = SionnaEnv(args.model_folder, unet_config=args.unet_config, VERBOSE=args.verbose, emit_cfr=args.emit_cfr)
        env.run()

        if args.single_run:
            break

