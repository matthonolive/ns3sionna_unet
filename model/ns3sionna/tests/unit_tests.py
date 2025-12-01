import os
import time
import numpy as np
import unittest
from common import message_pb2
from ns3sionna_server import SionnaEnv
import matplotlib.pyplot as plt
import seaborn as sns
import faulthandler
import sys

from ns3sionna_utils import MILLISECOND, SECOND

# Dump tracebacks to stderr or a file
faulthandler.enable(file=sys.stderr, all_threads=True)

# Optional: crash dump on segmentation fault
faulthandler.dump_traceback_later(10, repeat=False)  # after 10s, if not removed

'''
    Unittest for testing the server component of ns3sionna
    
    author: Zubow
'''
class TestServer(unittest.TestCase):

    def setUp(self):
        # This runs before every test
        self.env = SionnaEnv(rt_fast=False, VERBOSE=True)

        cwd = os.getcwd()
        print("Running unittest in current wdir:", cwd)

        self._start_time = time.time()


    def tearDown(self):
        # This runs after every test (optional)
        self.env.release()
        elapsed = time.time() - self._start_time
        print(f"{self.id()} took {elapsed:.4f} seconds")


    def _create_sim_init_constant_mob(self, mode=3):
        sim_info = message_pb2.Wrapper()
        sim_info.sim_init_msg.scene_fname               = "2_rooms_with_door/2_rooms_with_door_open.xml"
        sim_info.sim_init_msg.seed                      = 42
        sim_info.sim_init_msg.frequency                 = 5210
        sim_info.sim_init_msg.channel_bw                = 20
        sim_info.sim_init_msg.fft_size                  = 256
        sim_info.sim_init_msg.subcarrier_spacing        = 78125
        sim_info.sim_init_msg.mode                      = mode #1
        sim_info.sim_init_msg.sub_mode                  = 4 #1
        sim_info.sim_init_msg.min_coherence_time_ms     = 100000
        sim_info.sim_init_msg.time_evo_model            = 'position'

        n0 = sim_info.sim_init_msg.nodes.add()
        n0.id                                           = 0
        n0.constant_position_model.position.x           = 0.0
        n0.constant_position_model.position.y           = 0.0
        n0.constant_position_model.position.z           = 0.0

        n1 = sim_info.sim_init_msg.nodes.add()
        n1.id                                           = 1
        n1.constant_position_model.position.x           = 1.0
        n1.constant_position_model.position.y           = 2.0
        n1.constant_position_model.position.z           = 3.0

        return sim_info


    def _create_sim_init_wall_mob(self, mode=3, num_mobile_nodes=1):
        sim_info = message_pb2.Wrapper()
        sim_info.sim_init_msg.scene_fname               = "simple_room/simple_room_open.xml"
        sim_info.sim_init_msg.seed                      = 42
        sim_info.sim_init_msg.frequency                 = 5210
        sim_info.sim_init_msg.channel_bw                = 20
        sim_info.sim_init_msg.fft_size                  = 256
        sim_info.sim_init_msg.subcarrier_spacing        = 78125
        sim_info.sim_init_msg.mode                      = mode #1
        sim_info.sim_init_msg.sub_mode                  = 4 #1
        sim_info.sim_init_msg.min_coherence_time_ms     = 100000
        sim_info.sim_init_msg.time_evo_model            = 'position'

        # first node static
        n0 = sim_info.sim_init_msg.nodes.add()
        n0.id                                           = 0
        n0.constant_position_model.position.x           = 3.0
        n0.constant_position_model.position.y           = 2.0
        n0.constant_position_model.position.z           = 2.0

        for mob_user in range(num_mobile_nodes):
            # second++ node mobile with wall mode and constant speed of 1ms & uniform random direction
            n1 = sim_info.sim_init_msg.nodes.add()
            n1.id                                           = 1 + mob_user
            n1.random_walk_model.position.x                 = 4.0 + np.random.uniform(-1,1)
            n1.random_walk_model.position.y                 = 3.0 + np.random.uniform(-1,1)
            n1.random_walk_model.position.z                 = 1.0
            n1.random_walk_model.wall_value                 = 1
            n1.random_walk_model.speed.constant.value       = 4.0 + np.random.uniform(-1,1)
            n1.random_walk_model.direction.uniform.min      = -np.pi/10
            n1.random_walk_model.direction.uniform.max      = np.pi/10

        return sim_info


    def _create_sim_close_request(self):
        sim_info = message_pb2.Wrapper()
        sim_info.sim_close_request

        return sim_info


    def _create_channel_state_request(self, time, tx_node=0, rx_node=1):
        sim_info = message_pb2.Wrapper()
        sim_info.channel_state_request.tx_node = tx_node
        sim_info.channel_state_request.rx_node = rx_node
        sim_info.channel_state_request.time = time

        return sim_info


    def _create_channel_state_response(self):
        sim_info = message_pb2.Wrapper()

        return sim_info


    @unittest.skip("Not yet")
    def test_init(self):
        '''
        Test initialization phase for all 3 modes
        :return:
        '''
        for mode in [SionnaEnv.MODE_P2P]: #, SionnaEnv.MODE_P2MP, SionnaEnv.MODE_P2MP_LAH]:
            sim_init_msg = self._create_sim_init_constant_mob(mode=mode).sim_init_msg
            successful, error_msg = self.env.init_simulation_env(sim_init_msg)
            self.assertTrue(successful, error_msg)


    #@unittest.skip("Not yet")
    def test_get_csi(self):
        sim_init_msg = self._create_sim_init_constant_mob(mode=1).sim_init_msg

        successful, error_msg = self.env.init_simulation_env(sim_init_msg)
        self.assertTrue(successful, error_msg)

        csi_req = self._create_channel_state_request(time=0).channel_state_request
        csi_resp = self._create_channel_state_response()
        self.env.compute_cfr(csi_req, csi_resp)


    @unittest.skip("Not yet")
    def test_random_wall(self):
        sim_init_msg = self._create_sim_init_wall_mob().sim_init_msg

        successful, error_msg = self.env.init_simulation_env(sim_init_msg)
        self.assertTrue(successful, error_msg)

        time_arr = np.arange(50 * MILLISECOND, 2 * SECOND, 50 * MILLISECOND)

        csi_ts = []
        wb_loss = []
        delay = []
        for time in time_arr:
            csi_req = self._create_channel_state_request(time).channel_state_request
            csi_resp = self._create_channel_state_response()
            self.env.compute_cfr(csi_req, csi_resp)
            csi_ts.append(csi_resp.channel_state_response.csi[0].start_time)
            wb_loss.append(csi_resp.channel_state_response.csi[0].rx_nodes[0].wb_loss)
            delay.append(csi_resp.channel_state_response.csi[0].rx_nodes[0].delay)

        time_hist_0, pos_hist_0 = self.env._get_mobility_history(0)
        time_hist_1, pos_hist_1 = self.env._get_mobility_history(1)

        time_data_1 = time_hist_1
        mob_dt = np.asarray(time_data_1)
        mob_dt = mob_dt[2:] - mob_dt[1:-1]

        pos_data_0 = np.asarray(pos_hist_0)
        pos_data_1 = np.asarray(pos_hist_1)

        DO_PLOT = False
        if DO_PLOT:
            fig, axes = plt.subplots(2, 2, figsize=(10, 6))  # 2 row, 2 columns
            sns.histplot(mob_dt, ax=axes[0,0], kde=True)
            axes[0,0].set_xlabel('Mobility update dt [ns]')

            axes[0,1].scatter(pos_data_0[:,0], pos_data_0[:,1], 15, marker='p', alpha=0.5)
            axes[0,1].scatter(pos_data_1[:, 0], pos_data_1[:, 1], 5, marker='o', alpha=0.5)
            axes[0,1].grid(True)
            axes[0,1].set_aspect('equal')
            axes[0,1].set_xlim([self.env.bbox.min.x, self.env.bbox.max.x])
            axes[0,1].set_ylim([self.env.bbox.min.y, self.env.bbox.max.y])
            axes[0,1].set_xlabel('X coord (m)')
            axes[0,1].set_ylabel('Y coord (m)')

            axes[1,0].plot(np.asarray(csi_ts) / 1e9, wb_loss)
            axes[1,0].set_xlabel('Time (s)')
            axes[1,0].set_ylabel('Wideband loss (dB)')
            axes[1,0].grid(True)

            axes[1,1].plot(np.asarray(csi_ts) / 1e9, delay)
            axes[1,1].set_xlabel('Time (s)')
            axes[1,1].set_ylabel('Propagation delay (ns)')
            axes[1,1].grid(True)

            plt.tight_layout()
            plt.show()

    @unittest.skip("Not yet")
    def test_multi_user(self):

        '''
        Node 0 transmits to node 1; testing piggybacking of P2MP
        '''

        N = 3 # no. of mobile nodes
        sim_init_msg = self._create_sim_init_wall_mob(mode=2, num_mobile_nodes=N).sim_init_msg

        successful, error_msg = self.env.init_simulation_env(sim_init_msg)
        self.assertTrue(successful, error_msg)

        time_arr = np.arange(50 * MILLISECOND, 3000 * MILLISECOND, 50 * MILLISECOND)

        tx_node = 0
        rx_node = 1

        rx_nodes = list(self.env.node_info.keys())
        rx_nodes.remove(tx_node)

        csi_ts = []
        # for each rx node
        wb_loss_dict = {}
        delay_dict = {}
        for time in time_arr:
            csi_req = self._create_channel_state_request(time, tx_node, rx_node).channel_state_request
            csi_resp = self._create_channel_state_response()
            self.env.compute_cfr(csi_req, csi_resp)

            csi_ts.append(csi_resp.channel_state_response.csi[0].start_time)

            for rx_idx in range(len(csi_resp.channel_state_response.csi[0].rx_nodes)):

                rx_node_id = int(csi_resp.channel_state_response.csi[0].rx_nodes[rx_idx].id)
                wb_loss = csi_resp.channel_state_response.csi[0].rx_nodes[rx_idx].wb_loss
                delay = csi_resp.channel_state_response.csi[0].rx_nodes[rx_idx].delay

                if rx_node_id not in wb_loss_dict:
                    wb_loss_dict[rx_node_id] = []
                    delay_dict[rx_node_id] = []

                wb_loss_dict[rx_node_id].append(wb_loss)
                delay_dict[rx_node_id].append(delay)

        DO_PLOT = False
        if DO_PLOT:
            fig, axes = plt.subplots(2, 2, figsize=(10, 6))  # 2 row, 2 columns

            time_hist_0, pos_hist_0 = self.env._get_mobility_history(0)
            pos_data_0 = np.asarray(pos_hist_0)

            axes[0,0].scatter(pos_data_0[:,0], pos_data_0[:,1], 15, marker='p', alpha=0.5, label="tx")

            marker_labels = ['o', 's', 'v']
            for idx, rx_node_id in enumerate(rx_nodes):
                time_hist_tmp, pos_hist_tmp = self.env._get_mobility_history(rx_node_id)
                pos_data_tmp = np.asarray(pos_hist_tmp)
                axes[0,0].scatter(pos_data_tmp[:, 0], pos_data_tmp[:, 1], 5, marker=marker_labels[idx], alpha=0.5, label="rx" + str(rx_node_id))

            axes[0,0].grid(True)
            axes[0,0].set_aspect('equal')
            axes[0,0].set_xlim([self.env.bbox.min.x, self.env.bbox.max.x])
            axes[0,0].set_ylim([self.env.bbox.min.y, self.env.bbox.max.y])
            axes[0,0].set_xlabel('X coord (m)')
            axes[0,0].set_ylabel('Y coord (m)')
            axes[0,0].legend()

            for idx, rx_node_id in enumerate(rx_nodes):
                axes[0,1].plot(np.asarray(csi_ts) / 1e9, wb_loss_dict[rx_node_id], label="rx" + str(rx_node_id))
            axes[0,1].set_xlabel('Time (s)')
            axes[0,1].set_ylabel('Wideband loss (dB)')
            axes[0,1].grid(True)
            axes[0,1].legend()

            for idx, rx_node_id in enumerate(rx_nodes):
                axes[1,0].plot(np.asarray(csi_ts) / 1e9, delay_dict[rx_node_id], label="rx" + str(rx_node_id))
            axes[1,0].set_xlabel('Time (s)')
            axes[1,0].set_ylabel('Propagation delay (ns)')
            axes[1,0].grid(True)
            axes[1,0].legend()

            plt.tight_layout()
            plt.show()

    @unittest.skip("Not yet")
    def test_mode_p2mp_lah(self):

        '''
        Node 0 transmits to node 1; testing speculative channel computation
        '''

        N = 3 # no. of mobile nodes
        sim_init_msg = self._create_sim_init_wall_mob(mode=3, num_mobile_nodes=N).sim_init_msg

        successful, error_msg = self.env.init_simulation_env(sim_init_msg)
        self.assertTrue(successful, error_msg)


    @unittest.skip("Not yet")
    def test_coherence_time_mode1(self):
        K = 1 # one mobile user
        sim_init_msg = self._create_sim_init_wall_mob(mode=SionnaEnv.MODE_P2P, num_mobile_nodes=K).sim_init_msg

        successful, error_msg = self.env.init_simulation_env(sim_init_msg)
        self.assertTrue(successful, error_msg)

        time = 0 * MILLISECOND
        end_time = 200 * MILLISECOND

        csi_ts = []
        wb_loss = []
        delay = []
        while time < end_time:
            csi_req = self._create_channel_state_request(time).channel_state_request
            csi_resp = self._create_channel_state_response()
            self.env.compute_cfr(csi_req, csi_resp)
            # set time after Tc
            time = csi_resp.channel_state_response.csi[0].end_time
            csi_ts.append(csi_resp.channel_state_response.csi[0].start_time)
            wb_loss.append(csi_resp.channel_state_response.csi[0].rx_nodes[0].wb_loss)
            delay.append(csi_resp.channel_state_response.csi[0].rx_nodes[0].delay)

        time_hist_0, pos_hist_0 = self.env._get_mobility_history(0)
        time_hist_1, pos_hist_1 = self.env._get_mobility_history(1)

        csi_dt = np.asarray(csi_ts)
        csi_dt = csi_dt[2:] - csi_dt[1:-1]

        pos_data_0 = np.asarray(pos_hist_0)
        pos_data_1 = np.asarray(pos_hist_1)

        DO_PLOT = False
        if DO_PLOT:
            fig, axes = plt.subplots(2, 2, figsize=(10, 6))  # 2 row, 2 columns
            sns.histplot(csi_dt / 1e6, ax=axes[0,0], kde=True)
            axes[0,0].set_xlabel('CSI update dt [ms]')

            axes[0,1].scatter(pos_data_0[:,0], pos_data_0[:,1], 15, marker='p', alpha=0.5)
            axes[0,1].scatter(pos_data_1[:, 0], pos_data_1[:, 1], 5, marker='o', alpha=0.5)
            axes[0,1].grid(True)
            axes[0,1].set_aspect('equal')
            axes[0,1].set_xlim([self.env.bbox.min.x, self.env.bbox.max.x])
            axes[0,1].set_ylim([self.env.bbox.min.y, self.env.bbox.max.y])
            axes[0,1].set_xlabel('X coord (m)')
            axes[0,1].set_ylabel('Y coord (m)')

            axes[1,0].plot(np.asarray(csi_ts) / 1e9, wb_loss)
            axes[1,0].set_xlabel('Time (s)')
            axes[1,0].set_ylabel('Wideband loss (dB)')
            axes[1,0].grid(True)

            axes[1,1].plot(np.asarray(csi_ts) / 1e9, delay)
            axes[1,1].set_xlabel('Time (s)')
            axes[1,1].set_ylabel('Propagation delay (ns)')
            axes[1,1].grid(True)

            plt.tight_layout()
            plt.show()

if __name__ == "__main__":
    unittest.main()
