def print_sim_init(sim_init_msg):
    '''
    Debugging
    '''
    print("Sim init:")
    print("    Env:", sim_init_msg.scene_fname)
    print("    Seed:", sim_init_msg.seed, ", fc:", sim_init_msg.frequency, ", B:", sim_init_msg.channel_bw
          , ", FFT:", sim_init_msg.fft_size, ", dSC:", sim_init_msg.subcarrier_spacing
          , ", Tc_min:", sim_init_msg.min_coherence_time_ms, ", time_evo:", sim_init_msg.time_evo_model)

    print("Node Information:")
    for node_info in sim_init_msg.nodes:
        print("    Node ID:", node_info.id)

        if node_info.HasField("constant_position_model"):
            print("        Constant Position Model:")
            position = node_info.constant_position_model.position
            print("            Position:", position.x, position.y, position.z)

        elif node_info.HasField("random_walk_model"):
            print("        Random Walk Model:")
            random_walk_model = node_info.random_walk_model
            position = random_walk_model.position
            print("            Position:", position.x, position.y, position.z)

            if random_walk_model.HasField("wall_value"):
                print("            Mode Wall:", random_walk_model.wall_value)
            elif random_walk_model.HasField("time_value"):
                print("            Mode Time:", random_walk_model.time_value)
            elif random_walk_model.HasField("distance_value"):
                print("            Mode Distance:", random_walk_model.distance_value)

            speed = random_walk_model.speed
            if speed.HasField("uniform"):
                print(f"            Speed: Uniform({speed.uniform.min}, {speed.uniform.max})")
            elif speed.HasField("constant"):
                print(f"            Speed: Constant({speed.constant.value})")
            elif speed.HasField("normal"):
                print(f"            Speed: Normal({speed.normal.mean}, {speed.normal.variance})")

            direction = random_walk_model.direction
            if direction.HasField("uniform"):
                print(f"            Direction: Uniform({direction.uniform.min}, {direction.uniform.max})")
            elif direction.HasField("constant"):
                print(f"            Direction: Constant({direction.constant.value})")
            elif direction.HasField("normal"):
                print(f"            Direction: Normal({direction.normal.mean}, {direction.normal.variance})")


def print_csi_request(csi_req):
    print("%0.9fs: CSI request:" % (csi_req.time / 1e9))
    print("    NodeId1:  ", csi_req.tx_node)
    print("    NodeId2:  ", csi_req.rx_node)


def print_csi_response(simulation_time, node_a, node_b, node_a_pos, node_b_pos, delay, loss, ttl):
    print("%0.9fs: Propagation response:" % (simulation_time / 1e9))
    print("Tx:%d - Rx:%d" % (node_a, node_b))
    print("    posTx: ", node_a_pos)
    print("    posRx: ", node_b_pos)
    print("    delay:  ", delay)
    print("    loss:  ", loss)
    print("    ttl:", ttl)
