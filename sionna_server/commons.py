def print_simulation_info(simulation_info):
    """
    Debugging
    """
    print("Simulation Information:")
    print("    Environment:", simulation_info.scene_fname)
    print("    Seed:", simulation_info.seed)
    print("    Frequency:", simulation_info.frequency)
    print("    Channel-BW:", simulation_info.channel_bw)
    print("    FFT size:", simulation_info.fft_size)

    print("Node Information:")
    for node_info in simulation_info.nodes:
        print("Node ID:", node_info.id)

        if node_info.HasField("constant_position_model"):
            print("    Constant Position Model:")
            position = node_info.constant_position_model.position
            print("        Position:", position.x, position.y, position.z)

        elif node_info.HasField("random_walk_model"):
            print("    Random Walk Model:")
            random_walk_model = node_info.random_walk_model
            position = random_walk_model.position
            print("        Position:", position.x, position.y, position.z)

            if random_walk_model.HasField("time_value"):
                print("        Mode Time:", random_walk_model.time_value)
            elif random_walk_model.HasField("distance_value"):
                print("        Mode Distance:", random_walk_model.distance_value)

            speed = random_walk_model.speed
            if speed.HasField("uniform"):
                print(f"        Speed: Uniform({speed.uniform.min}, {speed.uniform.max})")
            elif speed.HasField("constant"):
                print(f"        Speed: Constant({speed.constant.value})")
            elif speed.HasField("normal"):
                print(f"        Speed: Normal({speed.normal.mean}, {speed.normal.variance})")

            direction = random_walk_model.direction
            if direction.HasField("uniform"):
                print(f"        Direction: Uniform({direction.uniform.min}, {direction.uniform.max})")
            elif direction.HasField("constant"):
                print(f"        Direction: Constant({direction.constant.value})")
            elif direction.HasField("normal"):
                print(f"        Direction: Normal({direction.normal.mean}, {direction.normal.variance})")


def print_csi_request(simulation_time, node_a, node_b):
    print("%0.9fs: Propagation request:" % (simulation_time / 1e9))
    print("    NodeId1:  ", node_a)
    print("    NodeId2:  ", node_b)
    print("    Timestamp:", simulation_time)


def print_csi_response(simulation_time, node_a, node_b, node_a_pos, node_b_pos, delay, loss, ttl):
    print("%0.9fs: Propagation response:" % (simulation_time / 1e9))
    print("Tx:%d - Rx:%d" % (node_a, node_b))
    print("    posTx: ", node_a_pos)
    print("    posRx: ", node_b_pos)
    print("    delay:  ", delay)
    print("    loss:  ", loss)
    print("    ttl:", ttl)
