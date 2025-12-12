#!/bin/bash

cd ../../

# some info about the machine
hostname
lscpu | grep "Model name"

echo "Test stationary / higher load"

./ns3 run "scratch/ns3-sionna/performance-ns3 --channel=1 --sim_max_stas=64 --mobile_scenario=false --mobile_speed=0.0 --udp_pkt_interval=20"

echo "Test low mobility / low load"

./ns3 run "scratch/ns3-sionna/performance-ns3 --channel=1 --sim_max_stas=64 --mobile_scenario=true --mobile_speed=1.0 --udp_pkt_interval=1000"

echo "Test high mobility / higher load"

./ns3 run "scratch/ns3-sionna/performance-ns3 --channel=1 --sim_max_stas=64 --mobile_scenario=true --mobile_speed=7.0 --udp_pkt_interval=20"
