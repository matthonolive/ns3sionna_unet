#!/bin/bash

cd ../../../

echo "Running ns3"
./ns3 run ns3sionna-example-mobility-sionna

echo "Plotting results"
python3 ./contrib/sionna/examples/plot_snr.py example-mobility-sionna-snr.csv