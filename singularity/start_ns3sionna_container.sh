#!/bin/bash

echo "Welcome to ns3sionna container"

echo "To run the server component:"
echo "source /opt/sionna-venv/bin/activate ; cd /opt/ns-allinone-3.40/ns-3.40/contrib/sionna/model/ns3sionna/ ; ./run_python_proto.sh"

echo "To run example ns3 script:"
echo "cd /opt/ns-allinone-3.40/ns-3.40/contrib/sionna/examples/ ; ./example-sionna-sensing-mobile.sh"


singularity shell --overlay overlay.img ubuntu-sionna-python.sif

