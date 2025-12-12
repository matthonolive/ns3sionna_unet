#!/bin/bash

echo "Starting server component of ns3sionna: normal mode using slower Python proto"
PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python python ns3sionna_server.py
