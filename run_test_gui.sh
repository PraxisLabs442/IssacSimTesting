#!/bin/bash
# Run VLA in Isaac Sim with GUI

cd /home/mpcr/Downloads/IsaacLab
./isaaclab.sh -p ~/Desktop/PraxisLabs/run_vla.py \
  --vla-device cuda:1 \
  --steps 200
