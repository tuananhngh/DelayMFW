#!/bin/bash
# Decentralized experiments: Fashion-MNIST, 30 agents, a growing number of agents with a large (101) delay.
set -euo pipefail

for nb in 2 5 10 20; do
    uv run delaymfw-decentralized --data_name=fashionmnist --nb_agents=30 --batch_size=2 --nb_classes=10 \
        --max_delay=1 --radius=24 --max_delay_agent=101 --nb_agent_delay="$nb" --runall=0
done
