# Handling Delayed Feedback in Distributed Learning: A Projection-Free Approach

This repository contains the companion code for the paper:

**"Handling Delayed Feedback in Distributed Online Optimization: A Projection-Free Approach"**
Nguyen, T.A., Kim Thang, N., Trystram, D. (2024). In: Bifet, A., et al. (eds) Machine Learning and Knowledge Discovery in Databases. Research Track. ECML PKDD 2024. Lecture Notes in Computer Science, vol 14941. Springer, Cham.
[https://doi.org/10.1007/978-3-031-70341-6_12](https://doi.org/10.1007/978-3-031-70341-6_12) | [arXiv:2402.02114](https://arxiv.org/abs/2402.02114)

## Overview

This code implements projection-free algorithms for distributed learning under delayed feedback. This is a Python port of the original Julia code. The repository includes implementations for both centralized and decentralized settings using Meta Frank-Wolfe algorithms that are robust to communication delays.

## Algorithms Implemented

### Centralized Setting
- **Delay-MFW**: Delay-tolerant Meta Frank-Wolfe algorithm
- **Bold-MFW**: Bold variant of Meta Frank-Wolfe
- **Delay-OFW**: Delay-tolerant Online Frank-Wolfe

### Decentralized Setting
- **Decentralized Delay-MFW**: Distributed Meta Frank-Wolfe with support for various network topologies (Erdős-Rényi, complete graphs, grid graphs, cycle graphs)

## Requirements

- [uv](https://docs.astral.sh/uv/) (it installs Python 3.12 and every dependency from `uv.lock`)

```bash
uv sync
```

Datasets (MNIST, Fashion-MNIST, CIFAR-10) are downloaded on first use into `~/.cache/delaymfw`
(override with `$DELAYMFW_DATA` or `--data_dir`).

## Repository Structure

- [src/delaymfw/algorithms/centralized.py](src/delaymfw/algorithms/centralized.py) - Delay-MFW, Bold-MFW, Delay-OFW
- [src/delaymfw/algorithms/decentralized.py](src/delaymfw/algorithms/decentralized.py) - Decentralized Delay-MFW
- [src/delaymfw/model.py](src/delaymfw/model.py) - Softmax regression loss, closed-form gradient, linear minimization oracle
- [src/delaymfw/data.py](src/delaymfw/data.py) - Dataset download/parsing and splitting across agents
- [src/delaymfw/graph.py](src/delaymfw/graph.py) - Network topologies, gossip matrix and spectral gap
- [src/delaymfw/cli/centralized.py](src/delaymfw/cli/centralized.py) - Centralized experiment entry point
- [src/delaymfw/cli/decentralized.py](src/delaymfw/cli/decentralized.py) - Decentralized experiment entry point
- [src/delaymfw/plotting.py](src/delaymfw/plotting.py) - Figures and CSV summaries
- [run-decentralized.sh](run-decentralized.sh) - Batch script for decentralized experiments
- [tests/](tests/) - Unit tests, including checks against line-by-line transcriptions of the original Julia code

## Usage

### Centralized Experiments

```bash
uv run delaymfw-centralized
```

This will:
- Load and process the MNIST dataset
- Run Delay-MFW, Bold-MFW, and Delay-OFW algorithms
- Test multiple delay settings (`--max_delay 61 81 101` by default)
- Save results and loss curves to `./result-centralized-ml/`

Other options: `--data_name`, `--batch_size`, `--radius`, `--K`, `--num_iters`, `--output_dir`, `--seed`, `--no_plot`.

### Decentralized Experiments

```bash
uv run delaymfw-decentralized --data_name=mnist --nb_agents=25 --batch_size=4 --nb_classes=10 --max_delay=10 --radius=32 --runall=true
```

**Parameters:**
- `--data_name`: Dataset name (`mnist`, `fashionmnist`, `cifar10`)
- `--nb_agents`: Number of agents in the network
- `--batch_size`: Batch size per agent
- `--nb_classes`: Number of classes
- `--max_delay`: Maximum delay for feedback
- `--max_delay_agent`: Fixed delay of the selected agents (when `runall=false`)
- `--nb_agent_delay`: Number of agents with high delay (when `runall=false`)
- `--radius`: Radius parameter for the constraint set
- `--runall`: If `true`, apply delay to all agents; if `false`, apply high delay only to selected agents
- `--graphs`: Topologies to run (default `er complete grid cycle`; `star` also available)
- `--edge_prob`, `--num_iters`, `--output_dir` (default `../result-decentralized-ml2`), `--seed`

For batch experiments:

```bash
bash run-decentralized.sh
```

### Plots

```bash
uv run delaymfw-plot centralized --results ./result-centralized-ml --delays 61 81 101
uv run delaymfw-plot decentralized --results ../result-decentralized-ml2/fashionmnist --iters 1000 --agents 30 --radius 24 --delays 1 31 101 501 --select 2 5 10 20 --max_delay_agent 501
```

### Tests

```bash
uv run pytest
```

## Key Features

- **Projection-Free Optimization**: Uses Linear Minimization Oracles (LMO) instead of projections
- **Delay Tolerance**: Algorithms handle arbitrary delays in gradient feedback
- **Vectorized Agents**: All agents of the decentralized setting are updated at once with NumPy
- **Multiple Network Topologies**: Supports Erdős-Rényi, complete, grid, cycle, and star graphs
- **Flexible Delay Models**: Uniform delay for all agents or selective high delay for specific agents

## Results

Results are saved as HDF5 files (`.h5`, one dataset per algorithm). The plotting command also reads
`.jld` files produced by the former Julia implementation.

## Citation

If you use this code in your research, please cite:

```bibtex
@InProceedings{10.1007/978-3-031-70341-6_12,
  author="Nguyen, Tuan-Anh and Kim Thang, Nguyen and Trystram, Denis",
  title="Handling Delayed Feedback in Distributed Online Optimization: A Projection-Free Approach",
  booktitle="Machine Learning and Knowledge Discovery in Databases. Research Track",
  year="2024",
  publisher="Springer Nature Switzerland",
  address="Cham",
  pages="197--211",
  isbn="978-3-031-70341-6"
}
```

## License

See the paper for more details on the theoretical guarantees and experimental setup.
