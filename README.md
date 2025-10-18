# Handling Delayed Feedback in Distributed Learning: A Projection-Free Approach

This repository contains the companion code for the paper:

**"Handling Delayed Feedback in Distributed Online Optimization: A Projection-Free Approach"**
Nguyen, T.A., Kim Thang, N., Trystram, D. (2024). In: Bifet, A., et al. (eds) Machine Learning and Knowledge Discovery in Databases. Research Track. ECML PKDD 2024. Lecture Notes in Computer Science, vol 14941. Springer, Cham.
[https://doi.org/10.1007/978-3-031-70341-6_12](https://doi.org/10.1007/978-3-031-70341-6_12) | [arXiv:2402.02114](https://arxiv.org/abs/2402.02114)

## Overview

This code implements projection-free algorithms for distributed learning under delayed feedback. The repository includes implementations for both centralized and decentralized settings using Meta Frank-Wolfe algorithms that are robust to communication delays.

## Algorithms Implemented

### Centralized Setting
- **Delay-MFW**: Delay-tolerant Meta Frank-Wolfe algorithm
- **Bold-MFW**: Bold variant of Meta Frank-Wolfe
- **Delay-OFW**: Delay-tolerant Online Frank-Wolfe

### Decentralized Setting
- **Decentralized Delay-MFW**: Distributed Meta Frank-Wolfe with support for various network topologies (Erdős-Rényi, complete graphs, grid graphs, cycle graphs)

## Requirements

- Julia (with the following packages):
  - MLDatasets
  - LinearAlgebra
  - Plots
  - Random
  - Flux
  - JLD
  - FileIO
  - SparseArrays
  - ProgressMeter
  - Distributed
  - LightGraphs
  - ArgParse

## Repository Structure

- [main-centralized-ml.jl](main-centralized-ml.jl) - Main script for centralized experiments
- [main-decentralized.jl](main-decentralized.jl) - Main script for decentralized experiments
- [centralized-algorithms-ml.jl](centralized-algorithms-ml.jl) - Implementation of centralized algorithms (Delay-MFW, Bold-MFW, Delay-OFW)
- [decentralized-algorithms-ml.jl](decentralized-algorithms-ml.jl) - Implementation of decentralized Delay-MFW
- [data-handler.jl](data-handler.jl) - Data processing utilities for distributing datasets across agents
- [graph_handler.jl](graph_handler.jl) - Network topology generation and weighted adjacency matrix computation
- [plot-result.py](plot-result.py) - Visualization script for experimental results
- [run-decentralized.sh](run-decentralized.sh) - Batch script for running decentralized experiments

## Usage

### Centralized Experiments

Run the centralized experiments on MNIST:

```bash
julia main-centralized-ml.jl
```

This will:
- Load and process the MNIST dataset
- Run Delay-MFW, Bold-MFW, and Delay-OFW algorithms
- Test multiple delay settings (max_delay = 61, 81, 101)
- Save results to `./result-centralized-ml/`

### Decentralized Experiments

Run decentralized experiments with command-line arguments:

```bash
julia main-decentralized.jl --data_name='mnist' --nb_agents=25 --batch_size=4 --nb_classes=10 --max_delay=10 --radius=32 --runall=true
```

**Parameters:**
- `--data_name`: Dataset name (`mnist`, `fashionmnist`, `cifar10`)
- `--nb_agents`: Number of agents in the network
- `--batch_size`: Batch size per agent
- `--nb_classes`: Number of classes
- `--max_delay`: Maximum delay for feedback
- `--max_delay_agent`: Maximum delay for selected agents (when `runall=false`)
- `--nb_agent_delay`: Number of agents with high delay (when `runall=false`)
- `--radius`: Radius parameter for the constraint set
- `--runall`: If `true`, apply delay to all agents; if `false`, apply high delay only to selected agents

For batch experiments, use the provided shell script:

```bash
bash run-decentralized.sh
```

## Key Features

- **Projection-Free Optimization**: Uses Linear Minimization Oracles (LMO) instead of projections
- **Delay Tolerance**: Algorithms handle arbitrary delays in gradient feedback
- **Distributed Computing**: Leverages Julia's `Distributed` package for parallel computation
- **Multiple Network Topologies**: Supports Erdős-Rényi, complete, grid, cycle, and star graphs
- **Flexible Delay Models**: Uniform delay for all agents or selective high delay for specific agents

## Results

Results are saved in JLD format and can be visualized using the plotting functions in the main scripts or the Python visualization script.

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
