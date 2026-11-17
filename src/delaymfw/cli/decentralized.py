"""Decentralized Delay-MFW over several network topologies (port of ``main-decentralized.jl``)."""

from __future__ import annotations

import argparse
import logging
import math
from pathlib import Path

import numpy as np

from delaymfw.algorithms import dec_delay_mfw
from delaymfw.cli import setup_logging, str2bool
from delaymfw.data import DATASETS, data_processing, load_dataset, uniform_delays
from delaymfw.graph import GRAPH_TYPES, generate_graph
from delaymfw.model import lmo_l1_columns
from delaymfw.results import RESULT_SUFFIX, save_results

log = logging.getLogger(__name__)


def build_weights(graph_types, nb_agents: int, p: float, rng: np.random.Generator):
    weights, gaps = [], []
    for g_type in graph_types:
        log.info("Generating %s graph", g_type)
        _, w, gap = generate_graph(g_type, nb_agents, p, seed=int(rng.integers(2**31)))
        weights.append((g_type, w))
        gaps.append((g_type, gap))
    log.info("Spectral Gap: %s", gaps)
    return weights


def run_experiment(dim, train_data, train_label, weights, delays, eta, radius, rng, path_for) -> None:
    num_iters = train_data.shape[0]
    for g_type, g_weight in weights:
        log.info("--------Running for %s--------", g_type)
        ddmfw = dec_delay_mfw(dim, train_data, train_label, g_weight, lmo_l1_columns, num_iters, eta, delays, radius, rng)
        save_results(path_for(g_type), {"ddmfw": ddmfw})


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_name", default="mnist", choices=DATASETS, help="Dataset name")
    parser.add_argument("--data_dir", type=Path, default=None, help="dataset cache (default: $DELAYMFW_DATA or ~/.cache/delaymfw)")
    parser.add_argument("--nb_agents", type=int, default=25, help="Number of agents")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--nb_classes", type=int, default=10, help="Number of classes")
    parser.add_argument("--max_delay", type=int, default=1, help="Maximum delay")
    parser.add_argument("--max_delay_agent", type=int, default=10, help="Maximum delay for selected agent")
    parser.add_argument("--nb_agent_delay", type=int, default=25, help="Number of agents with delay")
    parser.add_argument("--radius", type=float, default=32, help="Radius")
    parser.add_argument("--runall", type=str2bool, default=False, help="Delay for all agent or only selected agent")
    parser.add_argument("--graphs", nargs="+", default=["er", "complete", "grid", "cycle"], choices=GRAPH_TYPES)
    parser.add_argument("--edge_prob", type=float, default=0.3, help="edge probability of the Erdos-Renyi graph")
    parser.add_argument("--num_iters", type=int, default=None, help="truncate the number of rounds")
    parser.add_argument("--output_dir", type=Path, default=Path("../result-decentralized-ml2"))
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args(argv)
    setup_logging()
    log.info("%s", vars(args))

    rng = np.random.default_rng(args.seed)
    train_x, train_y = load_dataset(args.data_name, args.data_dir)
    flat_dim, train_data, train_label = data_processing(train_x, train_y, args.nb_agents, args.batch_size, rng)
    if args.num_iters is not None:
        train_data, train_label = train_data[: args.num_iters], train_label[: args.num_iters]
    dim = (flat_dim, args.nb_classes)
    num_iters, batch_size, _, num_agents = train_data.shape
    log.info("Number of agents: %d, Batch size: %d", num_agents, batch_size)
    log.info("Weight_Dim: %s, Feature: %d, Data size: %s, Label Size: %s", dim, flat_dim, train_data.shape, train_label.shape)

    out_dir = args.output_dir / args.data_name
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"{num_iters}-{args.max_delay}-{args.nb_agents}-{args.radius:g}-staticdelay-"

    weights = build_weights(args.graphs, args.nb_agents, args.edge_prob, rng)
    delays = uniform_delays(args.max_delay, (num_iters, args.nb_agents), rng)
    eta = 1 / math.sqrt(args.max_delay * num_iters)
    if args.runall:
        log.info("--------Delay for all agents--------")
        path_for = lambda g: out_dir / f"{prefix}{g}{RESULT_SUFFIX}"  # noqa: E731
    else:
        log.info("------DELAY FOR RANDOMLY SELECTED %d AGENTS------", args.nb_agent_delay)
        selected = rng.choice(args.nb_agents, size=min(args.nb_agent_delay, args.nb_agents), replace=False)
        delays[:, selected] = args.max_delay_agent
        tag = f"select_{args.nb_agent_delay}-{args.max_delay_agent}-amaxdelay-"
        path_for = lambda g: out_dir / f"{prefix}{tag}{g}{RESULT_SUFFIX}"  # noqa: E731
    run_experiment(dim, train_data, train_label, weights, delays, eta, args.radius, rng, path_for)


if __name__ == "__main__":
    main()
