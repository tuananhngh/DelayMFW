"""Centralized experiments: Delay-MFW vs Bold-MFW vs Delay-OFW (port of ``main-centralized-ml.jl``)."""

from __future__ import annotations

import argparse
import logging
import math
from pathlib import Path

import numpy as np

from delaymfw.algorithms import bold_mfw, d_ofw, delay_mfw
from delaymfw.cli import setup_logging
from delaymfw.data import DATASETS, data_processing, load_dataset, uniform_delays
from delaymfw.model import grad_fn, lmo_l1_columns, loss_fn
from delaymfw.results import RESULT_SUFFIX, load_result, save_results

log = logging.getLogger(__name__)


def run_setting(
    dim, data_x, data_y, num_iters, radius, max_delays, K, output_dir: Path, rng: np.random.Generator
) -> None:
    log.info("Running for %d Iterations and %d Subiterations", num_iters, K)
    for md in max_delays:
        log.info("Running for Max Delay %d", md)
        delay = uniform_delays(md, num_iters, rng)
        eta = 1 / math.sqrt(md * num_iters)
        common = (dim, data_x, data_y, loss_fn, grad_fn, lmo_l1_columns, num_iters, md, delay, eta, radius)
        log.info("------Running DMFW------")
        dmfw = delay_mfw(*common, K, rng)
        log.info("------Running Bold-MFW------")
        bmfw = bold_mfw(*common, K, rng)
        log.info("------Running DOFW------")
        dofw = d_ofw(*common)
        save_results(output_dir / f"{md}-delay{RESULT_SUFFIX}", {"dmfw": dmfw, "bmfw": bmfw, "dofw": dofw})


def plot_results(output_dir: Path, max_delays) -> None:
    """Save per-round and cumulative loss curves next to the results (replaces ``plot_loss``/``plot_result``)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = {"dmfw": "Delay-MFW", "bmfw": "Bold-MFW", "dofw": "Delay-OFW"}
    for md in max_delays:
        stem = output_dir / f"{md}-delay"
        curves = {key: load_result(stem, key) for key in labels}
        for cumulative in (False, True):
            fig, ax = plt.subplots()
            for key, name in labels.items():
                y = np.cumsum(curves[key]) if cumulative else curves[key]
                ax.plot(np.arange(1, len(y) + 1), y, label=name)
            ax.set_xlabel(f"Iterations t - Max Delay {md}")
            ax.set_ylabel("Cumulative Loss" if cumulative else "Loss")
            ax.legend(loc="upper left")
            fig.savefig(output_dir / f"{md}-delay-{'cumulative' if cumulative else 'loss'}.png", dpi=150)
            plt.close(fig)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_name", default="mnist", choices=DATASETS)
    parser.add_argument("--data_dir", type=Path, default=None, help="dataset cache (default: $DELAYMFW_DATA or ~/.cache/delaymfw)")
    parser.add_argument("--batch_size", type=int, default=60)
    parser.add_argument("--nb_classes", type=int, default=10)
    parser.add_argument("--radius", type=float, default=8)
    parser.add_argument("--K", type=int, default=31, help="number of Frank-Wolfe sub-iterations")
    parser.add_argument("--max_delay", type=int, nargs="+", default=[61, 81, 101])
    parser.add_argument("--num_iters", type=int, default=None, help="truncate the number of rounds")
    parser.add_argument("--output_dir", type=Path, default=Path("./result-centralized-ml"))
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--no_plot", action="store_true")
    args = parser.parse_args(argv)
    setup_logging()

    rng = np.random.default_rng(args.seed)
    train_x, train_y = load_dataset(args.data_name, args.data_dir)
    flat_dim, data_cell, label_cell = data_processing(train_x, train_y, 1, args.batch_size, rng)
    data_cell, label_cell = data_cell[..., 0], label_cell[..., 0]
    if args.num_iters is not None:
        data_cell, label_cell = data_cell[: args.num_iters], label_cell[: args.num_iters]
    dim = (flat_dim, args.nb_classes)
    num_iters = data_cell.shape[0]
    log.info("Feature: %s, Data size: %s, Label Size: %s", dim, data_cell.shape, label_cell.shape)

    run_setting(dim, data_cell, label_cell, num_iters, args.radius, args.max_delay, args.K, args.output_dir, rng)
    if not args.no_plot:
        plot_results(args.output_dir, args.max_delay)


if __name__ == "__main__":
    main()
