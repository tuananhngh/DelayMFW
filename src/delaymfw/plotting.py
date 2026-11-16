"""Figures and CSV summaries of experiment results (port of ``plot-result.py``).

Usage::

    delaymfw-plot centralized --results ./result-centralized-ml --delays 61 81 101
    delaymfw-plot decentralized --results ../result-decentralized-ml2/fashionmnist \
        --iters 1000 --agents 30 --radius 24 --delays 1 31 101 501 --select 2 5 10 20 --max_delay_agent 501

Figures are written to ``<results>/plots`` and CSV files to ``<results>`` (centralized) or
``<results>/csv-file`` (decentralized). Pass ``--show`` to also open them interactively.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from delaymfw.results import load_result

COLORS = mpl.colormaps["tab10"]
SHOW = False


def setup_style() -> None:
    mpl.rcParams["font.family"] = "serif"
    mpl.rcParams["font.serif"] = ["Times New Roman", "Times", "DejaVu Serif"]
    plt.rcParams["font.size"] = 16
    plt.rcParams["axes.linewidth"] = 2


def _every_100(n: int) -> list[int]:
    return list(range(0, n, 100))


def _style_ticks(ax) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    for axis in (ax.xaxis, ax.yaxis):
        axis.set_tick_params(which="major", size=5, width=2, direction="in")
        axis.set_tick_params(which="minor", size=3, width=1, direction="in")


def _finish(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    if SHOW:
        plt.show()
    plt.close(fig)


# --------------------------------------------------------------------------- centralized


def _load_centralized(path: Path, delay: int) -> dict[str, np.ndarray]:
    return {key: load_result(path / f"{delay}-delay", key) for key in ("dmfw", "dofw", "bmfw")}


def create_csv(delay: int, path: Path) -> None:
    r = _load_centralized(path, delay)
    pd.DataFrame(
        {
            "iteration": np.arange(1, len(r["dmfw"]) + 1),
            "DelMFW": np.cumsum(r["dmfw"]),
            "DOFW": np.cumsum(r["dofw"]),
            "BOLD-MFW": np.cumsum(r["bmfw"]),
        }
    ).to_csv(path / f"{delay}-delay-cumsum.csv", index=False)


def plot_compare_centralized(delay: int, path: Path) -> None:
    r = _load_centralized(path, delay)
    n = len(r["dmfw"])
    it = np.arange(1, n + 1)
    fig, ax = plt.subplots(figsize=(4, 2))
    ax.set_title(f"Maximum Delay: {delay}")
    _style_ticks(ax)
    for key, label, color, marker in (("dmfw", "DeLMFW", 0, "s"), ("dofw", "DOFW", 1, ">"), ("bmfw", "BOLD-MFW", 3, "o")):
        ax.plot(it, np.cumsum(r[key]), label=label, linewidth=2, color=COLORS(color), markevery=_every_100(n), marker=marker)
    ax.xaxis.set_major_locator(mpl.ticker.MultipleLocator(200))
    ax.xaxis.set_minor_locator(mpl.ticker.MultipleLocator(50))
    ax.set_xlim(-50, n)
    ax.set_xlabel("#Iterations", labelpad=10)
    ax.set_ylabel("Cumulative Loss", labelpad=10)
    ax.legend(loc="upper left", frameon=True, framealpha=1)
    _finish(fig, path / "plots" / f"{delay}-delay.png")


def compare_delay_centralized(delays: list[int], path: Path) -> None:
    totals = {"DeLMFW": [], "DOFW": [], "BOLD-MFW": []}
    for d in delays:
        r = _load_centralized(path, d)
        totals["DeLMFW"].append(r["dmfw"].sum())
        totals["DOFW"].append(r["dofw"].sum())
        totals["BOLD-MFW"].append(r["bmfw"].sum())
    pd.DataFrame({"delay": delays, **totals}).to_csv(path / "comparison-delay.csv", index=False)
    fig, ax = plt.subplots(figsize=(4, 4))
    _style_ticks(ax)
    for (label, values), color, marker in zip(totals.items(), (0, 1, 3), ("s", ">", "o"), strict=True):
        ax.plot(delays, values, label=label, linewidth=2, color=COLORS(color), marker=marker)
    ax.set_xticks(delays)
    ax.set_xlim(0, delays[-1] + 1)
    ax.set_xlabel("Maximum Delay", labelpad=10)
    ax.set_ylabel("Total Loss", labelpad=10)
    ax.legend(loc="center right", frameon=False)
    _finish(fig, path / "plots" / "comparison-delay.png")


# ------------------------------------------------------------------------- decentralized


class DecentralizedRuns:
    """Locates the result files written by ``delaymfw-decentralized``."""

    def __init__(self, path: Path, iters: int, agents: int, radius: str, tag: str) -> None:
        self.path, self.iters, self.agents, self.radius, self.tag = path, iters, agents, radius, tag

    def _prefix(self, delay: int) -> str:
        return f"{self.iters}-{delay}-{self.agents}-{self.radius}-{self.tag}"

    def uniform(self, delay: int, graph: str) -> np.ndarray:
        return load_result(self.path / f"{self._prefix(delay)}{graph}", "ddmfw")

    def selected(self, nb_selected: int, max_delay_agent: int, graph: str, base_delay: int = 1) -> np.ndarray:
        name = f"{self._prefix(base_delay)}select_{nb_selected}-{max_delay_agent}-amaxdelay-{graph}"
        return load_result(self.path / name, "ddmfw")


def plot_decentralized(runs: DecentralizedRuns, delay: int, graphs: list[str], out: Path, cumulative: bool = True) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    for g in graphs:
        loss = runs.uniform(delay, g)
        y = np.cumsum(loss) if cumulative else loss
        ax.plot(np.arange(1, len(y) + 1), y, label=f"De2MFW-{g}", marker="s", markevery=_every_100(len(y)))
    ax.set_xlim(-1, runs.iters)
    ax.set_xlabel("#Iterations", labelpad=10)
    ax.set_ylabel("Cumulative Regret" if cumulative else "Regret", labelpad=10)
    ax.legend(loc="upper left", frameon=False)
    _finish(fig, out / f"{delay}-delay-graphs.png")


def make_csv_select(runs: DecentralizedRuns, select_list: list[int], max_delay_agent: int, graphs: list[str], csv_dir: Path) -> None:
    columns: dict[str, np.ndarray] = {"iteration": np.arange(1, runs.iters + 1)}
    for s in select_list:
        for g in graphs:
            loss = runs.selected(s, max_delay_agent, g)
            columns[f"select-{s}-{g}-Loss"] = loss
            columns[f"select-{s}-{g}-cumsum"] = np.cumsum(loss)
    pd.DataFrame(columns).to_csv(csv_dir / f"select-max{max_delay_agent}-compare.csv", index=False)


def make_csv(runs: DecentralizedRuns, delays: list[int], graphs: list[str], csv_dir: Path) -> None:
    ite = np.arange(1, runs.iters + 1)
    for g in graphs:
        columns: dict[str, np.ndarray] = {"iteration": ite}
        for d in delays:
            loss = runs.uniform(d, g)
            columns[f"{d}-Loss"] = loss
            columns[f"{d}-cumsum"] = np.cumsum(loss)
        pd.DataFrame(columns).to_csv(csv_dir / f"{g}-compare.csv", index=False)
    for name, delay in (("non-delay", 1), *((str(d), d) for d in delays)):
        columns = {"iteration": ite}
        for g in graphs:
            loss = runs.uniform(delay, g)
            columns[f"{g}-Loss"] = loss
            columns[f"{g}-cumsum"] = np.cumsum(loss)
        pd.DataFrame(columns).to_csv(csv_dir / f"{name}-compare.csv", index=False)


def plot_compare_decentralized(delays: list[int], graphs: list[str], csv_dir: Path, cumsum: bool = False) -> None:
    """Ratio of each delayed run to the undelayed one, per graph and per delay."""
    suffix = "cumsum" if cumsum else "Loss"
    plots = csv_dir / "plots"
    non_delay = pd.read_csv(csv_dir / "non-delay-compare.csv")
    for g in graphs:
        data = pd.read_csv(csv_dir / f"{g}-compare.csv")
        n = len(data)
        fig, ax = plt.subplots(figsize=(6, 4))
        for d in delays:
            ax.plot(np.arange(1, n + 1), data[f"{d}-{suffix}"] / data[f"1-{suffix}"], label=f"{d}-delay", marker="s", markevery=_every_100(n))
        ax.set_xlim(-1, n)
        ax.set_xlabel("#Iterations", labelpad=10)
        ax.set_ylabel("Ratio", labelpad=10)
        ax.set_title(f"Graph: {g}")
        ax.legend(loc="upper left", frameon=False)
        _finish(fig, plots / f"{g}-delay-regret.pdf")
    for d in delays:
        data = pd.read_csv(csv_dir / f"{d}-compare.csv")
        n = len(data)
        ratios: dict[str, object] = {"iteration": np.arange(1, n + 1)}
        fig, ax = plt.subplots(figsize=(6, 4))
        for g in graphs:
            ratio = data[f"{g}-{suffix}"] / non_delay[f"{g}-{suffix}"]
            ratios[f"{g}-{suffix}"] = ratio
            ax.plot(np.arange(1, n + 1), ratio, label=f"{g}-graph", marker="s", markevery=_every_100(n))
        pd.DataFrame(ratios).to_csv(csv_dir / f"{d}-delay-ratio.csv", index=False)
        ax.set_xlim(-1, n)
        ax.set_xlabel("#Iterations", labelpad=10)
        ax.set_ylabel("Ratio", labelpad=10)
        ax.set_title(f"Delay: {d}")
        ax.legend(loc="lower right", frameon=False)
        _finish(fig, plots / f"{d}-graph-regret.pdf")


def plot_select_compare(csv_dir: Path, select_list: list[int], max_delay_agent: int, graphs: list[str]) -> None:
    data = pd.read_csv(csv_dir / f"select-max{max_delay_agent}-compare.csv")
    n = len(data)
    for s in select_list:
        fig, ax = plt.subplots(figsize=(6, 4))
        for g in graphs:
            ax.plot(np.arange(1, n + 1), data[f"select-{s}-{g}-cumsum"], label=f"{g}-graph", marker="s", markevery=_every_100(n))
        ax.set_xlim(-1, n)
        ax.set_xlabel("#Iterations", labelpad=10)
        ax.set_ylabel("Cumulative Loss", labelpad=10)
        ax.set_title(f"Select: {s}, Max Delay: {max_delay_agent}")
        ax.legend(loc="lower right", frameon=False)
        _finish(fig, csv_dir / "plots" / f"select-{s}-max{max_delay_agent}-regret.pdf")


def compare_increase_agent(csv_dir: Path, select_list: list[int], max_delay_agent: int, graphs: list[str]) -> pd.DataFrame:
    """Final cumulative loss as more agents get the large delay, plus % change vs. no delayed agent."""
    data = pd.read_csv(csv_dir / f"select-max{max_delay_agent}-compare.csv").iloc[-1]
    non_delay = pd.read_csv(csv_dir / "non-delay-compare.csv").iloc[-1]
    table = pd.DataFrame(
        {"agent": [0, *select_list]}
        | {g: [non_delay[f"{g}-cumsum"], *(data[f"select-{s}-{g}-cumsum"] for s in select_list)] for g in graphs}
    )
    table.to_csv(csv_dir / "compare-increase-agent.csv", index=True)
    return compute_percen_rows(table)


def compute_percen_rows(data: pd.DataFrame) -> pd.DataFrame:
    """Symmetric percentage difference of every row with respect to the first one."""
    first = data.iloc[0]
    return pd.DataFrame([(row - first) / ((row + first) / 2) * 100 for _, row in data.iterrows()]).reset_index(drop=True)


# ----------------------------------------------------------------------------------- CLI


def main(argv: list[str] | None = None) -> None:
    global SHOW
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--show", action="store_true", help="also display figures interactively")
    sub = parser.add_subparsers(dest="mode", required=True)

    cen = sub.add_parser("centralized")
    cen.add_argument("--results", type=Path, default=Path("./result-centralized-ml"))
    cen.add_argument("--delays", type=int, nargs="+", default=[61, 81, 101])

    dec = sub.add_parser("decentralized")
    dec.add_argument("--results", type=Path, required=True, help="e.g. ../result-decentralized-ml2/fashionmnist")
    dec.add_argument("--iters", type=int, default=1000)
    dec.add_argument("--agents", type=int, default=30)
    dec.add_argument("--radius", default="24")
    dec.add_argument("--tag", default="staticdelay-", help="file name tag after the radius ('' for legacy runs)")
    dec.add_argument("--delays", type=int, nargs="+", default=[1, 31, 101, 501], help="must include 1 (the undelayed baseline)")
    dec.add_argument("--graphs", nargs="+", default=["er", "grid", "complete", "cycle"])
    dec.add_argument("--select", type=int, nargs="*", default=[], help="nb_agent_delay values of selected-delay runs")
    dec.add_argument("--max_delay_agent", type=int, default=501)
    dec.add_argument("--cumsum", action=argparse.BooleanOptionalAction, default=True)

    args = parser.parse_args(argv)
    SHOW = args.show
    if not SHOW:
        mpl.use("Agg")
    setup_style()

    if args.mode == "centralized":
        for d in args.delays:
            create_csv(d, args.results)
            plot_compare_centralized(d, args.results)
        compare_delay_centralized(args.delays, args.results)
        return

    runs = DecentralizedRuns(args.results, args.iters, args.agents, args.radius, args.tag)
    csv_dir = args.results / "csv-file"
    csv_dir.mkdir(parents=True, exist_ok=True)
    for d in args.delays:
        plot_decentralized(runs, d, args.graphs, csv_dir / "plots", cumulative=args.cumsum)
    make_csv(runs, args.delays, args.graphs, csv_dir)
    plot_compare_decentralized(args.delays, args.graphs, csv_dir, cumsum=args.cumsum)
    if args.select:
        make_csv_select(runs, args.select, args.max_delay_agent, args.graphs, csv_dir)
        plot_select_compare(csv_dir, args.select, args.max_delay_agent, args.graphs)
        print(compare_increase_agent(csv_dir, args.select, args.max_delay_agent, args.graphs))


if __name__ == "__main__":
    main()
