"""End-to-end runs of the CLIs and plotting on a small synthetic dataset (no download)."""

import numpy as np
import pytest

from delaymfw import plotting
from delaymfw.cli import centralized, decentralized
from delaymfw.results import load_result


@pytest.fixture
def fake_dataset(monkeypatch):
    rng = np.random.default_rng(0)
    x = rng.random((1200, 20), dtype=np.float32)
    y = rng.integers(0, 10, size=1200)
    for module in (centralized, decentralized):
        monkeypatch.setattr(module, "load_dataset", lambda name, data_dir=None: (x, y))


def test_centralized_cli_and_plots(fake_dataset, tmp_path):
    out = tmp_path / "cen"
    centralized.main(["--batch_size", "6", "--max_delay", "3", "5", "--K", "4", "--output_dir", str(out)])
    for md in (3, 5):
        for key in ("dmfw", "bmfw", "dofw"):
            loss = load_result(out / f"{md}-delay", key)
            assert loss.shape == (200,) and np.all(np.isfinite(loss))
    assert (out / "3-delay-cumulative.png").exists()
    plotting.main(["centralized", "--results", str(out), "--delays", "3", "5"])
    assert (out / "comparison-delay.csv").exists()
    assert (out / "plots" / "5-delay.png").exists()


def test_decentralized_cli_and_plots(fake_dataset, tmp_path):
    out = tmp_path / "dec"
    common = ["--nb_agents", "6", "--batch_size", "4", "--radius", "4", "--num_iters", "20", "--output_dir", str(out), "--graphs", "complete", "cycle", "grid"]
    for md in (1, 3):
        decentralized.main([*common, "--max_delay", str(md), "--runall", "true"])
    for nb in (1, 2):
        decentralized.main([*common, "--max_delay", "1", "--runall", "0", "--nb_agent_delay", str(nb), "--max_delay_agent", "5"])
    res = out / "mnist"
    loss = load_result(res / "20-3-6-4-staticdelay-cycle", "ddmfw")
    assert loss.shape == (20,) and np.all(np.isfinite(loss))
    plotting.main(
        ["decentralized", "--results", str(res), "--iters", "20", "--agents", "6", "--radius", "4",
         "--delays", "1", "3", "--graphs", "complete", "cycle", "grid", "--select", "1", "2", "--max_delay_agent", "5"]
    )
    assert (res / "csv-file" / "compare-increase-agent.csv").exists()
    assert (res / "csv-file" / "plots" / "3-graph-regret.pdf").exists()
