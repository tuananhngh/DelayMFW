"""Saving and loading experiment results as HDF5 (replaces Julia's ``JLD``)."""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np

RESULT_SUFFIX = ".h5"


def save_results(path: str | Path, results: dict[str, np.ndarray]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as f:
        for key, value in results.items():
            f.create_dataset(key, data=np.asarray(value))
    return path


def load_result(path: str | Path, key: str) -> np.ndarray:
    """Read one array. ``path`` may omit the suffix; ``.h5`` then legacy Julia ``.jld`` are tried."""
    path = Path(path)
    candidates = [path] if path.suffix in (".h5", ".jld") else [Path(f"{path}.h5"), Path(f"{path}.jld")]
    for candidate in candidates:
        if candidate.exists():
            with h5py.File(candidate, "r") as f:
                return f[key][:]
    raise FileNotFoundError(f"No result file among {[str(c) for c in candidates]}")
