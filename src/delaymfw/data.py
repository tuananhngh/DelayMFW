"""Dataset loading (MNIST, Fashion-MNIST, CIFAR-10) and splitting across agents.

Replaces ``MLDatasets.jl``: the raw files are downloaded once into a cache directory
(``$DELAYMFW_DATA`` or ``~/.cache/delaymfw``) and parsed with NumPy. Images are returned
flattened as ``(samples, features)`` float32 arrays scaled to ``[0, 1]``, with 0-based labels.
"""

from __future__ import annotations

import gzip
import logging
import os
import shutil
import tarfile
import tempfile
import urllib.request
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

_IDX_DATASETS = {
    "mnist": "https://ossci-datasets.s3.amazonaws.com/mnist/",
    "fashionmnist": "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/",
}
_IDX_FILES = ("train-images-idx3-ubyte.gz", "train-labels-idx1-ubyte.gz")
_CIFAR10_URL = "https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz"

DATASETS = (*_IDX_DATASETS, "cifar10")


def default_data_dir() -> Path:
    return Path(os.environ.get("DELAYMFW_DATA", Path.home() / ".cache" / "delaymfw"))


def _download(url: str, dest: Path) -> Path:
    if dest.exists():
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    log.info("Downloading %s", url)
    with urllib.request.urlopen(url) as resp, tempfile.NamedTemporaryFile(dir=dest.parent, delete=False) as tmp:
        shutil.copyfileobj(resp, tmp)
    Path(tmp.name).replace(dest)
    return dest


def _read_idx(path: Path) -> np.ndarray:
    with gzip.open(path, "rb") as f:
        raw = f.read()
    ndim = raw[3]
    shape = tuple(int.from_bytes(raw[4 + 4 * i : 8 + 4 * i], "big") for i in range(ndim))
    return np.frombuffer(raw, dtype=np.uint8, offset=4 + 4 * ndim).reshape(shape)


def _load_idx(name: str, root: Path) -> tuple[np.ndarray, np.ndarray]:
    base = _IDX_DATASETS[name]
    img_path, lbl_path = (_download(base + f, root / name / f) for f in _IDX_FILES)
    images = _read_idx(img_path)
    labels = _read_idx(lbl_path)
    return images.reshape(len(images), -1), labels


def _load_cifar10(root: Path) -> tuple[np.ndarray, np.ndarray]:
    archive = _download(_CIFAR10_URL, root / "cifar10" / "cifar-10-binary.tar.gz")
    record = 1 + 3 * 32 * 32
    chunks = []
    with tarfile.open(archive, "r:gz") as tar:
        for i in range(1, 6):
            member = tar.extractfile(f"cifar-10-batches-bin/data_batch_{i}.bin")
            assert member is not None
            chunks.append(np.frombuffer(member.read(), dtype=np.uint8).reshape(-1, record))
    raw = np.concatenate(chunks)
    return raw[:, 1:], raw[:, 0]


def load_dataset(name: str, data_dir: Path | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Return the training split of ``name`` as ``(x, y)``: x is ``(N, features)`` in [0, 1]."""
    root = Path(data_dir) if data_dir is not None else default_data_dir()
    if name in _IDX_DATASETS:
        x, y = _load_idx(name, root)
    elif name == "cifar10":
        x, y = _load_cifar10(root)
    else:
        raise ValueError(f"Unknown dataset {name!r}, expected one of {DATASETS}")
    return x.astype(np.float32) / 255.0, y.astype(np.int64)


def data_processing(
    train_x: np.ndarray,
    train_y: np.ndarray,
    nb_agents: int,
    batch_size: int,
    rng: np.random.Generator,
) -> tuple[int, np.ndarray, np.ndarray]:
    """Shuffle the data, give each agent an equal chunk and cut the chunks into batches.

    Returns ``(features, data, labels)`` with ``data`` of shape
    ``(nb_batches, batch_size, features, nb_agents)`` and ``labels`` of shape
    ``(nb_batches, batch_size, nb_agents)``. Samples that do not fill a whole chunk or batch
    are dropped.
    """
    n_samples = train_x.shape[0]
    train_x = train_x.reshape(n_samples, -1)
    features = train_x.shape[1]
    log.info("Data Original Size %s", train_x.shape)
    chunk_size = n_samples // nb_agents
    nb_batches = chunk_size // batch_size
    used = nb_batches * batch_size
    perm = rng.permutation(n_samples)
    log.info("Data Shuffled")
    x = train_x[perm][: nb_agents * chunk_size].reshape(nb_agents, chunk_size, features)[:, :used]
    y = train_y[perm][: nb_agents * chunk_size].reshape(nb_agents, chunk_size)[:, :used]
    data = x.reshape(nb_agents, nb_batches, batch_size, features).transpose(1, 2, 3, 0)
    labels = y.reshape(nb_agents, nb_batches, batch_size).transpose(1, 2, 0)
    return features, np.ascontiguousarray(data), np.ascontiguousarray(labels)


def uniform_delays(max_delay: int, shape: int | tuple[int, ...], rng: np.random.Generator) -> np.ndarray:
    """Delays drawn uniformly from ``[ceil(0.1 * max_delay), max_delay]``."""
    low = int(np.ceil(0.1 * max_delay))
    return rng.integers(low, max_delay + 1, size=shape)
