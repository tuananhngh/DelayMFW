"""Centralized online algorithms under delayed feedback: Delay-MFW, Bold-MFW and Delay-OFW.

Conventions shared by all functions:

* ``data[t]`` is the ``(batch, features)`` batch revealed at round ``t`` and ``labels[t]`` its
  0-based class labels.
* ``delay[t] >= 1`` is the delay of the feedback of round ``t``: it becomes available at the
  end of round ``t + delay[t] - 1`` (a delay of 1 means "no delay").
* ``loss_fn(x, data, labels)`` / ``grad_fn(x, data, labels)`` evaluate the loss and its gradient
  and ``lmo(direction, radius)`` is the linear minimization oracle of the constraint set.
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict
from collections.abc import Callable

import numpy as np

log = logging.getLogger(__name__)

LossFn = Callable[[np.ndarray, np.ndarray, np.ndarray], float]
GradFn = Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray]
Lmo = Callable[[np.ndarray, float], np.ndarray]


def _log_progress(t: int, loss: float) -> None:
    if (t + 1) % 100 == 0:
        log.info("Loss %d: %s", t + 1, loss)


def bold_mfw(
    dim: tuple[int, ...],
    data: np.ndarray,
    labels: np.ndarray,
    loss_fn: LossFn,
    grad_fn: GradFn,
    lmo: Lmo,
    num_iters: int,
    max_delay: int,
    delay: np.ndarray,
    eta: float,
    radius: float,
    K: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Bold-MFW: one Meta-Frank-Wolfe instance per pending feedback slot."""
    loss_val = np.zeros(num_iters)
    start = time.time()
    grad_cell = np.zeros((max_delay + 1, K, *dim))
    is_free = np.zeros(max_delay + 1, dtype=int)
    for t in range(num_iters):
        dt, lb = data[t], labels[t]
        xs = np.zeros((K + 1, *dim))
        base_id = int(np.flatnonzero(is_free == 0)[0])
        is_free[base_id] = delay[t]
        for k in range(K):
            sigma = min(1.0, 1.0 / (k + 4))
            injected_noise = rng.random(dim) - 0.5
            v = lmo(eta * grad_cell[base_id, k] + injected_noise, radius)
            xs[k + 1] = (1 - sigma) * xs[k] + sigma * v
        loss_val[t] = loss_fn(xs[K], dt, lb)
        _log_progress(t, loss_val[t])
        for k in range(K):
            grad_cell[base_id, k] += grad_fn(xs[k], dt, lb)
        is_free[is_free > 0] -= 1
    log.info("Time taken: %.2f minutes", (time.time() - start) / 60)
    return loss_val


def delay_mfw(
    dim: tuple[int, ...],
    data: np.ndarray,
    labels: np.ndarray,
    loss_fn: LossFn,
    grad_fn: GradFn,
    lmo: Lmo,
    num_iters: int,
    max_delay: int,
    delay: np.ndarray,
    eta: float,
    radius: float,
    K: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Delay-MFW: Meta-Frank-Wolfe whose online oracles are fed delayed gradients."""
    del max_delay  # feedback is routed by arrival round; kept for signature parity
    loss_val = np.zeros(num_iters)
    start = time.time()
    grad_cell = np.zeros((K, *dim))
    # Sum of the K sub-iteration gradients, keyed by the round at which they arrive.
    pending: dict[int, np.ndarray] = {}
    for t in range(num_iters):
        dt, lb = data[t], labels[t]
        xs = np.zeros((K + 1, *dim))
        gs = np.empty((K, *dim))
        for k in range(K):
            sigma = min(1.0, 1.0 / (k + 4))
            gs[k] = grad_fn(xs[k], dt, lb)
            injected_noise = rng.random(dim) - 0.5
            v = lmo(eta * grad_cell[k] + injected_noise, radius)
            xs[k + 1] = (1 - sigma) * xs[k] + sigma * v
        loss_val[t] = loss_fn(xs[K], dt, lb)
        _log_progress(t, loss_val[t])
        arrival = t + int(delay[t]) - 1
        if arrival in pending:
            pending[arrival] += gs
        else:
            pending[arrival] = gs
        if t in pending:
            grad_cell += pending.pop(t)
    log.info("Time taken: %.2f minutes", (time.time() - start) / 60)
    return loss_val


def d_ofw(
    dim: tuple[int, ...],
    data: np.ndarray,
    labels: np.ndarray,
    loss_fn: LossFn,
    grad_fn: GradFn,
    lmo: Lmo,
    num_iters: int,
    max_delay: int,
    delay: np.ndarray,
    eta: float,
    radius: float,
) -> np.ndarray:
    """Delay-OFW: online Frank-Wolfe taking one step per received (delayed) gradient."""
    del max_delay
    x = np.full(dim, radius / np.prod(dim))
    x_1 = x.copy()
    loss_val = np.zeros(num_iters)
    G = np.zeros(dim)
    # Gradients keyed by arrival round, kept in the order of the round that produced them.
    pending: defaultdict[int, list[np.ndarray]] = defaultdict(list)
    start = time.time()
    for t in range(num_iters):
        dt, lb = data[t], labels[t]
        loss_val[t] = loss_fn(x, dt, lb)
        _log_progress(t, loss_val[t])
        pending[t + int(delay[t]) - 1].append(grad_fn(x, dt, lb))
        for g in pending.pop(t, []):
            G += g
            DF = eta * G + 2 * (x - x_1)
            v = lmo(DF, radius)
            delta = v - x
            dd = float(np.vdot(delta, delta))
            if dd == 0.0:  # x already is the FW vertex; the Julia code would produce NaN here
                continue
            sigma = min(-0.5 * float(np.vdot(delta, DF)) / dd, 1.0)
            x = (1 - sigma) * x + sigma * v
    log.info("Time taken: %.2f minutes", (time.time() - start) / 60)
    return loss_val
