"""Decentralized Delay-MFW over a communication graph.

Agents are the last axis of every array. The Julia version parallelized the per-agent work
with ``Distributed``; here it is vectorized with NumPy instead.
"""

from __future__ import annotations

import logging
import math
import time
from collections.abc import Callable

import numpy as np
from tqdm import tqdm

from delaymfw.model import batched_grad, global_losses

log = logging.getLogger(__name__)


def weighted_average(x: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Gossip step: ``out[..., j] = sum_i x[..., i] * weights[i, j]``."""
    return x @ weights


def dec_delay_mfw(
    dim: tuple[int, int],
    data: np.ndarray,
    labels: np.ndarray,
    weights: np.ndarray,
    lmo: Callable[[np.ndarray, float], np.ndarray],
    num_iters: int,
    eta: float,
    delays: np.ndarray,
    radius: float,
    rng: np.random.Generator,
    grad_fn: Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray] = batched_grad,
    losses_fn: Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray] = global_losses,
    progress: bool = True,
) -> np.ndarray:
    """Run decentralized Delay-MFW and return the per-round loss of the worst agent.

    Args:
        dim: ``(features, classes)`` shape of each agent's model.
        data: ``(num_iters, batch, features, agents)`` batches.
        labels: ``(num_iters, batch, agents)`` 0-based class labels.
        weights: ``(agents, agents)`` doubly stochastic gossip matrix.
        lmo: linear minimization oracle; must reduce axis 0 and keep the trailing axes.
        delays: ``(num_iters, agents)`` delays ``>= 1``; feedback of round ``s`` for agent ``i``
            arrives at round ``s + delays[s, i] - 1``.
        grad_fn: per-agent gradients, ``(F, C, N), (B, F, N), (B, N) -> (F, C, N)``.
        losses_fn: per-agent model loss averaged over every agent's batch, ``-> (N,)``.
    """
    num_agents = weights.shape[0]
    K = math.ceil(math.sqrt(num_iters))
    log.info("Running for %d Iterations and %d Subiterations", num_iters, K)
    t_start = time.time()
    rewards = np.zeros(num_iters)
    grad_cell = np.zeros((K, *dim, num_agents))
    injected_noise = rng.random((*dim, num_agents))
    # pending[arrival_round][agent] = sum of that agent's (K+1, F, C) gradient stacks due then.
    pending: dict[int, dict[int, np.ndarray]] = {}
    for t in tqdm(range(num_iters), disable=not progress):
        dt, lb = data[t], labels[t]
        xs = np.zeros((*dim, num_agents))
        tmp_grad = np.empty((K + 1, *dim, num_agents))
        for k in range(K):
            sigma = min(1.0, 1.0 / (k + 4))
            tmp_grad[k] = grad_fn(xs, dt, lb)
            v = lmo(eta * grad_cell[k] + injected_noise, radius)
            xs = (1 - sigma) * weighted_average(xs, weights) + sigma * v
        tmp_grad[K] = grad_fn(xs, dt, lb)
        rewards[t] = losses_fn(xs, dt, lb).max()

        for i in range(num_agents):
            arrival = t + int(delays[t, i]) - 1
            if arrival >= num_iters:
                continue
            bucket = pending.setdefault(arrival, {})
            if i in bucket:
                bucket[i] += tmp_grad[..., i]
            else:
                bucket[i] = tmp_grad[..., i].copy()

        due = pending.pop(t, None)
        if not due:
            continue  # nothing arrives: every gossip term below would be zero
        sum_delay_grad = np.zeros((K + 1, *dim, num_agents))
        for i, g in due.items():
            sum_delay_grad[..., i] = g
        gs = sum_delay_grad[0]
        for k in range(K):
            ds = weighted_average(gs, weights)
            grad_cell[k] += ds
            gs = sum_delay_grad[k + 1] - sum_delay_grad[k] + ds
    log.info("Time taken: %.2f minutes", (time.time() - t_start) / 60)
    return rewards
