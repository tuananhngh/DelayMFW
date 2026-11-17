"""Multinomial logistic regression loss, its gradient, and the linear minimization oracle.

Shapes follow the Julia code: ``weights`` is ``(features, classes)``, ``data`` is
``(batch, features)`` and ``labels`` is ``(batch,)`` with 0-based class indices.
"""

from __future__ import annotations

import numpy as np


def _log_softmax(logits: np.ndarray, axis: int = -1) -> np.ndarray:
    shifted = logits - logits.max(axis=axis, keepdims=True)
    return shifted - np.log(np.exp(shifted).sum(axis=axis, keepdims=True))


def loss_fn(weights: np.ndarray, data: np.ndarray, labels: np.ndarray) -> float:
    """Mean cross-entropy of ``softmax(data @ weights)`` (Flux ``crossentropy`` equivalent)."""
    log_p = _log_softmax(data @ weights, axis=1)
    return float(-log_p[np.arange(len(labels)), labels].mean())


def grad_fn(weights: np.ndarray, data: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Gradient of :func:`loss_fn` with respect to ``weights`` (closed form, replaces Flux autodiff)."""
    probs = np.exp(_log_softmax(data @ weights, axis=1))
    probs[np.arange(len(labels)), labels] -= 1.0
    return data.T @ probs / len(labels)


def batched_grad(weights: np.ndarray, data: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Per-agent gradients.

    ``weights`` is ``(F, C, N)``, ``data`` is ``(B, F, N)``, ``labels`` is ``(B, N)``.
    Returns ``(F, C, N)`` where slice ``[..., i]`` is ``grad_fn(weights[..., i], data[..., i], labels[:, i])``.
    """
    batch, _, n_agents = data.shape
    logits = np.einsum("bfi,fci->ibc", data, weights)
    probs = np.exp(_log_softmax(logits, axis=2))
    probs[np.arange(n_agents)[:, None], np.arange(batch)[None, :], labels.T] -= 1.0
    return np.einsum("bfi,ibc->fci", data, probs) / batch


def global_losses(weights: np.ndarray, data: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Loss of each agent's model averaged over every agent's batch.

    ``weights`` is ``(F, C, N)``, ``data`` is ``(B, F, N)``, ``labels`` is ``(B, N)``.
    Returns ``(N,)`` where entry ``i`` is ``mean_j loss_fn(weights[..., i], data[..., j], labels[:, j])``.
    """
    batch, _, n_agents = data.shape
    logits = np.einsum("bfj,fci->ijbc", data, weights)
    log_p = _log_softmax(logits, axis=3)
    picked = log_p[:, np.arange(n_agents)[:, None], np.arange(batch)[None, :], labels.T]
    return -picked.mean(axis=(1, 2))


def lmo_l1_columns(v: np.ndarray, radius: float = 1.0) -> np.ndarray:
    """Linear minimization oracle over the set of matrices whose columns lie in an l1 ball.

    For each column (axis 0 is reduced, any trailing axes are kept), puts ``-radius * sign(v)``
    on the entry of largest magnitude and zero elsewhere.
    """
    idx = np.abs(v).argmax(axis=0)
    picked = np.take_along_axis(v, idx[None, ...], axis=0)
    sol = np.zeros_like(v, dtype=float)
    np.put_along_axis(sol, idx[None, ...], -radius * np.sign(picked), axis=0)
    return sol
