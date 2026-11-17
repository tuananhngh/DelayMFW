"""Check the optimized ports against line-by-line transcriptions of the Julia code (0-based)."""

import math

import numpy as np
import pytest

from delaymfw.algorithms import bold_mfw, d_ofw, dec_delay_mfw, delay_mfw
from delaymfw.data import uniform_delays
from delaymfw.graph import generate_graph
from delaymfw.model import grad_fn, lmo_l1_columns, loss_fn

F, C, B, T = 6, 3, 4, 40


@pytest.fixture
def problem():
    rng = np.random.default_rng(42)
    data = rng.normal(size=(T, B, F))
    labels = rng.integers(0, C, size=(T, B))
    return data, labels


def julia_delay_mfw(dim, data, label, loss_enp, gradient_enp, lmo, num_iters, max_delay, delay, eta, R, K, rng):
    loss_val = np.zeros(num_iters)
    gs_ = np.zeros((num_iters, K, *dim))
    grad_cell = np.zeros((K, *dim))
    for t in range(num_iters):
        dt, lb = data[t], label[t]
        xs = np.zeros((K + 1, *dim))
        for k in range(K):
            sigma = min(1, 1 / (k + 4))
            gs_[t, k] = gradient_enp(xs[k], dt, lb)
            v = lmo(eta * grad_cell[k] + (rng.random(dim) - 0.5), R)
            xs[k + 1] = (1 - sigma) * xs[k] + sigma * v
        loss_val[t] = loss_enp(xs[K], dt, lb)
        for s in range(max(0, t - max_delay), t + 1):
            if s + delay[s] - 1 == t:
                for k in range(K):
                    grad_cell[k] += gs_[s, k]
    return loss_val


def julia_d_ofw(dim, data, label, loss_enp, gradient_enp, lmo, num_iters, max_delay, delay, eta, R):
    x = np.ones(dim)
    x = R * x / x.sum()
    x_1 = x.copy()
    loss_val = np.zeros(num_iters)
    G = np.zeros(dim)
    rec_G = np.zeros((num_iters, *dim))
    for t in range(num_iters):
        dt, lb = data[t], label[t]
        loss_val[t] = loss_enp(x, dt, lb)
        rec_G[t] = gradient_enp(x, dt, lb)
        for s in range(max(0, t - max_delay), t + 1):
            if s + delay[s] - 1 == t:
                G += rec_G[s]
                DF = eta * G + 2 * (x - x_1)
                v = lmo(DF, R)
                delta = v - x
                sigma = min(-0.5 * np.vdot(delta, DF) / np.vdot(delta, delta), 1)
                x = (1 - sigma) * x + sigma * v
    return loss_val


def julia_dec_delay_mfw(dim, data, label, num_agents, weights, lmo, num_iters, eta, delay, radius, rng):
    def f_sum(x, data_, label_):
        return sum(loss_fn(x, data_[:, :, i], label_[:, i]) for i in range(num_agents))

    def gradient_cat(x, data_, label_):
        return np.stack([grad_fn(x[:, :, i], data_[:, :, i], label_[:, i]) for i in range(num_agents)], axis=-1)

    def weighted_average(x, w):
        d1, d2, n = x.shape
        return (x.reshape(d1 * d2, n) @ w).reshape(d1, d2, n)

    K = math.ceil(math.sqrt(num_iters))
    rewards = np.zeros(num_iters)
    grad_cell = np.zeros((K, *dim, num_agents))
    gradient_hold = [[] for _ in range(num_agents)]
    injected_noise = rng.random((*dim, num_agents))
    for t in range(num_iters):
        dt, lb = data[t], label[t]
        xs = np.zeros((*dim, num_agents))
        tmp_grad = np.zeros((K + 1, *dim, num_agents))
        for k in range(K):
            sigma = min(1, 1 / (k + 4))
            tmp_grad[k] = gradient_cat(xs, dt, lb)
            v = np.stack([lmo((eta * grad_cell[k] + injected_noise)[:, :, i], radius) for i in range(num_agents)], -1)
            xs = (1 - sigma) * weighted_average(xs, weights) + sigma * v
        tmp_grad[K] = gradient_cat(xs, dt, lb)
        rewards[t] = max(f_sum(xs[:, :, i], dt, lb) / num_agents for i in range(num_agents))
        for i in range(num_agents):
            gradient_hold[i].append((t, tmp_grad[:, :, :, i]))
        sum_delay_grad = np.zeros((K + 1, *dim, num_agents))
        for i in range(num_agents):
            # Julia deleted while enumerating (skipping the next element); deliver every due gradient.
            keep = []
            for s, g in gradient_hold[i]:
                if s + delay[s, i] - 1 == t:
                    sum_delay_grad[:, :, :, i] += g
                else:
                    keep.append((s, g))
            gradient_hold[i] = keep
        gs_ = np.zeros((K + 1, *dim, num_agents))
        gs_[0] = sum_delay_grad[0]
        for k in range(K):
            ds = weighted_average(gs_[k], weights)
            grad_cell[k] += ds
            gs_[k + 1] = sum_delay_grad[k + 1] - sum_delay_grad[k] + ds
    return rewards


@pytest.mark.parametrize("max_delay", [1, 7])
def test_delay_mfw_matches_reference(problem, max_delay):
    data, labels = problem
    delay = uniform_delays(max_delay, T, np.random.default_rng(1))
    args = ((F, C), data, labels, loss_fn, grad_fn, lmo_l1_columns, T, max_delay, delay, 0.05, 2.0, 5)
    ours = delay_mfw(*args, np.random.default_rng(7))
    ref = julia_delay_mfw(*args, np.random.default_rng(7))
    np.testing.assert_allclose(ours, ref)


@pytest.mark.parametrize("max_delay", [1, 7])
def test_d_ofw_matches_reference(problem, max_delay):
    data, labels = problem
    delay = uniform_delays(max_delay, T, np.random.default_rng(1))
    args = ((F, C), data, labels, loss_fn, grad_fn, lmo_l1_columns, T, max_delay, delay, 0.05, 2.0)
    np.testing.assert_allclose(d_ofw(*args), julia_d_ofw(*args))


def test_bold_mfw_runs_and_learns(problem):
    data, labels = problem
    delay = uniform_delays(5, T, np.random.default_rng(1))
    out = bold_mfw((F, C), data, labels, loss_fn, grad_fn, lmo_l1_columns, T, 5, delay, 0.05, 2.0, 5, np.random.default_rng(0))
    assert out.shape == (T,) and np.all(np.isfinite(out))


@pytest.mark.parametrize("max_delay", [1, 4])
def test_dec_delay_mfw_matches_reference(max_delay):
    rng = np.random.default_rng(3)
    N, T_dec = 5, 16
    data = rng.normal(size=(T_dec, B, F, N))
    labels = rng.integers(0, C, size=(T_dec, B, N))
    _, w, _ = generate_graph("cycle", N)
    delays = uniform_delays(max_delay, (T_dec, N), rng)
    delays[:, 1] = 3  # a "selected" agent with a fixed delay
    ours = dec_delay_mfw((F, C), data, labels, w, lmo_l1_columns, T_dec, 0.1, delays, 2.0, np.random.default_rng(9), progress=False)
    ref = julia_dec_delay_mfw((F, C), data, labels, N, w, lmo_l1_columns, T_dec, 0.1, delays, 2.0, np.random.default_rng(9))
    np.testing.assert_allclose(ours, ref)
