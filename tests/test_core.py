import numpy as np
import pytest

from delaymfw.data import data_processing, uniform_delays
from delaymfw.graph import GRAPH_TYPES, compute_spectral_gap, generate_graph, grid_shape
from delaymfw.model import batched_grad, global_losses, grad_fn, lmo_l1_columns, loss_fn


@pytest.fixture
def rng():
    return np.random.default_rng(0)


def test_grad_matches_finite_differences(rng):
    w = rng.normal(size=(6, 4))
    x = rng.normal(size=(5, 6))
    y = rng.integers(0, 4, size=5)
    g = grad_fn(w, x, y)
    eps = 1e-6
    num = np.zeros_like(w)
    for idx in np.ndindex(w.shape):
        e = np.zeros_like(w)
        e[idx] = eps
        num[idx] = (loss_fn(w + e, x, y) - loss_fn(w - e, x, y)) / (2 * eps)
    np.testing.assert_allclose(g, num, atol=1e-7)


def test_loss_is_mean_cross_entropy(rng):
    w = rng.normal(size=(3, 4))
    x = rng.normal(size=(7, 3))
    y = rng.integers(0, 4, size=7)
    logits = x @ w
    p = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
    assert loss_fn(w, x, y) == pytest.approx(-np.log(p[np.arange(7), y]).mean())


def test_batched_helpers_match_per_agent_loops(rng):
    F, C, B, N = 5, 3, 4, 6
    w = rng.normal(size=(F, C, N))
    x = rng.normal(size=(B, F, N))
    y = rng.integers(0, C, size=(B, N))
    expected_grad = np.stack([grad_fn(w[..., i], x[..., i], y[:, i]) for i in range(N)], axis=-1)
    np.testing.assert_allclose(batched_grad(w, x, y), expected_grad)
    expected_loss = [np.mean([loss_fn(w[..., i], x[..., j], y[:, j]) for j in range(N)]) for i in range(N)]
    np.testing.assert_allclose(global_losses(w, x, y), expected_loss)


def test_lmo_matches_julia_loop(rng):
    v = rng.normal(size=(8, 5))
    expected = np.zeros_like(v)
    for col in range(v.shape[1]):
        row = np.argmax(np.abs(v[:, col]))
        expected[row, col] = -3 * np.sign(v[row, col])
    np.testing.assert_array_equal(lmo_l1_columns(v, 3), expected)
    stacked = rng.normal(size=(8, 5, 4))
    for i in range(4):
        np.testing.assert_array_equal(lmo_l1_columns(stacked, 2)[..., i], lmo_l1_columns(stacked[..., i], 2))


@pytest.mark.parametrize("graph_type", GRAPH_TYPES)
def test_gossip_matrix_is_doubly_stochastic(graph_type):
    graph, w, gap = generate_graph(graph_type, 30, p=0.5, seed=3)
    assert graph.number_of_nodes() == 30
    np.testing.assert_allclose(w, w.T)
    np.testing.assert_allclose(w.sum(axis=0), 1)
    np.testing.assert_allclose(w.sum(axis=1), 1)
    assert 0 < gap <= 1 + 1e-12
    assert gap == pytest.approx(compute_spectral_gap(w))


def test_grid_shape():
    assert grid_shape(30) == (5, 6)
    assert grid_shape(25) == (5, 5)


def test_data_processing_layout(rng):
    n, F, agents, batch = 50, 3, 2, 4
    x = np.arange(n * F, dtype=np.float32).reshape(n, F)
    y = np.arange(n)
    features, data, labels = data_processing(x, y, agents, batch, np.random.default_rng(1))
    perm = np.random.default_rng(1).permutation(n)
    chunk = n // agents
    assert features == F
    assert data.shape == (chunk // batch, batch, F, agents)
    for t in range(data.shape[0]):
        for i in range(agents):
            idx = perm[i * chunk + t * batch : i * chunk + (t + 1) * batch]
            np.testing.assert_array_equal(data[t, :, :, i], x[idx])
            np.testing.assert_array_equal(labels[t, :, i], y[idx])


def test_uniform_delays_range(rng):
    d = uniform_delays(101, 10_000, rng)
    assert d.min() == 11 and d.max() == 101
    assert uniform_delays(1, 5, rng).tolist() == [1] * 5
