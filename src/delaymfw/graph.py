"""Communication graphs and their Metropolis-Hastings gossip matrices."""

from __future__ import annotations

import math

import networkx as nx
import numpy as np

GRAPH_TYPES = ("er", "complete", "grid", "cycle", "star")


def grid_shape(n: int) -> tuple[int, int]:
    """Most square ``rows x cols`` factorization of ``n`` (e.g. 30 -> 5 x 6, 25 -> 5 x 5)."""
    rows = max(r for r in range(1, math.isqrt(n) + 1) if n % r == 0)
    return rows, n // rows


def make_graph(graph_type: str, nb_agents: int, p: float = 0.4, seed: int | None = None) -> nx.Graph:
    if graph_type == "er":
        graph = nx.erdos_renyi_graph(nb_agents, p, seed=seed)
    elif graph_type == "complete":
        graph = nx.complete_graph(nb_agents)
    elif graph_type == "cycle":
        graph = nx.cycle_graph(nb_agents)
    elif graph_type == "star":
        graph = nx.star_graph(nb_agents - 1)  # networkx counts the leaves, not the nodes
    elif graph_type == "grid":
        graph = nx.grid_2d_graph(*grid_shape(nb_agents))
    else:
        raise ValueError(f"Unknown graph type {graph_type!r}, expected one of {GRAPH_TYPES}")
    return nx.convert_node_labels_to_integers(graph, ordering="sorted")


def generate_weighted_adjacency_matrix(graph: nx.Graph) -> np.ndarray:
    """``W[i, j] = 1 / (1 + max(deg i, deg j))`` on edges, diagonal set so rows sum to 1."""
    if not nx.is_connected(graph):
        raise ValueError("Graph is not connected, retry with another seed or edge probability")
    n = graph.number_of_nodes()
    degrees = np.array([graph.degree(i) for i in range(n)])
    weight = np.zeros((n, n))
    for i, j in graph.edges():
        if i != j:
            weight[i, j] = weight[j, i] = 1.0 / (1.0 + max(degrees[i], degrees[j]))
    weight += np.diag(1.0 - weight.sum(axis=1))
    return weight


def compute_spectral_gap(weight_matrix: np.ndarray) -> float:
    eig = np.sort(np.linalg.eigvalsh(weight_matrix))[::-1]
    return float(eig[0] - eig[1])


def generate_graph(
    graph_type: str, nb_agents: int, p: float = 0.4, seed: int | None = None
) -> tuple[nx.Graph, np.ndarray, float]:
    graph = make_graph(graph_type, nb_agents, p, seed)
    weight = generate_weighted_adjacency_matrix(graph)
    return graph, weight, compute_spectral_gap(weight)
