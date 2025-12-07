"""
Event Graph Construction

Builds a graph where events are nodes and edges connect similar events.
Uses top-k neighbors approach from paper Section 3.2.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Set
import numpy as np
from .global_stats import GlobalStats
from .relevance import compute_relevance
from .embedding import cosine_similarity_matrix
from ..utils import StageLogger, MetricsCollector


@dataclass
class EventGraph:
    """
    Graph of events with weighted edges.

    Attributes:
        num_nodes: Number of events/nodes
        adjacency: {node_id: [(neighbor_id, weight), ...]}
    """
    num_nodes: int
    adjacency: Dict[int, List[Tuple[int, float]]] = field(default_factory=dict)

    def get_neighbors(self, node: int) -> List[Tuple[int, float]]:
        """Get neighbors of a node with weights."""
        return self.adjacency.get(node, [])

    def get_all_edges(self) -> List[Tuple[int, int, float]]:
        """Get all edges as (src, dst, weight) tuples."""
        edges = []
        for src, neighbors in self.adjacency.items():
            for dst, weight in neighbors:
                edges.append((src, dst, weight))
        return edges

    def to_undirected(self) -> "EventGraph":
        """Convert to undirected graph by adding reverse edges."""
        undirected_adj = {i: [] for i in range(self.num_nodes)}

        # Collect all edges
        edge_weights = {}
        for src, neighbors in self.adjacency.items():
            for dst, weight in neighbors:
                key = (min(src, dst), max(src, dst))
                # Keep max weight if duplicate
                if key not in edge_weights or weight > edge_weights[key]:
                    edge_weights[key] = weight

        # Add bidirectional edges
        for (a, b), weight in edge_weights.items():
            undirected_adj[a].append((b, weight))
            undirected_adj[b].append((a, weight))

        return EventGraph(num_nodes=self.num_nodes, adjacency=undirected_adj)

    def to_dict(self) -> dict:
        """Convert to serializable dict."""
        return {
            "num_nodes": self.num_nodes,
            "adjacency": {str(k): v for k, v in self.adjacency.items()},
        }


def compute_edge_weight(
    event_i,
    event_j,
    topic: str,
    global_stats: GlobalStats,
    cos_sim: float,
    em_n: int = 1,
    beta: float = 1.0
) -> float:
    """
    Compute edge weight between two events.

    From paper Equation 4:
    W_edge(e_i, e_j) = BoolEM_n(S_ei, S_ej) × [Σ Rel(ς,d) + cos(e_i, e_j)]

    Args:
        event_i: First event
        event_j: Second event
        topic: Current topic name
        global_stats: Global statistics
        cos_sim: Precomputed cosine similarity
        em_n: BoolEM_n threshold (min shared stakeholders)
        beta: Rel scaling parameter

    Returns:
        Edge weight (0 if BoolEM fails)
    """
    S_ei = set(event_i.stakeholders)
    S_ej = set(event_j.stakeholders)

    # Shared stakeholders
    shared = S_ei & S_ej

    # BoolEM_n check
    if len(shared) < em_n:
        return 0.0

    # Sum of Rel for shared stakeholders
    rel_sum = 0.0
    for s in shared:
        rel_sum += compute_relevance(s, topic, global_stats, beta)

    # Total weight = rel_sum + cosine similarity
    weight = rel_sum + cos_sim

    return weight


def build_event_graph(
    events: List,
    embeddings: np.ndarray,
    topic: str,
    global_stats: GlobalStats,
    top_k: int = 20,
    em_n: int = 1,
    beta: float = 1.0
) -> EventGraph:
    """
    Build event graph with top-k neighbors.

    From paper Section 3.2:
    "the top 20 similar events for every node"

    Args:
        events: List of Event objects
        embeddings: Event embeddings, shape (n_events, dim)
        topic: Current topic name
        global_stats: Global statistics
        top_k: Number of neighbors per node (default 20)
        em_n: BoolEM_n threshold
        beta: Rel scaling parameter

    Returns:
        EventGraph with top-k neighbors for each node
    """
    logger = StageLogger("stage2", "graph")
    metrics = MetricsCollector("stage2_graph")

    n = len(events)
    if n == 0:
        return EventGraph(num_nodes=0)

    # Compute cosine similarity matrix
    with metrics.timer("cosine_matrix"):
        cos_sim_matrix = cosine_similarity_matrix(embeddings)

    logger.info(f"Building graph for {n} events with top-{top_k} neighbors")

    adjacency = {}

    with metrics.timer("build_adjacency"):
        for i in range(n):
            # Compute weights to all other nodes
            weights = []
            for j in range(n):
                if i == j:
                    continue

                cos_sim = float(cos_sim_matrix[i, j])
                w = compute_edge_weight(
                    events[i], events[j],
                    topic, global_stats,
                    cos_sim, em_n, beta
                )

                if w > 0:
                    weights.append((j, w))

            # Sort by weight descending and keep top-k
            weights.sort(key=lambda x: x[1], reverse=True)
            adjacency[i] = weights[:top_k]

            metrics.count("edges_considered", len(weights))

    # Count total edges
    total_edges = sum(len(neighbors) for neighbors in adjacency.values())
    metrics.record("total_edges", total_edges)
    metrics.record("avg_neighbors", total_edges / n if n > 0 else 0)

    logger.info(
        f"Graph built: {n} nodes, {total_edges} edges",
        data={"avg_neighbors": total_edges / n if n > 0 else 0}
    )

    return EventGraph(num_nodes=n, adjacency=adjacency)
