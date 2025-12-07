"""
Stage 2: Event Clustering

This module implements event clustering using stakeholder-based relevance scores.
"""

from .global_stats import GlobalStats, build_global_stats
from .relevance import compute_relevance, compute_penalty, compute_reward
from .embedding import EmbeddingClient, LocalEmbeddingClient, cosine_similarity, cosine_similarity_matrix
from .graph import EventGraph, build_event_graph, compute_edge_weight
from .clustering import Cluster, cluster_events, find_connected_components, get_cluster_statistics


def get_embedding_client(config):
    """
    Get the appropriate embedding client based on config.

    Args:
        config: EmbeddingConfig object

    Returns:
        EmbeddingClient or LocalEmbeddingClient
    """
    if getattr(config, 'provider', 'api') == 'local':
        return LocalEmbeddingClient(config.local_model)
    else:
        return EmbeddingClient(config)


__all__ = [
    # Global stats
    "GlobalStats",
    "build_global_stats",
    # Relevance
    "compute_relevance",
    "compute_penalty",
    "compute_reward",
    # Embedding
    "EmbeddingClient",
    "LocalEmbeddingClient",
    "get_embedding_client",
    "cosine_similarity",
    "cosine_similarity_matrix",
    # Graph
    "EventGraph",
    "build_event_graph",
    "compute_edge_weight",
    # Clustering
    "Cluster",
    "cluster_events",
    "find_connected_components",
    "get_cluster_statistics",
]
