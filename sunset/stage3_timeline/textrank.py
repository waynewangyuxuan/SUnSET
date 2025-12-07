"""
TextRank for Representative Event Selection

Implements TextRank algorithm to select the most representative event from a cluster.
"""

import numpy as np
from typing import List
from ..stage2_clustering import Cluster, cosine_similarity_matrix


def textrank_select(
    cluster: Cluster,
    events: List,
    embeddings: np.ndarray,
    damping: float = 0.85,
    max_iter: int = 100,
    tol: float = 1e-6
) -> int:
    """
    Select the most representative event from a cluster using TextRank.

    Args:
        cluster: Cluster object
        events: List of all Event objects
        embeddings: Event embeddings, shape (n_events, dim)
        damping: Damping factor (default 0.85)
        max_iter: Maximum iterations (default 100)
        tol: Convergence tolerance (default 1e-6)

    Returns:
        Event ID of the most representative event
    """
    event_ids = cluster.event_ids
    n = len(event_ids)

    # Single event - return it
    if n == 1:
        return event_ids[0]

    # Get embeddings for this cluster
    cluster_embeds = np.array([embeddings[i] for i in event_ids])

    # Build similarity matrix
    sim_matrix = cosine_similarity_matrix(cluster_embeds)

    # Zero out diagonal (no self-similarity)
    np.fill_diagonal(sim_matrix, 0)

    # Normalize rows (stochastic matrix)
    row_sums = sim_matrix.sum(axis=1, keepdims=True)
    row_sums = np.maximum(row_sums, 1e-10)
    trans_matrix = sim_matrix / row_sums

    # Power iteration
    scores = np.ones(n) / n

    for _ in range(max_iter):
        new_scores = (1 - damping) / n + damping * (trans_matrix.T @ scores)

        if np.abs(new_scores - scores).sum() < tol:
            break

        scores = new_scores

    # Return highest scoring event
    best_idx = np.argmax(scores)
    best_event_id = event_ids[best_idx]

    return best_event_id


def select_representatives(
    ranked_clusters: List,
    events: List,
    embeddings: np.ndarray,
    damping: float = 0.85
) -> List:
    """
    Select representative events for all clusters.

    Args:
        ranked_clusters: List of RankedCluster objects
        events: List of all Event objects
        embeddings: Event embeddings
        damping: TextRank damping factor

    Returns:
        Updated ranked_clusters with representative_event_id set
    """
    for rc in ranked_clusters:
        rc.representative_event_id = textrank_select(
            rc.cluster, events, embeddings, damping
        )

    return ranked_clusters
