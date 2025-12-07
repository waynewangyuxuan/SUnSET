"""
Cluster Significance Computation

Implements Significance(C) from paper Equation 7.
"""

import math
from typing import List, Set, Tuple
from dataclasses import dataclass
from ..stage2_clustering import GlobalStats, Cluster, compute_relevance
from ..utils import StageLogger


def get_cluster_stakeholders(cluster: Cluster, events: List) -> Set[str]:
    """
    Get all unique stakeholders in a cluster.

    From paper Equation 6: S_C = ∪_{e∈C} {ς | ς ∈ S_e}

    Args:
        cluster: Cluster object
        events: List of all Event objects

    Returns:
        Set of unique stakeholder names
    """
    stakeholders = set()
    for event_id in cluster.event_ids:
        for s in events[event_id].stakeholders:
            stakeholders.add(s)
    return stakeholders


def compute_significance(
    cluster: Cluster,
    events: List,
    global_stats: GlobalStats,
    topic: str,
    beta: float = 1.0
) -> float:
    """
    Compute cluster significance.

    From paper Equation 7:
    Significance(C) = [1 + ln(|C|)] × (Σ_{ς∈S_C} Rel(ς,d) / |S_C|)

    Args:
        cluster: Cluster object
        events: List of all Event objects
        global_stats: Global statistics
        topic: Current topic name
        beta: Rel scaling parameter

    Returns:
        Significance score
    """
    # Get cluster stakeholders (Equation 6)
    S_C = get_cluster_stakeholders(cluster, events)

    if len(S_C) == 0:
        return 0.0

    # Size factor: [1 + ln(|C|)]
    cluster_size = len(cluster.event_ids)
    size_factor = 1 + math.log(cluster_size)

    # Sum of Rel over unique stakeholders
    rel_sum = 0.0
    for s in S_C:
        rel_sum += compute_relevance(s, topic, global_stats, beta)

    # Average relevance per stakeholder
    avg_rel = rel_sum / len(S_C)

    # Final significance
    significance = size_factor * avg_rel

    return significance


@dataclass
class RankedCluster:
    """Cluster with its significance score."""
    cluster: Cluster
    significance: float
    representative_event_id: int = -1

    def to_dict(self) -> dict:
        return {
            "cluster_id": self.cluster.id,
            "event_ids": self.cluster.event_ids,
            "significance": self.significance,
            "representative_event_id": self.representative_event_id,
        }


def rank_clusters(
    clusters: List[Cluster],
    events: List,
    global_stats: GlobalStats,
    topic: str,
    beta: float = 1.0
) -> List[RankedCluster]:
    """
    Rank clusters by significance.

    Args:
        clusters: List of clusters
        events: List of all Event objects
        global_stats: Global statistics
        topic: Current topic name
        beta: Rel scaling parameter

    Returns:
        List of RankedCluster, sorted by significance descending
    """
    logger = StageLogger("stage3", "significance")

    ranked = []
    for cluster in clusters:
        sig = compute_significance(cluster, events, global_stats, topic, beta)
        ranked.append(RankedCluster(cluster=cluster, significance=sig))

    # Sort by significance descending
    ranked.sort(key=lambda x: x.significance, reverse=True)

    # Log top clusters
    if ranked:
        top_sigs = [f"{r.significance:.4f}" for r in ranked[:5]]
        logger.info(
            f"Ranked {len(ranked)} clusters by significance",
            data={"top_5_significances": top_sigs}
        )

    return ranked
