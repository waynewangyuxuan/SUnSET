"""
Event Clustering

Clusters events using connected components on the event graph.
"""

from dataclasses import dataclass, field
from typing import List, Set, Dict
from collections import deque
from .graph import EventGraph
from ..utils import StageLogger, MetricsCollector


@dataclass
class Cluster:
    """
    A cluster of related events.

    Attributes:
        id: Cluster identifier
        event_ids: List of event IDs in this cluster
    """
    id: int
    event_ids: List[int] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.event_ids)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "event_ids": self.event_ids,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Cluster":
        return cls(id=data["id"], event_ids=data["event_ids"])


def find_connected_components(graph: EventGraph) -> List[Set[int]]:
    """
    Find connected components in the graph using BFS.

    Args:
        graph: Event graph (will be treated as undirected)

    Returns:
        List of sets, each containing node IDs in a component
    """
    # Make graph undirected
    undirected = graph.to_undirected()

    visited = set()
    components = []

    for start_node in range(graph.num_nodes):
        if start_node in visited:
            continue

        # BFS to find component
        component = set()
        queue = deque([start_node])

        while queue:
            node = queue.popleft()
            if node in visited:
                continue

            visited.add(node)
            component.add(node)

            for neighbor, _ in undirected.get_neighbors(node):
                if neighbor not in visited:
                    queue.append(neighbor)

        components.append(component)

    return components


def cluster_events(graph: EventGraph) -> List[Cluster]:
    """
    Cluster events using connected components.

    Args:
        graph: Event graph

    Returns:
        List of Cluster objects
    """
    logger = StageLogger("stage2", "clustering")
    metrics = MetricsCollector("stage2_clustering")

    with metrics.timer("find_components"):
        components = find_connected_components(graph)

    clusters = []
    for i, component in enumerate(components):
        cluster = Cluster(
            id=i,
            event_ids=sorted(list(component))
        )
        clusters.append(cluster)
        metrics.count("cluster_sizes", len(component))

    # Sort clusters by size (largest first)
    clusters.sort(key=lambda c: len(c), reverse=True)

    # Reassign IDs after sorting
    for i, cluster in enumerate(clusters):
        cluster.id = i

    # Log statistics
    sizes = [len(c) for c in clusters]
    singleton_count = sum(1 for s in sizes if s == 1)

    logger.info(
        f"Found {len(clusters)} clusters from {graph.num_nodes} events",
        data={
            "num_clusters": len(clusters),
            "largest_cluster": max(sizes) if sizes else 0,
            "singletons": singleton_count,
            "avg_size": sum(sizes) / len(sizes) if sizes else 0,
        }
    )

    metrics.record("num_clusters", len(clusters))
    metrics.record("singleton_clusters", singleton_count)

    return clusters


def get_cluster_statistics(clusters: List[Cluster]) -> Dict:
    """
    Compute statistics about clusters.

    Args:
        clusters: List of clusters

    Returns:
        Dict with cluster statistics
    """
    if not clusters:
        return {
            "num_clusters": 0,
            "total_events": 0,
            "avg_size": 0,
            "max_size": 0,
            "min_size": 0,
            "singletons": 0,
            "size_distribution": {},
        }

    sizes = [len(c) for c in clusters]

    # Size distribution buckets
    distribution = {}
    for size in sizes:
        bucket = str(size) if size <= 5 else "6+"
        distribution[bucket] = distribution.get(bucket, 0) + 1

    return {
        "num_clusters": len(clusters),
        "total_events": sum(sizes),
        "avg_size": sum(sizes) / len(sizes),
        "max_size": max(sizes),
        "min_size": min(sizes),
        "singletons": sum(1 for s in sizes if s == 1),
        "size_distribution": distribution,
    }
