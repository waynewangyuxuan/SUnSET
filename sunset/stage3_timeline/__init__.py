"""
Stage 3: Timeline Generation

This module implements timeline generation from event clusters.
"""

from .significance import (
    get_cluster_stakeholders,
    compute_significance,
    RankedCluster,
    rank_clusters,
)
from .textrank import textrank_select, select_representatives
from .generator import (
    TimelineEntry,
    Timeline,
    dedupe_by_date,
    generate_timeline,
    get_timeline_statistics,
)

__all__ = [
    # Significance
    "get_cluster_stakeholders",
    "compute_significance",
    "RankedCluster",
    "rank_clusters",
    # TextRank
    "textrank_select",
    "select_representatives",
    # Generator
    "TimelineEntry",
    "Timeline",
    "dedupe_by_date",
    "generate_timeline",
    "get_timeline_statistics",
]
