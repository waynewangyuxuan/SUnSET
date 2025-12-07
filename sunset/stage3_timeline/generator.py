"""
Timeline Generation

Generates the final timeline from ranked clusters.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional
import numpy as np
from .significance import RankedCluster, rank_clusters
from .textrank import select_representatives
from ..stage2_clustering import GlobalStats, Cluster
from ..utils import StageLogger, MetricsCollector


@dataclass
class TimelineEntry:
    """A single entry in the timeline."""
    date: str
    summary: str
    significance: float
    cluster_id: int = -1
    event_id: int = -1

    def to_dict(self) -> dict:
        return {
            "date": self.date,
            "summary": self.summary,
            "significance": self.significance,
            "cluster_id": self.cluster_id,
            "event_id": self.event_id,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "TimelineEntry":
        return cls(**data)


@dataclass
class Timeline:
    """A generated timeline."""
    entries: List[TimelineEntry] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.entries)

    def get_dates(self) -> List[str]:
        """Get list of dates in timeline."""
        return [e.date for e in self.entries]

    def get_summaries(self) -> List[str]:
        """Get list of summaries."""
        return [e.summary for e in self.entries]

    def to_dict(self) -> dict:
        return {
            "entries": [e.to_dict() for e in self.entries],
            "num_entries": len(self.entries),
        }

    def to_evaluation_format(self) -> Dict[str, List[str]]:
        """
        Convert to format expected by evaluation metrics.

        Returns:
            {date: [summary1, summary2, ...]}
        """
        result = {}
        for entry in self.entries:
            if entry.date not in result:
                result[entry.date] = []
            result[entry.date].append(entry.summary)
        return result

    @classmethod
    def from_dict(cls, data: dict) -> "Timeline":
        entries = [TimelineEntry.from_dict(e) for e in data["entries"]]
        return cls(entries=entries)


def dedupe_by_date(
    ranked_clusters: List[RankedCluster],
    events: List
) -> List[TimelineEntry]:
    """
    Create timeline entries, keeping only one per date (most significant).

    Args:
        ranked_clusters: Ranked clusters with representatives (sorted by significance)
        events: List of all Event objects

    Returns:
        List of TimelineEntry, one per unique date
    """
    seen_dates = set()
    entries = []

    for rc in ranked_clusters:
        if rc.representative_event_id < 0:
            continue

        event = events[rc.representative_event_id]
        date = event.date

        if date in seen_dates:
            continue

        seen_dates.add(date)
        entries.append(TimelineEntry(
            date=date,
            summary=event.summary,
            significance=rc.significance,
            cluster_id=rc.cluster.id,
            event_id=rc.representative_event_id,
        ))

    return entries


def generate_timeline(
    clusters: List[Cluster],
    events: List,
    embeddings: np.ndarray,
    global_stats: GlobalStats,
    topic: str,
    max_entries: int = 50,
    beta: float = 1.0,
    damping: float = 0.85
) -> Timeline:
    """
    Generate a timeline from clusters.

    Complete Stage 3 pipeline:
    1. Rank clusters by significance
    2. Select representative events via TextRank
    3. Dedupe by date
    4. Sort by date and limit

    Args:
        clusters: List of clusters from Stage 2
        events: List of all Event objects
        embeddings: Event embeddings
        global_stats: Global statistics
        topic: Current topic name
        max_entries: Maximum timeline entries
        beta: Rel scaling parameter
        damping: TextRank damping factor

    Returns:
        Generated Timeline
    """
    logger = StageLogger("stage3", "generator")
    metrics = MetricsCollector("stage3_timeline")

    if not clusters or not events:
        logger.warning("No clusters or events to generate timeline")
        return Timeline()

    # Step 1: Rank clusters by significance
    with metrics.timer("rank_clusters"):
        ranked = rank_clusters(clusters, events, global_stats, topic, beta)

    # Step 2: Select representative events via TextRank
    with metrics.timer("textrank"):
        ranked = select_representatives(ranked, events, embeddings, damping)

    # Step 3: Dedupe by date
    with metrics.timer("dedupe"):
        entries = dedupe_by_date(ranked, events)

    # Step 4: Sort by date and limit
    entries.sort(key=lambda e: e.date)
    entries = entries[:max_entries]

    timeline = Timeline(entries=entries)

    logger.info(
        f"Generated timeline with {len(timeline)} entries",
        data={
            "total_clusters": len(clusters),
            "unique_dates": len(timeline),
            "max_entries": max_entries,
        }
    )

    metrics.record("timeline_length", len(timeline))

    return timeline


def get_timeline_statistics(timeline: Timeline) -> Dict:
    """
    Compute statistics about a timeline.

    Args:
        timeline: Generated timeline

    Returns:
        Dict with timeline statistics
    """
    if not timeline.entries:
        return {
            "num_entries": 0,
            "date_range": None,
            "avg_significance": 0,
        }

    dates = timeline.get_dates()
    significances = [e.significance for e in timeline.entries]

    return {
        "num_entries": len(timeline),
        "date_range": {"start": min(dates), "end": max(dates)},
        "avg_significance": sum(significances) / len(significances),
        "max_significance": max(significances),
        "min_significance": min(significances),
    }
