"""
Global Statistics for Relevance Computation

Computes and stores stakeholder counts across ALL topics in the dataset.
Required for penalty calculation in Rel(ς, d).
"""

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Set
import json


@dataclass
class GlobalStats:
    """
    Global statistics across all topics.

    Attributes:
        count: count[stakeholder][topic] = number of occurrences
        topics: List of all topic names
    """
    count: Dict[str, Dict[str, int]] = field(default_factory=lambda: defaultdict(lambda: defaultdict(int)))
    topics: List[str] = field(default_factory=list)

    def get_stakeholder_counts(self, stakeholder: str) -> List[int]:
        """Get counts for a stakeholder across all topics."""
        return [self.count[stakeholder].get(topic, 0) for topic in self.topics]

    def get_total_count(self, stakeholder: str) -> int:
        """Get total count of stakeholder across all topics."""
        return sum(self.count[stakeholder].values())

    def get_topic_count(self, stakeholder: str, topic: str) -> int:
        """Get count of stakeholder in a specific topic."""
        return self.count[stakeholder].get(topic, 0)

    def get_all_stakeholders(self) -> Set[str]:
        """Get all unique stakeholders."""
        return set(self.count.keys())

    def to_dict(self) -> dict:
        """Convert to serializable dict."""
        return {
            "count": {s: dict(topics) for s, topics in self.count.items()},
            "topics": self.topics,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "GlobalStats":
        """Create from dict."""
        stats = cls()
        stats.topics = data["topics"]
        for stakeholder, topic_counts in data["count"].items():
            for topic, count in topic_counts.items():
                stats.count[stakeholder][topic] = count
        return stats

    def save(self, path: str):
        """Save to JSON file."""
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str) -> "GlobalStats":
        """Load from JSON file."""
        with open(path, "r") as f:
            return cls.from_dict(json.load(f))


def build_global_stats(all_events_by_topic: Dict[str, List]) -> GlobalStats:
    """
    Build global statistics from events across all topics.

    Args:
        all_events_by_topic: {topic_name: [Event, ...]} for ALL topics

    Returns:
        GlobalStats with stakeholder counts per topic
    """
    stats = GlobalStats()
    stats.topics = list(all_events_by_topic.keys())

    for topic_name, events in all_events_by_topic.items():
        for event in events:
            for stakeholder in event.stakeholders:
                stats.count[stakeholder][topic_name] += 1

    return stats
