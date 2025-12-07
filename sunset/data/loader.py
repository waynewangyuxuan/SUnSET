"""
Data Loader Module

Loads and processes Timeline17 dataset.
"""

import pickle
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..utils.logging import StageLogger


@dataclass
class Sentence:
    """A sentence from an article."""
    text: str
    mentioned_dates: List[str] = field(default_factory=list)


@dataclass
class Article:
    """A news article."""
    pub_date: str
    sentences: List[Sentence]

    @property
    def text(self) -> str:
        """Get full article text."""
        return " ".join(s.text for s in self.sentences)

    @property
    def num_sentences(self) -> int:
        return len(self.sentences)


@dataclass
class GoldTimeline:
    """A ground truth timeline."""
    name: str
    entries: Dict[str, List[str]]  # date -> list of summaries

    @property
    def dates(self) -> List[str]:
        return sorted(self.entries.keys())

    @property
    def num_entries(self) -> int:
        return len(self.entries)


@dataclass
class Topic:
    """A topic with articles and gold timelines."""
    name: str
    articles: List[Article]
    gold_timelines: List[GoldTimeline]

    @property
    def num_articles(self) -> int:
        return len(self.articles)

    @property
    def date_range(self) -> tuple:
        """Get (min_date, max_date) from articles."""
        dates = [a.pub_date for a in self.articles]
        return (min(dates), max(dates)) if dates else (None, None)


class Timeline17Dataset:
    """
    Timeline17 dataset loader and accessor.

    Usage:
        dataset = Timeline17Dataset.load("timeline17.pkl")

        for topic in dataset.topics:
            print(f"{topic.name}: {topic.num_articles} articles")

        # Access specific topic
        egypt = dataset.get_topic("EgyptianProtest")
    """

    def __init__(self, topics: List[Topic]):
        self.topics = topics
        self._topic_map = {t.name: t for t in topics}

    @classmethod
    def load(cls, path: str) -> "Timeline17Dataset":
        """Load dataset from pickle file."""
        logger = StageLogger("data", "loader", log_format="text")

        with open(path, "rb") as f:
            raw_data = pickle.load(f)

        topics = []
        for topic_name, topic_data in raw_data.items():
            # Parse articles
            articles = []
            for article_raw in topic_data.get("articles", []):
                sentences = []
                for sent_raw in article_raw.get("sentences", []):
                    if isinstance(sent_raw, dict):
                        sentences.append(Sentence(
                            text=sent_raw.get("text", ""),
                            mentioned_dates=sent_raw.get("mentioned_dates", []),
                        ))
                    elif isinstance(sent_raw, str):
                        sentences.append(Sentence(text=sent_raw))

                articles.append(Article(
                    pub_date=article_raw.get("pub_date", ""),
                    sentences=sentences,
                ))

            # Parse gold timelines
            gold_timelines = []
            for tl_name, tl_data in topic_data.get("gold_timelines", {}).items():
                gold_timelines.append(GoldTimeline(
                    name=tl_name,
                    entries=tl_data,
                ))

            topics.append(Topic(
                name=topic_name,
                articles=articles,
                gold_timelines=gold_timelines,
            ))

        logger.info(f"Loaded {len(topics)} topics from {path}")
        return cls(topics)

    def get_topic(self, name: str) -> Optional[Topic]:
        """Get a topic by name."""
        return self._topic_map.get(name)

    @property
    def topic_names(self) -> List[str]:
        """Get all topic names."""
        return list(self._topic_map.keys())

    def get_stats(self) -> Dict[str, Any]:
        """Get dataset statistics."""
        stats = {
            "num_topics": len(self.topics),
            "topics": {},
        }

        total_articles = 0
        total_gold_timelines = 0

        for topic in self.topics:
            min_date, max_date = topic.date_range
            topic_stats = {
                "num_articles": topic.num_articles,
                "num_gold_timelines": len(topic.gold_timelines),
                "date_range": {
                    "min": min_date,
                    "max": max_date,
                },
                "gold_timeline_lengths": [
                    gt.num_entries for gt in topic.gold_timelines
                ],
            }
            stats["topics"][topic.name] = topic_stats

            total_articles += topic.num_articles
            total_gold_timelines += len(topic.gold_timelines)

        stats["total_articles"] = total_articles
        stats["total_gold_timelines"] = total_gold_timelines

        return stats

    def __len__(self) -> int:
        return len(self.topics)

    def __iter__(self):
        return iter(self.topics)


def article_to_text(article: Article) -> str:
    """Convert article to plain text."""
    return article.text


def get_gold_timeline_dict(topic: Topic, timeline_index: int = 0) -> Dict[str, List[str]]:
    """
    Get gold timeline as dict {date: [summaries]}.

    Args:
        topic: Topic object
        timeline_index: Which gold timeline to use (default: first)

    Returns:
        Dict mapping date strings to lists of summary strings
    """
    if not topic.gold_timelines:
        return {}

    idx = min(timeline_index, len(topic.gold_timelines) - 1)
    return topic.gold_timelines[idx].entries
