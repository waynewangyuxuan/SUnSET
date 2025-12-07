"""
SUnSET Pipeline Runner

Complete pipeline for timeline summarization.
"""

import asyncio
from typing import Dict, List, Optional
from dataclasses import dataclass, field
import numpy as np

from ..config import Config
from ..data import Timeline17Dataset, Topic
from ..stage1_set import EventExtractor, Event
from ..stage2_clustering import (
    GlobalStats, build_global_stats,
    get_embedding_client,
    build_event_graph,
    cluster_events, Cluster,
)
from ..stage3_timeline import (
    generate_timeline, Timeline, get_timeline_statistics,
)
from ..evaluation import (
    EvalResult, evaluate_multi_gold, average_results,
    evaluate_with_tilse_multi_gold, is_tilse_available,
)
from ..utils import StageLogger, ArtifactManager, MetricsCollector


@dataclass
class TopicResult:
    """Result for a single topic."""
    topic_name: str
    events: List[Event]
    clusters: List[Cluster]
    timeline: Timeline
    eval_result: Optional[EvalResult] = None

    def to_dict(self) -> dict:
        return {
            "topic_name": self.topic_name,
            "num_events": len(self.events),
            "num_clusters": len(self.clusters),
            "timeline": self.timeline.to_dict(),
            "eval_result": self.eval_result.to_dict() if self.eval_result else None,
        }


@dataclass
class PipelineResult:
    """Complete pipeline result."""
    topic_results: Dict[str, TopicResult] = field(default_factory=dict)
    global_stats: Optional[GlobalStats] = None
    avg_result: Optional[EvalResult] = None

    def to_dict(self) -> dict:
        return {
            "topics": {k: v.to_dict() for k, v in self.topic_results.items()},
            "avg_result": self.avg_result.to_dict() if self.avg_result else None,
        }


class SUnSETPipeline:
    """
    Complete SUnSET pipeline.

    Runs:
    1. Stage 1: SET extraction on ALL topics (for global stats)
    2. Stage 2: Clustering for each topic
    3. Stage 3: Timeline generation for each topic
    4. Evaluation against gold timelines
    """

    def __init__(self, config: Config):
        self.config = config
        self.logger = StageLogger("pipeline", "main")
        self.metrics = MetricsCollector("pipeline")

    async def run(
        self,
        dataset: Timeline17Dataset,
        artifacts: Optional[ArtifactManager] = None,
        max_articles_per_topic: Optional[int] = None,
    ) -> PipelineResult:
        """
        Run the complete pipeline.

        Args:
            dataset: Timeline17 dataset
            artifacts: Optional artifact manager for saving results
            max_articles_per_topic: Limit articles per topic (for testing)

        Returns:
            PipelineResult with all topic results
        """
        result = PipelineResult()

        # Stage 1: Extract events from ALL topics
        self.logger.info(f"Starting pipeline for {len(dataset.topics)} topics")

        with self.metrics.timer("stage1_extraction"):
            all_events = await self._run_stage1(
                dataset, max_articles_per_topic
            )

        # Build global stats
        with self.metrics.timer("global_stats"):
            global_stats = build_global_stats(all_events)
            result.global_stats = global_stats

        self.logger.info(
            f"Built global stats: {len(global_stats.topics)} topics, "
            f"{len(global_stats.get_all_stakeholders())} stakeholders"
        )

        if artifacts:
            artifacts.save_json("global_stats.json", global_stats.to_dict())

        # Stage 2 & 3: Cluster and generate timeline for each topic
        embed_client = get_embedding_client(self.config.embedding)

        for topic in dataset.topics:
            topic_name = topic.name
            events = all_events[topic_name]

            self.logger.info(f"Processing topic: {topic_name} ({len(events)} events)")

            if len(events) < 2:
                self.logger.warning(f"Skipping {topic_name}: not enough events")
                continue

            # Get embeddings
            with self.metrics.timer("embedding"):
                texts = [e.summary for e in events]
                embeddings = await embed_client.embed_batch(texts)

            # Stage 2: Clustering
            with self.metrics.timer("stage2_clustering"):
                graph = build_event_graph(
                    events=events,
                    embeddings=embeddings,
                    topic=topic_name,
                    global_stats=global_stats,
                    top_k=self.config.pipeline.top_k,
                    em_n=self.config.pipeline.em_n,
                    beta=self.config.pipeline.beta,
                )
                clusters = cluster_events(graph)

            # Stage 3: Timeline generation
            with self.metrics.timer("stage3_timeline"):
                timeline = generate_timeline(
                    clusters=clusters,
                    events=events,
                    embeddings=embeddings,
                    global_stats=global_stats,
                    topic=topic_name,
                    max_entries=self.config.pipeline.max_timeline_entries,
                    beta=self.config.pipeline.beta,
                )

            # Evaluation
            eval_result = None
            if topic.gold_timelines:
                # Convert List[GoldTimeline] to Dict format
                gold_dict = {
                    gt.name: gt.entries
                    for gt in topic.gold_timelines
                }
                with self.metrics.timer("evaluation"):
                    eval_result = evaluate_multi_gold(
                        timeline, gold_dict
                    )

                self.logger.info(
                    f"  {topic_name}: AR-1={eval_result.ar1:.4f}, "
                    f"AR-2={eval_result.ar2:.4f}, Date-F1={eval_result.date_f1:.4f}"
                )

            # Store result
            topic_result = TopicResult(
                topic_name=topic_name,
                events=events,
                clusters=clusters,
                timeline=timeline,
                eval_result=eval_result,
            )
            result.topic_results[topic_name] = topic_result

            # Save artifacts
            if artifacts:
                artifacts.save_jsonl(
                    f"stage1/{topic_name}_events.jsonl",
                    [e.to_dict() for e in events]
                )
                artifacts.save_jsonl(
                    f"stage2/{topic_name}_clusters.jsonl",
                    [c.to_dict() for c in clusters]
                )
                artifacts.save_json(
                    f"stage3/{topic_name}_timeline.json",
                    timeline.to_dict()
                )
                if eval_result:
                    artifacts.save_json(
                        f"evaluation/{topic_name}_result.json",
                        eval_result.to_dict()
                    )

        await embed_client.close()

        # Compute average
        eval_results = {
            k: v.eval_result
            for k, v in result.topic_results.items()
            if v.eval_result is not None
        }
        if eval_results:
            result.avg_result = average_results(eval_results)
            self.logger.info(
                f"AVERAGE: AR-1={result.avg_result.ar1:.4f}, "
                f"AR-2={result.avg_result.ar2:.4f}, "
                f"Date-F1={result.avg_result.date_f1:.4f}"
            )

        if artifacts:
            artifacts.save_json("pipeline_result.json", result.to_dict())
            artifacts.save_json("metrics.json", self.metrics.get_report())

        return result

    async def _run_stage1(
        self,
        dataset: Timeline17Dataset,
        max_articles: Optional[int] = None
    ) -> Dict[str, List[Event]]:
        """Run Stage 1 extraction on all topics."""
        extractor = EventExtractor(self.config, max_concurrent=5)
        all_events = {}

        for topic in dataset.topics:
            articles = topic.articles
            if max_articles:
                articles = articles[:max_articles]

            self.logger.info(f"Extracting from {topic.name}: {len(articles)} articles")

            events = await extractor.extract_all(
                articles, include_stakeholders=True
            )
            all_events[topic.name] = events

            self.logger.info(f"  -> {len(events)} events extracted")

        return all_events


def generate_report(result: PipelineResult) -> str:
    """Generate a human-readable report."""
    lines = [
        "=" * 60,
        "SUnSET Pipeline Results",
        "=" * 60,
        "",
    ]

    # Per-topic results
    for topic_name, topic_result in result.topic_results.items():
        lines.append(f"Topic: {topic_name}")
        lines.append(f"  Events: {len(topic_result.events)}")
        lines.append(f"  Clusters: {len(topic_result.clusters)}")
        lines.append(f"  Timeline entries: {len(topic_result.timeline)}")

        if topic_result.eval_result:
            r = topic_result.eval_result
            lines.append(f"  AR-1:     {r.ar1:.4f}")
            lines.append(f"  AR-2:     {r.ar2:.4f}")
            lines.append(f"  Date-F1:  {r.date_f1:.4f}")

        lines.append("")

    # Average
    if result.avg_result:
        lines.append("=" * 60)
        lines.append(f"AVERAGE ({len(result.topic_results)} topics)")
        lines.append("=" * 60)
        r = result.avg_result
        lines.append(f"  AR-1:     {r.ar1:.4f}")
        lines.append(f"  AR-2:     {r.ar2:.4f}")
        lines.append(f"  Date-F1:  {r.date_f1:.4f}")

    return "\n".join(lines)
