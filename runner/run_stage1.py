"""
Stage 1: Event + Stakeholder Extraction

Extracts events and stakeholders from all articles, saves for reuse.
"""

import asyncio
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, ".")

from sunset.config import load_config
from sunset.data import Timeline17Dataset
from sunset.stage1_set import EventExtractor
from sunset.utils import ArtifactManager, StageLogger


async def main():
    parser = argparse.ArgumentParser(description="Stage 1: Event + Stakeholder Extraction")
    parser.add_argument("--config", "-c", default="config.yaml", help="Config file")
    parser.add_argument("--data", "-d", default="timeline17.pkl", help="Dataset path")
    parser.add_argument("--output", "-o", default="results", help="Output directory")
    parser.add_argument("--run-name", default="stage1_extraction", help="Run name")
    parser.add_argument("--max-concurrent", type=int, default=5, help="Max concurrent LLM calls")
    args = parser.parse_args()

    logger = StageLogger("stage1", "main", log_format="text")

    # Load config and data
    logger.info(f"Loading config from {args.config}")
    config = load_config(args.config)

    logger.info(f"Loading dataset from {args.data}")
    dataset = Timeline17Dataset.load(args.data)

    # Setup artifacts
    artifacts = ArtifactManager(args.output)
    artifacts.new_run(args.run_name)
    artifacts.save_json("config.json", config.to_dict())

    logger.info(f"Output: {artifacts.run_dir}")
    logger.info(f"Topics: {len(dataset.topics)}")

    # Extract events from all topics
    extractor = EventExtractor(config, max_concurrent=args.max_concurrent)
    all_events = {}
    total_events = 0
    total_articles = 0

    for topic in dataset.topics:
        topic_name = topic.name
        articles = topic.articles

        logger.info(f"\n{'='*50}")
        logger.info(f"Topic: {topic_name} ({len(articles)} articles)")
        logger.info(f"{'='*50}")

        # Extract with stakeholders
        events = await extractor.extract_all(articles, include_stakeholders=True)
        all_events[topic_name] = events

        total_events += len(events)
        total_articles += len(articles)

        # Save per-topic
        artifacts.save_jsonl(
            f"stage1/{topic_name}_events.jsonl",
            [e.to_dict() for e in events]
        )

        logger.info(f"  -> {len(events)} events extracted")

        # Show sample
        for e in events[:3]:
            logger.info(f"     {e.date}: {e.summary[:60]}...")
            logger.info(f"       Stakeholders: {e.stakeholders}")

    # Save combined
    combined = []
    for topic_name, events in all_events.items():
        for e in events:
            d = e.to_dict()
            d["topic"] = topic_name
            combined.append(d)

    artifacts.save_jsonl("stage1/all_events.jsonl", combined)

    # Summary
    logger.info(f"\n{'='*50}")
    logger.info(f"SUMMARY")
    logger.info(f"{'='*50}")
    logger.info(f"Total topics: {len(dataset.topics)}")
    logger.info(f"Total articles: {total_articles}")
    logger.info(f"Total events: {total_events}")
    logger.info(f"Avg events/article: {total_events/total_articles:.2f}")

    # Per-topic stats
    logger.info(f"\nPer-topic breakdown:")
    for topic_name, events in all_events.items():
        stakeholder_count = sum(len(e.stakeholders) for e in events)
        logger.info(f"  {topic_name}: {len(events)} events, {stakeholder_count} stakeholders")

    logger.info(f"\nResults saved to: {artifacts.run_dir}")
    logger.info(f"  - stage1/all_events.jsonl (combined)")
    logger.info(f"  - stage1/<topic>_events.jsonl (per-topic)")

    return all_events


if __name__ == "__main__":
    asyncio.run(main())
