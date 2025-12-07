"""
Stage 1 Test Script

Tests event extraction and coreference resolution on a small sample.
"""

import asyncio
import sys
sys.path.insert(0, ".")

from sunset.config import load_config
from sunset.data import Timeline17Dataset, article_to_text
from sunset.stage1_set import EventExtractor, CoreferenceResolver
from sunset.utils import ArtifactManager, StageLogger


async def test_event_extraction(config, dataset, artifacts):
    """Test event extraction on a few articles."""
    logger = StageLogger("test", "event_extraction", log_format="text")

    # Get a few articles from first topic
    topic = dataset.topics[0]
    sample_articles = topic.articles[:3]

    logger.info(f"Testing event extraction on {len(sample_articles)} articles from {topic.name}")

    extractor = EventExtractor(config, max_concurrent=5)

    # Extract events (without stakeholders first for speed)
    events = await extractor.extract_all(sample_articles, include_stakeholders=False)

    logger.info(f"Extracted {len(events)} events")

    # Print sample events
    for event in events[:5]:
        logger.info(
            f"Event: {event.date} - {event.summary[:80]}...",
            data={"id": event.id}
        )

    # Save results
    artifacts.save_jsonl("stage1/test_events.jsonl", [e.to_dict() for e in events])

    # Get metrics
    metrics = extractor.get_metrics_report()
    artifacts.save_json("stage1/test_extraction_metrics.json", metrics)
    logger.info("Extraction metrics", data=metrics)

    return events, sample_articles


async def test_stakeholder_extraction(config, events, articles, artifacts):
    """Test stakeholder extraction for a few events."""
    logger = StageLogger("test", "stakeholder_extraction", log_format="text")

    extractor = EventExtractor(config, max_concurrent=5)

    # Extract stakeholders for first 5 events
    for event in events[:5]:
        article = articles[event.source_article_index]
        stakeholders = await extractor.extract_stakeholders_for_event(event, article)
        event.stakeholders = stakeholders
        logger.info(
            f"Event {event.id}: {len(stakeholders)} stakeholders",
            data={"stakeholders": stakeholders}
        )

    # Save updated events
    artifacts.save_jsonl("stage1/test_events_with_stakeholders.jsonl", [e.to_dict() for e in events[:5]])

    return events


async def test_coreference(config, events, artifacts):
    """Test coreference resolution."""
    logger = StageLogger("test", "coreference", log_format="text")

    # Collect stakeholders
    stakeholders = set()
    for event in events[:5]:
        stakeholders.update(event.stakeholders)

    logger.info(f"Testing coreference for {len(stakeholders)} stakeholders")

    if not stakeholders:
        logger.warning("No stakeholders to resolve")
        return

    resolver = CoreferenceResolver(config.wikidata)

    # Resolve a few stakeholders
    for name in list(stakeholders)[:3]:
        mapping = await resolver.resolve(name)
        logger.info(
            f"Resolved: {name} -> {mapping.canonical_name}",
            data={"method": mapping.resolution_method, "qid": mapping.wikidata_id}
        )

    # Save cache
    resolver.cache.save()
    logger.info(f"Cache saved with {len(resolver.cache)} entries")


async def main():
    print("=== Stage 1 Test ===\n")

    # Load config and data
    config = load_config("config.yaml")
    dataset = Timeline17Dataset.load("timeline17.pkl")

    # Setup artifacts
    artifacts = ArtifactManager("results")
    artifacts.new_run("test_stage1")

    # Save config
    artifacts.save_json("config.json", config.to_dict())

    # Test event extraction
    print("\n--- Testing Event Extraction ---")
    events, articles = await test_event_extraction(config, dataset, artifacts)

    # Test stakeholder extraction
    print("\n--- Testing Stakeholder Extraction ---")
    events = await test_stakeholder_extraction(config, events, articles, artifacts)

    # Test coreference
    print("\n--- Testing Coreference Resolution ---")
    await test_coreference(config, events, artifacts)

    print("\n=== Stage 1 Test Complete ===")
    print(f"Results saved to: {artifacts.run_dir}")


if __name__ == "__main__":
    asyncio.run(main())
