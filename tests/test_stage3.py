"""
Stage 3 Test Script

Tests significance ranking, TextRank, and timeline generation.
"""

import asyncio
import sys
sys.path.insert(0, ".")

from sunset.config import load_config
from sunset.data import Timeline17Dataset
from sunset.stage1_set import EventExtractor
from sunset.stage2_clustering import (
    GlobalStats, build_global_stats,
    EmbeddingClient,
    build_event_graph,
    cluster_events,
)
from sunset.stage3_timeline import (
    compute_significance,
    rank_clusters,
    textrank_select,
    generate_timeline,
    get_timeline_statistics,
)
from sunset.utils import ArtifactManager, StageLogger


async def prepare_data(config, dataset, num_articles=3):
    """Extract events and cluster them for one topic."""
    logger = StageLogger("test", "prepare", log_format="text")

    # Extract from all topics for global stats
    extractor = EventExtractor(config, max_concurrent=5)
    all_events = {}

    for topic in dataset.topics:
        articles = topic.articles[:num_articles]
        events = await extractor.extract_all(articles, include_stakeholders=True)
        all_events[topic.name] = events
        logger.info(f"Extracted {len(events)} events from {topic.name}")

    # Build global stats
    global_stats = build_global_stats(all_events)
    logger.info(f"Built global stats: {len(global_stats.topics)} topics, {len(global_stats.get_all_stakeholders())} stakeholders")

    # Focus on first topic for clustering
    first_topic = dataset.topics[0].name
    events = all_events[first_topic]

    # Get embeddings
    embed_client = EmbeddingClient(config.embedding)
    texts = [e.summary for e in events]
    embeddings = await embed_client.embed_batch(texts)
    await embed_client.close()

    logger.info(f"Embeddings shape: {embeddings.shape}")

    # Build graph and cluster
    graph = build_event_graph(
        events=events,
        embeddings=embeddings,
        topic=first_topic,
        global_stats=global_stats,
        top_k=min(20, len(events) - 1),
        em_n=config.pipeline.em_n,
        beta=config.pipeline.beta,
    )

    clusters = cluster_events(graph)
    logger.info(f"Created {len(clusters)} clusters")

    return events, embeddings, clusters, global_stats, first_topic


def test_significance(clusters, events, global_stats, topic, beta):
    """Test significance computation."""
    logger = StageLogger("test", "significance", log_format="text")

    for cluster in clusters[:3]:
        sig = compute_significance(cluster, events, global_stats, topic, beta)
        logger.info(
            f"Cluster {cluster.id}: size={len(cluster)}, significance={sig:.4f}",
            data={"event_ids": cluster.event_ids}
        )


def test_textrank(clusters, events, embeddings):
    """Test TextRank representative selection."""
    logger = StageLogger("test", "textrank", log_format="text")

    for cluster in clusters[:3]:
        if len(cluster) == 1:
            rep_id = cluster.event_ids[0]
        else:
            rep_id = textrank_select(cluster, events, embeddings)

        rep_event = events[rep_id]
        logger.info(
            f"Cluster {cluster.id} representative: {rep_event.date} - {rep_event.summary[:60]}...",
            data={"event_id": rep_id}
        )


def test_timeline_generation(clusters, events, embeddings, global_stats, topic, config):
    """Test full timeline generation."""
    logger = StageLogger("test", "timeline", log_format="text")

    timeline = generate_timeline(
        clusters=clusters,
        events=events,
        embeddings=embeddings,
        global_stats=global_stats,
        topic=topic,
        max_entries=config.pipeline.max_timeline_entries,
        beta=config.pipeline.beta,
    )

    logger.info(f"Generated timeline with {len(timeline)} entries")

    # Show timeline entries
    for entry in timeline.entries:
        logger.info(
            f"  {entry.date}: {entry.summary[:60]}...",
            data={"significance": f"{entry.significance:.4f}"}
        )

    # Show statistics
    stats = get_timeline_statistics(timeline)
    logger.info("Timeline statistics", data=stats)

    return timeline


async def main():
    print("=== Stage 3 Test ===\n")

    # Load config and data
    config = load_config("config.yaml")
    dataset = Timeline17Dataset.load("timeline17.pkl")

    # Setup artifacts
    artifacts = ArtifactManager("results")
    artifacts.new_run("test_stage3")

    # Prepare data (Stage 1 + Stage 2)
    print("\n--- Preparing Data ---")
    events, embeddings, clusters, global_stats, topic = await prepare_data(
        config, dataset, num_articles=3
    )

    # Test significance
    print("\n--- Testing Significance ---")
    test_significance(clusters, events, global_stats, topic, config.pipeline.beta)

    # Test TextRank
    print("\n--- Testing TextRank ---")
    test_textrank(clusters, events, embeddings)

    # Test timeline generation
    print("\n--- Testing Timeline Generation ---")
    timeline = test_timeline_generation(
        clusters, events, embeddings, global_stats, topic, config
    )

    # Save results
    artifacts.save_json("stage3/timeline.json", timeline.to_dict())
    artifacts.save_json("stage3/timeline_eval_format.json", timeline.to_evaluation_format())

    print("\n=== Stage 3 Test Complete ===")
    print(f"Results saved to: {artifacts.run_dir}")


if __name__ == "__main__":
    asyncio.run(main())
