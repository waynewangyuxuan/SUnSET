"""
Stage 2 Test Script

Tests global stats, relevance computation, embedding, graph construction, and clustering.
"""

import asyncio
import sys
sys.path.insert(0, ".")

from sunset.config import load_config
from sunset.data import Timeline17Dataset
from sunset.stage1_set import EventExtractor
from sunset.stage2_clustering import (
    GlobalStats, build_global_stats,
    compute_relevance, compute_penalty, compute_reward,
    EmbeddingClient,
    build_event_graph,
    cluster_events, get_cluster_statistics,
)
from sunset.utils import ArtifactManager, StageLogger


async def extract_events_for_topics(config, dataset, num_articles_per_topic=3):
    """Extract events from a few articles per topic."""
    logger = StageLogger("test", "extraction", log_format="text")
    extractor = EventExtractor(config, max_concurrent=5)

    all_events = {}

    for topic in dataset.topics:
        articles = topic.articles[:num_articles_per_topic]
        events = await extractor.extract_all(articles, include_stakeholders=True)
        all_events[topic.name] = events
        logger.info(f"Extracted {len(events)} events from {topic.name}")

    return all_events


def test_global_stats(all_events):
    """Test global statistics computation."""
    logger = StageLogger("test", "global_stats", log_format="text")

    global_stats = build_global_stats(all_events)

    logger.info(f"Built global stats for {len(global_stats.topics)} topics")
    logger.info(f"Total unique stakeholders: {len(global_stats.get_all_stakeholders())}")

    # Show sample stakeholder stats
    stakeholders = list(global_stats.get_all_stakeholders())[:5]
    for s in stakeholders:
        counts = global_stats.get_stakeholder_counts(s)
        total = global_stats.get_total_count(s)
        logger.info(f"  {s}: total={total}, distribution={counts[:3]}...")

    return global_stats


def test_relevance(global_stats, topic):
    """Test relevance computation."""
    logger = StageLogger("test", "relevance", log_format="text")

    stakeholders = list(global_stats.get_all_stakeholders())[:10]

    logger.info(f"Testing relevance for {len(stakeholders)} stakeholders in {topic}")

    for s in stakeholders:
        P = compute_penalty(s, topic, global_stats)
        count = global_stats.get_topic_count(s, topic)
        R = compute_reward(count)
        Rel = compute_relevance(s, topic, global_stats, beta=1.0)

        logger.info(
            f"  {s}: P={P:.4f}, R={R:.4f}, Rel={Rel:.4f}",
            data={"count": count}
        )


async def test_embedding(config, events):
    """Test embedding computation."""
    logger = StageLogger("test", "embedding", log_format="text")

    client = EmbeddingClient(config.embedding)

    texts = [e.summary for e in events[:5]]
    logger.info(f"Testing embedding for {len(texts)} event summaries")

    try:
        embeddings = await client.embed_batch(texts)
        logger.info(f"Embeddings shape: {embeddings.shape}")

        # Show sample similarity
        if len(embeddings) >= 2:
            from sunset.stage2_clustering import cosine_similarity
            sim = cosine_similarity(embeddings[0], embeddings[1])
            logger.info(f"Similarity between event 0 and 1: {sim:.4f}")

        await client.close()
        return embeddings

    except Exception as e:
        logger.error(f"Embedding failed: {e}")
        await client.close()
        return None


async def test_graph_and_clustering(config, events, global_stats, topic):
    """Test graph construction and clustering."""
    logger = StageLogger("test", "graph_clustering", log_format="text")

    if len(events) < 2:
        logger.warning("Not enough events for clustering")
        return None

    # Get embeddings
    client = EmbeddingClient(config.embedding)
    texts = [e.summary for e in events]

    try:
        embeddings = await client.embed_batch(texts)
    except Exception as e:
        logger.error(f"Embedding failed: {e}")
        await client.close()
        return None

    await client.close()

    # Build graph
    logger.info(f"Building graph for {len(events)} events")
    graph = build_event_graph(
        events=events,
        embeddings=embeddings,
        topic=topic,
        global_stats=global_stats,
        top_k=min(20, len(events) - 1),
        em_n=config.pipeline.em_n,
        beta=config.pipeline.beta,
    )

    logger.info(
        f"Graph: {graph.num_nodes} nodes",
        data={"adjacency_sample": {k: len(v) for k, v in list(graph.adjacency.items())[:3]}}
    )

    # Cluster
    clusters = cluster_events(graph)
    stats = get_cluster_statistics(clusters)

    logger.info(
        f"Clustering result: {stats['num_clusters']} clusters",
        data=stats
    )

    return clusters


async def main():
    print("=== Stage 2 Test ===\n")

    # Load config and data
    config = load_config("config.yaml")
    dataset = Timeline17Dataset.load("timeline17.pkl")

    # Setup artifacts
    artifacts = ArtifactManager("results")
    artifacts.new_run("test_stage2")

    # Step 1: Extract events from all topics (small sample)
    print("\n--- Extracting Events from All Topics ---")
    all_events = await extract_events_for_topics(config, dataset, num_articles_per_topic=2)

    total_events = sum(len(events) for events in all_events.values())
    print(f"Total events extracted: {total_events}")

    # Step 2: Build global stats
    print("\n--- Building Global Statistics ---")
    global_stats = test_global_stats(all_events)
    artifacts.save_json("stage2/global_stats.json", global_stats.to_dict())

    # Step 3: Test relevance computation
    print("\n--- Testing Relevance Computation ---")
    first_topic = list(all_events.keys())[0]
    test_relevance(global_stats, first_topic)

    # Step 4: Test embedding
    print("\n--- Testing Embedding ---")
    first_topic_events = all_events[first_topic]
    embeddings = await test_embedding(config, first_topic_events)

    # Step 5: Test graph and clustering
    print("\n--- Testing Graph and Clustering ---")
    clusters = await test_graph_and_clustering(
        config, first_topic_events, global_stats, first_topic
    )

    if clusters:
        artifacts.save_jsonl(
            "stage2/test_clusters.jsonl",
            [c.to_dict() for c in clusters]
        )

    print("\n=== Stage 2 Test Complete ===")
    print(f"Results saved to: {artifacts.run_dir}")


if __name__ == "__main__":
    asyncio.run(main())
