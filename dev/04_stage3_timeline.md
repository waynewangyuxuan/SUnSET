# Stage 3: Timeline Generation (Paper Section 3.3)

## Overview

```
Clusters ──► Significance Ranking ──► TextRank ──► Timeline
             (Equation 7)              (select representative)
```

## 3.1 Cluster Stakeholders S_C

### Paper Formula (Equation 6)

```
S_C = ∪_{e∈C} {ς | ς ∈ S_e}
```

The set of all unique stakeholders across all events in cluster C.

### Pseudocode

```
FUNCTION get_cluster_stakeholders(cluster, events) -> Set[str]:
    S_C = set()

    FOR event_id IN cluster.event_ids:
        event = events[event_id]
        FOR ς IN event.stakeholders:
            S_C.add(ς)

    RETURN S_C
```

## 3.2 Cluster Significance

### Paper Formula (Equation 7)

```
Significance(C) = [1 + ln(|C|)] × (Σ_{ς∈S_C} Rel(ς,d) / |S_C|)
```

### Components

1. **Size Factor**: `[1 + ln(|C|)]`
   - |C| = number of events in cluster
   - Rewards larger clusters (logarithmically)
   - |C|=1 → factor=1.0
   - |C|=3 → factor=2.1
   - |C|=10 → factor=3.3

2. **Average Stakeholder Relevance**: `Σ Rel(ς,d) / |S_C|`
   - Sum Rel over unique stakeholders in cluster
   - **NOT** sum over events
   - Divide by number of unique stakeholders

### Pseudocode

```
FUNCTION compute_significance(cluster, events, global_stats, d, β=1.0) -> float:
    # Get cluster stakeholders (Equation 6)
    S_C = get_cluster_stakeholders(cluster, events)

    IF len(S_C) == 0:
        RETURN 0.0

    # Size factor
    size = len(cluster.event_ids)
    size_factor = 1 + log(size)

    # Sum of Rel over unique stakeholders
    rel_sum = 0.0
    FOR ς IN S_C:
        rel_sum += compute_relevance(ς, d, global_stats, β)

    # Average relevance per stakeholder
    avg_rel = rel_sum / len(S_C)

    # Final significance
    significance = size_factor * avg_rel

    RETURN significance
```

### Key Difference from Previous Design

| Aspect | Paper (Correct) | Previous (Wrong) |
|--------|-----------------|------------------|
| Rel scope | Stakeholder-level Rel(ς,d) | Event-level Rel(e) |
| Sum over | Unique stakeholders in S_C | Events in cluster |
| Denominator | \|S_C\| (unique stakeholders) | \|S_C\| (same, but numerator different) |

## 3.3 Rank Clusters

```
FUNCTION rank_clusters(clusters, events, global_stats, d, β=1.0) -> List[RankedCluster]:
    ranked = []

    FOR cluster IN clusters:
        sig = compute_significance(cluster, events, global_stats, d, β)
        ranked.append(RankedCluster(
            cluster = cluster,
            significance = sig
        ))

    # Sort by significance descending
    ranked.sort(key=lambda x: x.significance, reverse=True)

    RETURN ranked
```

## 3.4 TextRank for Representative Selection

### Paper Description (Section 3.3)

> "Events which are measured as significant will then be passed into TextRank to identify important nodes within existing clusters"

TextRank selects the most representative event from each cluster.

### TextRank Algorithm

```
FUNCTION textrank_select(cluster, events, embeddings, damping=0.85, max_iter=100) -> Event:
    event_ids = cluster.event_ids
    n = len(event_ids)

    IF n == 1:
        RETURN events[event_ids[0]]

    # Get embeddings for this cluster
    cluster_embeds = [embeddings[i] for i in event_ids]

    # Build similarity matrix
    sim_matrix = cosine_similarity(cluster_embeds)

    # Normalize rows (stochastic matrix)
    row_sums = sim_matrix.sum(axis=1, keepdims=True)
    row_sums = maximum(row_sums, 1e-10)
    trans_matrix = sim_matrix / row_sums

    # Power iteration
    scores = ones(n) / n

    FOR iter IN range(max_iter):
        new_scores = (1 - damping) / n + damping * trans_matrix.T @ scores

        IF abs(new_scores - scores).sum() < 1e-6:
            BREAK

        scores = new_scores

    # Return highest scoring event
    best_idx = argmax(scores)
    best_event_id = event_ids[best_idx]

    RETURN events[best_event_id]
```

## 3.5 Timeline Generation (TLG)

### Paper Description (Section 3.3)

> "The final set of nodes will subsequently be used in the final Timeline Generation (TLG)"

### Deduplication by Date

Multiple clusters may produce events for the same date. Keep only the most significant:

```
FUNCTION dedupe_by_date(ranked_clusters_with_reps) -> List[TimelineEntry]:
    seen_dates = set()
    entries = []

    FOR cluster IN ranked_clusters_with_reps:  # already sorted by significance
        date = cluster.representative.time

        IF date IN seen_dates:
            CONTINUE

        seen_dates.add(date)
        entries.append(TimelineEntry(
            date = date,
            summary = cluster.representative.event,
            significance = cluster.significance
        ))

    RETURN entries
```

### Determine Timeline Length

Paper follows prior work (Section 4):

> "given a set of temporally labelled news articles that is related to a broad topic... as well as the expected number of dates and the number of sentences to include in each date"

This suggests the expected timeline length is given as input (from dataset).

```
FUNCTION generate_timeline(entries, expected_dates, sentences_per_date=1) -> Timeline:
    # Sort by date
    entries.sort(key=lambda x: x.date)

    # Take top entries up to expected_dates
    entries = entries[:expected_dates]

    RETURN Timeline(entries=entries)
```

## 3.6 Complete Stage 3 Flow

```
FUNCTION stage3_timeline_generation(clusters, events, embeddings, global_stats, d, config) -> Timeline:
    # Step 1: Compute significance for all clusters
    ranked = rank_clusters(clusters, events, global_stats, d, config.beta)

    # Step 2: Select representative event for each cluster
    FOR cluster IN ranked:
        cluster.representative = textrank_select(cluster, events, embeddings)

    # Step 3: Dedupe by date
    entries = dedupe_by_date(ranked)

    # Step 4: Generate timeline
    timeline = generate_timeline(
        entries = entries,
        expected_dates = config.expected_dates
    )

    RETURN timeline
```

## 3.7 Relevance in Timeline Generation (Table 4)

Paper experiments show using Rel in both clustering AND timeline generation:

| Method | AR-1 | AR-2 | Date-F1 |
|--------|------|------|---------|
| TextRank only | 0.117 | 0.040 | 0.368 |
| TextRank + Rel | 0.129 | 0.047 | 0.389 |

The "+ Rel" means using Significance (Equation 7) for cluster ranking, not just cluster size.

## Output Format

```
Timeline:
    entries: [
        TimelineEntry(date="2011-01-25", summary="...", significance=0.85),
        TimelineEntry(date="2011-01-26", summary="...", significance=0.72),
        ...
    ]
```

## Key Implementation Notes

1. **Significance uses stakeholder-level Rel**: Sum over S_C, not events
2. **TextRank**: Standard algorithm on cosine similarity graph within cluster
3. **Date dedup**: Keep most significant cluster's representative per date
4. **Timeline length**: Typically specified by dataset/evaluation setup
