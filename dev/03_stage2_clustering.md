# Stage 2: Event Clustering (Paper Section 3.2)

## Overview

```
SET Triplets ──► Compute Rel(ς,d) ──► Build Graph (W_edge) ──► Clustering
                 for each stakeholder    top-20 neighbors
```

## 2.1 Relevance Score Rel(ς, d)

### Paper Formula (Equation 1)

```
Rel(ς, d) = β · P(ς, d) · R(count(ς_d))
```

Where:
- ς = stakeholder (person/organization)
- d = current topic (e.g., "EgyptianProtest")
- β = hyperparameter (tunable, paper uses 0.1 to 1.0)

**CRITICAL**: Rel is computed at **stakeholder level**, not event level.

## 2.2 Penalty Score P(ς, d)

### Paper Formula (Equation 2)

```
P(ς, d) = (σ_D / x̄) · √|D| · (count(ς_d) / count(ς_D))
         \_________/
              CV (coefficient of variation)
```

### Variables Explained

| Variable | Description | Scope |
|----------|-------------|-------|
| σ_D | Standard deviation of ς counts across all topics | Global (all D) |
| x̄ | Mean of ς counts across all topics | Global (all D) |
| \|D\| | Number of topics | Global |
| count(ς_d) | Occurrences of ς in current topic d | Topic-specific |
| count(ς_D) | Total occurrences of ς across all topics | Global |

### Paper's Intuition (Table 1)

- **High CV** = stakeholder appears in few topics (rare) → less penalty
- **Low CV** = stakeholder appears in many topics (common) → more penalty
- **High count(ς_d)/count(ς_D)** = concentrated in this topic → less penalty

### Pseudocode: Compute Penalty

```
FUNCTION compute_penalty(ς, d, global_stats) -> float:
    # Get counts across ALL topics
    counts_per_topic = []
    FOR topic IN all_topics:
        counts_per_topic.append(global_stats.count[ς][topic])

    # Coefficient of variation
    σ_D = std(counts_per_topic)
    x̄ = mean(counts_per_topic)

    IF x̄ == 0:
        RETURN 0  # stakeholder doesn't exist

    CV = σ_D / x̄

    # Topic-specific ratio
    count_ς_d = global_stats.count[ς][d]
    count_ς_D = sum(counts_per_topic)

    IF count_ς_D == 0:
        RETURN 0

    topic_ratio = count_ς_d / count_ς_D

    # Final penalty (normalized by √|D|)
    P = CV * sqrt(len(all_topics)) * topic_ratio

    # Boundary: 0 ≤ P ≤ 1 (proven in Appendix C)
    P = min(max(P, 0), 1)

    RETURN P
```

### Paper Proof (Appendix C)

The paper proves: **0 ≤ P ≤ 1**

Key insight: CV is bounded by √|D| when counts are non-negative integers.

## 2.3 Reward Score R(x)

### Paper Formula (Equation 3)

```
R(x) = tanh(x/10) = (e^(x/10) - e^(-x/10)) / (e^(x/10) + e^(-x/10))
```

Where x = count(ς_d) = stakeholder count in current topic

### Properties (Appendix D)

- R(0) = 0
- R(10) ≈ 0.76
- R(21) ≈ 0.97 (saturates)
- R(∞) → 1

### Pseudocode

```
FUNCTION compute_reward(count_ς_d) -> float:
    x = count_ς_d
    RETURN tanh(x / 10)
    # Equivalent: (exp(x/10) - exp(-x/10)) / (exp(x/10) + exp(-x/10))
```

## 2.4 Complete Relevance Computation

```
FUNCTION compute_relevance(ς, d, global_stats, β=1.0) -> float:
    P = compute_penalty(ς, d, global_stats)
    R = compute_reward(global_stats.count[ς][d])

    Rel = β * P * R

    RETURN Rel
```

### Building Global Stats

```
FUNCTION build_global_stats(all_topics_data) -> GlobalStats:
    # Count stakeholder occurrences per topic
    count = defaultdict(lambda: defaultdict(int))

    FOR topic_name, topic_data IN all_topics_data.items():
        FOR event IN topic_data.events:
            FOR ς IN event.stakeholders:
                count[ς][topic_name] += 1

    RETURN GlobalStats(count=count, topics=list(all_topics_data.keys()))
```

**IMPORTANT**: Global stats require data from ALL topics, not just the current one.

## 2.5 Edge Weight W_edge

### Paper Formula (Equation 4)

```
W_edge(e_i, e_j) = BoolEM_n(S_ei, S_ej) × [Σ_{ς_i=ς_j} Rel(ς,d) + cos(e_i, e_j)]
```

### Components

1. **BoolEM_n**: Boolean Exact Matching
   ```
   BoolEM_n(S_ei, S_ej) = 1 if |S_ei ∩ S_ej| ≥ n, else 0
   ```
   - n = minimum required shared stakeholders (paper experiments: n=0,1,2)
   - n=1 typically works best

2. **Σ Rel over shared stakeholders**:
   ```
   FOR ς IN (S_ei ∩ S_ej):
       sum += Rel(ς, d)
   ```

3. **Cosine similarity**:
   ```
   cos(e_i, e_j) = dot(embed(e_i), embed(e_j)) / (norm(e_i) * norm(e_j))
   ```
   - Paper uses GTE-Modernbert-Base for embedding

### Pseudocode

```
FUNCTION compute_edge_weight(e_i, e_j, d, global_stats, embeddings, n=1, β=1.0) -> float:
    S_ei = set(e_i.stakeholders)
    S_ej = set(e_j.stakeholders)

    shared = S_ei & S_ej

    # BoolEM check
    IF len(shared) < n:
        RETURN 0  # no edge

    # Sum of Rel for shared stakeholders
    rel_sum = 0
    FOR ς IN shared:
        rel_sum += compute_relevance(ς, d, global_stats, β)

    # Cosine similarity
    cos_sim = cosine_similarity(embeddings[e_i.id], embeddings[e_j.id])

    W = rel_sum + cos_sim

    RETURN W
```

## 2.6 Graph Construction: Top-20 Neighbors

### Paper Description (Section 3.2)

> "the clustering process uses the encoded event summary from a General Text Embedding (GTE) Model to obtain query-based cosine similarity scores combined with relevancy scores to generate the **top 20 similar events for every node**"

### Pseudocode

```
FUNCTION build_event_graph(events, embeddings, d, global_stats, k=20, n=1, β=1.0) -> Graph:
    N = len(events)
    adjacency = {}

    FOR i IN range(N):
        # Compute edge weights to all other events
        weights = []
        FOR j IN range(N):
            IF i == j:
                CONTINUE
            w = compute_edge_weight(events[i], events[j], d, global_stats, embeddings, n, β)
            IF w > 0:  # only consider positive edges
                weights.append((j, w))

        # Keep top-k neighbors
        weights.sort(key=lambda x: x[1], reverse=True)
        top_k = weights[:k]

        adjacency[i] = top_k

    RETURN Graph(nodes=range(N), adjacency=adjacency)
```

### Key Points

1. **NOT date-constrained**: Events from different dates CAN connect
2. **NOT threshold-based**: Always take top-20, regardless of absolute weight
3. **Directed graph**: Node i → top-20 neighbors (can be made undirected by union)

## 2.7 Clustering Algorithm

### Paper doesn't specify exact algorithm

The paper uses the constructed graph for clustering but doesn't explicitly state the algorithm. Common choices:

1. **Connected Components** (simplest)
2. **Louvain** (community detection)
3. **Label Propagation**

### Pseudocode: Connected Components

```
FUNCTION cluster_events(graph) -> List[Cluster]:
    # Make graph undirected
    undirected = make_undirected(graph)

    # Find connected components
    visited = set()
    clusters = []

    FOR node IN graph.nodes:
        IF node IN visited:
            CONTINUE

        # BFS/DFS to find component
        component = []
        queue = [node]

        WHILE queue:
            current = queue.pop()
            IF current IN visited:
                CONTINUE
            visited.add(current)
            component.append(current)

            FOR neighbor, weight IN undirected[current]:
                IF neighbor NOT IN visited:
                    queue.append(neighbor)

        clusters.append(Cluster(event_ids=component))

    RETURN clusters
```

## 2.8 Complete Stage 2 Flow

```
FUNCTION stage2_event_clustering(events, topic_name, global_stats, config) -> List[Cluster]:
    # Step 1: Embed all events
    texts = [e.event for e in events]
    embeddings = embed_model.encode(texts)

    # Step 2: Build graph with top-20 neighbors
    graph = build_event_graph(
        events = events,
        embeddings = embeddings,
        d = topic_name,
        global_stats = global_stats,
        k = 20,
        n = config.em_n,  # BoolEM_n threshold
        β = config.beta
    )

    # Step 3: Cluster
    clusters = cluster_events(graph)

    RETURN clusters
```

## Hyperparameters (from Paper Experiments)

| Parameter | Range Tested | Best Value |
|-----------|-------------|------------|
| β | 0.0 - 1.0 | 1.0 (T17), 0.9-1.0 (Crisis) |
| EM_n | 0, 1, 2 | 1 typically best |
| k (top-k) | - | 20 (fixed) |

## Output Format

```
List[Cluster]:
    Cluster(id=0, event_ids=[0, 5, 12, 23])
    Cluster(id=1, event_ids=[1, 3, 7])
    Cluster(id=2, event_ids=[2])  # singleton
    ...
```

## Critical Implementation Notes

1. **Global Stats**: Penalty P requires stats across ALL topics in dataset, not just current topic
2. **Top-20**: Always 20 neighbors, not a similarity threshold
3. **BoolEM_n**: Hard constraint - if shared stakeholders < n, edge weight = 0
4. **No date filtering**: Events across dates can cluster together
