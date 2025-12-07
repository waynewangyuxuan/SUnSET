# SUnSET Pipeline Overview (100% Paper Faithful)

## Paper Reference
- **Title**: SUnSET: Synergistic Understanding of Stakeholder, Events and Time
- **arXiv**: 2507.21903v2

## Architecture (Figure 2 in Paper)

```
Articles ──► SET Generation ──► Coreference ──► Event Clustering ──► Timeline Generation
                │                    │                 │                      │
                ▼                    ▼                 ▼                      ▼
         {date: event}          Wikidata KG      Graph + Rel(ς,d)      Significance + TextRank
         + Stakeholders          resolve            top-20 neighbors      Date-Event Pairs
```

## Core Innovation: Stakeholder-Level Relevance

Paper's key insight (Table 1):

| Stakeholder Type | Rarity Across Topics | Reoccurrence in Topic | Relevance |
|------------------|---------------------|----------------------|-----------|
| Significant      | Rare                | High                 | HIGH      |
| Normal           | Rare                | Low                  | Medium    |
| Normal           | Common              | High                 | Medium    |
| Irrelevant       | Common              | Low                  | LOW       |

## Key Formulas (Direct from Paper)

### Relevance (Equation 1)
```
Rel(ς, d) = β · P(ς, d) · R(count(ς_d))
```
- ς = stakeholder
- d = current topic
- β = hyperparameter (tunable, default 1.0)

### Penalty P(ς, d) (Equation 2)
```
P(ς, d) = CV · √|D| · (count(ς_d) / count(ς_D))

where CV = σ_D / x̄  (coefficient of variation)
```
- σ_D = std dev of stakeholder counts across ALL topics D
- x̄ = mean of stakeholder counts across ALL topics D
- |D| = number of topics
- count(ς_d) = occurrences in current topic d
- count(ς_D) = total occurrences across all topics

**Boundary**: 0 ≤ P ≤ 1 (proven in Appendix C)

### Reward R(x) (Equation 3)
```
R(x) = tanh(x/10) = (e^(x/10) - e^(-x/10)) / (e^(x/10) + e^(-x/10))
```
- x = count(ς_d) = stakeholder count in current topic
- Saturates at ~1.0 when x ≥ 21

### Edge Weight W_edge (Equation 4)
```
W_edge(e_i, e_j) = BoolEM_n(S_ei, S_ej) × [Σ_{ς_i=ς_j} Rel(ς,d) + cos(e_i, e_j)]
```
- BoolEM_n = 1 if |S_ei ∩ S_ej| ≥ n, else 0
- Sum Rel over shared stakeholders
- Add cosine similarity of event embeddings

### Graph Construction
- Each node connects to **top-20** highest W_edge neighbors
- **NOT** date-constrained
- **NOT** threshold-based

### Cluster Significance (Equation 7)
```
Significance(C) = [1 + ln(|C|)] × (Σ_{ς∈S_C} Rel(ς,d) / |S_C|)
```
- S_C = ∪_{e∈C} {ς | ς ∈ S_e} (Equation 6)
- Sum over unique stakeholders in cluster, NOT events

## Pipeline Stages

| Stage | Paper Section | Key Operation |
|-------|---------------|---------------|
| 1 | 3.1 | SET Generation: LLM extracts events + stakeholders |
| 1.5 | 3.1 + Appendix B | Coreference: Wikidata KG resolution |
| 2 | 3.2 | Event Clustering: Rel(ς,d) + BoolEM_n + top-20 graph |
| 3 | 3.3 | Timeline Generation: Significance + TextRank |

## Experimental Setup (Section 4)

| Component | Paper Setting |
|-----------|---------------|
| LLM | Qwen2.5-72B-Instruct |
| Embedding | GTE-Modernbert-Base |
| Datasets | Timeline17 (T17), Crisis |
| Best β | 1.0 for T17, 0.9-1.0 for Crisis |
| Best EM_n | n=1 typically sufficient |

## Our Setup

| Component | Our Setting |
|-----------|-------------|
| LLM | Qwen/Qwen3-32B |
| Embedding | qwen3-embed-0.6b |
| Dataset | Timeline17 only (for now) |
