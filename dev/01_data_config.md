# Data & Configuration (Paper Section 4)

## Package Management

Using `uv` for Python package management:

```bash
# Install dependencies
uv sync

# Run with uv
uv run python main.py
```

## Datasets

### Timeline17 (T17)

From paper:
> "T17 contains 19 timelines compiled from varying sources of online news sites, spanning 9 major topics from 2005-2013, each with 1-5 ground truth timelines"

| Property | Value |
|----------|-------|
| Topics | 9 |
| Total Timelines | 19 |
| Time Range | 2005-2013 |
| Ground Truth per Topic | 1-5 |

**Topics in T17**:
- EgyptianProtest
- Finan (Financial Crisis)
- MJ (Michael Jackson)
- SyrianCrisis
- IraqWar
- H1N1
- bpoil (BP Oil Spill)
- haiti
- LibyaWar

### Crisis Dataset

From paper:
> "Crisis has 22 annotated timelines covering 4 critical crisis events, each containing 4-7 ground truth timelines"

| Property | Value |
|----------|-------|
| Topics | 4 |
| Total Timelines | 22 |
| Ground Truth per Topic | 4-7 |

## Data Format (timeline17.pkl)

```python
{
    "topic_name": {
        "articles": [
            {
                "pub_date": "YYYY-MM-DD",
                "sentences": [
                    {"text": "...", "mentioned_dates": ["YYYY-MM-DD"]}
                ]
            }
        ],
        "gold_timelines": {
            "tl1": {
                "YYYY-MM-DD": ["event summary 1", "event summary 2"],
                ...
            },
            "tl2": {...}  # multiple annotators possible
        }
    }
}
```

## Data Structures

### Input: Article
```
Article:
    pub_date: str           # Publication date (YYYY-MM-DD)
    sentences: List[Sentence]

Sentence:
    text: str
    mentioned_dates: List[str]  # Dates mentioned in sentence
```

### Intermediate: SET Triplet
```
SET:
    stakeholders: List[str]  # Max 5, normalized entity names
    event: str               # Event summary
    time: str                # Event date (YYYY-MM-DD)
```

### Intermediate: Event (for clustering)
```
Event:
    id: int                  # Unique identifier
    stakeholders: List[str]
    event: str               # Summary text
    time: str                # Date
```

### Intermediate: Cluster
```
Cluster:
    id: int
    event_ids: List[int]
```

### Output: Timeline
```
Timeline:
    entries: List[TimelineEntry]

TimelineEntry:
    date: str
    summary: str
    significance: float  # Optional, for debugging
```

## Global Statistics (Required for Rel)

```
GlobalStats:
    # count[stakeholder][topic] = number of occurrences
    count: Dict[str, Dict[str, int]]

    # List of all topic names
    topics: List[str]
```

**CRITICAL**: GlobalStats must be computed across ALL topics before processing any single topic.

## Configuration

### Paper Settings (Section 4)

| Component | Paper Value |
|-----------|-------------|
| LLM | Qwen2.5-72B-Instruct |
| Embedding | GTE-Modernbert-Base |
| Deployment | VLLM |

### Our Settings

```python
Config:
    # LLM (SET Extraction) - Option 1: Local VLLM
    vllm_url: str = "http://ds-serv11.ucsd.edu:18000/v1"
    vllm_model: str = "Qwen/Qwen3-32B"

    # LLM (SET Extraction) - Option 2: OpenAI GPT-4
    openai_api_key: str = None  # Set via environment or config
    openai_model: str = "gpt-4o"

    # Which LLM to use: "vllm" or "openai"
    llm_provider: str = "openai"

    # Embedding (Event Clustering)
    embed_url: str = "http://ds-serv11.ucsd.edu:18003/v1"
    embed_model: str = "qwen3-embed-0.6b"

    # Proxy (Wikidata access)
    proxy_url: str = "https://proxy.frederickpi.com/proxy/random/normal"

    # Hyperparameters (from paper experiments)
    beta: float = 1.0       # Relevance scaling (Table 9-12)
    em_n: int = 1           # BoolEM_n threshold (Table 6)
    top_k: int = 20         # Top-k neighbors for graph

    # Stakeholder constraints
    max_stakeholders_per_event: int = 5
```

### LLM Provider Selection

```
FUNCTION get_llm_client(config) -> LLMClient:
    IF config.llm_provider == "openai":
        RETURN OpenAI(api_key=config.openai_api_key)
    ELSE:
        RETURN OpenAI(base_url=config.vllm_url, api_key="dummy")
```

## Pseudocode: Data Loading

```
FUNCTION load_dataset(path) -> Dict:
    data = pickle.load(path)
    RETURN data

FUNCTION get_articles(data, topic) -> List[Article]:
    RETURN data[topic]["articles"]

FUNCTION get_gold_timelines(data, topic) -> Dict[str, Dict]:
    RETURN data[topic]["gold_timelines"]

FUNCTION article_to_text(article) -> str:
    text = ""
    FOR sentence IN article["sentences"]:
        text += sentence["text"] + " "
    RETURN text.strip()
```

## Pseudocode: Build Global Stats

**Must be done before running pipeline on any topic.**

```
FUNCTION build_global_stats(dataset) -> GlobalStats:
    count = defaultdict(lambda: defaultdict(int))

    # First pass: run SET extraction on ALL topics
    FOR topic_name, topic_data IN dataset.items():
        FOR article IN topic_data["articles"]:
            events = extract_events(article)
            FOR event IN events:
                stakeholders = extract_stakeholders(event, article)
                FOR ς IN stakeholders:
                    count[ς][topic_name] += 1

    RETURN GlobalStats(
        count = count,
        topics = list(dataset.keys())
    )
```

## Pipeline Execution Order

```
1. Load dataset (all topics)
2. Run Stage 1 (SET extraction) on ALL topics
3. Build GlobalStats from all extracted stakeholders
4. For each topic:
   a. Run Stage 2 (Clustering) with GlobalStats
   b. Run Stage 3 (Timeline Generation)
   c. Evaluate against gold timelines
5. Report average metrics
```

## Key Implementation Notes

1. **Global Stats First**: Must process all topics to compute P(ς,d) correctly
2. **Cross-topic Information**: Penalty P uses statistics from ALL topics
3. **Multiple Gold Timelines**: Average evaluation across all available ground truths
4. **Wikidata Caching**: Cache resolved entities to avoid repeated API calls
