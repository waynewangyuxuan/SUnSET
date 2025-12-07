# Evaluation (Paper Section 4)

## Metrics (from Paper)

> "We incorporate a part of the Tilse framework and use three main scores to analyse the performance of the TLS task"

| Metric | Description |
|--------|-------------|
| AR-1 | Alignment-based ROUGE-1 F1-score (unigram overlap) |
| AR-2 | Alignment-based ROUGE-2 F1-score (bigram overlap) |
| Date-F1 | Date matching F1-score |

## Tilse Framework

We use [Tilse](https://github.com/complementizer/news-tls) for AR-1/AR-2 computation.

### Installation

```bash
pip install tilse
# or with uv:
uv add tilse
```

### Setup (Required)

Tilse requires pyrouge WordNet database. Run once:

```bash
cd /path/to/site-packages/pyrouge/tools/ROUGE-1.5.5/data
rm -f WordNet-2.0.exc.db
cd WordNet-2.0-Exceptions
rm -f WordNet-2.0.exc.db
./buildExeptionDB.pl . exc WordNet-2.0.exc.db
cd ..
ln -sf WordNet-2.0-Exceptions/WordNet-2.0.exc.db WordNet-2.0.exc.db
```

### Using Tilse for Evaluation

```python
from datetime import date
from tilse.data.timelines import Timeline, GroundTruth
from tilse.evaluation.rouge import TimelineRougeEvaluator

def evaluate_with_tilse(pred_entries, gold_timeline_dict):
    """
    Args:
        pred_entries: List[TimelineEntry] with date and summary
        gold_timeline_dict: Dict[str, List[str]] {date: [summaries]}

    Returns:
        (ar1_f1, ar2_f1)
    """
    # Convert to tilse format
    pred_dict = {}
    for entry in pred_entries:
        d = date.fromisoformat(entry.date)
        pred_dict[d] = [entry.summary]

    gold_dict = {}
    for date_str, summaries in gold_timeline_dict.items():
        d = date.fromisoformat(date_str)
        gold_dict[d] = summaries

    pred_timeline = Timeline(pred_dict)
    gold_timeline = Timeline(gold_dict)
    ground_truth = GroundTruth([gold_timeline])

    # Evaluate
    evaluator = TimelineRougeEvaluator(measures=["rouge_1", "rouge_2"])
    scores = evaluator.evaluate_concat(pred_timeline, ground_truth)

    ar1 = scores["rouge_1"]["f_score"]
    ar2 = scores["rouge_2"]["f_score"]

    return ar1, ar2
```

### Tested Output

```
Predicted dates: ['2011-01-25', '2011-01-26', '2011-01-28']
Gold dates:      ['2011-01-25', '2011-01-26', '2011-01-27']

rouge_1: precision=0.3846, recall=0.3571, f1=0.3704
rouge_2: precision=0.0000, recall=0.0000, f1=0.0000
```

## Date-F1

### Definition

Standard F1 score for date matching between predicted and gold timelines.

```
Precision = |pred_dates ∩ gold_dates| / |pred_dates|
Recall    = |pred_dates ∩ gold_dates| / |gold_dates|
Date-F1   = 2 × Precision × Recall / (Precision + Recall)
```

### Pseudocode

```
FUNCTION compute_date_f1(pred_timeline, gold_timeline) -> float:
    pred_dates = set(entry.date FOR entry IN pred_timeline.entries)
    gold_dates = set(gold_timeline.keys())

    IF not pred_dates OR not gold_dates:
        RETURN 0.0

    tp = len(pred_dates & gold_dates)

    precision = tp / len(pred_dates)
    recall = tp / len(gold_dates)

    IF precision + recall == 0:
        RETURN 0.0

    f1 = 2 * precision * recall / (precision + recall)
    RETURN f1
```

## Alignment-based ROUGE (AR)

### Paper Description (Section 4)

> "We use an Alignment-based ROUGE-1 F1-score (AR-1) to evaluate the semantic distance of unigram overlaps between generated timelines and the provided ground truth. We also used the corresponding bigram overlaps and scored the Alignment-based ROUGE-2 F1-score (AR-2)."

### Algorithm

Alignment-based ROUGE aligns events by date first, then computes ROUGE:

```
FOR each gold_date IN gold_timeline:
    gold_events = gold_timeline[gold_date]  # may have multiple summaries

    pred_event = find_pred_by_date(pred_timeline, gold_date)

    FOR gold_event IN gold_events:
        IF pred_event exists:
            score = ROUGE(pred_event.summary, gold_event)
        ELSE:
            score = 0
        scores.append(score)

RETURN mean(scores)
```

### ROUGE-N Implementation

```
FUNCTION rouge_n(candidate, reference, n) -> float:
    # Tokenize and lowercase
    cand_tokens = tokenize(candidate.lower())
    ref_tokens = tokenize(reference.lower())

    # Extract n-grams
    cand_ngrams = extract_ngrams(cand_tokens, n)
    ref_ngrams = extract_ngrams(ref_tokens, n)

    # Count
    cand_counts = Counter(cand_ngrams)
    ref_counts = Counter(ref_ngrams)

    # Overlap
    overlap = 0
    FOR ngram, count IN cand_counts.items():
        overlap += min(count, ref_counts.get(ngram, 0))

    # Precision & Recall
    precision = overlap / sum(cand_counts.values()) IF cand_counts ELSE 0
    recall = overlap / sum(ref_counts.values()) IF ref_counts ELSE 0

    # F1
    IF precision + recall == 0:
        RETURN 0.0

    f1 = 2 * precision * recall / (precision + recall)
    RETURN f1
```

### Complete AR Computation

```
FUNCTION compute_alignment_rouge(pred_timeline, gold_timeline, n=1) -> float:
    scores = []

    FOR gold_date, gold_events IN gold_timeline.items():
        # Find prediction for this date
        pred_event = None
        FOR entry IN pred_timeline.entries:
            IF entry.date == gold_date:
                pred_event = entry
                BREAK

        pred_text = pred_event.summary IF pred_event ELSE ""

        # Score against each gold event
        FOR gold_text IN gold_events:
            IF not pred_text:
                scores.append(0.0)
            ELSE:
                score = rouge_n(pred_text, gold_text, n)
                scores.append(score)

    RETURN mean(scores) IF scores ELSE 0.0
```

## Complete Evaluation

```
DATACLASS EvalResult:
    ar1: float
    ar2: float
    date_f1: float


FUNCTION evaluate(pred_timeline, gold_timeline) -> EvalResult:
    ar1 = compute_alignment_rouge(pred_timeline, gold_timeline, n=1)
    ar2 = compute_alignment_rouge(pred_timeline, gold_timeline, n=2)
    date_f1 = compute_date_f1(pred_timeline, gold_timeline)

    RETURN EvalResult(ar1=ar1, ar2=ar2, date_f1=date_f1)
```

## Multiple Gold Timelines

Some topics have multiple gold timelines (different annotators). Paper averages across all:

```
FUNCTION evaluate_multi_gold(pred_timeline, gold_timelines) -> EvalResult:
    all_ar1 = []
    all_ar2 = []
    all_date_f1 = []

    FOR timeline_name, gold_timeline IN gold_timelines.items():
        result = evaluate(pred_timeline, gold_timeline)
        all_ar1.append(result.ar1)
        all_ar2.append(result.ar2)
        all_date_f1.append(result.date_f1)

    RETURN EvalResult(
        ar1 = mean(all_ar1),
        ar2 = mean(all_ar2),
        date_f1 = mean(all_date_f1)
    )
```

## Evaluation Across All Topics

```
FUNCTION evaluate_all(pipeline, dataset) -> Dict[str, EvalResult]:
    results = {}

    FOR topic_name, topic_data IN dataset.items():
        pred_timeline = pipeline.run(topic_data)
        gold_timelines = topic_data["gold_timelines"]

        result = evaluate_multi_gold(pred_timeline, gold_timelines)
        results[topic_name] = result

    # Macro average
    avg = EvalResult(
        ar1 = mean(r.ar1 FOR r IN results.values()),
        ar2 = mean(r.ar2 FOR r IN results.values()),
        date_f1 = mean(r.date_f1 FOR r IN results.values())
    )
    results["AVERAGE"] = avg

    RETURN results
```

## Paper Results (Table 2)

### Crisis Dataset

| Method | LLM | AR-1 | AR-2 | Date-F1 |
|--------|-----|------|------|---------|
| CHRONOS | Qwen72B | 0.108 | 0.045 | 0.323 |
| LLM-TLS | Llama13B | 0.112 | 0.032 | 0.329 |
| LLM-TLS | Qwen72B | 0.111 | 0.036 | 0.326 |
| **SUnSET** | Qwen72B | **0.129** | **0.047** | **0.389** |
| SUnSET | GPT-4o | 0.107 | 0.036 | 0.381 |

### Timeline17 Dataset

| Method | LLM | AR-1 | AR-2 | Date-F1 |
|--------|-----|------|------|---------|
| CHRONOS | Qwen72B | 0.116 | 0.042 | 0.522 |
| LLM-TLS | Llama13B | 0.118 | 0.036 | 0.528 |
| LLM-TLS | Qwen72B | 0.114 | 0.040 | 0.543 |
| **SUnSET** | Qwen72B | **0.136** | **0.044** | **0.576** |
| SUnSET | GPT-4o | 0.120 | 0.039 | 0.590 |

## Expected Output Format

```
=== Evaluation Results ===

Topic: EgyptianProtest
  AR-1:     0.136
  AR-2:     0.044
  Date-F1:  0.576

Topic: SyrianCrisis
  AR-1:     0.128
  AR-2:     0.041
  Date-F1:  0.554

...

=== AVERAGE (9 topics) ===
  AR-1:     0.136
  AR-2:     0.044
  Date-F1:  0.576
```

## Implementation Notes

1. **Tilse framework**: Paper uses parts of https://github.com/complementizer/news-tls for evaluation
2. **Stemming**: Standard ROUGE implementations often use stemming (Porter stemmer)
3. **Tokenization**: Simple whitespace + punctuation split typically sufficient
