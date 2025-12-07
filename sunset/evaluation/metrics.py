"""
Evaluation Metrics for Timeline Summarization

Implements AR-1, AR-2, and Date-F1 metrics.
"""

from dataclasses import dataclass
from typing import List, Dict, Set
from collections import Counter
from datetime import date as datelib


@dataclass
class EvalResult:
    """Evaluation result for a single topic."""
    ar1: float
    ar2: float
    date_f1: float

    def to_dict(self) -> dict:
        return {
            "ar1": self.ar1,
            "ar2": self.ar2,
            "date_f1": self.date_f1,
        }


def compute_date_f1(pred_dates: Set[str], gold_dates: Set[str]) -> float:
    """
    Compute Date-F1 score.

    Args:
        pred_dates: Set of predicted dates (YYYY-MM-DD)
        gold_dates: Set of gold dates (YYYY-MM-DD)

    Returns:
        Date F1 score
    """
    if not pred_dates or not gold_dates:
        return 0.0

    tp = len(pred_dates & gold_dates)

    precision = tp / len(pred_dates)
    recall = tp / len(gold_dates)

    if precision + recall == 0:
        return 0.0

    f1 = 2 * precision * recall / (precision + recall)
    return f1


def tokenize(text: str) -> List[str]:
    """Simple tokenization: lowercase, split on whitespace and punctuation."""
    import re
    text = text.lower()
    # Split on non-alphanumeric characters
    tokens = re.findall(r'\b\w+\b', text)
    return tokens


def extract_ngrams(tokens: List[str], n: int) -> List[tuple]:
    """Extract n-grams from token list."""
    if len(tokens) < n:
        return []
    return [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]


def rouge_n(candidate: str, reference: str, n: int = 1) -> Dict[str, float]:
    """
    Compute ROUGE-N score.

    Args:
        candidate: Generated text
        reference: Reference text
        n: N-gram size (1 for ROUGE-1, 2 for ROUGE-2)

    Returns:
        Dict with precision, recall, f1
    """
    cand_tokens = tokenize(candidate)
    ref_tokens = tokenize(reference)

    cand_ngrams = extract_ngrams(cand_tokens, n)
    ref_ngrams = extract_ngrams(ref_tokens, n)

    if not cand_ngrams or not ref_ngrams:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    cand_counts = Counter(cand_ngrams)
    ref_counts = Counter(ref_ngrams)

    # Count overlap
    overlap = 0
    for ngram, count in cand_counts.items():
        overlap += min(count, ref_counts.get(ngram, 0))

    precision = overlap / sum(cand_counts.values())
    recall = overlap / sum(ref_counts.values())

    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)

    return {"precision": precision, "recall": recall, "f1": f1}


def compute_alignment_rouge(
    pred_timeline: Dict[str, str],
    gold_timeline: Dict[str, List[str]],
    n: int = 1
) -> Dict[str, float]:
    """
    Compute Alignment-based ROUGE score.

    Aligns events by date first, then computes ROUGE.

    Args:
        pred_timeline: {date: summary}
        gold_timeline: {date: [summary1, summary2, ...]}
        n: N-gram size (1 or 2)

    Returns:
        Dict with precision, recall, f1
    """
    all_precision = []
    all_recall = []
    all_f1 = []

    for gold_date, gold_summaries in gold_timeline.items():
        pred_text = pred_timeline.get(gold_date, "")

        for gold_text in gold_summaries:
            if not pred_text:
                all_precision.append(0.0)
                all_recall.append(0.0)
                all_f1.append(0.0)
            else:
                scores = rouge_n(pred_text, gold_text, n)
                all_precision.append(scores["precision"])
                all_recall.append(scores["recall"])
                all_f1.append(scores["f1"])

    if not all_f1:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    return {
        "precision": sum(all_precision) / len(all_precision),
        "recall": sum(all_recall) / len(all_recall),
        "f1": sum(all_f1) / len(all_f1),
    }


def evaluate(
    pred_timeline,
    gold_timeline: Dict[str, List[str]]
) -> EvalResult:
    """
    Evaluate a predicted timeline against a gold timeline.

    Args:
        pred_timeline: Timeline object with entries
        gold_timeline: {date: [summary1, summary2, ...]}

    Returns:
        EvalResult with AR-1, AR-2, Date-F1
    """
    # Convert pred_timeline to dict format
    pred_dict = {}
    pred_dates = set()
    for entry in pred_timeline.entries:
        pred_dict[entry.date] = entry.summary
        pred_dates.add(entry.date)

    gold_dates = set(gold_timeline.keys())

    # Compute metrics
    ar1_scores = compute_alignment_rouge(pred_dict, gold_timeline, n=1)
    ar2_scores = compute_alignment_rouge(pred_dict, gold_timeline, n=2)
    date_f1 = compute_date_f1(pred_dates, gold_dates)

    return EvalResult(
        ar1=ar1_scores["f1"],
        ar2=ar2_scores["f1"],
        date_f1=date_f1,
    )


def evaluate_multi_gold(
    pred_timeline,
    gold_timelines: Dict[str, Dict[str, List[str]]]
) -> EvalResult:
    """
    Evaluate against multiple gold timelines, averaging results.

    Args:
        pred_timeline: Timeline object
        gold_timelines: {timeline_name: {date: [summaries]}}

    Returns:
        Averaged EvalResult
    """
    all_ar1 = []
    all_ar2 = []
    all_date_f1 = []

    for tl_name, gold_tl in gold_timelines.items():
        result = evaluate(pred_timeline, gold_tl)
        all_ar1.append(result.ar1)
        all_ar2.append(result.ar2)
        all_date_f1.append(result.date_f1)

    if not all_ar1:
        return EvalResult(ar1=0.0, ar2=0.0, date_f1=0.0)

    return EvalResult(
        ar1=sum(all_ar1) / len(all_ar1),
        ar2=sum(all_ar2) / len(all_ar2),
        date_f1=sum(all_date_f1) / len(all_date_f1),
    )


def average_results(results: Dict[str, EvalResult]) -> EvalResult:
    """Compute average of multiple EvalResults."""
    if not results:
        return EvalResult(ar1=0.0, ar2=0.0, date_f1=0.0)

    ar1_vals = [r.ar1 for r in results.values()]
    ar2_vals = [r.ar2 for r in results.values()]
    date_f1_vals = [r.date_f1 for r in results.values()]

    return EvalResult(
        ar1=sum(ar1_vals) / len(ar1_vals),
        ar2=sum(ar2_vals) / len(ar2_vals),
        date_f1=sum(date_f1_vals) / len(date_f1_vals),
    )
