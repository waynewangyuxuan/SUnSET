"""
Tilse Framework Integration

Uses the Tilse framework for official AR-1/AR-2 computation.
"""

from datetime import date as datelib
from typing import Dict, List, Optional
from ..utils import StageLogger

logger = StageLogger("evaluation", "tilse")


def _parse_date(date_str: str) -> datelib:
    """Parse date string to datetime.date."""
    return datelib.fromisoformat(date_str)


def evaluate_with_tilse(
    pred_timeline,
    gold_timeline: Dict[str, List[str]]
) -> Dict[str, float]:
    """
    Evaluate using Tilse framework for AR-1/AR-2.

    Args:
        pred_timeline: Timeline object with entries
        gold_timeline: {date: [summaries]}

    Returns:
        Dict with ar1, ar2 scores
    """
    try:
        from tilse.data.timelines import Timeline, GroundTruth
        from tilse.evaluation.rouge import TimelineRougeEvaluator
    except ImportError:
        logger.warning("Tilse not installed, using fallback ROUGE implementation")
        return None

    # Convert predicted timeline to tilse format
    pred_dict = {}
    for entry in pred_timeline.entries:
        d = _parse_date(entry.date)
        if d not in pred_dict:
            pred_dict[d] = []
        pred_dict[d].append(entry.summary)

    # Convert gold timeline to tilse format
    gold_dict = {}
    for date_str, summaries in gold_timeline.items():
        d = _parse_date(date_str)
        gold_dict[d] = summaries

    # Create tilse objects
    pred_tl = Timeline(pred_dict)
    gold_tl = Timeline(gold_dict)
    ground_truth = GroundTruth([gold_tl])

    # Evaluate
    try:
        evaluator = TimelineRougeEvaluator(measures=["rouge_1", "rouge_2"])
        scores = evaluator.evaluate_concat(pred_tl, ground_truth)

        ar1 = scores["rouge_1"]["f_score"]
        ar2 = scores["rouge_2"]["f_score"]

        return {"ar1": ar1, "ar2": ar2}

    except Exception as e:
        logger.error(f"Tilse evaluation failed: {e}")
        return None


def evaluate_with_tilse_multi_gold(
    pred_timeline,
    gold_timelines: Dict[str, Dict[str, List[str]]]
) -> Optional[Dict[str, float]]:
    """
    Evaluate against multiple gold timelines using Tilse.

    Args:
        pred_timeline: Timeline object
        gold_timelines: {tl_name: {date: [summaries]}}

    Returns:
        Dict with averaged ar1, ar2 scores
    """
    all_ar1 = []
    all_ar2 = []

    for tl_name, gold_tl in gold_timelines.items():
        scores = evaluate_with_tilse(pred_timeline, gold_tl)
        if scores:
            all_ar1.append(scores["ar1"])
            all_ar2.append(scores["ar2"])

    if not all_ar1:
        return None

    return {
        "ar1": sum(all_ar1) / len(all_ar1),
        "ar2": sum(all_ar2) / len(all_ar2),
    }


def is_tilse_available() -> bool:
    """Check if Tilse is properly installed and configured."""
    try:
        from tilse.data.timelines import Timeline, GroundTruth
        from tilse.evaluation.rouge import TimelineRougeEvaluator

        # Quick test
        pred = Timeline({datelib(2011, 1, 1): ["Test event"]})
        gold = Timeline({datelib(2011, 1, 1): ["Test reference"]})
        gt = GroundTruth([gold])

        evaluator = TimelineRougeEvaluator(measures=["rouge_1"])
        evaluator.evaluate_concat(pred, gt)

        return True

    except Exception as e:
        logger.warning(f"Tilse not available: {e}")
        return False
