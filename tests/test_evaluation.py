"""
Evaluation Test Script

Tests AR-1, AR-2, and Date-F1 metrics.
"""

import sys
sys.path.insert(0, ".")

from sunset.evaluation import (
    EvalResult,
    compute_date_f1,
    rouge_n,
    compute_alignment_rouge,
    evaluate,
    evaluate_multi_gold,
    is_tilse_available,
    evaluate_with_tilse,
)
from sunset.stage3_timeline import Timeline, TimelineEntry
from sunset.utils import StageLogger


def test_date_f1():
    """Test Date-F1 computation."""
    logger = StageLogger("test", "date_f1", log_format="text")

    # Perfect match
    pred = {"2011-01-25", "2011-01-26", "2011-01-27"}
    gold = {"2011-01-25", "2011-01-26", "2011-01-27"}
    f1 = compute_date_f1(pred, gold)
    logger.info(f"Perfect match: F1 = {f1:.4f}")
    assert f1 == 1.0

    # Partial match
    pred = {"2011-01-25", "2011-01-26", "2011-01-28"}
    gold = {"2011-01-25", "2011-01-26", "2011-01-27"}
    f1 = compute_date_f1(pred, gold)
    logger.info(f"Partial match (2/3): F1 = {f1:.4f}")
    assert 0.6 < f1 < 0.7

    # No overlap
    pred = {"2011-01-28", "2011-01-29", "2011-01-30"}
    gold = {"2011-01-25", "2011-01-26", "2011-01-27"}
    f1 = compute_date_f1(pred, gold)
    logger.info(f"No overlap: F1 = {f1:.4f}")
    assert f1 == 0.0

    logger.info("✓ Date-F1 tests passed")


def test_rouge():
    """Test ROUGE computation."""
    logger = StageLogger("test", "rouge", log_format="text")

    # Identical text
    cand = "The quick brown fox jumps over the lazy dog"
    ref = "The quick brown fox jumps over the lazy dog"
    scores = rouge_n(cand, ref, n=1)
    logger.info(f"Identical: ROUGE-1 = {scores['f1']:.4f}")
    assert scores["f1"] == 1.0

    # Partial overlap
    cand = "The quick brown fox"
    ref = "The slow brown cat"
    scores = rouge_n(cand, ref, n=1)
    logger.info(f"Partial overlap: ROUGE-1 = {scores['f1']:.4f}")
    assert 0.4 < scores["f1"] < 0.7

    # No overlap
    cand = "hello world"
    ref = "goodbye universe"
    scores = rouge_n(cand, ref, n=1)
    logger.info(f"No overlap: ROUGE-1 = {scores['f1']:.4f}")
    assert scores["f1"] == 0.0

    # ROUGE-2
    cand = "The quick brown fox jumps over"
    ref = "The quick brown fox jumps high"
    scores = rouge_n(cand, ref, n=2)
    logger.info(f"ROUGE-2 test: F1 = {scores['f1']:.4f}")

    logger.info("✓ ROUGE tests passed")


def test_alignment_rouge():
    """Test alignment-based ROUGE."""
    logger = StageLogger("test", "alignment_rouge", log_format="text")

    pred = {
        "2011-01-25": "Protests begin in Egypt against Mubarak",
        "2011-01-26": "Violence erupts in Cairo streets",
    }
    gold = {
        "2011-01-25": ["Mass protests erupt in Egypt demanding Mubarak step down"],
        "2011-01-26": ["Violent clashes in Cairo between police and protesters"],
        "2011-01-27": ["Internet shutdown in Egypt"],  # No prediction for this
    }

    ar1 = compute_alignment_rouge(pred, gold, n=1)
    ar2 = compute_alignment_rouge(pred, gold, n=2)

    logger.info(f"Alignment ROUGE-1: {ar1['f1']:.4f}")
    logger.info(f"Alignment ROUGE-2: {ar2['f1']:.4f}")

    logger.info("✓ Alignment ROUGE tests passed")


def test_evaluate():
    """Test full evaluation."""
    logger = StageLogger("test", "evaluate", log_format="text")

    # Create predicted timeline
    pred = Timeline(entries=[
        TimelineEntry(date="2011-01-25", summary="Protests begin in Egypt", significance=0.9),
        TimelineEntry(date="2011-01-26", summary="Violence in Cairo", significance=0.8),
        TimelineEntry(date="2011-01-28", summary="Internet blocked", significance=0.7),
    ])

    # Create gold timeline
    gold = {
        "2011-01-25": ["Mass protests in Egypt demanding change"],
        "2011-01-26": ["Violence erupts between police and protesters"],
        "2011-01-27": ["Internet shutdown announced"],
    }

    result = evaluate(pred, gold)

    logger.info(f"Evaluation result:")
    logger.info(f"  AR-1:     {result.ar1:.4f}")
    logger.info(f"  AR-2:     {result.ar2:.4f}")
    logger.info(f"  Date-F1:  {result.date_f1:.4f}")

    assert result.ar1 >= 0 and result.ar1 <= 1
    assert result.ar2 >= 0 and result.ar2 <= 1
    assert result.date_f1 >= 0 and result.date_f1 <= 1

    logger.info("✓ Evaluate tests passed")


def test_tilse():
    """Test Tilse integration."""
    logger = StageLogger("test", "tilse", log_format="text")

    if not is_tilse_available():
        logger.warning("Tilse not available, skipping test")
        return

    pred = Timeline(entries=[
        TimelineEntry(date="2011-01-25", summary="Protests in Egypt", significance=0.9),
    ])

    gold = {
        "2011-01-25": ["Mass protests in Egypt"],
    }

    scores = evaluate_with_tilse(pred, gold)

    if scores:
        logger.info(f"Tilse AR-1: {scores['ar1']:.4f}")
        logger.info(f"Tilse AR-2: {scores['ar2']:.4f}")
        logger.info("✓ Tilse tests passed")
    else:
        logger.warning("Tilse evaluation returned None")


def main():
    print("=== Evaluation Test ===\n")

    print("\n--- Testing Date-F1 ---")
    test_date_f1()

    print("\n--- Testing ROUGE ---")
    test_rouge()

    print("\n--- Testing Alignment ROUGE ---")
    test_alignment_rouge()

    print("\n--- Testing Full Evaluate ---")
    test_evaluate()

    print("\n--- Testing Tilse ---")
    test_tilse()

    print("\n=== Evaluation Test Complete ===")


if __name__ == "__main__":
    main()
