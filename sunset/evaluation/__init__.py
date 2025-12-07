"""
Evaluation Module for Timeline Summarization

Implements AR-1, AR-2, and Date-F1 metrics.
"""

from .metrics import (
    EvalResult,
    compute_date_f1,
    rouge_n,
    compute_alignment_rouge,
    evaluate,
    evaluate_multi_gold,
    average_results,
)
from .tilse_wrapper import (
    evaluate_with_tilse,
    evaluate_with_tilse_multi_gold,
    is_tilse_available,
)

__all__ = [
    # Metrics
    "EvalResult",
    "compute_date_f1",
    "rouge_n",
    "compute_alignment_rouge",
    "evaluate",
    "evaluate_multi_gold",
    "average_results",
    # Tilse
    "evaluate_with_tilse",
    "evaluate_with_tilse_multi_gold",
    "is_tilse_available",
]
