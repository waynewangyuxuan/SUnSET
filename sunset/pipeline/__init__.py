"""
SUnSET Pipeline

Complete pipeline for timeline summarization.
"""

from .runner import (
    SUnSETPipeline,
    PipelineResult,
    TopicResult,
    generate_report,
)

__all__ = [
    "SUnSETPipeline",
    "PipelineResult",
    "TopicResult",
    "generate_report",
]
