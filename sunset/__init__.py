"""
SUnSET: Synergistic Understanding of Stakeholder, Events and Time

Timeline Summarization pipeline based on arXiv:2507.21903
"""

__version__ = "0.1.0"

from .config import Config, load_config
from .data import Timeline17Dataset
from .utils import (
    StageLogger,
    MetricsCollector,
    PipelineMetrics,
    ArtifactManager,
)

__all__ = [
    "Config",
    "load_config",
    "Timeline17Dataset",
    "StageLogger",
    "MetricsCollector",
    "PipelineMetrics",
    "ArtifactManager",
]
