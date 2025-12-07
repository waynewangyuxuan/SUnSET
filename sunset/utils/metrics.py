"""
Performance Metrics Module

Tracks timing, counts, and distributions for pipeline analysis.
"""

import time
import statistics
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class TimingStats:
    """Statistics for timing measurements."""
    count: int = 0
    total_ms: float = 0.0
    min_ms: float = float("inf")
    max_ms: float = 0.0
    samples: List[float] = field(default_factory=list)

    def add(self, duration_ms: float):
        self.count += 1
        self.total_ms += duration_ms
        self.min_ms = min(self.min_ms, duration_ms)
        self.max_ms = max(self.max_ms, duration_ms)
        self.samples.append(duration_ms)

    @property
    def avg_ms(self) -> float:
        return self.total_ms / self.count if self.count > 0 else 0.0

    @property
    def p50_ms(self) -> float:
        if not self.samples:
            return 0.0
        return statistics.median(self.samples)

    @property
    def p95_ms(self) -> float:
        if not self.samples:
            return 0.0
        sorted_samples = sorted(self.samples)
        idx = int(len(sorted_samples) * 0.95)
        return sorted_samples[min(idx, len(sorted_samples) - 1)]

    def to_dict(self) -> dict:
        return {
            "count": self.count,
            "total_ms": round(self.total_ms, 2),
            "avg_ms": round(self.avg_ms, 2),
            "min_ms": round(self.min_ms, 2) if self.count > 0 else None,
            "max_ms": round(self.max_ms, 2) if self.count > 0 else None,
            "p50_ms": round(self.p50_ms, 2),
            "p95_ms": round(self.p95_ms, 2),
        }


@dataclass
class CountStats:
    """Statistics for count measurements."""
    values: List[int] = field(default_factory=list)

    def add(self, value: int):
        self.values.append(value)

    @property
    def count(self) -> int:
        return len(self.values)

    @property
    def total(self) -> int:
        return sum(self.values)

    @property
    def avg(self) -> float:
        return self.total / self.count if self.count > 0 else 0.0

    @property
    def min(self) -> int:
        return min(self.values) if self.values else 0

    @property
    def max(self) -> int:
        return max(self.values) if self.values else 0

    def histogram(self, bins: int = 10) -> Dict[str, int]:
        """Create histogram of values."""
        if not self.values:
            return {}

        min_val = min(self.values)
        max_val = max(self.values)
        if min_val == max_val:
            return {str(min_val): len(self.values)}

        bin_width = (max_val - min_val) / bins
        hist = defaultdict(int)

        for v in self.values:
            bin_idx = min(int((v - min_val) / bin_width), bins - 1)
            bin_start = min_val + bin_idx * bin_width
            bin_end = bin_start + bin_width
            key = f"{bin_start:.1f}-{bin_end:.1f}"
            hist[key] += 1

        return dict(hist)

    def to_dict(self) -> dict:
        return {
            "count": self.count,
            "total": self.total,
            "avg": round(self.avg, 2),
            "min": self.min,
            "max": self.max,
            "histogram": self.histogram(),
        }


class MetricsCollector:
    """
    Collects and aggregates metrics for a pipeline stage.

    Usage:
        metrics = MetricsCollector("stage1")

        # Track timing
        with metrics.timer("extraction"):
            do_extraction()

        # Track counts
        metrics.count("events_per_article", 5)

        # Get report
        report = metrics.get_report()
    """

    def __init__(self, stage: str):
        self.stage = stage
        self.timings: Dict[str, TimingStats] = defaultdict(TimingStats)
        self.counts: Dict[str, CountStats] = defaultdict(CountStats)
        self.values: Dict[str, List[Any]] = defaultdict(list)
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None

    def start(self):
        """Mark the start of the stage."""
        self.start_time = time.time()

    def end(self):
        """Mark the end of the stage."""
        self.end_time = time.time()

    @property
    def total_duration_seconds(self) -> Optional[float]:
        if self.start_time and self.end_time:
            return self.end_time - self.start_time
        return None

    class _Timer:
        def __init__(self, collector: "MetricsCollector", name: str):
            self.collector = collector
            self.name = name
            self.start = None

        def __enter__(self):
            self.start = time.perf_counter()
            return self

        def __exit__(self, *args):
            duration_ms = (time.perf_counter() - self.start) * 1000
            self.collector.timings[self.name].add(duration_ms)

    def timer(self, name: str) -> _Timer:
        """Context manager for timing operations."""
        return self._Timer(self, name)

    def add_timing(self, name: str, duration_ms: float):
        """Add a timing measurement directly."""
        self.timings[name].add(duration_ms)

    def count(self, name: str, value: int):
        """Add a count measurement."""
        self.counts[name].add(value)

    def record(self, name: str, value: Any):
        """Record an arbitrary value."""
        self.values[name].append(value)

    def get_report(self) -> dict:
        """Generate a metrics report."""
        report = {
            "stage": self.stage,
            "timings": {k: v.to_dict() for k, v in self.timings.items()},
            "counts": {k: v.to_dict() for k, v in self.counts.items()},
            "values": {k: v for k, v in self.values.items()},
        }

        if self.total_duration_seconds is not None:
            report["total_duration_seconds"] = round(self.total_duration_seconds, 2)

        return report


class PipelineMetrics:
    """
    Aggregates metrics across all pipeline stages.

    Usage:
        pipeline_metrics = PipelineMetrics()
        stage1_metrics = pipeline_metrics.stage("stage1")
        # ... use stage1_metrics ...
        report = pipeline_metrics.get_full_report()
    """

    def __init__(self):
        self.stages: Dict[str, MetricsCollector] = {}
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None

    def start(self):
        """Mark the start of the pipeline."""
        self.start_time = time.time()

    def end(self):
        """Mark the end of the pipeline."""
        self.end_time = time.time()

    def stage(self, name: str) -> MetricsCollector:
        """Get or create a metrics collector for a stage."""
        if name not in self.stages:
            self.stages[name] = MetricsCollector(name)
        return self.stages[name]

    def get_full_report(self) -> dict:
        """Generate a full pipeline metrics report."""
        report = {
            "stages": {name: collector.get_report() for name, collector in self.stages.items()},
        }

        if self.start_time and self.end_time:
            report["total_duration_seconds"] = round(self.end_time - self.start_time, 2)

        return report
