"""
Structured Logging Module

Provides JSON-formatted logs with stage/component tracking and performance metrics.
"""

import json
import logging
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass, asdict
from datetime import datetime
from functools import wraps
from pathlib import Path
from typing import Any, Optional


@dataclass
class LogRecord:
    """Structured log record."""
    timestamp: str
    level: str
    stage: str
    component: str
    message: str
    data: Optional[dict] = None

    def to_json(self) -> str:
        d = asdict(self)
        if d["data"] is None:
            del d["data"]
        return json.dumps(d)


class JSONFormatter(logging.Formatter):
    """JSON log formatter."""

    def __init__(self, stage: str = "", component: str = ""):
        super().__init__()
        self.stage = stage
        self.component = component

    def format(self, record: logging.LogRecord) -> str:
        # Extract extra data if present
        data = getattr(record, "data", None)
        stage = getattr(record, "stage", self.stage)
        component = getattr(record, "component", self.component)

        log_record = LogRecord(
            timestamp=datetime.utcnow().isoformat() + "Z",
            level=record.levelname,
            stage=stage,
            component=component,
            message=record.getMessage(),
            data=data,
        )
        return log_record.to_json()


class TextFormatter(logging.Formatter):
    """Human-readable text formatter."""

    def format(self, record: logging.LogRecord) -> str:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        stage = getattr(record, "stage", "")
        component = getattr(record, "component", "")
        data = getattr(record, "data", None)

        prefix = f"[{timestamp}] [{record.levelname}]"
        if stage:
            prefix += f" [{stage}]"
        if component:
            prefix += f" [{component}]"

        msg = f"{prefix} {record.getMessage()}"
        if data:
            msg += f" | {json.dumps(data)}"
        return msg


class StageLogger:
    """
    Logger with stage and component context.

    Usage:
        logger = StageLogger("stage1", "event_extraction")
        logger.info("Extracted events", data={"count": 5, "duration_ms": 123})
    """

    def __init__(
        self,
        stage: str,
        component: str,
        level: str = "INFO",
        log_format: str = "json",
        log_file: Optional[str] = None,
    ):
        self.stage = stage
        self.component = component
        self.logger = logging.getLogger(f"sunset.{stage}.{component}")
        self.logger.setLevel(getattr(logging, level.upper()))

        # Prevent duplicate handlers
        if not self.logger.handlers:
            # Console handler
            console_handler = logging.StreamHandler(sys.stdout)
            if log_format == "json":
                console_handler.setFormatter(JSONFormatter(stage, component))
            else:
                console_handler.setFormatter(TextFormatter())
            self.logger.addHandler(console_handler)

            # File handler
            if log_file:
                file_handler = logging.FileHandler(log_file)
                file_handler.setFormatter(JSONFormatter(stage, component))
                self.logger.addHandler(file_handler)

    def _log(self, level: int, message: str, data: Optional[dict] = None):
        extra = {
            "stage": self.stage,
            "component": self.component,
            "data": data,
        }
        self.logger.log(level, message, extra=extra)

    def debug(self, message: str, data: Optional[dict] = None):
        self._log(logging.DEBUG, message, data)

    def info(self, message: str, data: Optional[dict] = None):
        self._log(logging.INFO, message, data)

    def warning(self, message: str, data: Optional[dict] = None):
        self._log(logging.WARNING, message, data)

    def error(self, message: str, data: Optional[dict] = None):
        self._log(logging.ERROR, message, data)

    @contextmanager
    def timed_operation(self, operation_name: str):
        """Context manager for timing operations."""
        start = time.perf_counter()
        try:
            yield
        finally:
            duration_ms = (time.perf_counter() - start) * 1000
            self.info(
                f"Completed {operation_name}",
                data={"operation": operation_name, "duration_ms": round(duration_ms, 2)},
            )


def timed(stage: str = "", component: str = ""):
    """Decorator for timing functions."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start = time.perf_counter()
            result = func(*args, **kwargs)
            duration_ms = (time.perf_counter() - start) * 1000

            logger = StageLogger(stage or "pipeline", component or func.__name__)
            logger.info(
                f"Function {func.__name__} completed",
                data={"duration_ms": round(duration_ms, 2)},
            )
            return result
        return wrapper
    return decorator


def timed_async(stage: str = "", component: str = ""):
    """Decorator for timing async functions."""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            start = time.perf_counter()
            result = await func(*args, **kwargs)
            duration_ms = (time.perf_counter() - start) * 1000

            logger = StageLogger(stage or "pipeline", component or func.__name__)
            logger.info(
                f"Async function {func.__name__} completed",
                data={"duration_ms": round(duration_ms, 2)},
            )
            return result
        return wrapper
    return decorator


def setup_logging(
    level: str = "INFO",
    log_format: str = "json",
    log_file: Optional[str] = None,
):
    """Setup root logging configuration."""
    root_logger = logging.getLogger("sunset")
    root_logger.setLevel(getattr(logging, level.upper()))

    # Clear existing handlers
    root_logger.handlers.clear()

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    if log_format == "json":
        console_handler.setFormatter(JSONFormatter())
    else:
        console_handler.setFormatter(TextFormatter())
    root_logger.addHandler(console_handler)

    # File handler
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(JSONFormatter())
        root_logger.addHandler(file_handler)
