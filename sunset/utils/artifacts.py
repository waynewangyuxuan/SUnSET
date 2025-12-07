"""
Artifacts Module

Handles saving and loading pipeline artifacts (JSON, JSONL, pickle, numpy).
"""

import json
import pickle
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Union

import numpy as np


class ArtifactManager:
    """
    Manages pipeline artifacts with timestamped run directories.

    Usage:
        artifacts = ArtifactManager("results")
        artifacts.new_run()  # Creates results/run_20241206_143022/

        # Save artifacts
        artifacts.save_json("stage1/events.json", events_data)
        artifacts.save_jsonl("stage1/events.jsonl", events_list)
        artifacts.save_numpy("stage2/embeddings.npy", embeddings)

        # Load artifacts
        events = artifacts.load_json("stage1/events.json")
    """

    def __init__(self, base_dir: str = "results"):
        self.base_dir = Path(base_dir)
        self.run_dir: Optional[Path] = None

    def new_run(self, run_id: Optional[str] = None) -> Path:
        """Create a new run directory."""
        if run_id is None:
            run_id = datetime.now().strftime("run_%Y%m%d_%H%M%S")

        self.run_dir = self.base_dir / run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)

        # Create subdirectories
        for subdir in ["logs", "stage1", "stage2", "stage3", "evaluation"]:
            (self.run_dir / subdir).mkdir(exist_ok=True)

        return self.run_dir

    def set_run(self, run_id: str):
        """Set the current run directory (for loading existing runs)."""
        self.run_dir = self.base_dir / run_id
        if not self.run_dir.exists():
            raise FileNotFoundError(f"Run directory not found: {self.run_dir}")

    def _get_path(self, relative_path: str) -> Path:
        """Get full path from relative path."""
        if self.run_dir is None:
            raise RuntimeError("No run directory set. Call new_run() first.")
        return self.run_dir / relative_path

    # JSON operations
    def save_json(self, relative_path: str, data: Any, indent: int = 2):
        """Save data as JSON."""
        path = self._get_path(relative_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=indent, ensure_ascii=False, default=str)

    def load_json(self, relative_path: str) -> Any:
        """Load JSON data."""
        path = self._get_path(relative_path)
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    # JSONL operations (for streaming large datasets)
    def save_jsonl(self, relative_path: str, items: List[Any]):
        """Save list of items as JSON Lines."""
        path = self._get_path(relative_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w", encoding="utf-8") as f:
            for item in items:
                f.write(json.dumps(item, ensure_ascii=False, default=str) + "\n")

    def append_jsonl(self, relative_path: str, item: Any):
        """Append a single item to JSONL file."""
        path = self._get_path(relative_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(item, ensure_ascii=False, default=str) + "\n")

    def load_jsonl(self, relative_path: str) -> List[Any]:
        """Load JSON Lines file."""
        path = self._get_path(relative_path)
        items = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    items.append(json.loads(line))
        return items

    def iter_jsonl(self, relative_path: str) -> Iterator[Any]:
        """Iterate over JSON Lines file (memory efficient)."""
        path = self._get_path(relative_path)
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    yield json.loads(line)

    # NumPy operations
    def save_numpy(self, relative_path: str, array: np.ndarray):
        """Save NumPy array."""
        path = self._get_path(relative_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, array)

    def load_numpy(self, relative_path: str) -> np.ndarray:
        """Load NumPy array."""
        path = self._get_path(relative_path)
        # Handle both .npy extension and without
        if not path.suffix:
            path = path.with_suffix(".npy")
        return np.load(path)

    # Pickle operations (for complex objects)
    def save_pickle(self, relative_path: str, obj: Any):
        """Save object as pickle."""
        path = self._get_path(relative_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "wb") as f:
            pickle.dump(obj, f)

    def load_pickle(self, relative_path: str) -> Any:
        """Load pickled object."""
        path = self._get_path(relative_path)
        with open(path, "rb") as f:
            return pickle.load(f)

    # Text operations
    def save_text(self, relative_path: str, text: str):
        """Save plain text."""
        path = self._get_path(relative_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w", encoding="utf-8") as f:
            f.write(text)

    def load_text(self, relative_path: str) -> str:
        """Load plain text."""
        path = self._get_path(relative_path)
        with open(path, "r", encoding="utf-8") as f:
            return f.read()

    # Utility methods
    def exists(self, relative_path: str) -> bool:
        """Check if artifact exists."""
        return self._get_path(relative_path).exists()

    def list_artifacts(self, subdir: str = "") -> List[str]:
        """List all artifacts in a subdirectory."""
        path = self._get_path(subdir)
        if not path.exists():
            return []
        return [str(p.relative_to(path)) for p in path.rglob("*") if p.is_file()]

    def get_run_info(self) -> Dict[str, Any]:
        """Get information about the current run."""
        if self.run_dir is None:
            return {}

        return {
            "run_id": self.run_dir.name,
            "run_dir": str(self.run_dir),
            "artifacts": self.list_artifacts(),
        }


def load_timeline17(path: str) -> Dict[str, Any]:
    """
    Load Timeline17 dataset from pickle file.

    Returns:
        Dict with topic names as keys, each containing:
        - articles: List of article dicts
        - gold_timelines: Dict of timeline name to timeline dict
    """
    with open(path, "rb") as f:
        return pickle.load(f)
