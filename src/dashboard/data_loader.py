"""Load evaluation artifacts for the dashboard."""
from __future__ import annotations

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def load_benchmark_summary(
    path: Path = Path("experiments/results/benchmark_summary.json"),
) -> dict:
    """Load benchmark_summary.json. Raises FileNotFoundError with setup instructions if missing."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Benchmark summary not found at {path}. "
            "Run the full pipeline: make download-data && make process-data && "
            "make finetune && make evaluate"
        )
    with open(path) as f:
        return json.load(f)


def load_approach_results(approach: str, results_dir: Path) -> dict:
    """Load individual results JSON (e.g. zero_shot_results.json) for a given approach."""
    results_dir = Path(results_dir)
    path = results_dir / f"{approach}_results.json"
    if not path.exists():
        logger.warning("Results file not found: %s", path)
        return {}
    with open(path) as f:
        return json.load(f)


_CM_FILENAMES = {
    "zero_shot": "zero_shot_cm.png",
    "few_shot": "few_shot_cm.png",
    "fine_tuned": "finetuned_cm.png",
}


def load_confusion_matrix_image(approach: str, cm_dir: Path) -> Path | None:
    """Return path to confusion matrix PNG or None if not found."""
    cm_dir = Path(cm_dir)
    filename = _CM_FILENAMES.get(approach, f"{approach}_cm.png")
    path = cm_dir / filename
    if path.exists():
        return path
    logger.warning("Confusion matrix not found: %s", path)
    return None


def get_available_approaches(summary: dict) -> list[str]:
    """Return list of approach names present in summary['approaches']."""
    return list(summary.get("approaches", {}).keys())
