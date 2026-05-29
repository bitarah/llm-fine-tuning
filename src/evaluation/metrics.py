"""Pure metric computation — no model loading, no file I/O."""
from __future__ import annotations

import difflib
import logging
from dataclasses import dataclass

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

logger = logging.getLogger(__name__)


@dataclass
class EvaluationResult:
    """Holds all computed metrics for one evaluation approach."""

    approach_name: str
    accuracy: float
    macro_f1: float
    weighted_f1: float
    macro_precision: float
    weighted_precision: float
    macro_recall: float
    weighted_recall: float
    per_class_f1: dict[str, float]
    per_class_precision: dict[str, float]
    per_class_recall: dict[str, float]
    confusion_matrix: list[list[int]]
    mean_latency_ms: float
    p95_latency_ms: float
    token_cost_per_1k: float
    total_samples: int
    correct_samples: int
    predictions: list[str]
    ground_truth: list[str]
    input_texts: list[str]
    raw_outputs: list[str]
    latencies_ms: list[float]


def compute_metrics(
    approach_name: str,
    predictions: list[str],
    ground_truth: list[str],
    latencies_ms: list[float],
    intent_list: list[str],
    input_texts: list[str],
    raw_outputs: list[str] | None = None,
) -> EvaluationResult:
    """Compute all metrics. Returns populated EvaluationResult."""
    accuracy = float(accuracy_score(ground_truth, predictions))
    macro_f1 = float(
        f1_score(ground_truth, predictions, average="macro", zero_division=0, labels=intent_list)
    )
    weighted_f1 = float(
        f1_score(ground_truth, predictions, average="weighted", zero_division=0, labels=intent_list)
    )

    report = classification_report(
        ground_truth,
        predictions,
        labels=intent_list,
        output_dict=True,
        zero_division=0,
    )
    per_class_f1 = {
        intent: report[intent]["f1-score"]
        for intent in intent_list
        if intent in report
    }
    per_class_precision = {
        intent: report[intent]["precision"]
        for intent in intent_list
        if intent in report
    }
    per_class_recall = {
        intent: report[intent]["recall"]
        for intent in intent_list
        if intent in report
    }
    macro_precision = float(report["macro avg"]["precision"])
    weighted_precision = float(report["weighted avg"]["precision"])
    macro_recall = float(report["macro avg"]["recall"])
    weighted_recall = float(report["weighted avg"]["recall"])

    cm = confusion_matrix(ground_truth, predictions, labels=intent_list)
    cm_list: list[list[int]] = cm.tolist()

    latency_arr = np.array(latencies_ms)
    mean_latency = float(np.mean(latency_arr))
    p95_latency = float(np.percentile(latency_arr, 95))

    token_cost = estimate_token_cost(input_texts)
    correct_samples = sum(p == g for p, g in zip(predictions, ground_truth))  # noqa: B905

    return EvaluationResult(
        approach_name=approach_name,
        accuracy=accuracy,
        macro_f1=macro_f1,
        weighted_f1=weighted_f1,
        macro_precision=macro_precision,
        weighted_precision=weighted_precision,
        macro_recall=macro_recall,
        weighted_recall=weighted_recall,
        per_class_f1=per_class_f1,
        per_class_precision=per_class_precision,
        per_class_recall=per_class_recall,
        confusion_matrix=cm_list,
        mean_latency_ms=mean_latency,
        p95_latency_ms=p95_latency,
        token_cost_per_1k=token_cost,
        total_samples=len(predictions),
        correct_samples=correct_samples,
        predictions=predictions,
        ground_truth=ground_truth,
        input_texts=input_texts,
        raw_outputs=raw_outputs if raw_outputs is not None else [],
        latencies_ms=latencies_ms,
    )


def normalise_prediction(raw_output: str, intent_list: list[str]) -> str:
    """
    Extract intent label from raw model output.
    Strategy (in order): exact match → substring match → difflib fuzzy match → 'unknown'.
    """
    stripped = raw_output.strip().lower()

    if stripped in intent_list:
        return stripped

    for intent in intent_list:
        if intent in stripped:
            return intent

    matches = difflib.get_close_matches(stripped, intent_list, n=1, cutoff=0.6)
    if matches:
        return matches[0]

    return "unknown"


def _normalise_with_strategy(raw_output: str, intent_list: list[str]) -> tuple[str, str]:
    """Return (prediction, strategy) where strategy is exact/substring/fuzzy/unknown."""
    stripped = raw_output.strip().lower()

    if stripped in intent_list:
        return stripped, "exact"

    for intent in intent_list:
        if intent in stripped:
            return intent, "substring"

    matches = difflib.get_close_matches(stripped, intent_list, n=1, cutoff=0.6)
    if matches:
        return matches[0], "fuzzy"

    return "unknown", "unknown"


def normalise_predictions_batch(
    raw_outputs: list[str],
    intent_list: list[str],
) -> tuple[list[str], dict[str, int]]:
    """Normalise all raw model outputs and return (predictions, strategy_counts)."""
    predictions: list[str] = []
    stats: dict[str, int] = {"exact": 0, "substring": 0, "fuzzy": 0, "unknown": 0}
    for raw in raw_outputs:
        pred, strategy = _normalise_with_strategy(raw, intent_list)
        predictions.append(pred)
        stats[strategy] = stats.get(strategy, 0) + 1
    return predictions, stats


def result_to_dict(result: EvaluationResult) -> dict:
    """Serialise EvaluationResult to a JSON-safe dict."""
    return {
        "approach_name": result.approach_name,
        "accuracy": result.accuracy,
        "macro_f1": result.macro_f1,
        "weighted_f1": result.weighted_f1,
        "macro_precision": result.macro_precision,
        "weighted_precision": result.weighted_precision,
        "macro_recall": result.macro_recall,
        "weighted_recall": result.weighted_recall,
        "per_class_f1": result.per_class_f1,
        "per_class_precision": result.per_class_precision,
        "per_class_recall": result.per_class_recall,
        "confusion_matrix": result.confusion_matrix,
        "mean_latency_ms": result.mean_latency_ms,
        "p95_latency_ms": result.p95_latency_ms,
        "token_cost_per_1k": result.token_cost_per_1k,
        "total_samples": result.total_samples,
        "correct_samples": result.correct_samples,
        "predictions": result.predictions,
        "ground_truth": result.ground_truth,
        "input_texts": result.input_texts,
        "raw_outputs": result.raw_outputs,
        "latencies_ms": result.latencies_ms,
    }


def estimate_token_cost(
    texts: list[str],
    model_name: str = "gpt-3.5-turbo",
    cost_per_1k_tokens: float = 0.002,
) -> float:
    """Estimate API cost per 1000 samples using tiktoken."""
    if not texts:
        return 0.0
    try:
        import tiktoken
        try:
            enc = tiktoken.encoding_for_model(model_name)
        except KeyError:
            enc = tiktoken.get_encoding("cl100k_base")
        total_tokens = sum(len(enc.encode(t)) for t in texts)
    except ImportError:
        # Fallback: rough word-based estimate (~1.3 tokens per word)
        total_tokens = int(sum(len(t.split()) * 1.3 for t in texts))

    avg_tokens = total_tokens / len(texts)
    return avg_tokens * cost_per_1k_tokens
