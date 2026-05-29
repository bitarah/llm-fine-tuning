"""Tests for src/evaluation — uses synthetic data, no real models."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from src.data.preprocessor import KNOWN_INTENTS
from src.evaluation.metrics import (
    EvaluationResult,
    compute_metrics,
    estimate_token_cost,
    normalise_prediction,
    result_to_dict,
)

# ─── Helpers ──────────────────────────────────────────────────────────────────

def make_fake_result(
    approach_name: str = "zero_shot",
    n: int = 27,
) -> EvaluationResult:
    """Build a minimal EvaluationResult using one correct prediction per intent."""
    predictions = KNOWN_INTENTS[:n]
    ground_truth = KNOWN_INTENTS[:n]
    latencies = [10.0] * n
    input_texts = [f"help with {i}" for i in predictions]
    return compute_metrics(
        approach_name=approach_name,
        predictions=predictions,
        ground_truth=ground_truth,
        latencies_ms=latencies,
        intent_list=KNOWN_INTENTS,
        input_texts=input_texts,
    )


# ─── compute_metrics tests ────────────────────────────────────────────────────

def test_compute_metrics_perfect_predictions():
    """All correct predictions yield accuracy=1.0 and macro_f1=1.0."""
    result = make_fake_result()
    assert result.accuracy == pytest.approx(1.0)
    assert result.macro_f1 == pytest.approx(1.0)


def test_compute_metrics_macro_f1_with_known_predictions():
    """Half-correct predictions yield accuracy < 1.0."""
    n = len(KNOWN_INTENTS)
    # First half correct, second half wrong (assign first intent to all wrong)
    predictions = list(KNOWN_INTENTS[:n // 2]) + [KNOWN_INTENTS[0]] * (n - n // 2)
    ground_truth = list(KNOWN_INTENTS)
    latencies = [5.0] * n
    input_texts = [f"q{i}" for i in range(n)]
    result = compute_metrics(
        approach_name="test",
        predictions=predictions,
        ground_truth=ground_truth,
        latencies_ms=latencies,
        intent_list=KNOWN_INTENTS,
        input_texts=input_texts,
    )
    assert result.accuracy < 1.0
    assert result.macro_f1 < 1.0
    assert result.correct_samples == n // 2


def test_compute_metrics_latency_stats():
    """Mean and P95 latency are computed correctly."""
    predictions = KNOWN_INTENTS
    ground_truth = KNOWN_INTENTS
    latencies = [float(i + 1) for i in range(27)]  # 1ms to 27ms
    result = compute_metrics(
        approach_name="test",
        predictions=list(predictions),
        ground_truth=list(ground_truth),
        latencies_ms=latencies,
        intent_list=KNOWN_INTENTS,
        input_texts=["x"] * 27,
    )
    assert result.mean_latency_ms == pytest.approx(14.0, abs=0.1)
    assert result.p95_latency_ms >= 25.0


def test_compute_metrics_confusion_matrix_shape():
    """Confusion matrix has shape (n_intents, n_intents)."""
    result = make_fake_result()
    n = len(KNOWN_INTENTS)
    assert len(result.confusion_matrix) == n
    assert all(len(row) == n for row in result.confusion_matrix)


# ─── normalise_prediction tests ───────────────────────────────────────────────

def test_normalise_prediction_exact_match():
    assert normalise_prediction("cancel_order", KNOWN_INTENTS) == "cancel_order"


def test_normalise_prediction_with_prefix():
    assert normalise_prediction("Intent: cancel_order", KNOWN_INTENTS) == "cancel_order"


def test_normalise_prediction_fuzzy_match():
    result = normalise_prediction("cansel_order", KNOWN_INTENTS)
    assert result == "cancel_order"


def test_normalise_prediction_returns_unknown():
    assert normalise_prediction("xyzzy", KNOWN_INTENTS) == "unknown"


def test_normalise_prediction_case_insensitive():
    assert normalise_prediction("CANCEL_ORDER", KNOWN_INTENTS) == "cancel_order"


def test_normalise_prediction_strips_whitespace():
    assert normalise_prediction("  track_order  ", KNOWN_INTENTS) == "track_order"


def test_normalise_prediction_substring_in_sentence():
    assert normalise_prediction("The intent is place_order here", KNOWN_INTENTS) == "place_order"


# ─── result_to_dict tests ─────────────────────────────────────────────────────

def test_result_to_dict_is_json_serialisable():
    result = make_fake_result()
    d = result_to_dict(result)
    json.dumps(d)  # must not raise


def test_result_to_dict_has_required_keys():
    result = make_fake_result()
    d = result_to_dict(result)
    required = {
        "approach_name", "accuracy", "macro_f1", "weighted_f1",
        "mean_latency_ms", "p95_latency_ms", "token_cost_per_1k",
        "total_samples", "correct_samples",
    }
    assert required.issubset(d.keys())


# ─── estimate_token_cost tests ────────────────────────────────────────────────

def test_estimate_token_cost_positive():
    texts = ["Hello world", "I need help with my order"]
    cost = estimate_token_cost(texts)
    assert cost > 0.0


def test_estimate_token_cost_empty_list():
    assert estimate_token_cost([]) == 0.0


# ─── benchmark_summary tests ──────────────────────────────────────────────────

def _make_summary(results: dict) -> dict:
    """Build a benchmark summary dict from fake results."""
    from src.config_loader import load_eval_config
    from src.evaluation.evaluator import _build_benchmark_summary

    config = load_eval_config()
    return _build_benchmark_summary(results, config)


def test_benchmark_summary_has_all_three_approaches():
    results = {
        "zero_shot": make_fake_result("zero_shot"),
        "few_shot": make_fake_result("few_shot"),
        "fine_tuned": make_fake_result("fine_tuned"),
    }
    summary = _make_summary(results)
    assert "zero_shot" in summary["approaches"]
    assert "few_shot" in summary["approaches"]
    assert "fine_tuned" in summary["approaches"]


def test_benchmark_summary_schema_has_required_keys():
    results = {
        "zero_shot": make_fake_result("zero_shot"),
        "few_shot": make_fake_result("few_shot"),
        "fine_tuned": make_fake_result("fine_tuned"),
    }
    summary = _make_summary(results)
    required_metric_keys = {
        "accuracy", "macro_f1", "weighted_f1",
        "mean_latency_ms", "p95_latency_ms",
        "token_cost_per_1k", "correct_samples", "total_samples",
    }
    for approach_key, approach_data in summary["approaches"].items():
        missing = required_metric_keys - approach_data.keys()
        assert not missing, f"{approach_key} is missing keys: {missing}"


def test_benchmark_summary_is_json_serialisable():
    results = {
        "zero_shot": make_fake_result("zero_shot"),
        "few_shot": make_fake_result("few_shot"),
        "fine_tuned": make_fake_result("fine_tuned"),
    }
    summary = _make_summary(results)
    json.dumps(summary)  # must not raise


def test_benchmark_summary_top_level_keys():
    results = {"zero_shot": make_fake_result("zero_shot")}
    summary = _make_summary(results)
    for key in ("generated_at", "base_model_id", "test_set_size", "intent_count", "approaches"):
        assert key in summary


# ─── confusion matrix PNG tests ───────────────────────────────────────────────

def test_confusion_matrix_png_created(tmp_path: Path):
    from src.evaluation.evaluator import plot_confusion_matrix

    result = make_fake_result()
    output_path = tmp_path / "test_cm.png"
    plot_confusion_matrix(result, KNOWN_INTENTS, output_path)
    assert output_path.exists()
    assert output_path.stat().st_size > 0


def test_confusion_matrix_png_title_contains_accuracy(tmp_path: Path):
    """Smoke test: plot_confusion_matrix runs without error for all approaches."""
    from src.evaluation.evaluator import plot_confusion_matrix

    for approach in ("zero_shot", "few_shot", "fine_tuned"):
        result = make_fake_result(approach_name=approach)
        out = tmp_path / f"{approach}_cm.png"
        plot_confusion_matrix(result, KNOWN_INTENTS, out)
        assert out.exists()


# ─── determinism test ─────────────────────────────────────────────────────────

def test_evaluation_is_deterministic():
    """Running metrics twice on same data yields identical results."""
    predictions = list(KNOWN_INTENTS)
    ground_truth = list(KNOWN_INTENTS)
    latencies = [10.0] * len(KNOWN_INTENTS)
    input_texts = ["q"] * len(KNOWN_INTENTS)

    r1 = compute_metrics("test", predictions, ground_truth, latencies, KNOWN_INTENTS, input_texts)
    r2 = compute_metrics("test", predictions, ground_truth, latencies, KNOWN_INTENTS, input_texts)

    assert r1.accuracy == r2.accuracy
    assert r1.macro_f1 == r2.macro_f1
    assert r1.mean_latency_ms == r2.mean_latency_ms
    assert r1.confusion_matrix == r2.confusion_matrix


# ─── load_few_shot_examples test ─────────────────────────────────────────────

def test_load_few_shot_examples(tmp_path: Path):
    from src.evaluation.baseline_fewshot import load_few_shot_examples

    fewshot_file = tmp_path / "fewshot_examples.jsonl"
    records = [
        {"intent": "cancel_order", "examples": ["I want to cancel", "Please cancel my order"]},
        {"intent": "track_order", "examples": ["Where is my package?"]},
    ]
    with open(fewshot_file, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    examples = load_few_shot_examples(fewshot_file)
    assert examples["cancel_order"] == ["I want to cancel", "Please cancel my order"]
    assert examples["track_order"] == ["Where is my package?"]


def test_load_few_shot_examples_raises_if_missing(tmp_path: Path):
    from src.evaluation.baseline_fewshot import load_few_shot_examples

    with pytest.raises(FileNotFoundError):
        load_few_shot_examples(tmp_path / "nonexistent.jsonl")


# ─── build_few_shot_prompt test ───────────────────────────────────────────────

def test_build_few_shot_prompt_is_deterministic():
    from src.evaluation.baseline_fewshot import build_few_shot_prompt

    examples = {intent: [f"help with {intent}"] for intent in KNOWN_INTENTS}
    p1 = build_few_shot_prompt("I need help", examples, n_examples=5, seed=42)
    p2 = build_few_shot_prompt("I need help", examples, n_examples=5, seed=42)
    assert p1 == p2


def test_build_few_shot_prompt_contains_instruction():
    from src.evaluation.baseline_fewshot import build_few_shot_prompt

    examples = {intent: [f"example for {intent}"] for intent in KNOWN_INTENTS}
    prompt = build_few_shot_prompt("Cancel my subscription", examples, seed=1)
    assert "Cancel my subscription" in prompt
