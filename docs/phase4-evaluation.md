# Phase 4 — Evaluation

## Overview

This phase evaluates all three approaches (zero-shot, few-shot, fine-tuned) on the held-out test split. It computes a comprehensive set of metrics, logs everything to MLflow, generates confusion matrix visualisations, and produces `benchmark_summary.json` that the dashboard (Phase 5) reads.

Read `design-overview.md` before starting. Phase 3 must be complete — `models/adapters/final/` and `models/training/adapter_info.json` must exist.

---

## Before You Start

```bash
make test-phase3
ls models/adapters/final/
cat models/training/adapter_info.json
```

---

## Evaluation Architecture

Three independent inference pipelines run against the same test set:

```
evaluator.py (orchestrator)
  ├── baseline_zeroshot.py   → zero-shot inference
  ├── baseline_fewshot.py    → few-shot inference
  ├── finetuned_runner.py    → fine-tuned adapter inference
  └── metrics.py             → metric computation (pure, no I/O)
```

---

## File: `src/evaluation/metrics.py`

**Responsibility:** Pure metric computation — no model loading, no file I/O.

### EvaluationResult Dataclass

```python
from dataclasses import dataclass, field

@dataclass
class EvaluationResult:
    approach_name: str
    accuracy: float
    macro_f1: float
    weighted_f1: float
    per_class_f1: dict[str, float]
    confusion_matrix: list[list[int]]       # normalised, nested list
    mean_latency_ms: float
    p95_latency_ms: float
    token_cost_per_1k: float
    total_samples: int
    correct_samples: int
    predictions: list[str]
    ground_truth: list[str]
    latencies_ms: list[float]
```

### Function Signatures

```python
def compute_metrics(
    approach_name: str,
    predictions: list[str],
    ground_truth: list[str],
    latencies_ms: list[float],
    intent_list: list[str],
    input_texts: list[str],
) -> EvaluationResult:
    """Compute all metrics. Returns populated EvaluationResult."""

def normalise_prediction(raw_output: str, intent_list: list[str]) -> str:
    """
    Extract intent label from raw model output.
    Strategy (in order): exact match → substring match → difflib fuzzy match → 'unknown'.
    """

def result_to_dict(result: EvaluationResult) -> dict:
    """Serialise EvaluationResult to a JSON-safe dict."""

def estimate_token_cost(
    texts: list[str],
    model_name: str = "gpt-3.5-turbo",
    cost_per_1k_tokens: float = 0.002,
) -> float:
    """Estimate API cost per 1000 samples using tiktoken."""
```

### Prediction Normalisation Detail

`normalise_prediction` must handle real model outputs like:
- `"cancel_order"` → exact match ✓
- `"Intent: cancel_order"` → substring match ✓
- `"The customer wants to cancel their order"` → fuzzy match (difflib) ✓
- `"xyzzy"` → `"unknown"`

Log normalisation statistics after each run:
`"Normalisation: {exact} exact, {substr} substring, {fuzzy} fuzzy, {unknown} unknown"`

---

## File: `src/evaluation/baseline_zeroshot.py`

**Responsibility:** Zero-shot inference on the full test set.

**Requirements:**
- Read `models/training/adapter_info.json` to determine framework (mlx or pytorch)
- Load base model (no adapter) using appropriate framework loader
- Load zero-shot template from `data/prompts/zeroshot_template.txt`
- For each sample: format prompt → run greedy inference → record raw output + latency
- Use `eval_config.yaml` settings: `max_new_tokens=20`, `temperature=0.0`
- Process in batches of `inference.batch_size`
- Show progress bar with `tqdm`

**Function signatures:**

```python
def load_base_model_for_inference(config: EvalConfig) -> tuple:
    """Load base model and tokenizer (no adapter). Auto-detects framework."""

def run_zero_shot_inference(
    model,
    tokenizer,
    test_samples: list[dict],
    config: EvalConfig,
) -> tuple[list[str], list[float]]:
    """Run zero-shot inference. Returns (raw_outputs, latencies_ms)."""

def evaluate_zero_shot(config: EvalConfig) -> EvaluationResult:
    """Full zero-shot evaluation pipeline. Returns EvaluationResult."""
```

---

## File: `src/evaluation/baseline_fewshot.py`

**Responsibility:** Few-shot inference on the full test set.

**Requirements:**
- Load few-shot examples from `data/prompts/fewshot_examples.jsonl`
- For each test sample: randomly sample 5 intents × 1 example each → format prompt
- The test sample's correct intent label must NOT be deterministically shown — examples are sampled randomly from all 27 intents
- Reuse the already-loaded base model (passed in, not reloaded)
- Truncate examples if prompt exceeds `model_max_length` tokens

**Function signatures:**

```python
def load_few_shot_examples(path: Path) -> dict[str, list[str]]:
    """Load the few-shot example bank. Returns {intent: [examples]}."""

def build_few_shot_prompt(
    instruction: str,
    examples: dict[str, list[str]],
    n_examples: int = 5,
    seed: int | None = None,
) -> str:
    """Build a few-shot prompt with n_examples randomly sampled diverse examples."""

def evaluate_few_shot(
    model,
    tokenizer,
    config: EvalConfig,
) -> EvaluationResult:
    """Full few-shot evaluation pipeline. Returns EvaluationResult."""
```

---

## File: `src/evaluation/finetuned_runner.py`

**Responsibility:** Load fine-tuned adapter and run inference.

**Requirements:**
- Read `models/training/adapter_info.json` to determine framework
- **MLX path:** `mlx_lm.utils.load(base_model_id, adapter_path=adapter_path)`
- **PyTorch path:** Load base model → `PeftModel.from_pretrained(base_model, adapter_path)`
- CRITICAL: Use the same chat-formatted prompt as fine-tuning (system + user messages), NOT the zero-shot template. The model will perform poorly if the inference prompt format differs from training format.

**Function signatures:**

```python
def load_finetuned_model(config: EvalConfig) -> tuple:
    """Load base model with LoRA adapter. Auto-detects framework from adapter_info.json."""

def build_finetuned_prompt(instruction: str, tokenizer) -> str:
    """Build inference prompt matching fine-tuning chat format exactly."""

def evaluate_finetuned(config: EvalConfig) -> EvaluationResult:
    """Full fine-tuned model evaluation pipeline. Returns EvaluationResult."""
```

---

## File: `src/evaluation/evaluator.py`

**Responsibility:** Orchestration only — runs all three evaluations, saves outputs, logs to MLflow. Must stay under 300 lines.

### Orchestration Flow

1. Load eval config from `configs/eval_config.yaml`
2. Load test data from `data/processed/test.jsonl`
3. Load base model once (shared by zero-shot and few-shot)
4. Run zero-shot → `EvaluationResult`
5. Run few-shot (reuse model) → `EvaluationResult`
6. Unload base model; load fine-tuned model
7. Run fine-tuned → `EvaluationResult`
8. Save all output files
9. Log all results to MLflow
10. Print benchmark table to stdout

### Output Files

```
experiments/results/benchmark_summary.json
experiments/results/zero_shot_results.json
experiments/results/few_shot_results.json
experiments/results/finetuned_results.json
experiments/results/confusion_matrices/zero_shot_cm.png
experiments/results/confusion_matrices/few_shot_cm.png
experiments/results/confusion_matrices/finetuned_cm.png
```

### `benchmark_summary.json` Schema

```json
{
  "generated_at": "2026-05-01T10:00:00Z",
  "base_model_id": "microsoft/Phi-3.5-mini-instruct",
  "test_set_size": 4030,
  "intent_count": 27,
  "approaches": {
    "zero_shot": {
      "accuracy": 0.72,
      "macro_f1": 0.71,
      "weighted_f1": 0.73,
      "mean_latency_ms": 145.3,
      "p95_latency_ms": 310.2,
      "token_cost_per_1k": 0.0018,
      "correct_samples": 2900,
      "total_samples": 4030
    },
    "few_shot": { },
    "fine_tuned": { }
  }
}
```

### Confusion Matrix Generation

```python
def plot_confusion_matrix(
    result: EvaluationResult,
    intent_list: list[str],
    output_path: Path,
) -> None:
    """Generate and save normalised confusion matrix PNG."""
```

Requirements:
- Normalised (proportion, not raw count)
- Intent labels on both axes, rotated 45°
- Title: `"{Approach} — Confusion Matrix | Accuracy: {acc:.1%}, Macro-F1: {f1:.3f}"`
- Colour map: `Blues`
- Figure size: 16×14 inches at 100 DPI

### MLflow Logging

```python
def log_all_results_to_mlflow(
    results: dict[str, EvaluationResult],
    config: EvalConfig,
) -> None:
    """Create one MLflow run per approach under the eval experiment."""
```

Each run tagged with `approach` and `base_model_id`. Log confusion matrix PNGs and `benchmark_summary.json` as artifacts.

**Module entry point:** Parse `--approaches` CLI flag (comma-separated, default: all three).

---

## File: `tests/test_evaluation.py`

Use synthetic predictions — no real models required.

**Required tests:**
```python
def test_compute_metrics_perfect_predictions():
    """All correct predictions yield accuracy=1.0 and macro_f1=1.0."""

def test_compute_metrics_macro_f1_with_known_predictions():

def test_normalise_prediction_exact_match():
    assert normalise_prediction("cancel_order", KNOWN_INTENTS) == "cancel_order"

def test_normalise_prediction_with_prefix():
    assert normalise_prediction("Intent: cancel_order", KNOWN_INTENTS) == "cancel_order"

def test_normalise_prediction_fuzzy_match():
    result = normalise_prediction("cansel_order", KNOWN_INTENTS)
    assert result == "cancel_order"

def test_normalise_prediction_returns_unknown():
    assert normalise_prediction("xyzzy", KNOWN_INTENTS) == "unknown"

def test_result_to_dict_is_json_serialisable():
    import json
    result = make_fake_result()
    json.dumps(result_to_dict(result))  # must not raise

def test_benchmark_summary_has_all_three_approaches():
def test_benchmark_summary_schema_has_required_keys():
def test_confusion_matrix_png_created(tmp_path):
def test_evaluation_is_deterministic():
    """Running metrics twice on same data yields identical results."""
```

---

## Acceptance Criteria Checklist

- [ ] `make evaluate` runs all three approaches end-to-end without error
- [ ] `experiments/results/benchmark_summary.json` exists with all three approach keys
- [ ] All metric keys present for each approach: accuracy, macro_f1, weighted_f1, mean_latency_ms, p95_latency_ms, token_cost_per_1k, correct_samples, total_samples
- [ ] Three confusion matrix PNGs exist in `experiments/results/confusion_matrices/`
- [ ] Three individual results JSON files saved
- [ ] MLflow eval experiment has three runs, each tagged with approach name
- [ ] Running `make evaluate` twice produces identical `benchmark_summary.json`
- [ ] Normalisation statistics logged for each approach
- [ ] `make test-phase4` passes all evaluation tests
- [ ] `evaluator.py` stays under 300 lines (orchestration only)

**Do not proceed to Phase 5 until all items are checked.**
