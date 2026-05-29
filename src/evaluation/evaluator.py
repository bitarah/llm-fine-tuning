"""Orchestrate all three evaluation approaches and persist results."""
from __future__ import annotations

import argparse
import gc
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from src.config_loader import EvalConfig, load_eval_config
from src.data.preprocessor import KNOWN_INTENTS
from src.evaluation.baseline_fewshot import evaluate_few_shot
from src.evaluation.baseline_zeroshot import (
    _load_test_data,
    load_base_model_for_inference,
    run_zero_shot_inference,
)
from src.evaluation.finetuned_runner import evaluate_finetuned
from src.evaluation.metrics import (
    EvaluationResult,
    compute_metrics,
    normalise_predictions_batch,
    result_to_dict,
)

matplotlib.use("Agg")
logger = logging.getLogger(__name__)


def plot_confusion_matrix(
    result: EvaluationResult,
    intent_list: list[str],
    output_path: Path,
) -> None:
    """Generate and save normalised confusion matrix PNG."""
    cm = np.array(result.confusion_matrix, dtype=float)
    row_sums = cm.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums == 0, 1, row_sums)
    cm_norm = cm / row_sums

    fig, ax = plt.subplots(figsize=(16, 14))
    im = ax.imshow(cm_norm, interpolation="nearest", cmap="Blues", vmin=0.0, vmax=1.0)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set(
        xticks=range(len(intent_list)),
        yticks=range(len(intent_list)),
        xticklabels=intent_list,
        yticklabels=intent_list,
        xlabel="Predicted",
        ylabel="True",
        title=(
            f"{result.approach_name.replace('_', ' ').title()} — Confusion Matrix | "
            f"Accuracy: {result.accuracy:.1%}, Macro-F1: {result.macro_f1:.3f}"
        ),
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    plt.setp(ax.get_yticklabels(), rotation=0)
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=100)
    plt.close(fig)
    logger.info("Confusion matrix saved to %s", output_path)


def _build_benchmark_summary(
    results: dict[str, EvaluationResult],
    config: EvalConfig,
) -> dict:
    """Assemble the benchmark_summary.json payload."""
    approaches: dict[str, dict] = {}
    for key, result in results.items():
        approaches[key] = {
            "accuracy": result.accuracy,
            "macro_precision": result.macro_precision,
            "weighted_precision": result.weighted_precision,
            "macro_recall": result.macro_recall,
            "weighted_recall": result.weighted_recall,
            "macro_f1": result.macro_f1,
            "weighted_f1": result.weighted_f1,
            "mean_latency_ms": result.mean_latency_ms,
            "p95_latency_ms": result.p95_latency_ms,
            "token_cost_per_1k": result.token_cost_per_1k,
            "correct_samples": result.correct_samples,
            "total_samples": result.total_samples,
        }
    first = next(iter(results.values()))
    return {
        "generated_at": datetime.now(tz=timezone.utc).isoformat(),  # noqa: UP017
        "base_model_id": config.model.base_model_id,
        "test_set_size": first.total_samples,
        "intent_count": len(KNOWN_INTENTS),
        "approaches": approaches,
    }


def _save_all_results(
    results: dict[str, EvaluationResult],
    summary: dict,
    results_dir: Path,
) -> None:
    """Write all JSON and PNG output files."""
    results_dir.mkdir(parents=True, exist_ok=True)
    cm_dir = results_dir / "confusion_matrices"
    cm_dir.mkdir(parents=True, exist_ok=True)

    (results_dir / "benchmark_summary.json").write_text(
        json.dumps(summary, indent=2)
    )

    file_map = {
        "zero_shot": "zero_shot_results.json",
        "few_shot": "few_shot_results.json",
        "fine_tuned": "finetuned_results.json",
    }
    cm_map = {
        "zero_shot": "zero_shot_cm.png",
        "few_shot": "few_shot_cm.png",
        "fine_tuned": "finetuned_cm.png",
    }

    for key, result in results.items():
        (results_dir / file_map[key]).write_text(
            json.dumps(result_to_dict(result), indent=2)
        )
        plot_confusion_matrix(result, KNOWN_INTENTS, cm_dir / cm_map[key])


def log_all_results_to_mlflow(
    results: dict[str, EvaluationResult],
    config: EvalConfig,
) -> None:
    """Create one MLflow run per approach under the eval experiment."""
    try:
        import mlflow

        mlflow.set_tracking_uri(config.evaluation.mlflow_tracking_uri)
        experiment = mlflow.get_experiment_by_name(config.evaluation.experiment_name)
        if experiment is None:
            experiment_id = mlflow.create_experiment(config.evaluation.experiment_name)
        else:
            experiment_id = experiment.experiment_id

        results_dir = Path(config.evaluation.results_dir)
        cm_dir = results_dir / "confusion_matrices"
        cm_map = {
            "zero_shot": "zero_shot_cm.png",
            "few_shot": "few_shot_cm.png",
            "fine_tuned": "finetuned_cm.png",
        }

        for key, result in results.items():
            with mlflow.start_run(experiment_id=experiment_id, run_name=key):
                mlflow.set_tags({"approach": key, "base_model_id": config.model.base_model_id})
                mlflow.log_metrics({
                    "accuracy": result.accuracy,
                    "macro_precision": result.macro_precision,
                    "weighted_precision": result.weighted_precision,
                    "macro_recall": result.macro_recall,
                    "weighted_recall": result.weighted_recall,
                    "macro_f1": result.macro_f1,
                    "weighted_f1": result.weighted_f1,
                    "mean_latency_ms": result.mean_latency_ms,
                    "p95_latency_ms": result.p95_latency_ms,
                    "token_cost_per_1k": result.token_cost_per_1k,
                })
                cm_path = cm_dir / cm_map.get(key, f"{key}_cm.png")
                if cm_path.exists():
                    mlflow.log_artifact(str(cm_path))
                summary_path = results_dir / "benchmark_summary.json"
                if summary_path.exists():
                    mlflow.log_artifact(str(summary_path))

        logger.info("MLflow logging complete for %d approaches", len(results))
    except Exception as exc:
        logger.warning("MLflow logging failed: %s", exc)


def _print_benchmark_table(results: dict[str, EvaluationResult]) -> None:
    """Print a formatted comparison table to stdout."""
    header = (
        f"{'Approach':<15} {'Accuracy':>10} {'Precision':>10} {'Recall':>8}"
        f" {'Macro-F1':>10} {'Wt-F1':>8} {'Latency(ms)':>12}"
    )
    print("\n" + "=" * len(header))
    print(header)
    print("-" * len(header))
    for key, r in results.items():
        print(
            f"{key:<15} {r.accuracy:>10.3f} {r.macro_precision:>10.3f}"
            f" {r.macro_recall:>8.3f} {r.macro_f1:>10.3f}"
            f" {r.weighted_f1:>8.3f} {r.mean_latency_ms:>12.1f}"
        )
    print("=" * len(header) + "\n")


def run_evaluation(config: EvalConfig, approaches: list[str]) -> dict[str, EvaluationResult]:
    """Run selected evaluation approaches and return results keyed by approach name."""
    results: dict[str, EvaluationResult] = {}
    results_dir = Path(config.evaluation.results_dir)

    model, tokenizer, _fw = load_base_model_for_inference(config)
    test_samples = _load_test_data(config)
    ground_truth = [s["intent"] for s in test_samples]
    input_texts = [s["instruction"] for s in test_samples]

    if "zero_shot" in approaches:
        print("Running zero-shot evaluation...")
        raw_outputs, latencies = run_zero_shot_inference(model, tokenizer, test_samples, config)
        predictions, stats = normalise_predictions_batch(raw_outputs, KNOWN_INTENTS)
        logger.info(
            "Normalisation: %d exact, %d substring, %d fuzzy, %d unknown",
            stats["exact"], stats["substring"], stats["fuzzy"], stats["unknown"],
        )
        print(
            f"[zero_shot] Normalisation: {stats['exact']} exact, "
            f"{stats['substring']} substring, {stats['fuzzy']} fuzzy, {stats['unknown']} unknown"
        )
        results["zero_shot"] = compute_metrics(
            "zero_shot", predictions, ground_truth, latencies, KNOWN_INTENTS, input_texts,
            raw_outputs=raw_outputs,
        )

    if "few_shot" in approaches:
        print("Running few-shot evaluation...")
        results["few_shot"] = evaluate_few_shot(model, tokenizer, config)

    del model, tokenizer
    gc.collect()

    if "fine_tuned" in approaches:
        print("Running fine-tuned evaluation...")
        results["fine_tuned"] = evaluate_finetuned(config)

    summary = _build_benchmark_summary(results, config)
    _save_all_results(results, summary, results_dir)
    log_all_results_to_mlflow(results, config)
    _print_benchmark_table(results)

    return results


def main() -> None:
    """Parse CLI args and run evaluation."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
    parser = argparse.ArgumentParser(description="Evaluate intent classification approaches")
    parser.add_argument(
        "--approaches",
        type=str,
        default="zero_shot,few_shot,fine_tuned",
        help="Comma-separated list of approaches to evaluate",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/eval_config.yaml"),
    )
    args = parser.parse_args()

    config = load_eval_config(args.config)
    approaches = [a.strip() for a in args.approaches.split(",")]
    run_evaluation(config, approaches)


if __name__ == "__main__":
    main()
