"""Few-shot inference on the full test set, reusing a pre-loaded base model."""
from __future__ import annotations

import json
import logging
import random
import time
from pathlib import Path
from typing import Any

from tqdm import tqdm

from src.config_loader import EvalConfig
from src.data.preprocessor import KNOWN_INTENTS
from src.evaluation.metrics import (
    EvaluationResult,
    compute_metrics,
    normalise_predictions_batch,
)
from src.training.config import get_device

logger = logging.getLogger(__name__)


def load_few_shot_examples(path: Path) -> dict[str, list[str]]:
    """Load the few-shot example bank. Returns {intent: [examples]}."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Few-shot examples not found at {path}. "
            "Run 'make process-data' to generate prompt files first."
        )
    examples: dict[str, list[str]] = {}
    with open(path) as f:
        for line in f:
            if line.strip():
                record = json.loads(line)
                examples[record["intent"]] = record["examples"]
    return examples


def build_few_shot_prompt(
    instruction: str,
    examples: dict[str, list[str]],
    n_examples: int = 5,
    seed: int | None = None,
) -> str:
    """Build a few-shot prompt with n_examples randomly sampled diverse examples."""
    rng = random.Random(seed)
    available = [intent for intent in KNOWN_INTENTS if examples.get(intent)]
    selected_intents = rng.sample(available, min(n_examples, len(available)))

    example_lines: list[str] = []
    for intent in selected_intents:
        ex = rng.choice(examples[intent])
        example_lines.append(f"Customer message: {ex}\nIntent: {intent}")

    examples_block = "\n\n".join(example_lines)
    intent_list_str = "\n".join(f"- {i}" for i in KNOWN_INTENTS)

    return (
        "You are a customer support intent classifier. Given a customer message, "
        "classify it into exactly one of the following intents:\n\n"
        f"{intent_list_str}\n\n"
        "Here are some examples:\n\n"
        f"{examples_block}\n\n"
        "Now classify this message:\n"
        f"Customer message: {instruction}\n"
        "Intent:"
    )


def _load_test_data(config: EvalConfig) -> list[dict]:
    """Load test split JSONL, capped by max_test_samples if set."""
    test_path = Path(config.evaluation.test_data)
    if not test_path.exists():
        raise FileNotFoundError(
            f"Test data not found at {test_path}. "
            "Run 'make process-data' to generate data splits first."
        )
    with open(test_path) as f:
        samples = [json.loads(line) for line in f if line.strip()]
    if config.evaluation.max_test_samples is not None:
        samples = samples[: config.evaluation.max_test_samples]
    return samples


def _run_batch_inference(
    model: Any,
    tokenizer: Any,
    prompts: list[str],
    config: EvalConfig,
    device: str,
) -> tuple[list[str], list[float]]:
    """Run inference on all prompts. Returns (raw_outputs, latencies_ms)."""
    raw_outputs: list[str] = []
    latencies_ms: list[float] = []

    for prompt in tqdm(prompts, desc="Few-shot inference"):
        start = time.perf_counter()
        if device == "mlx":
            from mlx_lm import generate as mlx_generate
            from mlx_lm.sample_utils import make_sampler

            output = mlx_generate(
                model,
                tokenizer,
                prompt=prompt,
                max_tokens=config.inference.max_new_tokens,
                sampler=make_sampler(config.inference.temperature),
                verbose=False,
            )
        else:
            import torch

            inputs = tokenizer(prompt, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=config.inference.max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                )
            new_tokens = output_ids[0, inputs["input_ids"].shape[1]:]
            output = tokenizer.decode(new_tokens, skip_special_tokens=True)
        latencies_ms.append((time.perf_counter() - start) * 1000.0)
        raw_outputs.append(output)

    return raw_outputs, latencies_ms


def evaluate_few_shot(
    model: Any,
    tokenizer: Any,
    config: EvalConfig,
) -> EvaluationResult:
    """Full few-shot evaluation pipeline. Returns EvaluationResult."""
    fewshot_path = Path("data/prompts/fewshot_examples.jsonl")
    examples = load_few_shot_examples(fewshot_path)
    test_samples = _load_test_data(config)
    device = get_device()

    prompts = [
        build_few_shot_prompt(
            sample["instruction"],
            examples,
            n_examples=config.few_shot.num_examples_per_intent,
            seed=config.inference.seed + idx,
        )
        for idx, sample in enumerate(test_samples)
    ]

    raw_outputs, latencies_ms = _run_batch_inference(model, tokenizer, prompts, config, device)

    predictions, stats = normalise_predictions_batch(raw_outputs, KNOWN_INTENTS)
    logger.info(
        "Normalisation: %d exact, %d substring, %d fuzzy, %d unknown",
        stats["exact"], stats["substring"], stats["fuzzy"], stats["unknown"],
    )
    print(
        f"[few_shot] Normalisation: {stats['exact']} exact, "
        f"{stats['substring']} substring, {stats['fuzzy']} fuzzy, {stats['unknown']} unknown"
    )

    ground_truth = [s["intent"] for s in test_samples]
    input_texts = [s["instruction"] for s in test_samples]

    return compute_metrics(
        approach_name="few_shot",
        predictions=predictions,
        ground_truth=ground_truth,
        latencies_ms=latencies_ms,
        intent_list=KNOWN_INTENTS,
        raw_outputs=raw_outputs,
        input_texts=input_texts,
    )
