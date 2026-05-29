"""Zero-shot inference on the full test set using the base model."""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

from tqdm import tqdm

from src.config_loader import EvalConfig, load_eval_config
from src.data.preprocessor import KNOWN_INTENTS
from src.evaluation.metrics import (
    EvaluationResult,
    compute_metrics,
    normalise_predictions_batch,
)
from src.training.config import get_device

logger = logging.getLogger(__name__)


def load_base_model_for_inference(config: EvalConfig) -> tuple:
    """Load base model and tokenizer (no adapter). Auto-detects framework."""
    device = get_device()

    if device == "mlx":
        from mlx_lm import load as mlx_load

        logger.info("Loading base model with MLX: %s", config.model.base_model_id)
        model, tokenizer = mlx_load(config.model.base_model_id)
        return model, tokenizer, "mlx"

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    logger.info("Loading base model with PyTorch (%s): %s", device, config.model.base_model_id)
    tokenizer = AutoTokenizer.from_pretrained(
        config.model.base_model_id,
        trust_remote_code=config.model.trust_remote_code,
    )
    model = AutoModelForCausalLM.from_pretrained(
        config.model.base_model_id,
        trust_remote_code=config.model.trust_remote_code,
        torch_dtype=torch.float16 if device in ("mps", "cuda") else torch.float32,
    )
    model = model.to(device)
    model.eval()
    return model, tokenizer, device


def _load_zeroshot_template() -> str:
    """Load the zero-shot prompt template from disk."""
    template_path = Path("data/prompts/zeroshot_template.txt")
    if not template_path.exists():
        raise FileNotFoundError(
            f"Zero-shot template not found at {template_path}. "
            "Run 'make process-data' to generate prompt templates first."
        )
    return template_path.read_text()


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


def _mlx_generate(model: Any, tokenizer: Any, prompt: str, config: EvalConfig) -> str:
    """Run greedy MLX inference for one prompt."""
    from mlx_lm import generate as mlx_generate
    from mlx_lm.sample_utils import make_sampler

    return mlx_generate(
        model,
        tokenizer,
        prompt=prompt,
        max_tokens=config.inference.max_new_tokens,
        sampler=make_sampler(config.inference.temperature),
        verbose=False,
    )


def _pytorch_generate(
    model: Any,
    tokenizer: Any,
    prompt: str,
    config: EvalConfig,
    device: str,
) -> str:
    """Run greedy PyTorch inference for one prompt."""
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
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


def run_zero_shot_inference(
    model: Any,
    tokenizer: Any,
    test_samples: list[dict],
    config: EvalConfig,
) -> tuple[list[str], list[float]]:
    """Run zero-shot inference. Returns (raw_outputs, latencies_ms)."""
    template = _load_zeroshot_template()
    intent_list_str = "\n".join(f"- {i}" for i in KNOWN_INTENTS)
    device = get_device()

    raw_outputs: list[str] = []
    latencies_ms: list[float] = []

    for sample in tqdm(test_samples, desc="Zero-shot inference"):
        prompt = template.format(
            intent_list=intent_list_str,
            instruction=sample["instruction"],
        )

        start = time.perf_counter()
        if device == "mlx":
            output = _mlx_generate(model, tokenizer, prompt, config)
        else:
            output = _pytorch_generate(model, tokenizer, prompt, config, device)
        latencies_ms.append((time.perf_counter() - start) * 1000.0)
        raw_outputs.append(output)

    return raw_outputs, latencies_ms


def evaluate_zero_shot(config: EvalConfig) -> EvaluationResult:
    """Full zero-shot evaluation pipeline. Returns EvaluationResult."""
    test_samples = _load_test_data(config)
    model, tokenizer, _framework = load_base_model_for_inference(config)

    raw_outputs, latencies_ms = run_zero_shot_inference(model, tokenizer, test_samples, config)

    predictions, stats = normalise_predictions_batch(raw_outputs, KNOWN_INTENTS)
    logger.info(
        "Normalisation: %d exact, %d substring, %d fuzzy, %d unknown",
        stats["exact"], stats["substring"], stats["fuzzy"], stats["unknown"],
    )
    print(
        f"[zero_shot] Normalisation: {stats['exact']} exact, "
        f"{stats['substring']} substring, {stats['fuzzy']} fuzzy, {stats['unknown']} unknown"
    )

    ground_truth = [s["intent"] for s in test_samples]
    input_texts = [s["instruction"] for s in test_samples]

    return compute_metrics(
        approach_name="zero_shot",
        predictions=predictions,
        ground_truth=ground_truth,
        latencies_ms=latencies_ms,
        intent_list=KNOWN_INTENTS,
        input_texts=input_texts,
        raw_outputs=raw_outputs,
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
    cfg = load_eval_config()
    result = evaluate_zero_shot(cfg)
    print(f"Zero-shot accuracy: {result.accuracy:.3f}, macro-F1: {result.macro_f1:.3f}")
