"""Load fine-tuned adapter and run inference on the test set."""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

from tqdm import tqdm

from src.config_loader import EvalConfig, load_eval_config
from src.data.formatter import SYSTEM_PROMPT
from src.data.preprocessor import KNOWN_INTENTS
from src.evaluation.metrics import (
    EvaluationResult,
    compute_metrics,
    normalise_predictions_batch,
)

logger = logging.getLogger(__name__)

_ADAPTER_INFO_PATH = Path("models/training/adapter_info.json")


def _read_adapter_info() -> dict:
    """Read adapter metadata from disk. Raises FileNotFoundError if missing."""
    if not _ADAPTER_INFO_PATH.exists():
        raise FileNotFoundError(
            f"Adapter info not found at {_ADAPTER_INFO_PATH}. "
            "Run 'make finetune' to train the model first."
        )
    with open(_ADAPTER_INFO_PATH) as f:
        return json.load(f)


def load_finetuned_model(config: EvalConfig) -> tuple:
    """Load base model with LoRA adapter. Auto-detects framework from adapter_info.json."""
    info = _read_adapter_info()
    framework = info["framework"]
    base_model_id = info["base_model_id"]
    adapter_path = Path(info["adapter_path"])

    if not adapter_path.exists():
        raise FileNotFoundError(
            f"Adapter weights not found at {adapter_path}. "
            "Run 'make finetune' to train the model first."
        )

    if framework == "mlx":
        from mlx_lm.utils import load as mlx_load

        logger.info("Loading fine-tuned model with MLX, adapter: %s", adapter_path)
        model, tokenizer = mlx_load(base_model_id, adapter_path=str(adapter_path))
        return model, tokenizer, "mlx"

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = "mps" if _has_mps() else ("cuda" if _has_cuda() else "cpu")
    logger.info("Loading fine-tuned model with PyTorch (%s)", device)
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_id,
        trust_remote_code=config.model.trust_remote_code,
    )
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        trust_remote_code=config.model.trust_remote_code,
        torch_dtype=torch.float16 if device in ("mps", "cuda") else torch.float32,
    )
    model = PeftModel.from_pretrained(base_model, str(adapter_path))
    model = model.to(device)
    model.eval()
    return model, tokenizer, device


def _has_mps() -> bool:
    """Check if MPS is available."""
    try:
        import torch
        return torch.backends.mps.is_available()
    except ImportError:
        return False


def _has_cuda() -> bool:
    """Check if CUDA is available."""
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


def build_finetuned_prompt(instruction: str, tokenizer: Any) -> str:
    """Build inference prompt matching fine-tuning chat format exactly."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Customer message: {instruction}"},
    ]
    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    # Fallback: manual formatting for tokenizers without apply_chat_template
    return (
        f"<|system|>\n{SYSTEM_PROMPT}\n"
        f"<|user|>\nCustomer message: {instruction}\n"
        "<|assistant|>\n"
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


def evaluate_finetuned(config: EvalConfig) -> EvaluationResult:
    """Full fine-tuned model evaluation pipeline. Returns EvaluationResult."""
    test_samples = _load_test_data(config)
    model, tokenizer, framework = load_finetuned_model(config)

    raw_outputs: list[str] = []
    latencies_ms: list[float] = []

    for sample in tqdm(test_samples, desc="Fine-tuned inference"):
        prompt = build_finetuned_prompt(sample["instruction"], tokenizer)
        start = time.perf_counter()

        if framework == "mlx":
            from mlx_lm import generate as mlx_generate
            from mlx_lm.sample_utils import make_sampler

            output = mlx_generate(
                model,
                tokenizer,
                prompt=prompt,
                max_tokens=config.inference.max_new_tokens,
                sampler=make_sampler(temp=config.inference.temperature),
                verbose=False,
            )
        else:
            import torch

            inputs = tokenizer(prompt, return_tensors="pt")
            inputs = {k: v.to(framework) for k, v in inputs.items()}
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

    predictions, stats = normalise_predictions_batch(raw_outputs, KNOWN_INTENTS)
    logger.info(
        "Normalisation: %d exact, %d substring, %d fuzzy, %d unknown",
        stats["exact"], stats["substring"], stats["fuzzy"], stats["unknown"],
    )
    print(
        f"[fine_tuned] Normalisation: {stats['exact']} exact, "
        f"{stats['substring']} substring, {stats['fuzzy']} fuzzy, {stats['unknown']} unknown"
    )

    ground_truth = [s["intent"] for s in test_samples]
    input_texts = [s["instruction"] for s in test_samples]

    return compute_metrics(
        approach_name="fine_tuned",
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
    result = evaluate_finetuned(cfg)
    print(f"Fine-tuned accuracy: {result.accuracy:.3f}, macro-F1: {result.macro_f1:.3f}")
