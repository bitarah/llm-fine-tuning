"""FastAPI route handlers for the intent classification API."""
from __future__ import annotations

import asyncio
import functools
import json
import logging
import time
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException

from src.data.formatter import SYSTEM_PROMPT, format_zero_shot_prompt
from src.data.preprocessor import KNOWN_INTENTS
from src.evaluation.metrics import _normalise_with_strategy
from src.serving import model_loader
from src.serving.schemas import ClassifyRequest, ClassifyResponse, ErrorResponse, HealthResponse

logger = logging.getLogger(__name__)

router = APIRouter()

_BENCHMARK_PATH = Path("experiments/results/benchmark_summary.json")
_FEWSHOT_EXAMPLES_PATH = Path("data/prompts/fewshot_examples.jsonl")

_CONFIDENCE_MAP = {
    "exact": 1.0,
    "substring": 0.85,
    "fuzzy": 0.7,
    "unknown": 0.0,
}


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Report server health and model status."""
    info = model_loader.get_device_info()
    return HealthResponse(
        status="ok",
        model_loaded=model_loader.is_model_loaded(),
        device=info["device"],
        base_model_id=info["base_model_id"],
    )


@router.post("/classify", response_model=ClassifyResponse, responses={500: {"model": ErrorResponse}})
async def classify(request: ClassifyRequest) -> ClassifyResponse:
    """Classify a customer message into one of 27 intents."""
    try:
        model, tokenizer = model_loader.get_model()
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    try:
        prompt = _build_prompt(request.text, request.approach, tokenizer)
        start = time.perf_counter()
        loop = asyncio.get_event_loop()
        raw_output = await loop.run_in_executor(
            model_loader.get_mlx_executor(),
            functools.partial(_run_inference, model, tokenizer, prompt),
        )
        latency_ms = (time.perf_counter() - start) * 1000.0

        prediction, strategy = _normalise_with_strategy(raw_output, KNOWN_INTENTS)
        confidence = _CONFIDENCE_MAP.get(strategy, 0.0)

        return ClassifyResponse(
            intent=prediction,
            confidence=confidence,
            approach_used=request.approach,
            latency_ms=latency_ms,
        )
    except Exception as exc:
        logger.exception("Inference error for approach=%s", request.approach)
        raise HTTPException(
            status_code=500,
            detail=f"Inference failed: {exc}",
        ) from exc


@router.get("/intents", response_model=list[str])
def list_intents() -> list[str]:
    """Return all 27 known intent labels."""
    return KNOWN_INTENTS


@router.get("/benchmark")
def benchmark() -> dict:
    """Return benchmark_summary.json contents."""
    if not _BENCHMARK_PATH.exists():
        raise HTTPException(
            status_code=404,
            detail=(
                f"Benchmark summary not found at {_BENCHMARK_PATH}. "
                "Run 'make evaluate' to generate evaluation results first."
            ),
        )
    with open(_BENCHMARK_PATH) as f:
        return json.load(f)


def _build_prompt(text: str, approach: str, tokenizer: Any) -> str:
    """Format the input text according to the requested approach."""
    if approach == "zero_shot":
        return format_zero_shot_prompt(text, KNOWN_INTENTS)

    if approach == "few_shot":
        return _build_few_shot_prompt(text)

    # Default: fine_tuned chat format
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Customer message: {text}"},
    ]
    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    return (
        f"<|system|>\n{SYSTEM_PROMPT}\n"
        f"<|user|>\nCustomer message: {text}\n"
        "<|assistant|>\n"
    )


def _build_few_shot_prompt(text: str) -> str:
    """Build a few-shot prompt, loading examples from disk if available."""
    if not _FEWSHOT_EXAMPLES_PATH.exists():
        # Fall back to zero-shot if examples not available
        return format_zero_shot_prompt(text, KNOWN_INTENTS)

    examples: dict[str, list[str]] = {}
    with open(_FEWSHOT_EXAMPLES_PATH) as f:
        for line in f:
            if line.strip():
                record = json.loads(line)
                examples[record["intent"]] = record["examples"]

    intent_list_str = "\n".join(f"- {intent}" for intent in KNOWN_INTENTS)
    example_block = ""
    for intent, exs in list(examples.items())[:3]:
        if exs:
            example_block += f'Customer message: "{exs[0]}"\nIntent: {intent}\n\n'

    return (
        f"Classify the customer message into exactly one intent from:\n"
        f"{intent_list_str}\n\n"
        f"Examples:\n{example_block}"
        f'Customer message: "{text}"\nIntent:'
    )


def _run_inference(model: Any, tokenizer: Any, prompt: str) -> str:
    """Run model inference for a single prompt. Returns raw text output."""
    device_info = model_loader.get_device_info()

    if device_info["framework"] == "mlx":
        from mlx_lm import generate as mlx_generate
        from mlx_lm.sample_utils import make_sampler

        return mlx_generate(
            model,
            tokenizer,
            prompt=prompt,
            max_tokens=20,
            sampler=make_sampler(temp=0.0),
            verbose=False,
        )

    import torch

    device = device_info["device"]
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=20,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    new_tokens = output_ids[0, inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)
