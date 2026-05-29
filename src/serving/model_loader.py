"""Singleton model loader — loads once on startup, cached in module-level variable."""
from __future__ import annotations

import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from src.config_loader import load_eval_config
from src.training.config import get_device

logger = logging.getLogger(__name__)

_model: Any = None
_tokenizer: Any = None
_model_loaded: bool = False
_framework: str = "unknown"
_base_model_id: str = ""
_device: str = ""

# Single-thread executor so MLX model loading and inference share the same thread.
# MLX GPU streams are thread-local; mixing threads causes "no Stream(gpu,1)" errors.
_mlx_executor: ThreadPoolExecutor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="mlx")

_ADAPTER_INFO_PATH = Path("models/training/adapter_info.json")


def load_model(config_path: Path = Path("configs/eval_config.yaml")) -> None:
    """Load fine-tuned model into memory. Called once on startup. Reads adapter_info.json."""
    global _model, _tokenizer, _model_loaded, _framework, _base_model_id, _device

    if not _ADAPTER_INFO_PATH.exists():
        raise FileNotFoundError(
            f"Adapter info not found at {_ADAPTER_INFO_PATH}. "
            "Run 'make finetune' to train the model first."
        )

    with open(_ADAPTER_INFO_PATH) as f:
        info = json.load(f)

    _framework = info["framework"]
    _base_model_id = info["base_model_id"]
    adapter_path = Path(info["adapter_path"])

    if not adapter_path.exists():
        raise FileNotFoundError(
            f"Adapter weights not found at {adapter_path}. "
            "Run 'make finetune' to train the model first."
        )

    start = time.perf_counter()

    if _framework == "mlx":
        from mlx_lm.utils import load as mlx_load

        logger.info("Loading fine-tuned model with MLX, adapter: %s", adapter_path)
        _model, _tokenizer = mlx_load(_base_model_id, adapter_path=str(adapter_path))
        _device = "mlx"
    else:
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer

        device = get_device()
        if device == "mlx":
            device = "mps" if _has_mps() else ("cuda" if _has_cuda() else "cpu")
        _device = device

        logger.info("Loading fine-tuned model with PyTorch (%s): %s", device, _base_model_id)
        config = load_eval_config(config_path)
        _tokenizer = AutoTokenizer.from_pretrained(
            _base_model_id,
            trust_remote_code=config.model.trust_remote_code,
        )
        base_model = AutoModelForCausalLM.from_pretrained(
            _base_model_id,
            trust_remote_code=config.model.trust_remote_code,
            torch_dtype=torch.float16 if device in ("mps", "cuda") else torch.float32,
        )
        _model = PeftModel.from_pretrained(base_model, str(adapter_path))
        _model = _model.to(device)
        _model.eval()

    elapsed = (time.perf_counter() - start) * 1000
    logger.info("Model loaded in %.1f ms on device=%s", elapsed, _device)
    _model_loaded = True


def get_model() -> tuple:
    """Return cached (model, tokenizer). Raises RuntimeError with instructions if not loaded."""
    if not _model_loaded:
        raise RuntimeError(
            "Model is not loaded. "
            "Run 'make finetune' to train the model, then restart the server."
        )
    return _model, _tokenizer


def is_model_loaded() -> bool:
    """Return whether model has been successfully loaded."""
    return _model_loaded


def get_mlx_executor() -> ThreadPoolExecutor:
    """Return the shared single-thread executor for all MLX operations."""
    return _mlx_executor


def get_device_info() -> dict:
    """Return dict with: device, framework, base_model_id."""
    return {
        "device": _device or get_device(),
        "framework": _framework,
        "base_model_id": _base_model_id,
    }


def _has_mps() -> bool:
    """Check MPS availability."""
    try:
        import torch
        return torch.backends.mps.is_available()
    except ImportError:
        return False


def _has_cuda() -> bool:
    """Check CUDA availability."""
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False
