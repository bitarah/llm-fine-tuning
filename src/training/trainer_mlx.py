"""MLX LoRA fine-tuning entry point for Apple Silicon."""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

from src.training.callbacks import MLflowCallback
from src.training.config import TrainingConfig, get_device, load_training_config, log_device_info

logger = logging.getLogger(__name__)


# ─── MLX bridge callback ───────────────────────────────────────────────────────

class _MLXTrainingCallback:
    """Bridge between mlx-lm's duck-typed callback and MLflowCallback."""

    def __init__(self, mlflow_cb: MLflowCallback) -> None:
        """Init with a configured MLflowCallback."""
        self._cb = mlflow_cb
        self._epoch = 0
        self._step_losses: list[float] = []

    def on_train_loss_report(self, train_info: dict) -> None:
        """Forward step metrics to MLflowCallback."""
        step = train_info.get("iteration", 0)
        loss = float(train_info.get("train_loss", 0.0))
        lr = float(train_info.get("learning_rate", 0.0))
        self._cb.on_step(step, loss, lr)
        self._step_losses.append(loss)

    def on_val_loss_report(self, val_info: dict) -> None:
        """Forward validation metrics to MLflowCallback as epoch boundary."""
        self._epoch += 1
        val_loss = float(val_info.get("val_loss", 0.0))
        avg_train = (
            sum(self._step_losses) / len(self._step_losses) if self._step_losses else 0.0
        )
        self._cb.on_epoch_end(self._epoch, avg_train, val_loss)
        self._step_losses.clear()


# ─── Model loading ─────────────────────────────────────────────────────────────

def load_and_quantize_model(config: TrainingConfig) -> tuple[Any, Any]:
    """Load base model, apply 4-bit quantisation if enabled. Returns (model, tokenizer)."""
    from mlx_lm import load as mlx_load
    from mlx_lm.utils import quantize_model as mlx_quantize_model

    logger.info("Loading model: %s", config.model.base_model_id)
    model, tokenizer, model_cfg = mlx_load(
        config.model.base_model_id,
        tokenizer_config={"trust_remote_code": config.model.trust_remote_code},
        return_config=True,
    )

    if config.quantization.enabled:
        logger.info("Applying %d-bit quantisation", config.quantization.bits)
        model, model_cfg = mlx_quantize_model(
            model, model_cfg, group_size=64, bits=config.quantization.bits
        )

    from mlx.utils import tree_flatten as _tree_flatten
    total_params = sum(p.size for _, p in _tree_flatten(model.parameters()))
    print(f"Model loaded: {total_params:,} total parameters")

    return model, tokenizer


# ─── LoRA setup ────────────────────────────────────────────────────────────────

def apply_lora(model: Any, config: TrainingConfig) -> Any:
    """Apply LoRA layers to model. Returns modified model."""
    from mlx_lm.tuner.utils import linear_to_lora_layers

    model.freeze()

    lora_cfg = {
        "rank": config.lora.rank,
        "alpha": config.lora.alpha,
        "scale": config.lora.alpha / config.lora.rank,
        "dropout": config.lora.dropout,
    }
    linear_to_lora_layers(model, num_layers=-1, config=lora_cfg)

    from mlx.utils import tree_flatten as _tree_flatten
    trainable = sum(p.size for _, p in _tree_flatten(model.trainable_parameters()))
    total = sum(p.size for _, p in _tree_flatten(model.parameters()))
    pct = 100.0 * trainable / total if total > 0 else 0.0
    print(f"LoRA applied, trainable params: {trainable:,} ({pct:.1f}%)")

    return model


# ─── Dataset loading ───────────────────────────────────────────────────────────

def _validate_jsonl(path: Path) -> None:
    """Raise ValueError if any JSONL row is missing the 'messages' key."""
    with open(path) as f:
        for i, line in enumerate(f):
            if not line.strip():
                continue
            row = json.loads(line)
            if "messages" not in row:
                raise ValueError(
                    f"Row {i} in {path} is missing 'messages' key. "
                    "Run 'make process-data' to regenerate fine-tune JSONL files."
                )


def _load_datasets(config: TrainingConfig, tokenizer: Any) -> tuple[Any, Any]:
    """Load and validate fine-tune JSONL files into mlx-lm dataset objects."""
    train_path = Path(config.paths.train_data)
    val_path = Path(config.paths.val_data)

    if not train_path.exists():
        raise FileNotFoundError(
            f"Training data not found at {train_path}. "
            "Run 'make process-data' to generate it first."
        )
    if not val_path.exists():
        raise FileNotFoundError(
            f"Validation data not found at {val_path}. "
            "Run 'make process-data' to generate it first."
        )

    from mlx_lm.tuner.datasets import CacheDataset, ChatDataset

    _validate_jsonl(train_path)
    _validate_jsonl(val_path)

    with open(train_path) as f:
        train_data = [json.loads(line) for line in f if line.strip()]
    with open(val_path) as f:
        val_data = [json.loads(line) for line in f if line.strip()]

    logger.info("Loaded %d train / %d val examples", len(train_data), len(val_data))
    return (
        CacheDataset(ChatDataset(train_data, tokenizer)),
        CacheDataset(ChatDataset(val_data, tokenizer)),
    )


# ─── Adapter saving ────────────────────────────────────────────────────────────

def save_adapter(model: Any, tokenizer: Any, output_dir: Path) -> None:
    """Save LoRA adapter weights and tokenizer config to output_dir."""
    import mlx.core as mx
    from mlx.utils import tree_flatten

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    weights = dict(tree_flatten(model.trainable_parameters()))
    mx.save_safetensors(str(output_dir / "adapters.safetensors"), weights)

    adapter_cfg = {
        "model_type": "lora",
        "base_model_id": None,  # filled by adapter_info.json
    }
    with open(output_dir / "adapter_config.json", "w") as f:
        json.dump(adapter_cfg, f, indent=2)

    try:
        tokenizer.save_pretrained(str(output_dir))
    except Exception:
        logger.debug("Tokenizer does not support save_pretrained; skipping tokenizer save.")

    logger.info("Adapter saved to %s", output_dir)


def _write_adapter_info(config: TrainingConfig, adapter_path: Path, framework: str) -> None:
    """Write adapter_info.json to models/training/ for downstream phases."""
    info_dir = Path("models/training")
    info_dir.mkdir(parents=True, exist_ok=True)
    info = {
        "framework": framework,
        "base_model_id": config.model.base_model_id,
        "adapter_path": str(adapter_path),
    }
    info_path = info_dir / "adapter_info.json"
    with open(info_path, "w") as f:
        json.dump(info, f, indent=2)
    logger.info("Adapter info written to %s", info_path)


# ─── Training loop ─────────────────────────────────────────────────────────────

def run_training(config: TrainingConfig) -> Path:
    """Execute full MLX training loop. Returns path to saved adapter."""
    import mlx.optimizers as optim
    from mlx_lm.tuner.trainer import TrainingArgs
    from mlx_lm.tuner.trainer import train as mlx_train

    output_dir = Path(config.paths.output_dir) / "final"
    checkpoint_dir = Path(config.paths.output_dir) / "checkpoints"

    mlflow_cb = MLflowCallback(
        tracking_uri=config.paths.mlflow_tracking_uri,
        experiment_name=config.logging.experiment_name,
    )
    mlflow_cb.on_train_begin(config)
    bridge_cb = _MLXTrainingCallback(mlflow_cb)

    model, tokenizer = load_and_quantize_model(config)
    model = apply_lora(model, config)
    train_dataset, val_dataset = _load_datasets(config, tokenizer)

    steps_per_epoch = max(1, len(train_dataset) // config.training.batch_size)
    total_iters = config.training.num_epochs * steps_per_epoch

    lr_schedule = optim.cosine_decay(config.training.learning_rate, total_iters)
    optimizer = optim.AdamW(
        learning_rate=lr_schedule,
        weight_decay=config.training.weight_decay,
    )

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    adapter_file = checkpoint_dir / "adapters.safetensors"

    training_args = TrainingArgs(
        batch_size=config.training.batch_size,
        iters=total_iters,
        val_batches=max(1, len(val_dataset) // config.training.batch_size),
        steps_per_report=config.logging.log_every_n_steps,
        steps_per_eval=config.logging.eval_every_n_steps,
        steps_per_save=config.logging.save_every_n_steps,
        adapter_file=str(adapter_file),
        max_seq_length=config.model.model_max_length,
        grad_accumulation_steps=config.training.gradient_accumulation_steps,
    )

    try:
        mlx_train(
            model=model,
            optimizer=optimizer,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            args=training_args,
            training_callback=bridge_cb,
        )
    except KeyboardInterrupt:
        interrupted_dir = Path(config.paths.output_dir) / "interrupted"
        print("\nInterrupted — saving adapter to interrupted/")
        save_adapter(model, tokenizer, interrupted_dir)
        _write_adapter_info(config, interrupted_dir, "mlx")
        raise

    save_adapter(model, tokenizer, output_dir)
    _write_adapter_info(config, output_dir, "mlx")
    mlflow_cb.on_train_end(output_dir)

    return output_dir


def _resume_from_checkpoint(config: TrainingConfig) -> int:
    """Read latest checkpoint step. Raises FileNotFoundError if no checkpoint exists."""
    latest_txt = Path(config.paths.output_dir) / "checkpoints" / "latest.txt"
    if not latest_txt.exists():
        raise FileNotFoundError(
            f"No checkpoint found at {latest_txt}. "
            "Remove --resume flag to start training from scratch."
        )
    return int(latest_txt.read_text().strip())


# ─── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    """Parse CLI args and run MLX training."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")

    parser = argparse.ArgumentParser(description="Fine-tune with MLX LoRA")
    parser.add_argument("--config", type=Path, default=Path("configs/training_config.yaml"))
    parser.add_argument("--resume", action="store_true", help="Resume from latest checkpoint")
    args = parser.parse_args()

    config = load_training_config(args.config)
    log_device_info()

    device = get_device()
    if device != "mlx":
        print(
            f"WARNING: device={device}. MLX (Metal) is unavailable. "
            "Run 'make finetune-pytorch' to use the PyTorch fallback trainer."
        )
        sys.exit(1)

    if args.resume:
        step = _resume_from_checkpoint(config)
        logger.info("Resuming from step %d", step)

    adapter_path = run_training(config)
    print(f"Training complete. Adapter saved to: {adapter_path}")


if __name__ == "__main__":
    main()
