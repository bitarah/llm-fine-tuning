"""PyTorch MPS/CUDA fallback trainer using HuggingFace PEFT and TRL."""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

from src.training.callbacks import MLflowCallback
from src.training.config import TrainingConfig, get_device, load_training_config, log_device_info

logger = logging.getLogger(__name__)


# ─── Device / dtype helpers ────────────────────────────────────────────────────

def _get_torch_dtype(device: str) -> Any:
    """Return appropriate torch dtype for the target device."""
    import torch
    if device == "cuda":
        return torch.float16
    # MPS does not support bfloat16 reliably; CPU and MPS use float32
    return torch.float32


def _get_quantization_config(device: str, config: TrainingConfig) -> Any | None:
    """Return BitsAndBytesConfig for CUDA, None for MPS/CPU (unsupported)."""
    if not config.quantization.enabled or device != "cuda":
        return None
    import torch
    from transformers import BitsAndBytesConfig
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )


# ─── Model loading ─────────────────────────────────────────────────────────────

def load_model(config: TrainingConfig) -> tuple[Any, Any]:
    """Load causal LM and tokenizer with device-appropriate dtype and quantization."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = get_device()
    torch_dtype = _get_torch_dtype(device)
    quant_cfg = _get_quantization_config(device, config)

    if device == "cpu":
        logger.warning("Training on CPU — this will be very slow.")

    tokenizer = AutoTokenizer.from_pretrained(
        config.model.base_model_id,
        trust_remote_code=config.model.trust_remote_code,
        model_max_length=config.model.model_max_length,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs: dict[str, Any] = {
        "trust_remote_code": config.model.trust_remote_code,
        "torch_dtype": torch_dtype,
    }
    if quant_cfg is not None:
        model_kwargs["quantization_config"] = quant_cfg
    else:
        device_map = {"": device} if device in ("mps", "cpu") else "auto"
        model_kwargs["device_map"] = device_map

    model = AutoModelForCausalLM.from_pretrained(config.model.base_model_id, **model_kwargs)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model loaded: {total_params:,} total parameters")
    return model, tokenizer


# ─── LoRA setup ────────────────────────────────────────────────────────────────

def apply_lora(model: Any, config: TrainingConfig) -> Any:
    """Apply PEFT LoRA adapter to model. Returns the wrapped model."""
    from peft import LoraConfig, TaskType, get_peft_model

    peft_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=config.lora.rank,
        lora_alpha=config.lora.alpha,
        lora_dropout=config.lora.dropout,
        target_modules=config.lora.target_modules,
        bias="none",
    )
    model = get_peft_model(model, peft_cfg)
    trainable, total = model.get_nb_trainable_parameters()
    pct = 100.0 * trainable / total if total > 0 else 0.0
    print(f"LoRA applied, trainable params: {trainable:,} ({pct:.1f}%)")
    return model


# ─── Dataset helpers ───────────────────────────────────────────────────────────

def _validate_training_data(config: TrainingConfig) -> None:
    """Raise FileNotFoundError if fine-tune data files are missing."""
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


def _make_formatting_func(tokenizer: Any) -> Any:
    """Return a function that applies the model's chat template to a messages record."""
    def _format(record: dict) -> str:
        return tokenizer.apply_chat_template(
            record["messages"],
            tokenize=False,
            add_generation_prompt=False,
        )
    return _format


# ─── Training ──────────────────────────────────────────────────────────────────

def _make_hf_callback(mlflow_cb: MLflowCallback, config: TrainingConfig) -> Any:
    """Return a HuggingFace TrainerCallback that delegates to MLflowCallback."""
    from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments

    class _HFCallback(TrainerCallback):
        def on_log(
            self,
            args: TrainingArguments,
            state: TrainerState,
            control: TrainerControl,
            logs: dict | None = None,
            **kwargs: Any,
        ) -> None:
            if logs is None:
                return
            step = state.global_step
            if "loss" in logs:
                mlflow_cb.on_step(step, logs["loss"], logs.get("learning_rate", 0.0))

        def on_epoch_end(
            self,
            args: TrainingArguments,
            state: TrainerState,
            control: TrainerControl,
            **kwargs: Any,
        ) -> None:
            epoch = int(state.epoch or 0)
            last_log = state.log_history[-1] if state.log_history else {}
            mlflow_cb.on_epoch_end(
                epoch,
                last_log.get("loss", 0.0),
                last_log.get("eval_loss", 0.0),
            )

    return _HFCallback()


def run_training(config: TrainingConfig) -> Path:
    """Execute full PyTorch training loop. Returns path to saved adapter."""
    from datasets import load_dataset as hf_load_dataset
    from trl import SFTConfig, SFTTrainer

    _validate_training_data(config)

    output_dir = Path(config.paths.output_dir) / "final"
    output_dir.mkdir(parents=True, exist_ok=True)

    mlflow_cb = MLflowCallback(
        tracking_uri=config.paths.mlflow_tracking_uri,
        experiment_name=config.logging.experiment_name,
    )
    mlflow_cb.on_train_begin(config)

    model, tokenizer = load_model(config)
    model = apply_lora(model, config)

    dataset = hf_load_dataset(
        "json",
        data_files={
            "train": config.paths.train_data,
            "validation": config.paths.val_data,
        },
    )

    formatting_func = _make_formatting_func(tokenizer)
    hf_callback = _make_hf_callback(mlflow_cb, config)

    device = get_device()
    sft_cfg = SFTConfig(
        output_dir=str(output_dir),
        num_train_epochs=config.training.num_epochs,
        per_device_train_batch_size=config.training.batch_size,
        gradient_accumulation_steps=config.training.gradient_accumulation_steps,
        learning_rate=config.training.learning_rate,
        lr_scheduler_type=config.training.lr_scheduler,
        warmup_ratio=config.training.warmup_ratio,
        weight_decay=config.training.weight_decay,
        max_grad_norm=config.training.max_grad_norm,
        seed=config.training.seed,
        max_seq_length=config.model.model_max_length,
        logging_steps=config.logging.log_every_n_steps,
        eval_steps=config.logging.eval_every_n_steps,
        save_steps=config.logging.save_every_n_steps,
        eval_strategy="steps",
        save_strategy="steps",
        fp16=(device == "cuda"),
        bf16=False,
        report_to=[],  # disable HF's built-in reporting; we use MLflowCallback
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_cfg,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        formatting_func=formatting_func,
        callbacks=[hf_callback],
    )

    trainer.train()

    save_adapter(model, tokenizer, output_dir)
    _write_adapter_info(config, output_dir, "pytorch")
    mlflow_cb.on_train_end(output_dir)

    return output_dir


# ─── Adapter saving ────────────────────────────────────────────────────────────

def save_adapter(model: Any, tokenizer: Any, output_dir: Path) -> None:
    """Save PEFT LoRA adapter and tokenizer to output_dir."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
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
    with open(info_dir / "adapter_info.json", "w") as f:
        json.dump(info, f, indent=2)
    logger.info("Adapter info written to %s/adapter_info.json", info_dir)


# ─── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    """Parse CLI args and run PyTorch MPS/CUDA training."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")

    parser = argparse.ArgumentParser(description="Fine-tune with PyTorch MPS/CUDA")
    parser.add_argument("--config", type=Path, default=Path("configs/training_config.yaml"))
    parser.add_argument("--resume", action="store_true", help="Resume from last checkpoint")
    args = parser.parse_args()

    config = load_training_config(args.config)
    log_device_info()

    adapter_path = run_training(config)
    print(f"Training complete. Adapter saved to: {adapter_path}")


if __name__ == "__main__":
    main()
