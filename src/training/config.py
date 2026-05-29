"""Training configuration dataclasses and device utilities."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import yaml
from dotenv import load_dotenv

logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """Model identity and context window settings."""

    base_model_id: str
    model_max_length: int = 2048
    trust_remote_code: bool = True


@dataclass
class LoraConfig:
    """LoRA adapter structure settings."""

    rank: int = 16
    alpha: int = 32
    dropout: float = 0.05
    target_modules: list[str] = field(default_factory=list)


@dataclass
class QuantizationConfig:
    """Model weight quantization settings."""

    enabled: bool = True
    bits: int = 4


@dataclass
class TrainingParams:
    """Training loop hyperparameters."""

    num_epochs: int = 3
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    lr_scheduler: str = "cosine"
    warmup_ratio: float = 0.05
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    seed: int = 42


@dataclass
class PathsConfig:
    """File system paths for data, adapters, and experiment tracking."""

    train_data: str = "data/processed/finetune_train.jsonl"
    val_data: str = "data/processed/finetune_val.jsonl"
    output_dir: str = "models/adapters"
    mlflow_tracking_uri: str = "experiments/mlflow"


@dataclass
class LoggingConfig:
    """Logging and checkpoint frequency settings."""

    log_every_n_steps: int = 10
    eval_every_n_steps: int = 100
    save_every_n_steps: int = 200
    experiment_name: str = "intent-classification"
    run_name: str = "phi35-lora-run"


@dataclass
class TrainingConfig:
    """Composite training configuration with fully typed subconfigs."""

    model: ModelConfig
    lora: LoraConfig
    quantization: QuantizationConfig
    training: TrainingParams
    paths: PathsConfig
    logging: LoggingConfig


def load_training_config(path: Path = Path("configs/training_config.yaml")) -> TrainingConfig:
    """Load YAML and return typed TrainingConfig. Raises FileNotFoundError if missing."""
    load_dotenv()
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Training config not found at {path}. "
            "Run 'make install' to set up the project first."
        )
    with open(path) as f:
        data = yaml.safe_load(f)

    return TrainingConfig(
        model=ModelConfig(
            base_model_id=data["model"]["base_model_id"],
            model_max_length=data["model"].get("model_max_length", 2048),
            trust_remote_code=data["model"].get("trust_remote_code", True),
        ),
        lora=LoraConfig(
            rank=data["lora"]["rank"],
            alpha=data["lora"]["alpha"],
            dropout=data["lora"]["dropout"],
            target_modules=data["lora"]["target_modules"],
        ),
        quantization=QuantizationConfig(
            enabled=data["quantization"]["enabled"],
            bits=data["quantization"]["bits"],
        ),
        training=TrainingParams(
            num_epochs=data["training"]["num_epochs"],
            batch_size=data["training"]["batch_size"],
            gradient_accumulation_steps=data["training"]["gradient_accumulation_steps"],
            learning_rate=data["training"]["learning_rate"],
            lr_scheduler=data["training"]["lr_scheduler"],
            warmup_ratio=data["training"]["warmup_ratio"],
            weight_decay=data["training"]["weight_decay"],
            max_grad_norm=data["training"]["max_grad_norm"],
            seed=data["training"]["seed"],
        ),
        paths=PathsConfig(
            train_data=data["paths"]["train_data"],
            val_data=data["paths"]["val_data"],
            output_dir=data["paths"]["output_dir"],
            mlflow_tracking_uri=data["paths"]["mlflow_tracking_uri"],
        ),
        logging=LoggingConfig(
            log_every_n_steps=data["logging"]["log_every_n_steps"],
            eval_every_n_steps=data["logging"]["eval_every_n_steps"],
            save_every_n_steps=data["logging"]["save_every_n_steps"],
            experiment_name=data["logging"]["experiment_name"],
            run_name=data["logging"]["run_name"],
        ),
    )


def get_device() -> str:
    """Return best available compute device string: mlx, mps, cuda, or cpu."""
    try:
        import mlx.core as mx
        if mx.metal.is_available():
            return "mlx"
    except (ImportError, Exception):
        pass
    try:
        import torch
        if torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
    except ImportError:
        pass
    return "cpu"


def log_device_info() -> None:
    """Print which device is being used and why."""
    device = get_device()
    reasons = {
        "mlx": "Apple Silicon — MLX Metal GPU (native)",
        "mps": "Apple Silicon — PyTorch MPS (MLX unavailable)",
        "cuda": "NVIDIA GPU — CUDA",
        "cpu": "CPU only — no GPU acceleration available",
    }
    print(f"device = {device}  [{reasons[device]}]")
