"""Load and parse configuration from YAML files and environment variables."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import yaml
from dotenv import load_dotenv

logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """Model configuration settings."""

    base_model_id: str
    model_max_length: int = 2048
    trust_remote_code: bool = True
    adapter_path: str | None = None


@dataclass
class LoRAConfig:
    """LoRA (Low-Rank Adaptation) configuration."""

    rank: int
    alpha: int
    dropout: float
    target_modules: list[str]


@dataclass
class QuantizationConfig:
    """Quantization settings."""

    enabled: bool
    bits: int


@dataclass
class TrainingConfig:
    """Training hyperparameters and paths."""

    model: ModelConfig
    lora: LoRAConfig
    quantization: QuantizationConfig
    training: dict  # Contains num_epochs, batch_size, learning_rate, etc.
    paths: dict  # Contains train_data, val_data, output_dir, mlflow_tracking_uri
    logging: dict  # Contains log_every_n_steps, eval_every_n_steps, etc.


@dataclass
class InferenceConfig:
    """Inference settings."""

    max_new_tokens: int
    temperature: float
    do_sample: bool
    batch_size: int
    seed: int


@dataclass
class FewShotConfig:
    """Few-shot prompt configuration."""

    num_examples_per_intent: int
    example_selection: str


@dataclass
class EvaluationConfig:
    """Evaluation settings."""

    test_data: str
    results_dir: str
    mlflow_tracking_uri: str
    experiment_name: str
    max_test_samples: int | None = None


@dataclass
class EvalConfig:
    """Evaluation configuration."""

    model: ModelConfig
    inference: InferenceConfig
    few_shot: FewShotConfig
    evaluation: EvaluationConfig
    approaches: list[dict]


def load_training_config(path: Path | None = None) -> TrainingConfig:
    """Load training config from YAML file and environment variables.

    Args:
        path: Path to training_config.yaml. Defaults to configs/training_config.yaml.

    Returns:
        TrainingConfig object with all settings.

    Raises:
        FileNotFoundError: If config file does not exist.
        EnvironmentError: If required environment variables are missing.
    """
    load_dotenv()

    if path is None:
        path = Path("configs/training_config.yaml")
    else:
        path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"Training config not found at {path}. "
            "Create it with: cp configs/training_config.yaml.example configs/training_config.yaml"
        )

    with open(path) as f:
        data = yaml.safe_load(f)

    if not data:
        raise ValueError(f"Config file is empty: {path}")

    model = ModelConfig(
        base_model_id=data["model"]["base_model_id"],
        model_max_length=data["model"].get("model_max_length", 2048),
        trust_remote_code=data["model"].get("trust_remote_code", True),
    )

    lora = LoRAConfig(
        rank=data["lora"]["rank"],
        alpha=data["lora"]["alpha"],
        dropout=data["lora"]["dropout"],
        target_modules=data["lora"]["target_modules"],
    )

    quantization = QuantizationConfig(
        enabled=data["quantization"]["enabled"],
        bits=data["quantization"]["bits"],
    )

    return TrainingConfig(
        model=model,
        lora=lora,
        quantization=quantization,
        training=data["training"],
        paths=data["paths"],
        logging=data["logging"],
    )


def load_eval_config(path: Path | None = None) -> EvalConfig:
    """Load evaluation config from YAML file and environment variables.

    Args:
        path: Path to eval_config.yaml. Defaults to configs/eval_config.yaml.

    Returns:
        EvalConfig object with all settings.

    Raises:
        FileNotFoundError: If config file does not exist.
        EnvironmentError: If required environment variables are missing.
    """
    load_dotenv()

    if path is None:
        path = Path("configs/eval_config.yaml")
    else:
        path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"Evaluation config not found at {path}. "
            "Create it with: cp configs/eval_config.yaml.example configs/eval_config.yaml"
        )

    with open(path) as f:
        data = yaml.safe_load(f)

    if not data:
        raise ValueError(f"Config file is empty: {path}")

    model = ModelConfig(
        base_model_id=data["model"]["base_model_id"],
        adapter_path=data["model"].get("adapter_path"),
        trust_remote_code=data["model"].get("trust_remote_code", True),
    )

    inference = InferenceConfig(
        max_new_tokens=data["inference"]["max_new_tokens"],
        temperature=data["inference"]["temperature"],
        do_sample=data["inference"]["do_sample"],
        batch_size=data["inference"]["batch_size"],
        seed=data["inference"]["seed"],
    )

    few_shot = FewShotConfig(
        num_examples_per_intent=data["few_shot"]["num_examples_per_intent"],
        example_selection=data["few_shot"]["example_selection"],
    )

    evaluation = EvaluationConfig(
        test_data=data["evaluation"]["test_data"],
        results_dir=data["evaluation"]["results_dir"],
        mlflow_tracking_uri=data["evaluation"]["mlflow_tracking_uri"],
        experiment_name=data["evaluation"]["experiment_name"],
        max_test_samples=data["evaluation"].get("max_test_samples"),
    )

    return EvalConfig(
        model=model,
        inference=inference,
        few_shot=few_shot,
        evaluation=evaluation,
        approaches=data["approaches"],
    )


if __name__ == "__main__":
    config = load_training_config()
    print(f"Training config loaded: {config.model.base_model_id}")
    eval_config = load_eval_config()
    print(f"Eval config loaded: {eval_config.model.base_model_id}")
