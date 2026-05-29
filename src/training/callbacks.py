"""MLflow logging integration for training loops."""
from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

from src.training.config import TrainingConfig, get_device

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


def get_or_create_experiment(name: str, tracking_uri: str) -> str:
    """Return MLflow experiment ID, creating it if it does not exist."""
    try:
        import mlflow
        mlflow.set_tracking_uri(tracking_uri)
        experiment = mlflow.get_experiment_by_name(name)
        if experiment is None:
            return mlflow.create_experiment(name)
        return experiment.experiment_id
    except Exception as exc:
        logger.warning("MLflow experiment setup failed: %s", exc)
        return ""


def _timestamp() -> str:
    """Return current time as a compact string for run naming."""
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def _flatten_config(config: TrainingConfig) -> dict[str, str]:
    """Flatten all config fields with dot notation for MLflow params."""
    return {
        "model.base_model_id": config.model.base_model_id,
        "model.model_max_length": str(config.model.model_max_length),
        "lora.rank": str(config.lora.rank),
        "lora.alpha": str(config.lora.alpha),
        "lora.dropout": str(config.lora.dropout),
        "lora.target_modules": ",".join(config.lora.target_modules),
        "quantization.enabled": str(config.quantization.enabled),
        "quantization.bits": str(config.quantization.bits),
        "training.num_epochs": str(config.training.num_epochs),
        "training.batch_size": str(config.training.batch_size),
        "training.gradient_accumulation_steps": str(config.training.gradient_accumulation_steps),
        "training.learning_rate": str(config.training.learning_rate),
        "training.lr_scheduler": config.training.lr_scheduler,
        "training.warmup_ratio": str(config.training.warmup_ratio),
        "training.weight_decay": str(config.training.weight_decay),
        "training.max_grad_norm": str(config.training.max_grad_norm),
        "training.seed": str(config.training.seed),
    }


class MLflowCallback:
    """MLflow logging callback for training loops."""

    def __init__(self, tracking_uri: str, experiment_name: str) -> None:
        """Initialize with MLflow connection settings."""
        self._tracking_uri = tracking_uri
        self._experiment_name = experiment_name

    def on_train_begin(self, config: TrainingConfig) -> None:
        """Start MLflow run and log all hyperparameters as params."""
        try:
            import mlflow
            mlflow.set_tracking_uri(self._tracking_uri)
            experiment_id = get_or_create_experiment(self._experiment_name, self._tracking_uri)
            run_name = f"{config.logging.run_name}-{_timestamp()}"
            mlflow.start_run(experiment_id=experiment_id, run_name=run_name)
            mlflow.log_params(_flatten_config(config))
            mlflow.set_tags(
                {
                    "device": get_device(),
                    "base_model_id": config.model.base_model_id,
                    "framework": "mlx",
                }
            )
        except Exception as exc:
            logger.warning("MLflow on_train_begin failed: %s", exc)

    def on_step(self, step: int, loss: float, lr: float) -> None:
        """Log step-level metrics: train_loss, learning_rate."""
        try:
            import mlflow
            mlflow.log_metrics({"train_loss": loss, "learning_rate": lr}, step=step)
        except Exception as exc:
            logger.warning("MLflow on_step failed at step %d: %s", step, exc)

    def on_epoch_end(self, epoch: int, train_loss: float, val_loss: float) -> None:
        """Log epoch-level metrics: train_loss_epoch, val_loss_epoch."""
        try:
            import mlflow
            mlflow.log_metrics(
                {"train_loss_epoch": train_loss, "val_loss_epoch": val_loss},
                step=epoch,
            )
        except Exception as exc:
            logger.warning("MLflow on_epoch_end failed at epoch %d: %s", epoch, exc)

    def on_train_end(self, adapter_path: Path) -> None:
        """Log adapter artifact and end the MLflow run."""
        try:
            import mlflow
            if adapter_path.exists():
                mlflow.log_artifact(str(adapter_path), artifact_path="adapter")
            mlflow.end_run()
        except Exception as exc:
            logger.warning("MLflow on_train_end failed: %s", exc)
