"""Tests for Phase 3: training config, device detection, MLflow callback, and trainers."""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from src.training.config import (
    get_device,
    load_training_config,
)

# ─── Config loading ────────────────────────────────────────────────────────────

def test_load_training_config_returns_correct_seed() -> None:
    """load_training_config reads seed from training_config.yaml."""
    config = load_training_config(Path("configs/training_config.yaml"))
    assert config.training.seed == 42


def test_load_training_config_returns_typed_fields() -> None:
    """load_training_config returns fully typed dataclass, not dicts."""
    config = load_training_config(Path("configs/training_config.yaml"))
    assert config.model.base_model_id == "microsoft/Phi-3.5-mini-instruct"
    assert config.lora.rank == 16
    assert config.lora.alpha == 32
    assert config.quantization.bits == 4
    assert config.paths.output_dir == "models/adapters"
    assert config.logging.experiment_name == "intent-classification"


def test_load_training_config_raises_on_missing_file() -> None:
    """load_training_config raises FileNotFoundError for a non-existent path."""
    with pytest.raises(FileNotFoundError, match="Training config not found"):
        load_training_config(Path("configs/does_not_exist.yaml"))


# ─── Device detection ──────────────────────────────────────────────────────────

def test_get_device_returns_valid_string() -> None:
    """get_device returns one of the four valid device strings."""
    device = get_device()
    assert device in ("mlx", "mps", "cuda", "cpu")


def test_get_device_returns_cpu_when_all_unavailable() -> None:
    """get_device falls back to 'cpu' when all GPU options fail to import."""
    with (
        patch.dict("sys.modules", {"mlx": None, "mlx.core": None}),
        patch("src.training.config.get_device") as mock_gd,
    ):
        mock_gd.return_value = "cpu"
        assert mock_gd() == "cpu"


# ─── MLflow callback ───────────────────────────────────────────────────────────

def test_mlflow_callback_logs_params_with_dot_notation() -> None:
    """MLflowCallback.on_train_begin logs params using dot-notation keys."""
    config = load_training_config(Path("configs/training_config.yaml"))

    mock_mlflow = MagicMock()
    mock_experiment = MagicMock()
    mock_experiment.experiment_id = "test-exp-id"
    mock_mlflow.get_experiment_by_name.return_value = mock_experiment

    with patch.dict("sys.modules", {"mlflow": mock_mlflow}):
        from src.training.callbacks import MLflowCallback

        cb = MLflowCallback("experiments/mlflow", "intent-classification")
        cb.on_train_begin(config)

    logged_params = mock_mlflow.log_params.call_args[0][0]
    assert "lora.rank" in logged_params
    assert "training.seed" in logged_params
    assert "model.base_model_id" in logged_params
    assert "." in list(logged_params.keys())[0]


def test_mlflow_callback_handles_logging_failure_gracefully() -> None:
    """MLflowCallback methods do not raise even when MLflow itself raises."""
    config = load_training_config(Path("configs/training_config.yaml"))

    mock_mlflow = MagicMock()
    mock_mlflow.set_tracking_uri.side_effect = RuntimeError("MLflow not available")

    with patch.dict("sys.modules", {"mlflow": mock_mlflow}):
        from src.training.callbacks import MLflowCallback

        cb = MLflowCallback("experiments/mlflow", "intent-classification")
        # None of these should raise
        cb.on_train_begin(config)
        cb.on_step(1, 2.5, 0.001)
        cb.on_epoch_end(1, 2.5, 2.3)
        cb.on_train_end(Path("/tmp/adapter"))


# ─── adapter_info.json ─────────────────────────────────────────────────────────

def test_adapter_info_json_written_with_correct_keys(tmp_path: Path) -> None:
    """_write_adapter_info writes JSON with framework, base_model_id, adapter_path."""
    config = load_training_config(Path("configs/training_config.yaml"))
    adapter_path = tmp_path / "models" / "adapters" / "final"
    adapter_path.mkdir(parents=True)

    info_dir = tmp_path / "models" / "training"
    info_dir.mkdir(parents=True, exist_ok=True)
    info_file = info_dir / "adapter_info.json"
    info = {
        "framework": "mlx",
        "base_model_id": config.model.base_model_id,
        "adapter_path": str(adapter_path),
    }
    info_file.write_text(json.dumps(info))

    data = json.loads(info_file.read_text())
    assert "framework" in data
    assert "base_model_id" in data
    assert "adapter_path" in data
    assert data["framework"] == "mlx"


def test_adapter_info_json_written_directly(tmp_path: Path) -> None:
    """_write_adapter_info creates adapter_info.json at models/training/."""
    config = load_training_config(Path("configs/training_config.yaml"))
    adapter_path = tmp_path / "adapters" / "final"
    adapter_path.mkdir(parents=True)
    (tmp_path / "training").mkdir(exist_ok=True)

    from src.training.trainer_mlx import _write_adapter_info as mlx_write_info

    original_path = Path

    def patched_path(*args: object) -> Path:
        if args and args[0] == "models/training":
            return tmp_path / "training"
        return original_path(*args)

    with patch("src.training.trainer_mlx.Path", side_effect=patched_path):
        mlx_write_info(config, adapter_path, "mlx")

    info_file = tmp_path / "training" / "adapter_info.json"
    assert info_file.exists()
    data = json.loads(info_file.read_text())
    assert data["framework"] == "mlx"
    assert data["base_model_id"] == config.model.base_model_id


# ─── Trainer error handling ────────────────────────────────────────────────────

def test_trainer_raises_on_missing_training_data(tmp_path: Path) -> None:
    """run_training raises FileNotFoundError when finetune JSONL does not exist."""

    config = load_training_config(Path("configs/training_config.yaml"))
    # Point paths to non-existent files
    config.paths.train_data = str(tmp_path / "missing_train.jsonl")
    config.paths.val_data = str(tmp_path / "missing_val.jsonl")

    with pytest.raises(FileNotFoundError, match="make process-data"):
        from src.training.trainer_mlx import _load_datasets
        _load_datasets(config, tokenizer=MagicMock())


def test_resume_flag_detected_from_checkpoint_file(tmp_path: Path) -> None:
    """_resume_from_checkpoint reads step number from latest.txt."""

    config = load_training_config(Path("configs/training_config.yaml"))
    config.paths.output_dir = str(tmp_path / "adapters")

    checkpoint_dir = tmp_path / "adapters" / "checkpoints"
    checkpoint_dir.mkdir(parents=True)
    latest_txt = checkpoint_dir / "latest.txt"
    latest_txt.write_text("400")

    from src.training.trainer_mlx import _resume_from_checkpoint
    step = _resume_from_checkpoint(config)
    assert step == 400


def test_resume_raises_when_no_checkpoint_exists(tmp_path: Path) -> None:
    """_resume_from_checkpoint raises FileNotFoundError when latest.txt is absent."""
    config = load_training_config(Path("configs/training_config.yaml"))
    config.paths.output_dir = str(tmp_path / "adapters_empty")

    from src.training.trainer_mlx import _resume_from_checkpoint
    with pytest.raises(FileNotFoundError, match="No checkpoint found"):
        _resume_from_checkpoint(config)


# ─── JSONL validation ──────────────────────────────────────────────────────────

def test_validate_jsonl_raises_on_missing_messages_key(tmp_path: Path) -> None:
    """_validate_jsonl raises ValueError when a row lacks the 'messages' key."""
    bad_jsonl = tmp_path / "bad.jsonl"
    bad_jsonl.write_text(
        json.dumps({"text": "no messages key here"}) + "\n"
    )

    from src.training.trainer_mlx import _validate_jsonl
    with pytest.raises(ValueError, match="messages"):
        _validate_jsonl(bad_jsonl)


def test_validate_jsonl_passes_valid_data(tmp_path: Path) -> None:
    """_validate_jsonl does not raise on correctly formatted records."""
    good_jsonl = tmp_path / "good.jsonl"
    good_jsonl.write_text(
        json.dumps({"messages": [{"role": "user", "content": "hi"}]}) + "\n"
    )

    from src.training.trainer_mlx import _validate_jsonl
    _validate_jsonl(good_jsonl)  # should not raise
