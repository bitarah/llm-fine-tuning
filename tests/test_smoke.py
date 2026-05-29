"""Smoke tests — verify environment and project structure."""
import sys
import warnings
from pathlib import Path


def test_python_version():
    """Python 3.11 or higher is required."""
    assert sys.version_info >= (3, 11), f"Python 3.11+ required, got {sys.version}"


def test_required_directories_exist():
    """All required project directories must exist."""
    root = Path(__file__).parent.parent
    required = [
        "data/raw", "data/processed", "data/prompts",
        "models/adapters", "models/exports", "models/training",
        "experiments/mlflow", "experiments/results",
        "src/data", "src/training", "src/evaluation",
        "src/serving", "src/dashboard", "tests", "configs", "docker",
    ]
    for d in required:
        assert (root / d).is_dir(), f"Missing directory: {d}"


def test_config_files_exist():
    """Training and eval config files must exist."""
    root = Path(__file__).parent.parent
    assert (root / "configs" / "training_config.yaml").exists()
    assert (root / "configs" / "eval_config.yaml").exists()


def test_env_example_exists():
    """'.env.example' must be present."""
    root = Path(__file__).parent.parent
    assert (root / ".env.example").exists()


def test_src_importable():
    """src package must be importable."""
    import src  # noqa: F401


def test_core_dependencies_importable():
    """Core dependencies must be importable."""
    import datasets  # noqa: F401
    import fastapi  # noqa: F401
    import mlflow  # noqa: F401
    import sklearn  # noqa: F401
    import streamlit  # noqa: F401
    import torch  # noqa: F401
    import transformers  # noqa: F401


def test_mlx_or_mps_available():
    """At least one accelerator should be available (soft warning on non-Apple hardware)."""
    mlx_ok = False
    mps_ok = False
    try:
        import mlx.core as mx  # noqa: F401
        mlx_ok = True
    except ImportError:
        pass
    try:
        import torch
        mps_ok = torch.backends.mps.is_available()
    except Exception:
        pass
    if not mlx_ok and not mps_ok:
        warnings.warn(
            "Neither MLX nor MPS available — training will fall back to CPU.",
            UserWarning,
            stacklevel=2,
        )


def test_config_loader_works():
    """Config loader must load training config without error."""
    from src.config_loader import load_training_config
    config = load_training_config()
    assert config.model.base_model_id is not None
    assert config.training["seed"] == 42
