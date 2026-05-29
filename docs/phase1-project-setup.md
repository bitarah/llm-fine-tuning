# Phase 1 — Project Setup

## Overview

This phase creates the full project skeleton: directory structure, dependency management, configuration files, developer tooling, and a smoke test. No model or data code is written here. The goal is a clean, reproducible development environment that every subsequent phase builds on.

Read `design-overview.md` before starting to understand the full project architecture and directory layout.

---

## Prerequisites

- macOS with Apple Silicon (M1/M2/M3/M4) — primary target
- Python 3.11+ installed (pyenv or Homebrew)
- `git` available
- Xcode Command Line Tools installed (for MLX Metal compilation)
- 16GB unified memory minimum recommended

---

## Step 1 — Directory Structure

Create the full directory tree exactly as specified in `design-overview.md`. Use `mkdir -p` for nested paths. Add a `.gitkeep` in every empty leaf directory.

Required directories:
```
data/raw/
data/processed/
data/prompts/
models/adapters/
models/exports/
models/training/
experiments/mlflow/
experiments/results/confusion_matrices/
src/data/
src/training/
src/evaluation/
src/serving/
src/dashboard/components/
tests/
configs/
notebooks/
docker/
```

All `src/` subdirectories must have an `__init__.py`. The root `src/` directory must also have `__init__.py`.

---

## Step 2 — Python Package Setup

Create `pyproject.toml` to make `src` importable as `intent_pipeline`:

```toml
[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.backends.legacy:build"

[project]
name = "intent-pipeline"
version = "0.1.0"
requires-python = ">=3.11"

[tool.setuptools.packages.find]
where = ["."]
include = ["src*"]

[tool.ruff]
line-length = 100
target-version = "py311"
src = ["src"]

[tool.ruff.lint]
select = ["E", "W", "F", "I", "B", "C4", "UP"]
ignore = ["E501"]

[tool.mypy]
python_version = "3.11"
warn_return_any = true
warn_unused_configs = true
ignore_missing_imports = true

[tool.pytest.ini_options]
testpaths = ["tests"]
asyncio_mode = "auto"
addopts = "-v"
```

Install in editable mode: `pip install -e .`

---

## Step 3 — Dependencies

### `requirements.txt` (all platforms)

```
# Core ML
torch>=2.3.0
transformers>=4.45.0
datasets>=2.20.0
peft>=0.12.0
trl>=0.11.0
accelerate>=0.34.0
tokenizers>=0.19.0

# Evaluation
deepeval>=1.0.0
scikit-learn>=1.5.0
numpy>=1.26.0

# Experiment tracking
mlflow>=2.15.0

# API serving
fastapi>=0.115.0
uvicorn[standard]>=0.30.0
pydantic>=2.8.0
httpx>=0.27.0

# Dashboard
streamlit>=1.38.0
plotly>=5.24.0

# Utilities
python-dotenv>=1.0.0
pyyaml>=6.0
tiktoken>=0.7.0
tqdm>=4.66.0
pandas>=2.2.0
matplotlib>=3.9.0

# Testing
pytest>=8.3.0
pytest-asyncio>=0.24.0
pytest-cov>=5.0.0

# Linting & typing
ruff>=0.6.0
mypy>=1.11.0
```

### `requirements-mlx.txt` (Apple Silicon only — install after requirements.txt)

```
mlx>=0.18.0
mlx-lm>=0.18.0
```

---

## Step 4 — Configuration Files

### `configs/training_config.yaml`

```yaml
model:
  base_model_id: "microsoft/Phi-3.5-mini-instruct"
  model_max_length: 2048
  trust_remote_code: true

lora:
  rank: 16
  alpha: 32
  dropout: 0.05
  target_modules:
    - "q_proj"
    - "k_proj"
    - "v_proj"
    - "o_proj"
    - "gate_proj"
    - "up_proj"
    - "down_proj"

quantization:
  enabled: true
  bits: 4

training:
  num_epochs: 3
  batch_size: 4
  gradient_accumulation_steps: 4
  learning_rate: 2.0e-4
  lr_scheduler: "cosine"
  warmup_ratio: 0.05
  weight_decay: 0.01
  max_grad_norm: 1.0
  seed: 42

paths:
  train_data: "data/processed/finetune_train.jsonl"
  val_data: "data/processed/finetune_val.jsonl"
  output_dir: "models/adapters"
  mlflow_tracking_uri: "experiments/mlflow"

logging:
  log_every_n_steps: 10
  eval_every_n_steps: 100
  save_every_n_steps: 200
  experiment_name: "intent-classification"
  run_name: "phi35-lora-run"
```

### `configs/eval_config.yaml`

```yaml
model:
  base_model_id: "microsoft/Phi-3.5-mini-instruct"
  adapter_path: "models/adapters/final"
  trust_remote_code: true

inference:
  max_new_tokens: 20
  temperature: 0.0
  do_sample: false
  batch_size: 8
  seed: 42

few_shot:
  num_examples_per_intent: 5
  example_selection: "random"

evaluation:
  test_data: "data/processed/test.jsonl"
  results_dir: "experiments/results"
  mlflow_tracking_uri: "experiments/mlflow"
  experiment_name: "intent-classification-eval"

approaches:
  - name: "zero_shot"
    enabled: true
  - name: "few_shot"
    enabled: true
  - name: "fine_tuned"
    enabled: true
```

---

## Step 5 — Environment Variables

Create `.env.example`:

```dotenv
# Copy to .env and fill in values — never commit .env

HF_TOKEN=your_huggingface_token_here
MLFLOW_TRACKING_URI=experiments/mlflow
API_HOST=0.0.0.0
API_PORT=8000
LOG_LEVEL=info
BASE_MODEL_ID=microsoft/Phi-3.5-mini-instruct
ADAPTER_PATH=models/adapters/final
STREAMLIT_PORT=8501
```

Create `src/config_loader.py`:
- Load `.env` using `python-dotenv`
- Load a YAML config file by path
- Return a typed config object using `dataclasses`
- Function: `load_training_config(path: Path) -> TrainingConfig`
- Function: `load_eval_config(path: Path) -> EvalConfig`
- Raise a clear `EnvironmentError` if required variables are missing

---

## Step 6 — Makefile

```makefile
.PHONY: install install-mlx lint lint-fix typecheck test \
        test-phase1 test-phase2 test-phase3 test-phase4 test-phase5 \
        download-data process-data finetune finetune-pytorch \
        evaluate serve dashboard docker-build docker-up clean

install:
	pip install -e .
	pip install -r requirements.txt

install-mlx:
	pip install -r requirements-mlx.txt

lint:
	ruff check src/ tests/

lint-fix:
	ruff check --fix src/ tests/

typecheck:
	mypy src/ --ignore-missing-imports

test:
	pytest tests/ -v --cov=src --cov-report=term-missing

test-phase1:
	pytest tests/test_smoke.py -v

test-phase2:
	pytest tests/test_data.py -v

test-phase3:
	pytest tests/test_training.py -v

test-phase4:
	pytest tests/test_evaluation.py -v

test-phase5:
	pytest tests/test_serving.py -v

download-data:
	python -m src.data.downloader

process-data:
	python -m src.data.preprocessor
	python -m src.data.splitter
	python -m src.data.formatter

finetune:
	python -m src.training.trainer_mlx

finetune-pytorch:
	python -m src.training.trainer_pytorch

evaluate:
	python -m src.evaluation.evaluator

serve:
	uvicorn src.serving.app:app --host 0.0.0.0 --port 8000 --reload

dashboard:
	streamlit run src/dashboard/app.py --server.port 8501

docker-build:
	docker-compose -f docker/docker-compose.yml build

docker-up:
	docker-compose -f docker/docker-compose.yml up

clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf .pytest_cache .mypy_cache .ruff_cache
```

---

## Step 7 — Smoke Tests (`tests/test_smoke.py`)

```python
"""Smoke tests — verify environment and project structure."""
import sys
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
    import torch          # noqa: F401
    import transformers   # noqa: F401
    import datasets       # noqa: F401
    import mlflow         # noqa: F401
    import fastapi        # noqa: F401
    import streamlit      # noqa: F401
    import sklearn        # noqa: F401


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
        import warnings
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
    assert config.training.seed == 42
```

---

## Step 8 — README.md Structure

Write a complete `README.md` with these sections:
1. Project title and one-paragraph description
2. Architecture diagram (ASCII or Mermaid)
3. Prerequisites (Python version, Apple Silicon note, HuggingFace token)
4. Quick Start — exact commands from `git clone` to first passing test
5. Phase-by-phase run guide linking each `make` command
6. Configuration guide — how to find and override configs
7. MLflow UI — `mlflow ui --backend-store-uri experiments/mlflow`
8. Docker — `docker-compose up` instructions
9. Project structure (copy directory tree from `design-overview.md`)
10. License (MIT)

---

## Step 9 — `.gitignore`

```gitignore
__pycache__/
*.py[cod]
*.egg-info/
dist/
build/
.env
.venv/
venv/
data/raw/
data/processed/
models/
experiments/mlflow/
.pytest_cache/
.coverage
htmlcov/
.vscode/
.idea/
.DS_Store
*.npz
*.safetensors
```

---

## Acceptance Criteria Checklist

- [ ] `make install` completes without errors on Apple Silicon
- [ ] `make install-mlx` installs MLX (Apple Silicon) or prints clear skip message
- [ ] `make lint` — ruff reports zero errors
- [ ] `make test-phase1` — all smoke tests pass
- [ ] Directory tree matches `design-overview.md` exactly (`find src -type f` to verify)
- [ ] `python -c "import src"` succeeds
- [ ] Both config YAML files parse without error
- [ ] `.env.example` present; `.env` absent from repo
- [ ] `README.md` has all 10 sections
- [ ] `pyproject.toml` includes ruff, mypy, and pytest config

**Do not proceed to Phase 2 until all items are checked.**
