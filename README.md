# LLM Intent Classification — Portfolio Project

This project demonstrates a complete applied AI engineering workflow to fine-tune a small open-source LLM for e-commerce customer support intent classification. It compares three approaches — zero-shot prompting, few-shot prompting, and LoRA fine-tuning — and exposes results through a Streamlit evaluation dashboard backed by MLflow experiment tracking.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    Data Processing                          │
│  Download → Clean → Split → Format (3 approaches)          │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│                   Fine-Tuning (MLX)                         │
│  LoRA/QLoRA Training on Apple Silicon (MPS/Metal)          │
│  MLflow Tracking (hyperparams, loss, artifacts)            │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│                    Evaluation                               │
│  Zero-shot | Few-shot | Fine-tuned                         │
│  Metrics: Accuracy, F1, Latency, Cost                      │
└────────────────────┬────────────────────────────────────────┘
                     │
        ┌────────────┴────────────┐
        ▼                         ▼
┌────────────────┐      ┌──────────────────┐
│   FastAPI      │      │    Streamlit     │
│   /classify    │      │    Dashboard     │
│   /health      │      │ Benchmark Table  │
│   /intents     │      │ Confusion Matrix │
└────────────────┘      │   Live Predict   │
                        └──────────────────┘
```

---

## Prerequisites

- **macOS with Apple Silicon** (M1/M2/M3/M4) — primary target
- **Python 3.11+** installed (via pyenv or Homebrew)
- **Xcode Command Line Tools** installed (for MLX Metal compilation)
- **16GB+ unified memory** recommended
- **HuggingFace token** (for model downloads)

To install Xcode tools:
```bash
xcode-select --install
```

To generate a HuggingFace token:
1. Visit https://huggingface.co/settings/tokens
2. Create a token with "read" access
3. Copy the token and save it

---

## Quick Start

Clone the repository and run the setup:

```bash
git clone <repo-url>
cd llm-fine-tuning

# Install Python dependencies
make install
make install-mlx

# Verify everything works
make lint
make test-phase1
```

All tests should pass. If any fail, check the Prerequisites section.

---

## Phase-by-Phase Run Guide

Run each phase in order. Each `make` command will execute one complete phase.

### Phase 1: Project Setup ✓ (Complete)
```bash
make test-phase1    # Verify environment and configs
```

### Phase 2: Data Processing
```bash
make download-data  # Download Bitext dataset (~26k samples, 27 intents)
make process-data   # Clean, split (70/15/15), format for all 3 approaches
make test-phase2    # Verify data integrity
```

### Phase 3: Fine-Tuning
```bash
make finetune       # Train LoRA adapter on Apple Silicon (MLX)
                    # Falls back to PyTorch MPS if MLX unavailable
make test-phase3    # Verify trainer and adapter
```

### Phase 4: Evaluation
```bash
make evaluate       # Run zero-shot, few-shot, fine-tuned on test set
                    # Compute accuracy, F1, latency, cost metrics
make test-phase4    # Verify evaluation pipeline
```

### Phase 5: Dashboard & Serving
```bash
make serve          # Start FastAPI on port 8000
make dashboard      # Start Streamlit on port 8501

# In separate terminals:
curl -X POST http://localhost:8000/classify -H "Content-Type: application/json" \
  -d '{"text": "Where is my order?"}'

open http://localhost:8501
```

All phases are deterministic — running the same `make` command twice produces identical results.

---

## Configuration Guide

### Training Configuration

Edit `configs/training_config.yaml` to adjust:

| Setting | Purpose | Default |
|---------|---------|---------|
| `model.base_model_id` | HuggingFace model to fine-tune | `microsoft/Phi-3.5-mini-instruct` |
| `lora.rank` | LoRA rank (higher = more parameters) | `16` |
| `training.num_epochs` | Training epochs | `3` |
| `training.batch_size` | Batch size (reduce if OOM) | `4` |
| `training.learning_rate` | Learning rate | `2.0e-4` |

### Evaluation Configuration

Edit `configs/eval_config.yaml` to adjust:

| Setting | Purpose | Default |
|---------|---------|---------|
| `inference.max_new_tokens` | Max output tokens | `20` |
| `few_shot.num_examples_per_intent` | Examples per intent class | `5` |
| `approaches[*].enabled` | Which approaches to run | All enabled |

### Environment Variables

Copy `.env.example` to `.env` and fill in your values:

```bash
cp .env.example .env
```

Required variables:
- `HF_TOKEN` — Your HuggingFace API token

Optional variables:
- `MLFLOW_TRACKING_URI` — MLflow backend (default: `experiments/mlflow`)
- `API_HOST` / `API_PORT` — FastAPI address (default: `0.0.0.0:8000`)
- `STREAMLIT_PORT` — Streamlit port (default: `8501`)

---

## MLflow Experiment Tracking

After training, view metrics and artifacts:

```bash
mlflow ui --backend-store-uri experiments/mlflow
```

Then open http://localhost:5000 in your browser.

You'll see:
- Training loss curves (per epoch)
- Hyperparameters (learning rate, LoRA rank, etc.)
- LoRA adapter artifacts
- Evaluation metrics (accuracy, F1, latency)

---

## Docker Deployment

Build and run both the API and dashboard in containers:

```bash
make docker-build    # Build images
make docker-up       # Start services

# Wait ~30s for services to start
curl http://localhost:8000/health     # Verify API
open http://localhost:8501             # View dashboard
```

Services:
- **API**: http://localhost:8000 (FastAPI + Uvicorn)
- **Dashboard**: http://localhost:8501 (Streamlit)

To stop:
```bash
docker-compose -f docker/docker-compose.yml down
```

---

## Project Structure

```
project-root/
├── data/
│   ├── raw/                        # Downloaded dataset
│   ├── processed/                  # Cleaned, split, formatted
│   └── prompts/                    # Prompt templates
├── models/
│   ├── adapters/                   # LoRA adapter weights
│   ├── exports/                    # Merged model exports
│   └── training/                   # Adapter metadata
├── experiments/
│   ├── mlflow/                     # MLflow tracking store
│   └── results/                    # Benchmark results & matrices
├── src/
│   ├── data/
│   │   ├── downloader.py           # HuggingFace dataset download
│   │   ├── preprocessor.py         # Cleaning, intent extraction
│   │   ├── splitter.py             # 70/15/15 stratified split
│   │   └── formatter.py            # 3 prompt templates
│   ├── training/
│   │   ├── config.py               # Hyperparameter dataclass + device detection
│   │   ├── trainer_mlx.py          # MLX LoRA fine-tuning entry
│   │   ├── trainer_pytorch.py      # PyTorch MPS fallback
│   │   └── callbacks.py            # MLflow logging
│   ├── evaluation/
│   │   ├── evaluator.py            # Orchestration loop
│   │   ├── metrics.py              # Accuracy, F1, latency, cost
│   │   ├── baseline_zeroshot.py    # Zero-shot runner
│   │   ├── baseline_fewshot.py     # Few-shot runner
│   │   └── finetuned_runner.py     # Fine-tuned inference
│   ├── serving/
│   │   ├── app.py                  # FastAPI application
│   │   ├── router.py               # Endpoints
│   │   ├── schemas.py              # Pydantic models
│   │   └── model_loader.py         # Caching
│   └── dashboard/
│       ├── app.py                  # Streamlit entry
│       ├── data_loader.py          # MLflow artifact reader
│       └── components/
│           ├── benchmark_table.py  # 3-way comparison
│           ├── confusion_matrix.py # Per-model matrices
│           ├── latency_chart.py    # Scatter plot
│           └── live_predict.py     # Input panel
├── tests/
│   ├── test_smoke.py               # Environment checks
│   ├── test_data.py                # Data pipeline
│   ├── test_training.py            # Training + MLflow
│   ├── test_evaluation.py          # Metrics & inference
│   └── test_serving.py             # API endpoints
├── configs/
│   ├── training_config.yaml        # Hyperparameters
│   └── eval_config.yaml            # Evaluation settings
├── notebooks/
│   └── exploration.ipynb           # EDA & prototyping
├── docker/
│   ├── Dockerfile                  # Container image
│   └── docker-compose.yml          # Multi-service orchestration
├── requirements.txt                # Core ML dependencies
├── requirements-mlx.txt            # Apple Silicon (MLX)
├── pyproject.toml                  # Package config + ruff/mypy/pytest
├── Makefile                        # Developer shortcuts
├── README.md                       # This file
└── .gitignore                      # Git exclusions
```

---

## License

MIT License — see LICENSE file for details.

---

## Contact

For questions or issues, open a GitHub issue in this repository.
