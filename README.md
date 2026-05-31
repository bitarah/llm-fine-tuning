# 🤖 LLM Intent Classification — Fine-Tuning Pipeline

> **End-to-end LLM fine-tuning on Apple Silicon for e-commerce customer support intent classification**

[![Python](https://img.shields.io/badge/python-3.11+-blue.svg)]() [![MLX](https://img.shields.io/badge/MLX-Apple%20Silicon-orange)]() [![MLflow](https://img.shields.io/badge/tracking-MLflow-blue)]() [![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

A full-stack ML engineering project that fine-tunes **Phi-3.5-mini-instruct** with LoRA/QLoRA on Apple Silicon, benchmarks it against zero-shot and few-shot baselines, and serves results through a FastAPI endpoint and Streamlit evaluation dashboard backed by MLflow experiment tracking.

## 🎯 Project Overview

This project demonstrates **end-to-end LLM engineering skills** for applied NLP:

- **Real Dataset**: Bitext customer support dataset (~26k samples, 27 intent classes)
- **3-Way Benchmark**: Zero-shot vs. few-shot vs. LoRA fine-tuned — same model, same test set
- **Apple Silicon Native**: MLX-based training with MPS/PyTorch fallback
- **Production Serving**: FastAPI REST API + Streamlit evaluation dashboard
- **Full Experiment Tracking**: MLflow logs hyperparameters, loss curves, and adapter artifacts

## 📊 Results

| Approach | Accuracy | Macro F1 | Mean Latency |
|---|---|---|---|
| Zero-shot | 69.7% | 67.1% | 2,954 ms |
| Few-shot | 70.4% | 69.3% | 3,314 ms |
| **LoRA Fine-tuned** | **99.4%** | **99.3%** | **925 ms** |

Fine-tuning with LoRA yields a **+29.7pp accuracy gain** over zero-shot while running **3.2× faster** at inference.

## ✨ Key Features

### 🗂️ Data Pipeline
- **Bitext Dataset**: 26k e-commerce support utterances across 27 intent classes
- **Stratified Splits**: 70/15/15 train/val/test with fixed seed (reproducible)
- **3 Prompt Formats**: Separate formatters for zero-shot, few-shot, and fine-tune JSONL

### 🏋️ Fine-Tuning (MLX / PyTorch)
- **LoRA**: Rank-16 adapters on all attention + MLP projection layers
- **4-bit Quantization**: QLoRA for reduced memory footprint on unified memory
- **MLflow Tracking**: Loss curves, hyperparameters, and adapter artifacts per run
- **Device-Aware**: MLX → MPS → CUDA → CPU priority chain

### 📐 Evaluation
- **3-Way Comparison**: Zero-shot, few-shot, and fine-tuned evaluated on identical 1,000-sample test set
- **Metrics**: Accuracy, macro/weighted F1, P95 latency, per-1k token cost
- **Confusion Matrices**: Per-approach, saved to `experiments/results/`

### 🚀 Serving & Dashboard
- **FastAPI**: `/classify`, `/health`, `/intents` endpoints
- **Streamlit**: Benchmark table, confusion matrix viewer, live prediction panel
- **Docker**: Multi-service compose for API + dashboard

## 🏗️ Architecture

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

## 📁 Project Structure

```
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
│
├── configs/
│   ├── training_config.yaml        # Hyperparameters
│   └── eval_config.yaml            # Evaluation settings
│
├── data/
│   ├── raw/                        # Downloaded dataset
│   ├── processed/                  # Cleaned, split, formatted
│   └── prompts/                    # Prompt templates
│
├── models/
│   ├── adapters/                   # LoRA adapter weights
│   └── training/                   # Adapter metadata
│
├── experiments/
│   ├── mlflow/                     # MLflow tracking store
│   └── results/                    # Benchmark results & matrices
│
├── tests/                          # pytest suite (mocked, no real data needed)
├── docker/                         # Dockerfile + docker-compose
├── Makefile                        # Developer shortcuts
└── requirements.txt / requirements-mlx.txt
```

## 🚀 Quick Start

### Prerequisites
- **macOS with Apple Silicon** (M1/M2/M3/M4) — primary target
- **Python 3.11+**
- **Xcode Command Line Tools** (`xcode-select --install`)
- **16GB+ unified memory** recommended
- **HuggingFace token** with read access

### Setup

```bash
git clone https://github.com/bitarah/llm-fine-tuning.git
cd llm-fine-tuning

# Copy and fill in your HuggingFace token
cp .env.example .env

# Install dependencies
make install
make install-mlx   # Apple Silicon only

# Verify environment
make lint
make test-phase1
```

### Run All Phases

```bash
# Phase 2: Download and process data
make download-data   # ~26k samples, 27 intents
make process-data    # Clean → split → format

# Phase 3: Fine-tune
make finetune        # LoRA training on MLX (falls back to PyTorch MPS)

# Phase 4: Evaluate all 3 approaches
make evaluate

# Phase 5: Serve results
make serve           # FastAPI on :8000
make dashboard       # Streamlit on :8501
```

### Docker (Easiest Setup)

```bash
make docker-build
make docker-up

# API:       http://localhost:8000
# Dashboard: http://localhost:8501

make docker-down
```

### Example API Call

```bash
curl -X POST http://localhost:8000/classify \
  -H "Content-Type: application/json" \
  -d '{"text": "Where is my order?"}'
```

## 🔬 Machine Learning Details

### Model
- **Base**: `microsoft/Phi-3.5-mini-instruct` (3.8B parameters)
- **Adapter**: LoRA rank-16, alpha-32, targeting all attention + MLP projections
- **Quantization**: 4-bit QLoRA for memory efficiency

### Training Configuration

| Hyperparameter | Value |
|---|---|
| LoRA rank | 16 |
| LoRA alpha | 32 |
| Quantization | 4-bit |
| Epochs | 3 |
| Batch size | 4 |
| Gradient accumulation | 4 steps |
| Learning rate | 2e-4 |
| LR scheduler | cosine |

### Dataset

- **Source**: [Bitext Customer Support](https://huggingface.co/datasets/bitext/Bitext-customer-support-llm-chatbot-training-dataset)
- **Size**: ~26,000 utterances
- **Classes**: 27 e-commerce intents (order status, returns, billing, shipping, etc.)
- **Split**: 70% train / 15% val / 15% test (stratified, seed 42)

## 📈 MLflow Experiment Tracking

```bash
mlflow ui --backend-store-uri experiments/mlflow
# Open http://localhost:5000
```

Tracks per-run: training loss curves, hyperparameters, LoRA adapter artifacts, and evaluation metrics.

## ⚙️ Configuration

Edit `configs/training_config.yaml` to tune hyperparameters. Edit `configs/eval_config.yaml` to adjust inference settings (token limits, few-shot examples per intent, which approaches to run).

Environment variables (`.env`):

| Variable | Purpose | Default |
|---|---|---|
| `HF_TOKEN` | HuggingFace API token | required |
| `MLFLOW_TRACKING_URI` | MLflow backend | `experiments/mlflow` |
| `API_HOST` / `API_PORT` | FastAPI address | `0.0.0.0:8000` |
| `STREAMLIT_PORT` | Dashboard port | `8501` |

## 🎓 Skills Demonstrated

**For LLM / Applied ML Engineering Roles:**
- ✅ LLM fine-tuning with LoRA/QLoRA (parameter-efficient training)
- ✅ Apple Silicon-native ML (MLX framework)
- ✅ Rigorous benchmarking (zero-shot vs. few-shot vs. fine-tuned)
- ✅ Experiment tracking with MLflow
- ✅ REST API serving with FastAPI
- ✅ Interactive evaluation dashboard (Streamlit)
- ✅ Containerized deployment (Docker Compose)
- ✅ Clean ML project structure with full test coverage
- ✅ Configuration-driven pipelines (no hardcoded values)

## 📝 License

MIT License — see [LICENSE](LICENSE) file for details.

## 👤 Author

**Bita Rahmat Zadeh**
- Portfolio: [bitarah.github.io](https://bitarah.github.io/)
- LinkedIn: [linkedin.com/in/bita-rahmat-zadeh-240a3b1b0](https://www.linkedin.com/in/bita-rahmat-zadeh-240a3b1b0/)
- GitHub: [@bitarah](https://github.com/bitarah)

## 🙏 Acknowledgments

- [Bitext](https://huggingface.co/datasets/bitext/Bitext-customer-support-llm-chatbot-training-dataset) for the customer support dataset
- [Microsoft](https://huggingface.co/microsoft/Phi-3.5-mini-instruct) for Phi-3.5-mini-instruct
- [Apple MLX](https://github.com/ml-explore/mlx) for the Apple Silicon ML framework
- [HuggingFace](https://huggingface.co/) for model hosting and the `transformers` / `peft` ecosystem

---

⭐ **Star this repo** if you find it useful for your LLM fine-tuning projects!
