# LLM Intent Classification — Design Overview

## Project Summary

This project builds an end-to-end pipeline to fine-tune a small open-source LLM for e-commerce customer support intent classification. It compares three approaches — zero-shot prompting, few-shot prompting, and QLoRA fine-tuning — and exposes all results through a Streamlit evaluation dashboard backed by MLflow experiment tracking.

The project is designed to run natively on Apple Silicon (MPS/Metal) using the `mlx-lm` framework, with PyTorch MPS as a fallback. All components are modular, testable, and production-aware.

---

## Business Goal

Demonstrate a complete applied AI engineering workflow: data acquisition → preprocessing → fine-tuning → evaluation → serving. The deliverable is a reproducible, documented project suitable for a senior AI/ML engineer portfolio targeting roles requiring LLM fine-tuning, evaluation, RAG/NLP, and production integration.

---

## Architecture Overview

```
project-root/
├── data/
│   ├── raw/                        # Downloaded dataset
│   ├── processed/                  # Cleaned, split, formatted
│   └── prompts/                    # Prompt templates
├── models/
│   ├── adapters/                   # Saved LoRA/QLoRA adapter weights
│   └── exports/                    # Merged model exports (optional)
├── experiments/
│   └── mlflow/                     # MLflow tracking store (local)
├── src/
│   ├── data/
│   │   ├── downloader.py           # Dataset download logic
│   │   ├── preprocessor.py         # Cleaning, normalisation
│   │   ├── splitter.py             # Train/val/test split
│   │   └── formatter.py            # Prompt formatting per approach
│   ├── training/
│   │   ├── config.py               # Training hyperparameters (dataclass)
│   │   ├── trainer_mlx.py          # MLX LoRA fine-tuning entry point
│   │   ├── trainer_pytorch.py      # PyTorch MPS fallback trainer
│   │   └── callbacks.py            # MLflow logging callbacks
│   ├── evaluation/
│   │   ├── evaluator.py            # Core evaluation loop
│   │   ├── metrics.py              # Accuracy, macro-F1, latency, cost
│   │   ├── baseline_zeroshot.py    # Zero-shot inference runner
│   │   ├── baseline_fewshot.py     # Few-shot inference runner
│   │   └── finetuned_runner.py     # Fine-tuned model inference
│   ├── serving/
│   │   ├── app.py                  # FastAPI application
│   │   ├── router.py               # /classify and /health endpoints
│   │   ├── schemas.py              # Pydantic request/response models
│   │   └── model_loader.py         # Model loading & caching
│   └── dashboard/
│       ├── app.py                  # Streamlit entry point
│       ├── components/
│       │   ├── benchmark_table.py  # Three-way comparison table
│       │   ├── confusion_matrix.py # Per-model confusion matrix
│       │   ├── latency_chart.py    # Latency vs accuracy scatter
│       │   └── live_predict.py     # Live inference input panel
│       └── data_loader.py          # Reads MLflow artifacts
├── tests/
│   ├── test_smoke.py
│   ├── test_data.py
│   ├── test_training.py
│   ├── test_evaluation.py
│   └── test_serving.py
├── configs/
│   ├── training_config.yaml
│   └── eval_config.yaml
├── notebooks/
│   └── exploration.ipynb
├── docker/
│   ├── Dockerfile
│   └── docker-compose.yml
├── requirements.txt
├── requirements-mlx.txt
├── README.md
├── CLAUDE.md
└── Makefile
```

---

## Dataset

- **Source:** Bitext Customer Support LLM Chatbot Training Dataset
- **HuggingFace ID:** `bitext/Bitext-customer-support-llm-chatbot-training-dataset`
- **Size:** ~26,872 labeled samples
- **Intents:** 27 intents across 10 categories (order status, returns, billing, account, shipping, etc.)
- **Columns used:** `instruction` (user utterance), `intent` (label), `category`
- **Splits:** 70% train / 15% validation / 15% test (stratified by intent)

---

## Models

### Base Model
- **Primary:** `microsoft/Phi-3.5-mini-instruct` (3.8B params — fits in 16GB unified RAM)
- **Alternative:** `mistralai/Mistral-7B-Instruct-v0.3` (requires 24GB+ for QLoRA)
- **Selection rationale:** Phi-3.5-mini delivers strong instruction-following at low memory cost on Apple Silicon.

### Fine-Tuning Method
- **Framework:** `mlx-lm` (Apple MLX) — Apple Silicon native LoRA/QLoRA
- **Technique:** LoRA (rank 16, alpha 32) — QLoRA-equivalent via MLX's native quantisation
- **Quantisation:** 4-bit via MLX quantise utilities
- **Fallback:** PyTorch MPS with HuggingFace `peft` + `trl` SFTTrainer

### Three Evaluation Approaches
1. **Zero-shot:** Base model + minimal system prompt, no examples
2. **Few-shot:** 5 examples per intent class injected into context
3. **Fine-tuned:** LoRA adapter loaded alongside base model

---

## Evaluation Metrics

| Metric | Tool | Purpose |
|---|---|---|
| Accuracy | sklearn | Overall correctness |
| Macro-F1 | sklearn | Balanced class performance |
| Per-class F1 | sklearn | Identify weak intents |
| Inference latency (ms) | `time` module | Production viability |
| P95 latency (ms) | numpy | Tail latency for SLA planning |
| Estimated token cost | tiktoken | Cloud API cost proxy |
| Confusion matrix | matplotlib | Error pattern analysis |

All metrics are logged to **MLflow** with full hyperparameter metadata per run.

---

## Technology Stack

| Layer | Tool | Version target |
|---|---|---|
| Language | Python | 3.11+ |
| Fine-tuning (primary) | mlx-lm | latest |
| Fine-tuning (fallback) | HuggingFace TRL + PEFT | latest |
| Base model hub | HuggingFace `transformers` | latest |
| Experiment tracking | MLflow | 2.x |
| API serving | FastAPI + Uvicorn | latest |
| Dashboard | Streamlit | latest |
| Containerisation | Docker | 24+ |
| Testing | pytest | latest |
| Linting | ruff | latest |
| Type checking | mypy | latest |

---

## Key Design Principles

1. **MPS/Metal first.** All training code defaults to Apple Silicon (`mlx` → `mps` → `cuda` → `cpu`).
2. **Modular files.** Every source file stays under 1000 lines. Each file has a single responsibility.
3. **Config-driven.** Hyperparameters live in YAML configs and dataclasses — never hardcoded in logic files.
4. **Reproducible.** Every experiment is seeded and tracked in MLflow. Results are reproducible from a single command.
5. **Testable.** All core logic has unit tests. Data pipelines, metric calculations, and API schemas are tested independently.
6. **Production-aware.** The FastAPI service handles errors, validates inputs, logs requests, and is containerised.

---

## Phase Summary

| Phase | Focus | Key Output |
|---|---|---|
| 1 | Project setup | Repo skeleton, dependencies, configs, Makefile |
| 2 | Data processing | Downloaded dataset, cleaned splits, formatted prompts |
| 3 | Fine-tuning | Trained LoRA adapter, MLflow run, model saved |
| 4 | Evaluation | Benchmark results for all 3 approaches, MLflow artifacts |
| 5 | Dashboard | Streamlit dashboard + FastAPI service + Docker |

See `phases.md` for sequencing and dependencies between phases.
