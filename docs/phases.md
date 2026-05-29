# Project Phases — Sequencing & Dependencies

## How to Use This Document

This file is the master sequencing guide. Each phase has its own detailed spec file. Claude Code should:

1. Read `design-overview.md` once at the start to understand the full architecture
2. Read this file to understand phase ordering and dependencies
3. Read the current phase spec file before beginning any implementation
4. After completing a phase, verify every acceptance criterion before stopping

Each phase is designed for an isolated Claude Code session with a low context window. Phase spec files are self-contained — they include all necessary context without needing prior phase specs.

---

## Phase Dependency Graph

```
Phase 1 (Setup)
    └── Phase 2 (Data)
            └── Phase 3 (Fine-Tuning)
                    └── Phase 4 (Evaluation)
                            └── Phase 5 (Dashboard)
```

Each phase depends strictly on the previous. Do not skip phases.

---

## Phase 1 — Project Setup
**Spec:** `phase1-project-setup.md`

**Goal:** Create the full repository skeleton, install all dependencies, and verify the development environment works end-to-end before any model or data work begins.

**Produces:**
- Full directory structure (as defined in `design-overview.md`)
- `requirements.txt` and `requirements-mlx.txt`
- `configs/training_config.yaml` and `configs/eval_config.yaml`
- `Makefile` with developer shortcuts
- `README.md` with setup instructions
- `.env.example` with required environment variables
- `src/config_loader.py`
- `tests/test_smoke.py` with smoke tests passing
- `ruff` and `mypy` configurations in `pyproject.toml`
- `CLAUDE.md` skill file

**Acceptance Criteria:**
- [ ] `make install` completes without errors on Apple Silicon
- [ ] `make install-mlx` installs MLX packages
- [ ] `make lint` passes (ruff zero errors)
- [ ] `make test-phase1` — all smoke tests pass
- [ ] Directory tree matches `design-overview.md` exactly
- [ ] `python -c "import src"` succeeds
- [ ] MLX import succeeds on Apple Silicon (or graceful fallback message on non-Apple hardware)

**Does NOT include:** Any data, model, or training code.

---

## Phase 2 — Data Processing
**Spec:** `phase2-data-processing.md`

**Goal:** Download, clean, split, and format the Bitext dataset into all three prompt formats (zero-shot, few-shot, fine-tuning).

**Depends on:** Phase 1 complete.

**Produces:**
- `data/raw/bitext_raw.parquet` and `data/raw/sample.jsonl`
- `data/processed/train.jsonl`, `val.jsonl`, `test.jsonl`
- `data/processed/finetune_train.jsonl`, `finetune_val.jsonl`
- `data/prompts/zeroshot_template.txt`
- `data/prompts/fewshot_examples.jsonl`
- `src/data/downloader.py`, `preprocessor.py`, `splitter.py`, `formatter.py`
- `tests/test_data.py`

**Acceptance Criteria:**
- [ ] `make download-data` downloads and saves raw dataset
- [ ] `make process-data` produces all processed files
- [ ] All 27 intent classes present in all splits
- [ ] No label leakage between train/val/test
- [ ] Fine-tune JSONL is valid for `mlx-lm`
- [ ] `make test-phase2` passes all data tests

**Does NOT include:** Any training or model loading code.

---

## Phase 3 — Fine-Tuning
**Spec:** `phase3-finetuning.md`

**Goal:** Fine-tune Phi-3.5-mini-instruct using LoRA via `mlx-lm`. Log all hyperparameters and training metrics to MLflow.

**Depends on:** Phase 2 complete.

**Produces:**
- `src/training/config.py`, `trainer_mlx.py`, `trainer_pytorch.py`, `callbacks.py`
- `models/adapters/final/` — saved LoRA adapter weights
- `models/training/adapter_info.json` — framework metadata for downstream phases
- `tests/test_training.py`
- MLflow experiment with at least one completed run

**Acceptance Criteria:**
- [ ] `make finetune` completes without error on Apple Silicon using MLX
- [ ] Training logs confirm MLX/Metal is being used
- [ ] LoRA adapter saved to `models/adapters/final/`
- [ ] `models/training/adapter_info.json` written correctly
- [ ] MLflow run contains: all hyperparameters, loss curves, adapter artifact
- [ ] Training is resumable with `--resume` flag
- [ ] `make test-phase3` passes all training tests

**Does NOT include:** Evaluation or serving code.

---

## Phase 4 — Evaluation
**Spec:** `phase4-evaluation.md`

**Goal:** Evaluate all three approaches (zero-shot, few-shot, fine-tuned) on the held-out test split. Compute and log all metrics to MLflow.

**Depends on:** Phase 3 complete.

**Produces:**
- `src/evaluation/evaluator.py`, `metrics.py`, `baseline_zeroshot.py`, `baseline_fewshot.py`, `finetuned_runner.py`
- `experiments/results/benchmark_summary.json`
- `experiments/results/zero_shot_results.json`, `few_shot_results.json`, `finetuned_results.json`
- `experiments/results/confusion_matrices/` — one PNG per approach
- `tests/test_evaluation.py`
- MLflow eval experiment with three runs (one per approach)

**Acceptance Criteria:**
- [ ] `make evaluate` runs all three approaches end-to-end without error
- [ ] All metrics (accuracy, macro-F1, latency, P95, token cost) computed for all three approaches
- [ ] `benchmark_summary.json` exists with all three approaches and all metric keys
- [ ] Three confusion matrix PNGs generated and saved
- [ ] Evaluation is deterministic — same results on re-run
- [ ] `make test-phase4` passes all evaluation tests

**Does NOT include:** Dashboard or serving code.

---

## Phase 5 — Dashboard & Serving
**Spec:** `phase5-dashboard.md`

**Goal:** Build the Streamlit evaluation dashboard, FastAPI serving endpoint, and Docker packaging.

**Depends on:** Phase 4 complete.

**Produces:**
- `src/serving/app.py`, `router.py`, `schemas.py`, `model_loader.py`
- `src/dashboard/app.py` and all `components/` files
- `src/dashboard/data_loader.py`
- `docker/Dockerfile` and `docker-compose.yml`
- `tests/test_serving.py`

**Acceptance Criteria:**
- [ ] `make serve` starts FastAPI on port 8000; `GET /health` returns 200
- [ ] `POST /classify` accepts `{"text": "..."}` and returns intent + confidence
- [ ] `GET /intents` returns all 27 intent names
- [ ] `GET /benchmark` returns summary JSON
- [ ] `make dashboard` starts Streamlit on port 8501 without error
- [ ] Dashboard Tab 1: benchmark table + scatter chart
- [ ] Dashboard Tab 2: confusion matrix images
- [ ] Dashboard Tab 3: live predict panel (with graceful fallback if API is down)
- [ ] `docker-compose up` builds and starts both services
- [ ] `make test-phase5` passes all serving tests

---

## Cross-Phase Rules

Apply across all phases and every implementation session:

1. **File line limit:** No source file in `src/` may exceed 1000 lines. Split if approaching limit.
2. **Single responsibility:** Logic files contain no I/O; I/O files contain no business logic.
3. **No hardcoded values:** Paths, model names, seeds, thresholds always come from `configs/*.yaml` or env vars.
4. **Type annotations:** All public functions and class methods must be fully typed.
5. **Docstrings:** All public functions require a one-line docstring minimum.
6. **Tests before done:** A phase is not complete until `make test-phase{N}` passes.
7. **MPS first:** Device priority is always `mlx` → `mps` → `cuda` → `cpu`.
8. **Fail fast clearly:** Missing files raise `FileNotFoundError` with the exact `make` command to fix it.
