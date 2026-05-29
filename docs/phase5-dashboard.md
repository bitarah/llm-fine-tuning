# Phase 5 — Dashboard & Serving

## Overview

This phase builds the Streamlit evaluation dashboard, FastAPI serving endpoint, and Docker packaging. The dashboard reads from `experiments/results/benchmark_summary.json` produced in Phase 4. The FastAPI service loads the fine-tuned model and serves live predictions.

Read `design-overview.md` before starting. Phase 4 must be complete.

---

## Before You Start

```bash
make test-phase4
cat experiments/results/benchmark_summary.json   # must exist
ls experiments/results/confusion_matrices/        # must show 3 PNGs
```

---

## Part A — FastAPI Serving

### File: `src/serving/schemas.py`

**Responsibility:** Pydantic request/response models only — no logic.

```python
from pydantic import BaseModel, Field

class ClassifyRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=1000)
    approach: str = Field(default="fine_tuned")  # zero_shot | few_shot | fine_tuned

class ClassifyResponse(BaseModel):
    intent: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    approach_used: str
    latency_ms: float

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    device: str
    base_model_id: str

class ErrorResponse(BaseModel):
    error: str
    detail: str
```

### File: `src/serving/model_loader.py`

**Responsibility:** Singleton model loader — loads once on startup, cached in module-level variable.

```python
_model = None
_tokenizer = None
_model_loaded: bool = False

def load_model(config_path: Path = Path("configs/eval_config.yaml")) -> None:
    """Load fine-tuned model into memory. Called once on startup. Reads adapter_info.json."""

def get_model() -> tuple:
    """Return cached (model, tokenizer). Raises RuntimeError with instructions if not loaded."""

def is_model_loaded() -> bool:
    """Return whether model has been successfully loaded."""

def get_device_info() -> dict:
    """Return dict with: device, framework, base_model_id."""
```

- Auto-detect framework from `models/training/adapter_info.json`
- Log load time and device on startup
- `RuntimeError` message must tell user to run `make finetune` if adapter is missing

### File: `src/serving/router.py`

**Responsibility:** FastAPI route handlers.

#### `GET /health` → `HealthResponse`
Always returns 200. Reports `model_loaded: false` if model is not ready (never 503).

#### `POST /classify` → `ClassifyResponse`
- Route to zero_shot / few_shot / fine_tuned inference based on `approach`
- Time with `time.perf_counter()`
- Parse output with `normalise_prediction()` from `src.evaluation.metrics`
- Confidence: `1.0` exact match, `0.7` fuzzy match, `0.0` unknown
- On model error: HTTP 500 with `ErrorResponse`

#### `GET /intents` → `list[str]`
Returns all 27 intent names. No model required.

#### `GET /benchmark` → `dict`
Returns contents of `experiments/results/benchmark_summary.json`.
Returns HTTP 404 with helpful message if file not found.

### File: `src/serving/app.py`

```python
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from .router import router
from .model_loader import load_model

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup."""
    load_model()
    yield

def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="Intent Classification API",
        description="LLM-based e-commerce customer support intent classifier",
        version="0.1.0",
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(router, prefix="/api/v1")
    return app

app = create_app()
```

---

## Part B — Streamlit Dashboard

### File: `src/dashboard/data_loader.py`

```python
def load_benchmark_summary(
    path: Path = Path("experiments/results/benchmark_summary.json"),
) -> dict:
    """Load benchmark_summary.json. Raises FileNotFoundError with setup instructions if missing."""

def load_approach_results(approach: str, results_dir: Path) -> dict:
    """Load individual results JSON (e.g. zero_shot_results.json) for a given approach."""

def load_confusion_matrix_image(approach: str, cm_dir: Path) -> Path | None:
    """Return path to confusion matrix PNG or None if not found."""

def get_available_approaches(summary: dict) -> list[str]:
    """Return list of approach names present in summary['approaches']."""
```

### File: `src/dashboard/components/benchmark_table.py`

Render a formatted comparison table using `st.dataframe` with Pandas Styler.

**Columns:** Approach | Accuracy | Macro-F1 | Mean Latency (ms) | P95 Latency (ms) | Cost/1k

**Formatting:**
- Accuracy and F1 as percentages (e.g. `72.4%`)
- Latency as `145ms`
- Cost as `$0.0018`
- Highlight the best-performing cell in each metric column with green background

Include an interpretive caption: e.g. "Fine-tuning improves accuracy by X% over zero-shot at Y× higher latency."

```python
def render_benchmark_table(summary: dict) -> None:
    """Render three-way benchmark comparison table with metric highlighting."""
```

### File: `src/dashboard/components/confusion_matrix.py`

```python
def render_confusion_matrices(cm_dir: Path, available_approaches: list[str]) -> None:
    """
    Render confusion matrix PNGs side-by-side using st.columns.
    Each image has approach name as caption.
    Show placeholder message if an image is missing.
    """
```

### File: `src/dashboard/components/latency_chart.py`

```python
def render_latency_scatter(summary: dict) -> None:
    """
    Plotly scatter: X=mean latency (ms), Y=accuracy (%).
    One point per approach, labelled.
    Bubble size proportional to token cost.
    Tooltip: approach, accuracy, macro-F1, latency, cost.
    Title: 'Accuracy vs. Latency Trade-off'.
    """
```

Use `plotly.express.scatter`. Render with `st.plotly_chart(fig, use_container_width=True)`.

### File: `src/dashboard/components/live_predict.py`

```python
def render_live_predict_panel(
    api_base_url: str = "http://localhost:8000/api/v1",
) -> None:
    """
    Streamlit form: text input, approach dropdown, submit button.
    On submit: POST to /classify, show intent, confidence bar, latency badge.
    Use httpx (not requests).
    If API unreachable: st.warning with 'make serve' command — do not crash.
    """
```

### File: `src/dashboard/app.py`

**Page config:** `st.set_page_config(layout="wide", page_title="Intent Dashboard")`

**Sidebar:**
- About text (one paragraph)
- MLflow UI link: `http://localhost:5000`
- API status: green/red dot based on `/health` ping (wrapped in try/except)

**Main content — four tabs:**

**Tab 1: "Benchmark Results"**
- `render_benchmark_table(summary)`
- `render_latency_scatter(summary)`

**Tab 2: "Confusion Matrices"**
- `render_confusion_matrices(cm_dir, approaches)`
- Short explanation of how to read a normalised confusion matrix

**Tab 3: "Live Predict"**
- `render_live_predict_panel()`
- Note: "Requires FastAPI server — run `make serve` in a separate terminal"

**Tab 4: "About"**
- Project description
- Tech stack table
- Phase summary table

**Error handling:** If `benchmark_summary.json` is missing, show `st.error` with the commands to run phases 1–4. Do not crash.

---

## Part C — Docker

### `docker/Dockerfile`

```dockerfile
FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY configs/ ./configs/
COPY pyproject.toml .
RUN pip install -e .

EXPOSE 8000 8501

CMD ["uvicorn", "src.serving.app:app", "--host", "0.0.0.0", "--port", "8000"]
```

Note: MLX is not used in Docker — Metal GPU is not accessible inside containers on macOS. Document this in README. The Docker image uses PyTorch CPU for inference, which is slower but portable.

### `docker/docker-compose.yml`

```yaml
version: "3.9"

services:
  api:
    build:
      context: ..
      dockerfile: docker/Dockerfile
    ports:
      - "8000:8000"
    volumes:
      - ../data:/app/data:ro
      - ../models:/app/models:ro
      - ../experiments:/app/experiments:ro
      - ../configs:/app/configs:ro
    environment:
      - LOG_LEVEL=info
    command: uvicorn src.serving.app:app --host 0.0.0.0 --port 8000

  dashboard:
    build:
      context: ..
      dockerfile: docker/Dockerfile
    ports:
      - "8501:8501"
    volumes:
      - ../data:/app/data:ro
      - ../experiments:/app/experiments:ro
    environment:
      - API_BASE_URL=http://api:8000/api/v1
    depends_on:
      - api
    command: streamlit run src/dashboard/app.py --server.port 8501 --server.address 0.0.0.0
```

---

## File: `tests/test_serving.py`

Use `fastapi.testclient.TestClient` with mocked model loader — no real model loading in tests.

```python
import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
from src.serving.app import create_app

@pytest.fixture
def client():
    with patch("src.serving.model_loader.load_model"):
        with patch("src.serving.model_loader.get_model") as mock_get:
            mock_get.return_value = (MagicMock(), MagicMock())
            with patch("src.serving.model_loader.is_model_loaded", return_value=True):
                app = create_app()
                with TestClient(app) as c:
                    yield c

def test_health_returns_200(client): ...
def test_health_response_has_required_keys(client): ...
def test_classify_returns_200_with_valid_input(client): ...
def test_classify_rejects_empty_text(client): ...
def test_classify_rejects_text_over_1000_chars(client): ...
def test_classify_returns_500_on_model_error(client): ...
def test_intents_endpoint_returns_27_items(client): ...
def test_benchmark_endpoint_returns_dict(client, tmp_path): ...
def test_benchmark_endpoint_returns_404_when_missing(client): ...
```

---

## Acceptance Criteria Checklist

- [ ] `make serve` starts FastAPI on port 8000
- [ ] `curl http://localhost:8000/api/v1/health` returns `{"status":"ok",...}`
- [ ] `POST /classify` with `{"text":"I want to cancel my order"}` returns intent + confidence
- [ ] `GET /intents` returns array of exactly 27 strings
- [ ] `GET /benchmark` returns the summary JSON (or 404 with message if missing)
- [ ] `make dashboard` starts Streamlit on port 8501 without error
- [ ] Tab 1 benchmark table shows all three approaches with metric highlighting
- [ ] Tab 1 scatter plot renders without error
- [ ] Tab 2 shows confusion matrix images
- [ ] Tab 3 live predict works when API running; shows warning when API is down
- [ ] Dashboard handles missing `benchmark_summary.json` gracefully (no crash)
- [ ] `docker-compose up` builds and starts both services
- [ ] `make test-phase5` passes all serving tests
- [ ] No file in `src/serving/` or `src/dashboard/` exceeds 1000 lines
- [ ] CORS enabled on FastAPI app

---

## Final End-to-End Verification

After Phase 5, verify the full pipeline runs start-to-finish:

```bash
make install && make install-mlx
make download-data && make process-data
make finetune
make evaluate
make serve &
make dashboard
```

- [ ] All `make` commands complete without error
- [ ] `make test` passes all tests across all phases
- [ ] `make lint` reports zero ruff errors
- [ ] MLflow UI shows training and evaluation runs
- [ ] Dashboard renders real results

**Phase 5 complete = project complete.**
