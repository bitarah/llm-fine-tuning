"""Streamlit evaluation dashboard entry point."""
from __future__ import annotations

import os
from pathlib import Path

import streamlit as st

from src.dashboard.components.benchmark_table import render_benchmark_table
from src.dashboard.components.confusion_matrix import render_confusion_matrices
from src.dashboard.components.latency_chart import render_latency_scatter
from src.dashboard.components.live_predict import render_live_predict_panel
from src.dashboard.components.sample_explorer import render_sample_explorer
from src.dashboard.data_loader import get_available_approaches, load_benchmark_summary

_RESULTS_DIR = Path("experiments/results")
_CM_DIR = _RESULTS_DIR / "confusion_matrices"
_MLFLOW_URL = "http://localhost:5000"
_API_BASE_URL = os.environ.get("API_BASE_URL", "http://localhost:8000/api/v1")

st.set_page_config(layout="wide", page_title="Intent Dashboard")


def _api_status() -> bool:
    """Return True if the FastAPI server is reachable."""
    try:
        import httpx
        with httpx.Client(timeout=2.0) as client:
            resp = client.get(f"{_API_BASE_URL}/health")
            return resp.status_code == 200
    except Exception:
        return False


def _render_sidebar() -> None:
    """Render sidebar with project info and live status indicators."""
    with st.sidebar:
        st.title("Intent Dashboard")
        st.markdown(
            "End-to-end pipeline comparing zero-shot prompting, few-shot prompting, "
            "and QLoRA fine-tuning for e-commerce customer support intent classification."
        )
        st.divider()
        st.markdown(f"[Open MLflow UI]({_MLFLOW_URL})")
        st.divider()
        api_up = _api_status()
        dot = "🟢" if api_up else "🔴"
        st.markdown(f"{dot} API {'online' if api_up else 'offline'}")
        if not api_up:
            st.caption("Run `make serve` to start the API.")


def main() -> None:
    """Render the full dashboard."""
    _render_sidebar()

    try:
        summary = load_benchmark_summary(_RESULTS_DIR / "benchmark_summary.json")
        approaches = get_available_approaches(summary)
    except FileNotFoundError as exc:
        st.error(
            str(exc) + "\n\nRun the full pipeline:\n"
            "```\nmake download-data && make process-data && make finetune && make evaluate\n```"
        )
        return

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Benchmark Results",
        "Confusion Matrices",
        "Sample Explorer",
        "Live Predict",
        "About",
    ])

    with tab1:
        st.header("Benchmark Results")
        render_benchmark_table(summary)
        st.divider()
        render_latency_scatter(summary)

    with tab2:
        st.header("Confusion Matrices")
        st.markdown(
            "Each cell shows the fraction of true-class samples predicted as each class. "
            "Bright diagonal = accurate; off-diagonal entries reveal systematic confusions."
        )
        render_confusion_matrices(_CM_DIR, approaches)

    with tab3:
        st.header("Sample Explorer")
        st.markdown(
            "Inspect per-sample predictions. Filter by correct/incorrect or by intent to spot "
            "systematic errors and compare what each approach gets wrong."
        )
        render_sample_explorer(_RESULTS_DIR, approaches)

    with tab4:
        st.header("Live Predict")
        render_live_predict_panel(_API_BASE_URL)

    with tab5:
        st.header("About")
        st.markdown(
            "This project builds an end-to-end pipeline to fine-tune a small open-source LLM "
            "for e-commerce customer support intent classification. It compares three approaches — "
            "zero-shot prompting, few-shot prompting, and QLoRA fine-tuning — across accuracy, "
            "latency, and estimated API cost."
        )
        st.subheader("Tech Stack")
        st.table({
            "Layer": ["Fine-tuning", "Base Model", "Experiment Tracking", "API", "Dashboard"],
            "Tool": ["mlx-lm / HuggingFace PEFT", "Phi-3.5-mini-instruct", "MLflow", "FastAPI", "Streamlit"],
        })
        st.subheader("Pipeline Phases")
        st.table({
            "Phase": [1, 2, 3, 4, 5],
            "Focus": [
                "Project setup",
                "Data processing",
                "Fine-tuning",
                "Evaluation",
                "Dashboard & Serving",
            ],
            "Key Output": [
                "Repo skeleton, configs, Makefile",
                "Cleaned splits, prompt templates",
                "Trained LoRA adapter, MLflow run",
                "Benchmark results, confusion matrices",
                "FastAPI service, Streamlit dashboard, Docker",
            ],
        })


main()
