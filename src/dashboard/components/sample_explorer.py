"""Per-sample prediction explorer for debugging classification behaviour."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from src.dashboard.data_loader import load_approach_results


def _build_sample_df(approach: str, results_dir: Path) -> pd.DataFrame | None:
    """Load per-sample data for one approach and return as a DataFrame."""
    data = load_approach_results(approach, results_dir)
    if not data:
        return None

    input_texts = data.get("input_texts", [])
    predictions = data.get("predictions", [])
    ground_truth = data.get("ground_truth", [])
    raw_outputs = data.get("raw_outputs", [])
    latencies = data.get("latencies_ms", [])

    if not predictions or not ground_truth:
        return None

    n = len(predictions)
    rows = []
    for i in range(n):
        pred = predictions[i] if i < len(predictions) else ""
        truth = ground_truth[i] if i < len(ground_truth) else ""
        rows.append({
            "Input": input_texts[i] if i < len(input_texts) else "",
            "Raw Output": raw_outputs[i] if i < len(raw_outputs) else "",
            "Predicted": pred,
            "Actual": truth,
            "Correct": pred == truth,
            "Latency (ms)": round(latencies[i], 1) if i < len(latencies) else None,
        })
    return pd.DataFrame(rows)


def render_sample_explorer(results_dir: Path, approaches: list[str]) -> None:
    """Render filterable per-sample prediction table for each approach."""
    approach_labels = {a: a.replace("_", " ").title() for a in approaches}

    selected = st.selectbox(
        "Approach",
        options=approaches,
        format_func=lambda a: approach_labels[a],
    )

    df = _build_sample_df(selected, results_dir)
    if df is None:
        st.warning(f"No sample data found for {approach_labels[selected]}. Re-run `make evaluate`.")
        return

    col1, col2 = st.columns(2)
    with col1:
        filter_correct = st.selectbox("Show", ["All", "Correct only", "Incorrect only"])
    with col2:
        all_intents = sorted(set(df["Actual"].tolist()))
        filter_intent = st.selectbox("Filter by actual intent", ["All"] + all_intents)

    filtered = df.copy()
    if filter_correct == "Correct only":
        filtered = filtered[filtered["Correct"]]
    elif filter_correct == "Incorrect only":
        filtered = filtered[~filtered["Correct"]]
    if filter_intent != "All":
        filtered = filtered[filtered["Actual"] == filter_intent]

    total = len(filtered)
    correct = filtered["Correct"].sum()
    st.caption(f"{correct}/{total} correct ({correct/total:.0%})" if total else "No samples match filters.")

    def _colour_correct(val: bool) -> str:
        return "background-color: #d4edda" if val else "background-color: #f8d7da"

    styled = filtered.style.applymap(_colour_correct, subset=["Correct"])
    st.dataframe(styled, use_container_width=True, height=500)
