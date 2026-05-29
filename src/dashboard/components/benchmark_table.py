"""Render the three-way benchmark comparison table."""
from __future__ import annotations

import pandas as pd
import streamlit as st


def render_benchmark_table(summary: dict) -> None:
    """Render three-way benchmark comparison table with metric highlighting."""
    approaches = summary.get("approaches", {})
    if not approaches:
        st.warning("No benchmark data available.")
        return

    rows = []
    for name, metrics in approaches.items():
        rows.append({
            "Approach": name.replace("_", " ").title(),
            "_approach_key": name,
            "Accuracy": metrics.get("accuracy", 0.0),
            "Precision": metrics.get("macro_precision", 0.0),
            "Recall": metrics.get("macro_recall", 0.0),
            "Macro-F1": metrics.get("macro_f1", 0.0),
            "Mean Latency (ms)": metrics.get("mean_latency_ms", 0.0),
            "P95 Latency (ms)": metrics.get("p95_latency_ms", 0.0),
            "Cost/1k": metrics.get("token_cost_per_1k", 0.0),
        })

    df = pd.DataFrame(rows).drop(columns=["_approach_key"])

    def highlight_best(col: pd.Series) -> list[str]:
        """Green background on best value; lower is better for latency/cost."""
        if col.name in ("Mean Latency (ms)", "P95 Latency (ms)", "Cost/1k"):
            best = col.min()
        else:
            best = col.max()
        return ["background-color: #d4edda" if v == best else "" for v in col]

    numeric_cols = ["Accuracy", "Precision", "Recall", "Macro-F1", "Mean Latency (ms)", "P95 Latency (ms)", "Cost/1k"]

    styled = (
        df.style
        .apply(highlight_best, subset=numeric_cols)
        .format({
            "Accuracy": "{:.1%}",
            "Precision": "{:.1%}",
            "Recall": "{:.1%}",
            "Macro-F1": "{:.1%}",
            "Mean Latency (ms)": "{:.0f}ms",
            "P95 Latency (ms)": "{:.0f}ms",
            "Cost/1k": "${:.4f}",
        })
    )

    st.dataframe(styled, use_container_width=True)

    # Interpretive caption
    if len(rows) >= 2:
        first = rows[0]
        last = rows[-1]
        acc_delta = (last["Accuracy"] - first["Accuracy"]) * 100
        lat_ratio = last["Mean Latency (ms)"] / max(first["Mean Latency (ms)"], 1e-6)
        direction = "improves" if acc_delta > 0 else "reduces"
        st.caption(
            f"Fine-tuning {direction} accuracy by {abs(acc_delta):.1f}% over zero-shot "
            f"at {lat_ratio:.1f}× the latency."
        )
