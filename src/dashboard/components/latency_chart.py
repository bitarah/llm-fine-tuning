"""Render accuracy vs. latency scatter plot."""
from __future__ import annotations

import plotly.express as px
import streamlit as st


def render_latency_scatter(summary: dict) -> None:
    """
    Plotly scatter: X=mean latency (ms), Y=accuracy (%).
    One point per approach, labelled.
    Bubble size proportional to token cost.
    Tooltip: approach, accuracy, macro-F1, latency, cost.
    Title: 'Accuracy vs. Latency Trade-off'.
    """
    approaches = summary.get("approaches", {})
    if not approaches:
        st.warning("No data available for scatter plot.")
        return

    rows = []
    for name, metrics in approaches.items():
        rows.append({
            "Approach": name.replace("_", " ").title(),
            "Mean Latency (ms)": metrics.get("mean_latency_ms", 0.0),
            "Accuracy (%)": metrics.get("accuracy", 0.0) * 100,
            "Macro-F1": metrics.get("macro_f1", 0.0),
            "Cost/1k": metrics.get("token_cost_per_1k", 0.0),
        })

    fig = px.scatter(
        rows,
        x="Mean Latency (ms)",
        y="Accuracy (%)",
        size="Cost/1k",
        text="Approach",
        hover_data=["Approach", "Accuracy (%)", "Macro-F1", "Mean Latency (ms)", "Cost/1k"],
        title="Accuracy vs. Latency Trade-off",
        size_max=40,
    )
    fig.update_traces(textposition="top center")
    fig.update_layout(showlegend=False)

    st.plotly_chart(fig, use_container_width=True)
