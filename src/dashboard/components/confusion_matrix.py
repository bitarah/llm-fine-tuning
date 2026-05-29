"""Render confusion matrix PNGs in the dashboard."""
from __future__ import annotations

from pathlib import Path

import streamlit as st

from src.dashboard.data_loader import load_confusion_matrix_image


def render_confusion_matrices(cm_dir: Path, available_approaches: list[str]) -> None:
    """
    Render confusion matrix PNGs side-by-side using st.columns.
    Each image has approach name as caption.
    Show placeholder message if an image is missing.
    """
    cm_dir = Path(cm_dir)
    cols = st.columns(len(available_approaches)) if available_approaches else []

    for col, approach in zip(cols, available_approaches, strict=False):
        image_path = load_confusion_matrix_image(approach, cm_dir)
        label = approach.replace("_", " ").title()
        with col:
            if image_path is not None:
                st.image(str(image_path), caption=label, use_container_width=True)
            else:
                st.info(f"No confusion matrix found for {label}. Run `make evaluate` first.")
