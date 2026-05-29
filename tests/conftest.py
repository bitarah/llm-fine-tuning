"""Shared pytest fixtures for all test modules."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from src.data.preprocessor import KNOWN_INTENTS


@pytest.fixture
def sample_dataframe() -> pd.DataFrame:
    """Synthetic DataFrame with correct schema (27 intents, 10 rows each = 270 rows)."""
    rows = []
    for i, intent in enumerate(KNOWN_INTENTS):
        for j in range(10):
            rows.append({
                "instruction": f"I need help with {intent.replace('_', ' ')} unique query number {j}",
                "intent": intent,
                "category": f"category_{i % 10}",
                "text_length": len(f"I need help with {intent.replace('_', ' ')} unique query number {j}"),
                "intent_id": i,
            })
    return pd.DataFrame(rows)


@pytest.fixture
def tmp_output_dir(tmp_path: Path) -> Path:
    """Temporary directory for file output tests."""
    return tmp_path


@pytest.fixture
def mock_config():
    """TrainingConfig loaded from actual YAML files."""
    from src.config_loader import load_training_config
    return load_training_config()
