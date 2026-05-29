"""Create stratified train/val/test splits and save as JSONL."""
from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)


def split_dataset(
    df: pd.DataFrame,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return (train_df, val_df, test_df) with stratified intent distribution."""
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-9, (
        "Ratios must sum to 1.0"
    )

    train_df, temp_df = train_test_split(
        df,
        test_size=(val_ratio + test_ratio),
        stratify=df["intent"],
        random_state=seed,
    )

    relative_val = val_ratio / (val_ratio + test_ratio)
    val_df, test_df = train_test_split(
        temp_df,
        test_size=(1.0 - relative_val),
        stratify=temp_df["intent"],
        random_state=seed,
    )

    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)


def save_split_as_jsonl(df: pd.DataFrame, output_path: Path) -> None:
    """Save DataFrame as JSONL, one record per line."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cols = [c for c in ["instruction", "intent", "category", "intent_id"] if c in df.columns]
    with open(output_path, "w") as f:
        for _, row in df[cols].iterrows():
            f.write(json.dumps(row.to_dict()) + "\n")
    logger.info(f"Saved {len(df)} rows to {output_path}")


def verify_no_leakage(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> None:
    """Assert no instruction appears in more than one split. Raises AssertionError if leakage found."""
    train_set = set(train_df["instruction"])
    val_set = set(val_df["instruction"])
    test_set = set(test_df["instruction"])

    train_val = train_set & val_set
    train_test = train_set & test_set
    val_test = val_set & test_set

    leaks = []
    if train_val:
        leaks.append(f"train/val overlap: {len(train_val)} rows")
    if train_test:
        leaks.append(f"train/test overlap: {len(train_test)} rows")
    if val_test:
        leaks.append(f"val/test overlap: {len(val_test)} rows")

    if leaks:
        raise AssertionError("Label leakage detected: " + ", ".join(leaks))

    logger.info("No label leakage detected across splits.")


if __name__ == "__main__":

    cleaned_path = Path("data/processed/cleaned.parquet")
    if not cleaned_path.exists():
        raise FileNotFoundError(
            f"Cleaned dataset not found at {cleaned_path}. "
            "Run 'python -m src.data.preprocessor' first."
        )

    df = pd.read_parquet(cleaned_path)
    train_df, val_df, test_df = split_dataset(df)
    verify_no_leakage(train_df, val_df, test_df)

    output_dir = Path("data/processed")
    save_split_as_jsonl(train_df, output_dir / "train.jsonl")
    save_split_as_jsonl(val_df, output_dir / "val.jsonl")
    save_split_as_jsonl(test_df, output_dir / "test.jsonl")

    print(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")
    print("\nIntent distribution in train:")
    print(train_df["intent"].value_counts().to_string())
