"""Download the Bitext customer support dataset from HuggingFace."""
from __future__ import annotations

import logging
import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

DATASET_ID = "bitext/Bitext-customer-support-llm-chatbot-training-dataset"


def download_dataset(
    dataset_id: str = DATASET_ID,
    output_dir: Path = Path("data/raw"),
    force_redownload: bool = False,
) -> Path:
    """Download Bitext dataset from HuggingFace. Returns path to saved Parquet file."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    parquet_path = output_dir / "bitext_raw.parquet"

    if parquet_path.exists() and not force_redownload:
        logging.info(f"File exists, skipping: {parquet_path}")
        return parquet_path

    import datasets as hf_datasets  # lazy import — skipped by early return in tests

    hf_token = os.getenv("HF_TOKEN")
    try:
        dataset = hf_datasets.load_dataset(dataset_id, token=hf_token)
    except Exception as exc:
        raise ConnectionError(
            f"Failed to download dataset '{dataset_id}'. "
            "Check your internet connection and HF_TOKEN in .env. "
            f"Original error: {exc}"
        ) from exc

    split_name = "train" if "train" in dataset else list(dataset.keys())[0]
    df = dataset[split_name].to_pandas()

    df.to_parquet(parquet_path, index=False)
    logger.info(f"Saved raw dataset to {parquet_path}")

    sample_path = output_dir / "sample.jsonl"
    df.head(100).to_json(sample_path, orient="records", lines=True)
    logger.info(f"Saved 100-row sample to {sample_path}")

    return parquet_path


def print_dataset_summary(df: pd.DataFrame) -> None:
    """Print row count, column names, intent distribution, and category counts."""
    print(f"Total rows: {len(df)}")
    print(f"Columns: {list(df.columns)}")
    if "intent" in df.columns:
        print(f"\nIntent distribution ({df['intent'].nunique()} unique):")
        print(df["intent"].value_counts().to_string())
    if "category" in df.columns:
        print(f"\nCategory distribution ({df['category'].nunique()} unique):")
        print(df["category"].value_counts().to_string())


if __name__ == "__main__":
    parquet_path = download_dataset()
    df = pd.read_parquet(parquet_path)
    print_dataset_summary(df)
