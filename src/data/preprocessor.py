"""Clean and normalise the raw Bitext dataset."""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

KNOWN_INTENTS: list[str] = sorted([
    "cancel_order",
    "change_order",
    "change_shipping_address",
    "check_cancellation_fee",
    "check_invoice",
    "check_payment_methods",
    "check_refund_policy",
    "complaint",
    "contact_customer_service",
    "contact_human_agent",
    "create_account",
    "delete_account",
    "delivery_options",
    "delivery_period",
    "edit_account",
    "get_invoice",
    "get_refund",
    "newsletter_subscription",
    "payment_issue",
    "place_order",
    "recover_password",
    "registration_problems",
    "review",
    "set_up_shipping_address",
    "switch_account",
    "track_order",
    "track_refund",
])


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Apply all cleaning steps. Returns cleaned DataFrame."""
    initial_count = len(df)

    df = df.dropna(subset=["instruction", "intent"])
    df = df[df["instruction"].str.strip() != ""]
    df = df[df["intent"].str.strip() != ""]

    df = df.copy()
    df["instruction"] = df["instruction"].str.strip()

    df["intent"] = df["intent"].str.lower().str.replace(" ", "_", regex=False)

    df = df.drop_duplicates(subset=["instruction"], keep="first")

    df = df[df["instruction"].str.len().between(5, 500)]

    df, unknown = validate_intents(df)

    df = df.copy()
    df["text_length"] = df["instruction"].str.len()
    df["intent_id"] = df["intent"].map(
        {intent: idx for idx, intent in enumerate(KNOWN_INTENTS)}
    )

    logger.info(f"Cleaned dataset: {initial_count} → {len(df)} rows")
    return df.reset_index(drop=True)


def validate_intents(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Check for unknown intents. Returns (clean_df, list_of_unknown_intents)."""
    known_set = set(KNOWN_INTENTS)
    unknown = [i for i in df["intent"].unique() if i not in known_set]
    if unknown:
        for intent in unknown:
            logger.warning(f"Unknown intent dropped: '{intent}'")
    clean_df = df[df["intent"].isin(known_set)].copy()
    return clean_df, unknown


def get_intent_to_id_mapping() -> dict[str, int]:
    """Return sorted intent-to-integer-index mapping (27 entries)."""
    return {intent: idx for idx, intent in enumerate(KNOWN_INTENTS)}


def load_raw_dataset(path: Path = Path("data/raw/bitext_raw.parquet")) -> pd.DataFrame:
    """Load raw Parquet. Raises FileNotFoundError with helpful message if missing."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Raw dataset not found at {path}. "
            "Run 'make download-data' to download the dataset first."
        )
    return pd.read_parquet(path)


if __name__ == "__main__":
    df = load_raw_dataset()
    print(f"Loaded {len(df)} rows")
    cleaned = clean_dataset(df)
    output_path = Path("data/processed/cleaned.parquet")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cleaned.to_parquet(output_path, index=False)
    print(f"Before: {len(df)} rows → After: {len(cleaned)} rows")
    print(f"Saved cleaned dataset to {output_path}")
