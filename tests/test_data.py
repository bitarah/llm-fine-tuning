"""Tests for data pipeline: downloader, preprocessor, splitter, formatter."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from src.data.formatter import (
    format_finetune_record,
    format_zero_shot_prompt,
    select_few_shot_examples,
)
from src.data.preprocessor import (
    KNOWN_INTENTS,
    clean_dataset,
    get_intent_to_id_mapping,
    validate_intents,
)
from src.data.splitter import (
    split_dataset,
    verify_no_leakage,
)

# ─── Downloader ────────────────────────────────────────────────────────────────


def test_download_skips_if_file_exists(tmp_path: Path) -> None:
    """download_dataset returns early if parquet already exists."""
    from src.data.downloader import download_dataset

    parquet_path = tmp_path / "bitext_raw.parquet"
    parquet_path.touch()

    result = download_dataset(output_dir=tmp_path, force_redownload=False)
    assert result == parquet_path


def test_download_is_idempotent(tmp_path: Path) -> None:
    """Calling download_dataset twice with force_redownload=False skips second call."""
    from src.data.downloader import download_dataset

    parquet_path = tmp_path / "bitext_raw.parquet"
    parquet_path.touch()
    original_mtime = parquet_path.stat().st_mtime

    download_dataset(output_dir=tmp_path, force_redownload=False)
    assert parquet_path.stat().st_mtime == original_mtime


# ─── Preprocessor ──────────────────────────────────────────────────────────────


def test_clean_removes_nulls() -> None:
    """clean_dataset drops rows with null instruction or intent."""
    df = pd.DataFrame({
        "instruction": ["hello", None, "world", ""],
        "intent": ["cancel_order", "track_order", None, "get_refund"],
        "category": ["a", "b", "c", "d"],
    })
    result = clean_dataset(df)
    assert result["instruction"].isna().sum() == 0
    assert result["intent"].isna().sum() == 0
    assert len(result) < len(df)


def test_clean_removes_short_instructions() -> None:
    """clean_dataset removes instructions shorter than 5 chars."""
    df = pd.DataFrame({
        "instruction": ["hi", "hello there friend", "ok"],
        "intent": ["cancel_order", "track_order", "get_refund"],
        "category": ["a", "b", "c"],
    })
    result = clean_dataset(df)
    assert all(result["instruction"].str.len() >= 5)


def test_clean_removes_duplicates() -> None:
    """clean_dataset keeps only the first occurrence of duplicate instructions."""
    df = pd.DataFrame({
        "instruction": ["I want to cancel", "I want to cancel", "Track my order please"],
        "intent": ["cancel_order", "cancel_order", "track_order"],
        "category": ["a", "a", "b"],
    })
    result = clean_dataset(df)
    assert len(result) == 2


def test_validate_intents_flags_unknowns() -> None:
    """validate_intents returns unknown intents and drops them."""
    df = pd.DataFrame({
        "instruction": ["help me now", "cancel this order please"],
        "intent": ["unknown_intent_xyz", "cancel_order"],
        "category": ["a", "b"],
    })
    clean, unknowns = validate_intents(df)
    assert "unknown_intent_xyz" in unknowns
    assert "unknown_intent_xyz" not in clean["intent"].values
    assert "cancel_order" in clean["intent"].values


def test_intent_to_id_mapping_has_27_entries() -> None:
    """get_intent_to_id_mapping returns exactly 27 entries."""
    mapping = get_intent_to_id_mapping()
    assert len(mapping) == 27


def test_intent_to_id_mapping_is_sorted() -> None:
    """get_intent_to_id_mapping keys are in sorted order."""
    mapping = get_intent_to_id_mapping()
    keys = list(mapping.keys())
    assert keys == sorted(keys)


# ─── Splitter ──────────────────────────────────────────────────────────────────


def test_split_ratios_approximately_correct(sample_dataframe: pd.DataFrame) -> None:
    """split_dataset produces splits within 5% of requested ratios."""
    train, val, test = split_dataset(sample_dataframe)
    total = len(sample_dataframe)
    assert abs(len(train) / total - 0.70) < 0.05
    assert abs(len(val) / total - 0.15) < 0.05
    assert abs(len(test) / total - 0.15) < 0.05


def test_all_intents_in_all_splits(sample_dataframe: pd.DataFrame) -> None:
    """All 27 intent classes must appear in train, val, and test."""
    train, val, test = split_dataset(sample_dataframe)
    for split_df in (train, val, test):
        for intent in KNOWN_INTENTS:
            assert intent in split_df["intent"].values, f"Missing intent '{intent}' in split"


def test_no_instruction_leakage_between_splits(sample_dataframe: pd.DataFrame) -> None:
    """verify_no_leakage raises no error when splits are clean."""
    train, val, test = split_dataset(sample_dataframe)
    verify_no_leakage(train, val, test)


def test_no_leakage_raises_on_overlap() -> None:
    """verify_no_leakage raises AssertionError when instructions overlap."""
    shared_instruction = "I need help with cancel order example 0"
    train = pd.DataFrame({"instruction": [shared_instruction, "unique train row"]})
    val = pd.DataFrame({"instruction": [shared_instruction, "unique val row"]})
    test = pd.DataFrame({"instruction": ["unique test row"]})
    with pytest.raises(AssertionError, match="leakage"):
        verify_no_leakage(train, val, test)


def test_splits_are_deterministic_with_seed(sample_dataframe: pd.DataFrame) -> None:
    """Same seed produces identical splits."""
    train1, val1, test1 = split_dataset(sample_dataframe, seed=42)
    train2, val2, test2 = split_dataset(sample_dataframe, seed=42)
    pd.testing.assert_frame_equal(train1.reset_index(drop=True), train2.reset_index(drop=True))
    pd.testing.assert_frame_equal(val1.reset_index(drop=True), val2.reset_index(drop=True))


# ─── Formatter ─────────────────────────────────────────────────────────────────


def test_zero_shot_prompt_contains_instruction() -> None:
    """format_zero_shot_prompt includes the instruction text."""
    prompt = format_zero_shot_prompt("Where is my order?", KNOWN_INTENTS)
    assert "Where is my order?" in prompt


def test_zero_shot_prompt_contains_all_27_intents() -> None:
    """format_zero_shot_prompt lists all 27 intent names."""
    prompt = format_zero_shot_prompt("test instruction here", KNOWN_INTENTS)
    for intent in KNOWN_INTENTS:
        assert intent in prompt, f"Intent '{intent}' missing from zero-shot prompt"


def test_few_shot_examples_has_5_per_intent(sample_dataframe: pd.DataFrame) -> None:
    """select_few_shot_examples returns up to 5 examples per intent."""
    examples = select_few_shot_examples(sample_dataframe, n_per_intent=5)
    assert len(examples) == 27
    for _intent, exs in examples.items():
        assert len(exs) <= 5


def test_finetune_format_has_three_messages() -> None:
    """format_finetune_record returns a dict with exactly 3 messages."""
    record = format_finetune_record("I want to cancel my order.", "cancel_order")
    assert "messages" in record
    assert len(record["messages"]) == 3


def test_finetune_format_system_role_present() -> None:
    """format_finetune_record includes a system role message."""
    record = format_finetune_record("Track my package.", "track_order")
    roles = [m["role"] for m in record["messages"]]
    assert "system" in roles


def test_finetune_format_assistant_content_equals_intent() -> None:
    """format_finetune_record assistant message content matches intent."""
    record = format_finetune_record("I need a refund.", "get_refund")
    assistant_msg = next(m for m in record["messages"] if m["role"] == "assistant")
    assert assistant_msg["content"] == "get_refund"
