# Phase 2 — Data Processing

## Overview

This phase downloads the Bitext customer support dataset, cleans and normalises it, creates stratified train/val/test splits, and formats the data into three prompt structures: zero-shot, few-shot, and fine-tuning (chat) format.

Read `design-overview.md` before starting. Phase 1 must be complete — directory structure, configs, and `make install` must already work.

---

## Before You Start

```bash
make test-phase1     # must pass
ls configs/          # must show both YAML files
```

---

## Dataset Details

- **HuggingFace ID:** `bitext/Bitext-customer-support-llm-chatbot-training-dataset`
- **Columns used:** `instruction` (input text), `intent` (label), `category`
- **Total rows:** ~26,872 | **Intent classes:** 27 | **Categories:** 10

### All 27 Intent Classes (sorted)
```
cancel_order, change_order, change_shipping_address, check_cancellation_fee,
check_invoice, check_payment_methods, check_refund_policy, complaint,
contact_customer_service, contact_human_agent, create_account, delete_account,
delivery_options, delivery_period, edit_account, get_invoice, get_refund,
newsletter_subscription, payment_issue, place_order, recover_password,
registration_problems, review, set_up_shipping_address, switch_account,
track_order, track_refund
```

---

## File: `src/data/downloader.py`

**Responsibility:** Download the raw dataset from HuggingFace and save to `data/raw/`.

**Requirements:**
- Use `datasets.load_dataset()` to fetch the dataset
- Save as Parquet to `data/raw/bitext_raw.parquet`
- Save a 100-row sample as `data/raw/sample.jsonl` for quick inspection
- Skip download if file already exists (idempotent) — respect `force_redownload` flag
- Print summary: total rows, column names, intent value counts
- Read `HF_TOKEN` from environment for authenticated downloads
- Raise `ConnectionError` with a helpful message on network failure

**Function signatures:**

```python
def download_dataset(
    dataset_id: str = "bitext/Bitext-customer-support-llm-chatbot-training-dataset",
    output_dir: Path = Path("data/raw"),
    force_redownload: bool = False,
) -> Path:
    """Download Bitext dataset from HuggingFace. Returns path to saved Parquet file."""

def print_dataset_summary(df: pd.DataFrame) -> None:
    """Print row count, column names, intent distribution, and category counts."""
```

**Module entry point (`__main__`):** Call `download_dataset()` then `print_dataset_summary()`.

---

## File: `src/data/preprocessor.py`

**Responsibility:** Clean and normalise the raw dataset.

**Cleaning steps (in order):**
1. Drop rows where `instruction` or `intent` is null or empty string
2. Strip leading/trailing whitespace from `instruction`
3. Normalise intent labels to lowercase with underscores
4. Remove duplicate `instruction` rows (keep first)
5. Filter out instructions shorter than 5 characters or longer than 500 characters
6. Validate all intent values are in the known 27-intent list — log unknowns as warnings and drop those rows
7. Add `text_length` column (character count of `instruction`)
8. Add `intent_id` column (integer index of intent in sorted intent list)

**Function signatures:**

```python
KNOWN_INTENTS: list[str] = [...]  # full sorted list of 27 intents

def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Apply all cleaning steps. Returns cleaned DataFrame."""

def validate_intents(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Check for unknown intents. Returns (clean_df, list_of_unknown_intents)."""

def get_intent_to_id_mapping() -> dict[str, int]:
    """Return sorted intent-to-integer-index mapping (27 entries)."""

def load_raw_dataset(path: Path = Path("data/raw/bitext_raw.parquet")) -> pd.DataFrame:
    """Load raw Parquet. Raises FileNotFoundError with helpful message if missing."""
```

**Module entry point:** Load raw data → clean → save to `data/processed/cleaned.parquet` → print before/after row counts.

---

## File: `src/data/splitter.py`

**Responsibility:** Create stratified train/val/test splits and save as JSONL.

**Split ratios:** Train 70% / Val 15% / Test 15%

**Requirements:**
- Use `sklearn.model_selection.train_test_split` with `stratify=df["intent"]`
- Use `seed=42` for reproducibility
- Save splits as JSONL to `data/processed/train.jsonl`, `val.jsonl`, `test.jsonl`
- Each JSONL row: `{"instruction": "...", "intent": "...", "category": "...", "intent_id": 0}`
- Verify no intent class is missing from any split
- Print split statistics: row counts and per-intent distribution

**Function signatures:**

```python
def split_dataset(
    df: pd.DataFrame,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return (train_df, val_df, test_df) with stratified intent distribution."""

def save_split_as_jsonl(df: pd.DataFrame, output_path: Path) -> None:
    """Save DataFrame as JSONL, one record per line."""

def verify_no_leakage(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> None:
    """Assert no instruction appears in more than one split. Raises AssertionError if leakage found."""
```

**Module entry point:** Load `cleaned.parquet` → split → verify no leakage → save all JSONL files → print statistics.

---

## File: `src/data/formatter.py`

**Responsibility:** Format processed splits into three prompt structures.

### Format 1 — Zero-Shot Prompt Template
Saved to `data/prompts/zeroshot_template.txt`:

```
You are a customer support intent classifier. Given a customer message, classify it into exactly one of the following intents:

{intent_list}

Respond with ONLY the intent name, nothing else.

Customer message: {instruction}
Intent:
```

Where `{intent_list}` is the sorted list of 27 intents, one per line with a dash prefix.

### Format 2 — Few-Shot Examples
Select 5 diverse examples per intent from the training set (not from val/test).
Save to `data/prompts/fewshot_examples.jsonl`. Each row:
`{"intent": "cancel_order", "examples": ["I want to cancel...", ...]}`

Few-shot prompt is assembled at runtime by selecting 5 random intents × 1 example each,
then appending the test query. Keep total prompt under `model_max_length` tokens.

### Format 3 — Fine-Tuning Chat Format (mlx-lm)
Each JSONL row is a chat conversation:

```json
{
  "messages": [
    {"role": "system", "content": "You are a customer support intent classifier. Classify the customer message into exactly one intent. Respond with only the intent name."},
    {"role": "user", "content": "Customer message: {instruction}"},
    {"role": "assistant", "content": "{intent}"}
  ]
}
```

Save to `data/processed/finetune_train.jsonl` (training only) and `data/processed/finetune_val.jsonl`.

**Function signatures:**

```python
def format_zero_shot_prompt(instruction: str, intent_list: list[str]) -> str:
    """Format a single instruction using the zero-shot template."""

def select_few_shot_examples(
    train_df: pd.DataFrame,
    n_per_intent: int = 5,
    seed: int = 42,
) -> dict[str, list[str]]:
    """Return dict mapping intent to list of n example instructions."""

def format_few_shot_prompt(
    instruction: str,
    examples: dict[str, list[str]],
    n_intents_to_show: int = 5,
    seed: int | None = None,
) -> str:
    """Format a single instruction using randomly sampled few-shot examples."""

def format_finetune_record(instruction: str, intent: str) -> dict:
    """Return a single chat-formatted record for mlx-lm fine-tuning."""

def save_few_shot_examples(
    examples: dict[str, list[str]],
    output_path: Path = Path("data/prompts/fewshot_examples.jsonl"),
) -> None:
    """Persist the few-shot example bank as JSONL."""

def format_and_save_finetune_data(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    output_dir: Path = Path("data/processed"),
) -> None:
    """Format and save fine-tune JSONL files for train and val splits."""
```

**Module entry point:** Load all splits → create all formats → save all output files → print file paths and record counts.

---

## File: `tests/test_data.py`

Use `pytest` fixtures with small synthetic DataFrames (10–50 rows). Do not depend on the real downloaded dataset in unit tests. Create `tests/conftest.py` with a `sample_dataframe` fixture.

**Required tests:**
```python
# Downloader
def test_download_is_idempotent():
def test_download_skips_if_file_exists():

# Preprocessor
def test_clean_removes_nulls():
def test_clean_removes_short_instructions():
def test_clean_removes_duplicates():
def test_validate_intents_flags_unknowns():
def test_intent_to_id_mapping_has_27_entries():
def test_intent_to_id_mapping_is_sorted():

# Splitter
def test_split_ratios_approximately_correct():
def test_all_intents_in_all_splits():
def test_no_instruction_leakage_between_splits():
def test_splits_are_deterministic_with_seed():

# Formatter
def test_zero_shot_prompt_contains_instruction():
def test_zero_shot_prompt_contains_all_27_intents():
def test_few_shot_examples_has_5_per_intent():
def test_finetune_format_has_three_messages():
def test_finetune_format_system_role_present():
def test_finetune_format_assistant_content_equals_intent():
```

---

## Output Files Checklist

After `make process-data`, all must exist:

```
data/raw/bitext_raw.parquet
data/raw/sample.jsonl
data/processed/cleaned.parquet
data/processed/train.jsonl
data/processed/val.jsonl
data/processed/test.jsonl
data/processed/finetune_train.jsonl
data/processed/finetune_val.jsonl
data/prompts/zeroshot_template.txt
data/prompts/fewshot_examples.jsonl
```

---

## Acceptance Criteria Checklist

- [ ] `make download-data` completes and saves `data/raw/bitext_raw.parquet`
- [ ] Running `make download-data` a second time skips download (idempotent)
- [ ] `make process-data` runs all preprocessing, splitting, and formatting end-to-end
- [ ] All 27 intent classes present in train, val, and test splits
- [ ] `verify_no_leakage()` passes — no instruction in more than one split
- [ ] `data/processed/finetune_train.jsonl` is valid JSONL with `messages` key in every row
- [ ] Zero-shot template file contains all 27 intents
- [ ] `data/prompts/fewshot_examples.jsonl` has exactly 27 entries
- [ ] `make test-phase2` passes all data tests
- [ ] No hardcoded file paths in any `src/data/` file

**Do not proceed to Phase 3 until all items are checked.**
