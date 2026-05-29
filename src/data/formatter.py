"""Format processed dataset splits into zero-shot, few-shot, and fine-tuning formats."""
from __future__ import annotations

import json
import logging
import random
from pathlib import Path

import pandas as pd

from src.data.preprocessor import KNOWN_INTENTS

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are a customer support intent classifier. "
    "Classify the customer message into exactly one intent. "
    "Respond with only the intent name."
)

ZERO_SHOT_TEMPLATE = (
    "You are a customer support intent classifier. Given a customer message, "
    "classify it into exactly one of the following intents:\n\n"
    "{intent_list}\n\n"
    "Respond with ONLY the intent name, nothing else.\n\n"
    "Customer message: {instruction}\n"
    "Intent:"
)


def format_zero_shot_prompt(instruction: str, intent_list: list[str]) -> str:
    """Format a single instruction using the zero-shot template."""
    formatted_list = "\n".join(f"- {intent}" for intent in intent_list)
    return ZERO_SHOT_TEMPLATE.format(intent_list=formatted_list, instruction=instruction)


def select_few_shot_examples(
    train_df: pd.DataFrame,
    n_per_intent: int = 5,
    seed: int = 42,
) -> dict[str, list[str]]:
    """Return dict mapping intent to list of n example instructions."""
    rng = random.Random(seed)
    examples: dict[str, list[str]] = {}
    for intent in KNOWN_INTENTS:
        intent_rows = train_df[train_df["intent"] == intent]["instruction"].tolist()
        n = min(n_per_intent, len(intent_rows))
        examples[intent] = rng.sample(intent_rows, n) if n > 0 else []
    return examples


def format_few_shot_prompt(
    instruction: str,
    examples: dict[str, list[str]],
    n_intents_to_show: int = 5,
    seed: int | None = None,
) -> str:
    """Format a single instruction using randomly sampled few-shot examples."""
    rng = random.Random(seed)
    available_intents = [i for i in KNOWN_INTENTS if examples.get(i)]
    selected_intents = rng.sample(available_intents, min(n_intents_to_show, len(available_intents)))

    example_lines = []
    for intent in selected_intents:
        ex = rng.choice(examples[intent])
        example_lines.append(f"Customer message: {ex}\nIntent: {intent}")

    examples_block = "\n\n".join(example_lines)
    intent_list = "\n".join(f"- {i}" for i in KNOWN_INTENTS)

    return (
        "You are a customer support intent classifier. Given a customer message, "
        "classify it into exactly one of the following intents:\n\n"
        f"{intent_list}\n\n"
        "Here are some examples:\n\n"
        f"{examples_block}\n\n"
        "Now classify this message:\n"
        f"Customer message: {instruction}\n"
        "Intent:"
    )


def format_finetune_record(instruction: str, intent: str) -> dict:
    """Return a single chat-formatted record for mlx-lm fine-tuning."""
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Customer message: {instruction}"},
            {"role": "assistant", "content": intent},
        ]
    }


def save_few_shot_examples(
    examples: dict[str, list[str]],
    output_path: Path = Path("data/prompts/fewshot_examples.jsonl"),
) -> None:
    """Persist the few-shot example bank as JSONL."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for intent, exs in examples.items():
            f.write(json.dumps({"intent": intent, "examples": exs}) + "\n")
    logger.info(f"Saved {len(examples)} intent example sets to {output_path}")


def format_and_save_finetune_data(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    output_dir: Path = Path("data/processed"),
) -> None:
    """Format and save fine-tune JSONL files for train and val splits."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for split_name, split_df in [("finetune_train", train_df), ("finetune_val", val_df)]:
        output_path = output_dir / f"{split_name}.jsonl"
        with open(output_path, "w") as f:
            for _, row in split_df.iterrows():
                record = format_finetune_record(row["instruction"], row["intent"])
                f.write(json.dumps(record) + "\n")
        logger.info(f"Saved {len(split_df)} records to {output_path}")


if __name__ == "__main__":

    train_path = Path("data/processed/train.jsonl")
    val_path = Path("data/processed/val.jsonl")

    if not train_path.exists():
        raise FileNotFoundError(
            f"Train split not found at {train_path}. "
            "Run 'python -m src.data.splitter' first."
        )

    train_df = pd.read_json(train_path, lines=True)
    val_df = pd.read_json(val_path, lines=True)

    zeroshot_template_path = Path("data/prompts/zeroshot_template.txt")
    zeroshot_template_path.parent.mkdir(parents=True, exist_ok=True)
    intent_list_str = "\n".join(f"- {i}" for i in KNOWN_INTENTS)
    template_content = (
        "You are a customer support intent classifier. Given a customer message, "
        "classify it into exactly one of the following intents:\n\n"
        f"{intent_list_str}\n\n"
        "Respond with ONLY the intent name, nothing else.\n\n"
        "Customer message: {instruction}\n"
        "Intent:"
    )
    zeroshot_template_path.write_text(template_content)
    print(f"Saved zero-shot template to {zeroshot_template_path}")

    few_shot_examples = select_few_shot_examples(train_df)
    fewshot_path = Path("data/prompts/fewshot_examples.jsonl")
    save_few_shot_examples(few_shot_examples, fewshot_path)
    print(f"Saved few-shot examples to {fewshot_path} ({len(few_shot_examples)} intents)")

    format_and_save_finetune_data(train_df, val_df)
    print(f"Saved finetune_train.jsonl ({len(train_df)} rows)")
    print(f"Saved finetune_val.jsonl ({len(val_df)} rows)")
