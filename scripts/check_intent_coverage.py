"""Check that KNOWN_INTENTS exactly covers all intent labels in the training data."""
from __future__ import annotations

import json
import sys
from pathlib import Path

# Resolve project root regardless of where the script is called from
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.preprocessor import KNOWN_INTENTS

DATA_FILES = [
    PROJECT_ROOT / "data/processed/train.jsonl",
    PROJECT_ROOT / "data/processed/val.jsonl",
    PROJECT_ROOT / "data/processed/test.jsonl",
]


def load_intents_from_file(path: Path) -> set[str]:
    intents = set()
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                record = json.loads(line)
                intent = record.get("intent", "").strip()
                if intent:
                    intents.add(intent)
    return intents


def main() -> None:
    known = set(KNOWN_INTENTS)
    data_intents: set[str] = set()

    print("Scanning data files...")
    for path in DATA_FILES:
        if not path.exists():
            print(f"  MISSING  {path.relative_to(PROJECT_ROOT)}")
            continue
        found = load_intents_from_file(path)
        print(f"  {len(found):>3} intents  {path.relative_to(PROJECT_ROOT)}")
        data_intents |= found

    print(f"\nKNOWN_INTENTS : {len(known)} entries")
    print(f"Data intents  : {len(data_intents)} unique labels\n")

    in_data_not_known = sorted(data_intents - known)
    in_known_not_data = sorted(known - data_intents)
    exact_match = known == data_intents

    if in_data_not_known:
        print(f"[PROBLEM] In data but NOT in KNOWN_INTENTS ({len(in_data_not_known)}):")
        for i in in_data_not_known:
            print(f"    '{i}'")
    else:
        print("[OK] No data intents are missing from KNOWN_INTENTS.")

    if in_known_not_data:
        print(f"\n[PROBLEM] In KNOWN_INTENTS but NOT in data ({len(in_known_not_data)}):")
        for i in in_known_not_data:
            print(f"    '{i}'")
    else:
        print("[OK] No KNOWN_INTENTS entries are absent from data.")

    # String-level sanity: check for whitespace, casing, or invisible character issues
    print("\nString-level checks:")
    mismatches = []
    for intent in sorted(data_intents & known):
        # Verify the string bytes are identical (catches invisible chars / encoding issues)
        if intent.encode() != intent.strip().lower().encode():
            mismatches.append(intent)
    if mismatches:
        print(f"  [PROBLEM] Strings with whitespace or casing anomalies: {mismatches}")
    else:
        print("  [OK] All shared labels are clean (no whitespace/casing issues).")

    print()
    if exact_match:
        print("RESULT: EXACT MATCH — KNOWN_INTENTS perfectly covers the data.")
    else:
        print("RESULT: MISMATCH — see problems above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
