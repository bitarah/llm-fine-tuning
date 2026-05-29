---
name: implement
description: Read phase documentation and execute implementation with full testing
trigger: "implement \"{phase}\""
---

# /implement "{phase}" Skill

**Purpose:** Phase-driven development. Read the phase spec, implement everything in it, test thoroughly, and stop before the next phase.

**Invocation:** `/implement "phase1-overview"` or `/implement "phase2-data-processing"`

---

## Execution Flow

### 1. Read All Documentation (5 minutes)

Before writing ANY code:

```bash
cat docs/design-overview.md
cat docs/phases.md
cat docs/{phase}*.md
find src/ -name "*.py" 2>/dev/null | head -50
make test 2>/dev/null || echo "No tests yet"
```

Skip any read that fails — that file may not exist yet.

**Output:** You now understand:
- The full project architecture
- The complete phase dependency graph
- The current phase's exact requirements
- What code already exists to avoid duplication
- The current test state

---

### 2. Implement the Phase (per spec)

Work through the phase spec **top to bottom**. For each file:

1. Create it at the exact path specified
2. Implement all functions with exact signatures shown
3. Add full type annotations to every public function/method
4. Add one-line docstrings to every public function
5. Add `if __name__ == "__main__":` entry points where required
6. Check line count: if >900 lines, split immediately

**Separation of Concerns:**
- **Logic files:** Pure functions, no I/O, no model loading
- **I/O files:** File reading/writing and API calls only, no business logic
- **Orchestration files:** Call other modules only, contain no inline logic

**No Hardcoded Values:**
- All paths, model names, seeds, thresholds from `configs/*.yaml` or env vars
- Load via `src/config_loader.py` or `os.getenv()`
- Never inline: `model_id = "microsoft/Phi-3.5-mini-instruct"`
- Always: `model_id = config.model.base_model_id`

**Device Priority:**
- Use `get_device()` from `src/training/config.py`
- Order: `mlx` → `mps` → `cuda` → `cpu`
- Never hardcode `"cuda"` or `"cpu"`

**Fail Fast:**
When a required file is missing:
```python
if not adapter_path.exists():
    raise FileNotFoundError(
        f"Adapter not found at {adapter_path}. "
        "Run 'make finetune' to train the model first."
    )
```

**Logging:**
- `logging` module for library code (info/warning/debug)
- `print()` only for CLI-facing summaries (dataset stats, training progress)
- Never use `print()` for debugging

**Idempotency:**
```python
if output_path.exists() and not force_redownload:
    logging.info(f"File exists, skipping: {output_path}")
    return output_path
```

---

### 3. Write Tests (same session)

For each implementation file, write `tests/test_{module}.py`:

- Use `pytest` fixtures
- Create synthetic test data (DataFrames, dicts)
- Mock all external dependencies:
  - Model loading (mlx_lm, transformers, peft)
  - MLflow runs
  - HuggingFace dataset downloads
  - Network calls
- Use `unittest.mock.patch`

**Create `tests/conftest.py` with shared fixtures:**
```python
@pytest.fixture
def sample_dataframe():
    """Synthetic DataFrame with correct schema (27 intents, ~50 rows)"""
    pass

@pytest.fixture
def tmp_output_dir():
    """Temporary directory for file output tests"""
    pass

@pytest.fixture
def mock_config():
    """TrainingConfig/EvalConfig loaded from actual YAML files"""
    pass
```

---

### 4. Phase Completion (5 minutes)

At the end of the phase:

1. **Lint:** `make lint` — fix all ruff errors
2. **Test:** `make test-phase{N}` — all tests must pass
3. **Verify:** Go through **Acceptance Criteria Checklist** at bottom of phase spec, item by item
4. **Report:**

```
Phase {N} complete.
Checked: {X}/{X} acceptance criteria.
Files created:
  - src/...
  - tests/...
  - configs/...
Tests: {X} passed, 0 failed
Next: Start a new session with /implement "phase{N+1}-..."
```

**Do not proceed to the next phase. Stop here.**

---

## Rules (Apply Every Session)

### File Size Guard
```bash
wc -l src/path/to/file.py
```
If >900 lines, immediately refactor:
1. Identify cohesive groups of functions
2. Extract to new helper module in same directory
3. Update imports in original
4. Verify both <900 lines
5. Run tests to confirm nothing broke

The 1000-line limit is a hard ceiling.

### What NOT To Do
- Do not create files or modules not in `design-overview.md` or phase spec
- Do not change directory structure from `design-overview.md`
- Do not install packages not in `requirements.txt` / `requirements-mlx.txt`
- Do not start the next phase in this session
- Do not leave `TODO` comments — implement or raise clear exception
- Do not use `assert` for runtime validation — use explicit `if/raise`

### Ambiguity Handling
If a spec detail is ambiguous:
1. Choose the simplest option that satisfies stated acceptance criteria
2. Add comment: `# NOTE: spec ambiguous — chose X because Y`
3. Implement and move on

---

## Key File Locations

| What | Where |
|---|---|
| Full architecture | `docs/design-overview.md` |
| Phase sequencing | `docs/phases.md` |
| Training config | `configs/training_config.yaml` |
| Eval config | `configs/eval_config.yaml` |
| Config loader | `src/config_loader.py` |
| Device detection | `src/training/config.get_device()` |
| Intent list | `src/data/preprocessor.KNOWN_INTENTS` |
| Adapter metadata | `models/training/adapter_info.json` |
| Benchmark results | `experiments/results/benchmark_summary.json` |
| MLflow store | `experiments/mlflow/` |

---

## Success Criteria

You are done when:
- ✅ All files in the spec exist with correct content
- ✅ All functions have full type annotations and one-line docstrings
- ✅ No file exceeds 900 lines (1000 is hard ceiling)
- ✅ `make lint` passes (zero ruff errors)
- ✅ `make test-phase{N}` passes (zero test failures)
- ✅ Every acceptance criterion in the phase spec is checked ✓
- ✅ You have NOT started the next phase

**Report completion. Stop. Wait for new session.**
