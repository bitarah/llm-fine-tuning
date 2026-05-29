# CLAUDE.md — LLM Fine-Tuning Portfolio Project

This document defines the rules, workflow, and constraints for implementing this **portfolio project**. It is read automatically by Claude Code at the start of every session.

**Key mindset:** This is portfolio-quality code (well-structured, readable, thoroughly tested), NOT production code (no edge-case handling for scenarios that can't happen, no defensive over-engineering, no backwards-compatibility shims).

---

## Development Workflow

### Session Protocol

Every session starts with `/implement "phase{N}-{name}"`. The skill will:

1. Read `docs/design-overview.md`, `docs/phases.md`, and the current phase spec
2. Implement the phase completely per the spec
3. Test everything with `make test-phase{N}`
4. Verify all acceptance criteria
5. Report completion and stop (do not start the next phase)

**You do not run `/implement` manually.** The user types `/implement "phase1-overview"` and you execute it.

---

## Your Role

- **Implement exactly what the spec says.** Do not improvise architecture or add features beyond the current phase.
- **Build one phase per session.** After `make test-phase{N}` passes and all acceptance criteria are met, stop.
- **Do not start the next phase.** That happens in a new session.
- **Read specs fully before writing code.** The first 5 minutes of every session are reading the docs, not coding.

---

## Code Quality Rules

### Separation of Concerns
- **Logic files** (`metrics.py`, `formatter.py`, `splitter.py`): pure functions, no file I/O, no model loading
- **I/O files** (`downloader.py`, `model_loader.py`): file reading/writing and API calls only, no business logic
- **Orchestration files** (`evaluator.py`, `app.py`): call other modules only, contain no inline logic
- If a function does more than one thing, split it

### No Hardcoded Values
Every path, model name, seed, threshold, and port must come from:
- `configs/training_config.yaml` or `configs/eval_config.yaml` (loaded via `src/config_loader.py`)
- Environment variables loaded from `.env` via `python-dotenv`
- Function parameters with sensible defaults that trace back to configs

Never: `model_id = "microsoft/Phi-3.5-mini-instruct"` inline in a training function.
Always: `model_id = config.model.base_model_id`

### Device Priority
All device selection must use `get_device()` from `src/training/config.py`.
Priority order: `mlx` → `mps` → `cuda` → `cpu`.
Never hardcode `"cuda"` or `"cpu"` in inference or training code.

### Type Annotations
- All public functions and class methods require full type annotations
- No `Any` unless absolutely unavoidable (mocked objects, dynamic configs)
- Return types always explicit

Example:
```python
def evaluate_intent(text: str, model_id: str) -> dict[str, float]:
    """Predict intent class and return confidence scores."""
    pass
```

### Docstrings
- All public functions require **one-line** docstrings minimum
- No multi-paragraph docstrings; no "Args:" or "Returns:" blocks
- Write only the WHY, not the WHAT (code already shows WHAT)
- Example: `"""Pre-tokenize text and cache results to avoid re-computation."""`

### Comments
- Default to **no comments**
- Add one only when the WHY is non-obvious:
  - A hidden constraint or subtle invariant
  - A workaround for a specific bug
  - Behavior that would surprise a reader
- Never explain WHAT the code does — well-named identifiers already do that
- Never reference the current task or callers — those rot in the code

### Fail Fast with Clarity
When a required file is missing, raise a descriptive error:

```python
if not adapter_path.exists():
    raise FileNotFoundError(
        f"Adapter not found at {adapter_path}. "
        "Run 'make finetune' to train the model first."
    )
```

The error message must tell the user exactly which `make` command fixes the problem.

### Logging
- Use Python's `logging` module for info/warning/debug messages in library code
- Use `print()` only for CLI-facing summaries (dataset statistics, training progress, final results)
- Never use `print()` for debugging — use `logging.debug()`

### Idempotency
Download and processing steps must check if outputs already exist:
```python
if output_path.exists() and not force_redownload:
    logging.info(f"File exists, skipping: {output_path}")
    return output_path
```

---

## Testing Rules

### Writing Tests
- Write tests in `tests/test_{module}.py` alongside each implementation file
- Use `pytest` fixtures — create synthetic DataFrames and dicts, never depend on real downloaded data
- Mock all external dependencies with `unittest.mock.patch`:
  - Model loading (mlx_lm, transformers, peft)
  - MLflow runs
  - HuggingFace dataset downloads
  - Network calls
- After completing all files in a phase, run `make test-phase{N}` and fix all failures

### conftest.py
Create `tests/conftest.py` with shared fixtures:
- `sample_dataframe` — synthetic DataFrame with correct schema (27 intents, ~50 rows)
- `tmp_output_dir` — temporary directory for file output tests
- `mock_config` — a TrainingConfig/EvalConfig loaded from actual YAML files

### Phase Completion
A phase is not complete until `make test-phase{N}` passes with **zero failures**.

---

## File Size Guard

Before marking any file complete, check its size:
```bash
wc -l src/path/to/file.py
```

If the file exceeds **900 lines**, immediately refactor:
1. Identify cohesive groups of functions (e.g. all tokenization logic, all MLflow logic)
2. Extract them into a new helper module in the same directory (e.g. `src/training/mlflow_utils.py`)
3. Update imports in the original file
4. Verify both files are under 900 lines
5. Run tests again to confirm nothing broke

The 1000-line limit is a hard ceiling. 900 lines is the refactor trigger.

---

## Phase Completion Protocol

At the end of every phase:

1. **Lint:** `make lint` — fix all ruff errors before declaring done
2. **Test:** `make test-phase{N}` — all tests must pass
3. **Verify:** Go through the **Acceptance Criteria Checklist** at the bottom of the phase spec, item by item
4. **Report a completion summary:**

```
Phase {N} complete.
Checked: {X}/{X} acceptance criteria.
Files created:
  - src/...
  - tests/...
Tests: {X} passed, 0 failed
Next: Start a new session with /implement "phase{N+1}-..."
```

**Do not proceed to the next phase. Stop here.**

---

## Handling Spec Ambiguity

If a spec detail is ambiguous:
1. Choose the simplest option that satisfies the stated acceptance criteria
2. Add a comment: `# NOTE: spec ambiguous — chose X because Y`
3. Implement and move on — do not pause to ask

---

## What NOT To Do

- **Do not create files or modules** not described in `design-overview.md` or the phase spec
- **Do not change the directory structure** from what `design-overview.md` defines
- **Do not install packages** not in `requirements.txt` or `requirements-mlx.txt` without noting it
- **Do not start the next phase** in this session
- **Do not leave `TODO` comments** — either implement it or raise a clear exception
- **Do not use `assert`** for runtime validation in production code — use explicit `if/raise`
- **Do not over-engineer** for hypothetical future requirements. Three similar lines is better than a premature abstraction.
- **Do not add error handling** for scenarios that can't happen. Trust internal code and framework guarantees.
- **Do not add feature flags** or backwards-compatibility shims when you can just change the code.

---

## Portfolio vs. Production Mindset

**This is a portfolio project, not production code. Implications:**

✅ **DO:**
- Write clean, readable, well-tested code
- Follow the spec exactly
- Use good abstractions and separation of concerns
- Document your thinking in git commits
- Make it portfolio-ready to show an interviewer

❌ **DON'T:**
- Add error handling for impossible edge cases (e.g., "what if PyTorch is installed but not importable?")
- Over-engineer for 3+ years of maintenance
- Add configuration for every parameter (some can be constants)
- Implement graceful degradation for missing dependencies
- Support multiple platforms if the spec targets Apple Silicon
- Spend time on non-functional requirements (performance optimization, monitoring, alerting)

---

## Quick Reference — Key File Locations

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

## For Every Session

1. **Read** `docs/design-overview.md`, `docs/phases.md`, current phase spec
2. **Implement** all files in the phase spec, top to bottom
3. **Test** with `make test-phase{N}`
4. **Verify** all acceptance criteria
5. **Report** completion
6. **Stop** — do not start the next phase

The `/implement` skill automates all of this. Just type `/implement "phase{N}-{name}"` and it handles the rest.
