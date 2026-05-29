# Phase 3 — Fine-Tuning

## Overview

This phase fine-tunes Phi-3.5-mini-instruct using LoRA via Apple MLX as the primary framework. All training metrics and hyperparameters are logged to MLflow. A PyTorch MPS fallback trainer is also implemented for non-MLX environments.

Read `design-overview.md` before starting. Phases 1 and 2 must be complete — the fine-tuning JSONL data must exist at `data/processed/finetune_train.jsonl`.

---

## Before You Start

```bash
make test-phase2
ls data/processed/finetune_train.jsonl    # must exist
ls data/processed/finetune_val.jsonl      # must exist
```

---

## Hardware & Framework Strategy

### Primary Path: Apple MLX
`mlx-lm` provides native LoRA fine-tuning on Apple Silicon via Metal GPU.
- Uses Metal Performance Shaders via MLX's unified memory model
- Native 4-bit quantisation with `mlx_lm.utils.quantize`
- LoRA via `mlx_lm.lora` — no CUDA dependencies
- Adapter saved as `.npz` weights to `models/adapters/`

### Fallback Path: PyTorch MPS + HuggingFace TRL
If MLX is unavailable:
- `peft` for LoRA adapter injection
- `trl.SFTTrainer` for supervised fine-tuning
- `torch.backends.mps.is_available()` to route to MPS device
- Note: PyTorch MPS does not support BitsAndBytes quantisation — use fp16 with LoRA only

### Device Detection (implement in `src/training/config.py`)

```python
def get_device() -> str:
    """Return best available compute device: mlx, mps, cuda, or cpu."""
    try:
        import mlx.core as mx
        mx.metal.is_available()
        return "mlx"
    except (ImportError, Exception):
        pass
    import torch
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"
```

---

## File: `src/training/config.py`

**Responsibility:** Training configuration dataclasses and device utilities. Keep under 150 lines.

Define nested dataclasses mirroring `configs/training_config.yaml`:
- `ModelConfig` — base_model_id, model_max_length, trust_remote_code
- `LoraConfig` — rank, alpha, dropout, target_modules
- `QuantizationConfig` — enabled, bits
- `TrainingParams` — num_epochs, batch_size, gradient_accumulation_steps, learning_rate, lr_scheduler, warmup_ratio, weight_decay, max_grad_norm, seed
- `PathsConfig` — train_data, val_data, output_dir, mlflow_tracking_uri
- `LoggingConfig` — log_every_n_steps, eval_every_n_steps, save_every_n_steps, experiment_name, run_name
- `TrainingConfig` — composite of all above

**Function signatures:**

```python
def load_training_config(path: Path = Path("configs/training_config.yaml")) -> TrainingConfig:
    """Load YAML and return typed TrainingConfig. Raises FileNotFoundError if missing."""

def get_device() -> str:
    """Return best available compute device string: mlx, mps, cuda, or cpu."""

def log_device_info() -> None:
    """Print which device is being used and why."""
```

---

## File: `src/training/callbacks.py`

**Responsibility:** MLflow logging integration for training loops.

**`MLflowCallback` class:**

```python
class MLflowCallback:
    def on_train_begin(self, config: TrainingConfig) -> None:
        """Start MLflow run and log all hyperparameters as params."""

    def on_step(self, step: int, loss: float, lr: float) -> None:
        """Log step-level metrics: train_loss, learning_rate."""

    def on_epoch_end(self, epoch: int, train_loss: float, val_loss: float) -> None:
        """Log epoch-level metrics: train_loss_epoch, val_loss_epoch."""

    def on_train_end(self, adapter_path: Path) -> None:
        """Log adapter artifact and end the MLflow run."""
```

**Additional functions:**

```python
def get_or_create_experiment(name: str, tracking_uri: str) -> str:
    """Return MLflow experiment ID, creating it if it does not exist."""
```

**Rules:**
- Flatten all config fields with dot notation for MLflow params (e.g. `lora.rank`, `training.seed`)
- Wrap all MLflow operations in try/except — logging failure must never crash training
- Tag each run with: `device`, `base_model_id`, `framework` (mlx or pytorch)

---

## File: `src/training/trainer_mlx.py`

**Responsibility:** Primary MLX fine-tuning entry point.

### Implementation Steps

**1. Model Loading**
```python
def load_and_quantize_model(config: TrainingConfig) -> tuple:
    """Load base model, apply 4-bit quantisation if enabled. Returns (model, tokenizer)."""
    # model, tokenizer = mlx_lm.utils.load(config.model.base_model_id)
    # if config.quantization.enabled:
    #     mlx_lm.utils.quantize(model, bits=config.quantization.bits)
```
Log total and trainable parameter counts before training.

**2. LoRA Setup**
```python
def apply_lora(model, config: TrainingConfig):
    """Apply LoRA layers to model. Returns modified model."""
```
Print: "LoRA applied to {n} layers, trainable params: {count} ({pct:.1f}%)"

**3. Data Loading**
- Load `finetune_train.jsonl` and `finetune_val.jsonl` using `mlx_lm.utils.load_dataset`
- Validate JSONL has `messages` key in every row before starting (fail fast)

**4. Training Loop**
```python
def run_training(config: TrainingConfig) -> Path:
    """Execute full training loop. Returns path to saved adapter."""
```
- Use `mlx_lm.lora.train()` or equivalent MLX training utilities
- Log via `MLflowCallback` every `log_every_n_steps`
- Evaluate on validation set every `eval_every_n_steps`
- Save checkpoint every `save_every_n_steps` to `models/adapters/checkpoints/step_{N}/`
- Write `models/adapters/checkpoints/latest.txt` with current step number
- Handle `KeyboardInterrupt` gracefully — save adapter before exiting

**5. Adapter Saving**
```python
def save_adapter(model, tokenizer, output_dir: Path) -> None:
    """Save LoRA adapter weights and tokenizer config to output_dir."""
```
After saving, write `models/training/adapter_info.json`:
```json
{"framework": "mlx", "base_model_id": "...", "adapter_path": "models/adapters/final"}
```

**Module entry point:** Parse optional `--config` and `--resume` CLI args. Check device is `mlx`. Call `run_training()`.

---

## File: `src/training/trainer_pytorch.py`

**Responsibility:** PyTorch MPS fallback trainer. Mirrors the MLX trainer interface.

**Device setup:**
- MPS: `fp16=False`, `bf16=False`, `load_in_4bit=False` (MPS does not support BitsAndBytes)
- CUDA: use `BitsAndBytesConfig` for 4-bit
- CPU: fp32, warn user training will be slow

**Implementation:**
- Load with `transformers.AutoModelForCausalLM.from_pretrained()`
- Apply PEFT: `peft.LoraConfig` → `peft.get_peft_model()`
- Load data with `datasets.load_dataset("json", ...)` → tokenise with model's chat template
- Train with `trl.SFTTrainer`
- Use same `MLflowCallback`
- Save to same `models/adapters/final/` path
- Write `adapter_info.json` with `"framework": "pytorch"`

---

## File: `tests/test_training.py`

Do not load real models in tests — mock all model/tokenizer objects.

**Required tests:**
```python
def test_load_training_config_returns_correct_seed():
def test_load_training_config_raises_on_missing_file():
def test_get_device_returns_valid_string():
    # must return one of: "mlx", "mps", "cuda", "cpu"
def test_mlflow_callback_logs_params_with_dot_notation(mock_mlflow):
def test_mlflow_callback_handles_logging_failure_gracefully():
def test_adapter_info_json_written_with_correct_keys(tmp_path):
def test_trainer_raises_on_missing_training_data():
def test_resume_flag_detected_from_checkpoint_file(tmp_path):
```

Use `unittest.mock.patch` to mock `mlx_lm`, `torch`, and `mlflow` imports.

---

## MLflow Experiment Structure

After training, MLflow must show:

```
Experiment: intent-classification
└── Run: phi35-lora-run-{timestamp}
    ├── Parameters: model.base_model_id, lora.rank, lora.alpha,
    │               training.num_epochs, training.learning_rate, ...
    ├── Metrics: train_loss (per step), train_loss_epoch, val_loss_epoch (per epoch)
    ├── Tags: device=mlx, framework=mlx, base_model_id=...
    └── Artifacts: training_config.yaml, adapter/ (weights)
```

---

## Checkpoint & Resume

1. Save checkpoint every `save_every_n_steps` to `models/adapters/checkpoints/step_{N}/`
2. Write `models/adapters/checkpoints/latest.txt` with the step number
3. On `--resume`: read `latest.txt`, load that checkpoint, continue training from that step
4. On `KeyboardInterrupt`: save final adapter to `models/adapters/interrupted/` before exiting

---

## Acceptance Criteria Checklist

- [ ] `make finetune` completes without error on Apple Silicon using MLX
- [ ] Training logs confirm `device = mlx` (Metal is being used)
- [ ] LoRA adapter saved to `models/adapters/final/`
- [ ] `models/training/adapter_info.json` exists with correct `framework` and `adapter_path` keys
- [ ] MLflow run visible via `mlflow ui --backend-store-uri experiments/mlflow`
- [ ] MLflow run contains: all hyperparameters, step-level and epoch-level loss curves, adapter artifact
- [ ] Training resumable with `--resume` flag
- [ ] `make test-phase3` passes all training tests
- [ ] `trainer_mlx.py` under 1000 lines; `trainer_pytorch.py` under 1000 lines
- [ ] No hardcoded model names or paths in trainer files

**Do not proceed to Phase 4 until all items are checked.**
