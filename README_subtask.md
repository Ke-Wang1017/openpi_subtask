# Pi0.5 Subtask Training Guide (LIBERO)

## Overview

Pi0.5 with subtask capability supports **two training strategies**:

| Strategy | Description |
|----------|-------------|
| **Joint Training** | Train subtask prediction, discrete action token prediction, and continuous action generation in a single run |
| **Knowledge Insulation** | Two-stage training: first finetune the VLM, then finetune the action expert ([paper](https://www.pi.website/research/knowledge_insulation)) |

---

## Training Strategies

### Strategy A: Joint Training (All Three Losses)

Train all three loss components simultaneously. This uses
`libero_pi05_subtask_hybrid` in `src/openpi/training/config.py`:

```python
# Mode: Subtask + FAST + Flow (Hybrid — all three losses)
TrainConfig(
    name="libero_pi05_subtask_hybrid",
    exp_name="libero_subtask_hybrid",
    model=pi05_config.Pi05Config(
        action_horizon=20,
        max_token_len=192,
        discrete_state_input=False,
        subtask_loss_weight=0.15,
        fast_token_loss_weight=0.15,
        flow_matching_loss_weight=1.0,
        fast_tokenizer_path="physical-intelligence/fast",
    ),
    weight_loader=weight_loaders.CheckpointWeightLoader(
        "/home/kewang/.cache/openpi/openpi-assets/checkpoints/pi05_base/params"
    ),
    data=LeRobotLiberoSubtaskDataConfig(
        repo_id="KeWangRobotics/libero_10_subtasks",
        base_config=DataConfig(
            asset_id="libero_subtask",
            use_quantile_norm=True,  # Quantile normalization for gripper actions
        ),
    ),
    lr_schedule=_optimizer.CosineDecaySchedule(
        warmup_steps=3000,
        peak_lr=2.5e-5,
        decay_steps=150_000,
        decay_lr=2.5e-6,
    ),
    num_train_steps=40_000,
    save_interval=5000,
    batch_size=64,
    fsdp_devices=1,
    ema_decay=0.999,
),
```

---

### Strategy B: Knowledge Insulation (Two Stages)

#### Stage 1 — Finetune the VLM (subtask + FAST token loss)

The VLM is finetuned while the action expert is frozen. Only subtask prediction
and discrete action token losses are used.

```python
# Mode: Subtask + FAST Token (discrete action tokens)
TrainConfig(
    name="libero_pi05_subtask_fast",
    exp_name="libero_subtask_fast",
    model=pi05_config.Pi05Config(
        action_horizon=25,
        max_token_len=256,
        discrete_state_input=False,
        subtask_loss_weight=10.0,
        fast_token_loss_weight=1.0,
        flow_matching_loss_weight=0.0,  # Disabled
        fast_tokenizer_path="physical-intelligence/fast",
    ),
    weight_loader=weight_loaders.CheckpointWeightLoader(
        "/home/kewang/.cache/openpi/openpi-assets/checkpoints/pi05_base/params"
    ),
    data=LeRobotLiberoSubtaskDataConfig(
        repo_id="KeWangRobotics/libero_10_subtasks",
        base_config=DataConfig(
            asset_id="libero_subtask",
            use_quantile_norm=True,
        ),
    ),
    lr_schedule=_optimizer.CosineDecaySchedule(
        warmup_steps=3000,
        peak_lr=2.5e-5,
        decay_steps=150_000,
        decay_lr=2.5e-6,
    ),
    num_train_steps=20_000,
    save_interval=4000,
    batch_size=512,
    fsdp_devices=8,
    ema_decay=0.999,
    wandb_enabled=True,
),
```

#### Stage 2 — Finetune the Action Expert (flow matching loss only)

The VLM is frozen and only the action expert is trained using flow matching loss.
Gradients are blocked from the VLM via `freeze_filter`. The checkpoint is
initialized from Stage 1.

```python
# Mode: Action Expert only (flow matching)
TrainConfig(
    name="libero_pi05_action_expert",
    exp_name="libero_action_expert",
    model=pi05_config.Pi05Config(
        action_horizon=25,
        max_token_len=256,
        discrete_state_input=False,
        subtask_loss_weight=0.0,       # Disabled
        fast_token_loss_weight=0.0,    # Disabled
        flow_matching_loss_weight=1.0,
        fast_tokenizer_path="physical-intelligence/fast",
    ),
    weight_loader=weight_loaders.CheckpointWeightLoader(
        "/home/kewang/.cache/openpi/openpi-checkpoints/libero_pi05_subtask_fast/my_experiment/12000/params"
    ),
    data=LeRobotLiberoSubtaskDataConfig(
        repo_id="KeWangRobotics/libero_10_subtasks",
        base_config=DataConfig(
            asset_id="libero_subtask",
            use_quantile_norm=True,
        ),
    ),
    lr_schedule=_optimizer.CosineDecaySchedule(
        warmup_steps=3000,
        peak_lr=2.5e-5,
        decay_steps=150_000,
        decay_lr=2.5e-6,
    ),
    num_train_steps=8_000,
    save_interval=2000,
    batch_size=512,
    fsdp_devices=8,
    ema_decay=0.999,
    wandb_enabled=True,
    freeze_filter=nnx.All(
        nnx.Param,
        nnx_utils.PathRegex(".*llm.*"),             # Freeze all LLM layers
        nnx.Not(nnx_utils.PathRegex(".*llm.*_1.*")), # Exclude action expert branch
    ),
),
```

---

## Setup

### 1. Download the FAST Tokenizer

```bash
python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download(repo_id="physical-intelligence/fast")
PY
```

### 2. Download the Pi0.5 Base Model

```bash
python - <<'PY'
from openpi.training import config as _config
from openpi.shared import download

config = _config.get_config("pi05_base")
checkpoint_dir = download.maybe_download("gs://openpi-assets/checkpoints/pi05_base")
PY
```

---

## Running Training

### Option A: Joint Training

```bash
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py libero_pi05_subtask_hybrid \
  --exp-name=my_experiment_all \
  --overwrite
```

### Option B: Knowledge Insulation

**Phase 1** — Finetune the VLM:

```bash
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py libero_pi05_subtask_fast \
  --exp-name=my_experiment_all \
  --overwrite
```

**Phase 2** — Finetune the action expert (resume from Phase 1 checkpoint):

```bash
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py libero_pi05_action_expert \
  --exp-name=my_experiment_all \
  --overwrite
```

---

## Evaluation

First, see the [LIBERO README](README.md) to set up the environment.

### Synchronous Server + Client

**Start the Pi0.5 server:**

```bash
export OPENPI_DATA_HOME=$HOME/.cache/openpi
python scripts/async_pi05/sync_pi05_websocket_server.py \
  --config libero_pi05_action_expert \
  --checkpoint PATH_TO_CHECKPOINT \
  --gpu-id 0 \
  --host 0.0.0.0 \
  --port 8765
```

**Start the LIBERO client:**

```bash
source examples/libero/.venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$PWD/third_party/libero
python examples/libero/main_subtask.py --host 127.0.0.1 --port 8765
```

---

### Asynchronous Server + Client

Use the async stack for true non-blocking inference.

**Start the async Pi0.5 server:**

```bash
export OPENPI_DATA_HOME=$HOME/.cache/openpi
python scripts/async_pi05/async_pi05_websocket_server.py \
  --config libero_pi05_action_expert \
  --checkpoint PATH_TO_CHECKPOINT \
  --gpu-id 0 \
  --host 0.0.0.0 \
  --port 8765
```

**Test with a single query:**

```bash
python scripts/async_pi05/async_pi05_client.py \
  --host 127.0.0.1 \
  --port 8765 \
  --high-level-prompt "Pick up the flashcard on the table"
```

**Run the async LIBERO evaluation:**

```bash
source examples/libero/.venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$PWD/third_party/libero
python examples/libero/main_subtask_async.py --host 127.0.0.1 --port 8765
```
