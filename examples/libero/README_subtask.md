To run a training, first download the FAST tokenizer with bash command:
```bash
python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download(repo_id="physical-intelligence/fast")
PY
```
python - <<'PY'
from openpi.training import config as _config
from openpi.policies import policy_config
from openpi.shared import download

config = _config.get_config("pi05_base")
checkpoint_dir = download.maybe_download("gs://openpi-assets/checkpoints/pi05_base")
PY

Then we can kick off training! There are two choices:
1, First you can train with three phases as the original PI05 paper:
```python
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py libero_pi05_subtask_hybrid --exp-name=my_experiment_all --overwrit
e
```
2, Or you can train with the Knowledge Insulation which has two phases:
For phase one which train the VLM with FAST tokenizer
```python
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py libero_pi05_subtask_fast --exp-name=my_experiment_all --overwrit
e
```
For phase two in which the VLM is frozen and the action expert is finetuned,
and the checkpoint is resumed from the phase 1 model:

```python
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py libero_pi05_action_expert --exp-name=my_experiment_all --overwrit
e
```


### Run LIBERO subtask evaluation (sync server + sync client)
First see the [LIBERO README](README.md) to setup the environment. Then run:


### Start the sync Pi0.5 server
```bash
export OPENPI_DATA_HOME=$HOME/.cache/openpi
python scripts/async_pi05_websocket_server.py \
  --config libero_pi05_action_expert \
  --checkpoint PATH_TO_CHECKPOINT \
  --gpu-id 0 \
  --host 0.0.0.0 \
  --port 8765
```

### Start the sync LIBERO client
```bash
# Create virtual environment
source examples/libero/.venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$PWD/third_party/libero
```
```bash
python examples/libero/main_subtask.py --host 127.0.0.1 --port 8765
```
