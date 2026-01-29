### Run LIBERO subtask evaluation (sync server + sync client)
First see the [LIBERO README](README.md) to setup the environment. Then run:


### Start the sync Pi0.5 server
```bash
export OPENPI_DATA_HOME=$HOME/.cache/openpi
python scripts/sync_pi05_websocket_server.py \
  --config libero_pi05_action_expert \
  --checkpoint /home/kewang/checkpoints/4000 \
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
python examples/libero/main_subtask_synchronous.py --host 127.0.0.1 --port 8765
```
