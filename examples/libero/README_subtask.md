### Start the async server
```python
export OPENPI_DATA_HOME=$HOME/.cache/openpi
python scripts/async_pi05/async_pi05_websocket_server.py \
  --config libero_pi05_action_expert \
  --checkpoint /home/kewang/checkpoints/4000 \
  --host 0.0.0.0 \
  --port 8765

```

### Start the async LIBERO clint 

```python
python examples/libero/main_subtask.py --host 127.0.0.1 --port 8765
```