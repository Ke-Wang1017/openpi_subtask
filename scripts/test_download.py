from openpi.training import config as _config
from openpi.policies import policy_config
from openpi.shared import download

config = _config.get_config("pi05_droid")
# checkpoint_dir = download.maybe_download("gs://openpi-assets/checkpoints/pi05_base")
checkpoint_dir = download.maybe_download("gs://big_vision/paligemma_tokenizer.model")