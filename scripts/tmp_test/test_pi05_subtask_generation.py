import os
# GPU memory optimization settings
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.5"  # Use only 50% of GPU memory
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"  # Don't preallocate all memory
os.environ["JAX_ENABLE_X64"] = "false"  # Use 32-bit for memory efficiency
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"  # Use platform allocator for better memory management
# Remove invalid XLA flags
os.environ["OPENPI_DATA_HOME"] = "/root/.cache/openpi"

import numpy as np
import jax
import cv2
import gc
from flax import nnx
from openpi.models import model as _model
import openpi.shared.nnx_utils as nnx_utils
import jax.numpy as jnp
from openpi.training.config import get_config
from openpi.models.tokenizer import PaligemmaTokenizer
from openpi.models.model import Observation
from openpi.models.pi0 import make_attn_mask

PALIGEMMA_EOS_TOKEN = 1
max_decoding_steps = 100
temperature = 0.1

### Step 1: Initialize model and load pretrained params
# Use the original config for compatibility with pretrained weights
config = get_config("right_pi05_20")
model_rng = jax.random.key(0)
rng = jax.random.key(0)
model = config.model.create(model_rng)

# Load pretrained params
graphdef, state = nnx.split(model)
loader = config.weight_loader
params = nnx.state(model)
# Convert frozen params to bfloat16.
params = nnx_utils.state_map(params, config.freeze_filter, lambda p: p.replace(p.value.astype(jnp.bfloat16)))

params_shape = params.to_pure_dict()
loaded_params = loader.load(params_shape)
state.replace_by_pure_dict(loaded_params)
model = nnx.merge(graphdef, state)
# Clear memory after model loading and set model to eval mode
gc.collect()
jax.clear_caches()
model.eval()  # Set to evaluation mode to reduce memory usage

### Step 2: Construct an observation batch
# load 3 images from tmp_test as uint8 format
img_share_path = '/workspace/openpi/scripts/tmp_test'
img_name_list = ['faceImg.png', 'leftImg.png', 'rightImg.png']
img_list = []
for img_name in img_name_list:
    img_path = os.path.join(img_share_path, img_name)
    img = cv2.imread(img_path)
    img_list.append(img)
# Convert images from [0, 255] to [-1, 1] range as expected by the model
img_dict = {
    "base_0_rgb": jnp.array(img_list[0][np.newaxis, :, :, :]).astype(jnp.float32) / 127.5 - 1.0,
    "left_wrist_0_rgb": jnp.array(img_list[1][np.newaxis, :, :, :]).astype(jnp.float32) / 127.5 - 1.0,
    "right_wrist_0_rgb": jnp.array(img_list[2][np.newaxis, :, :, :]).astype(jnp.float32) / 127.5 - 1.0,
}


# Tokenize the prompt
high_level_prompt = 'Pick up the flashcard on the table'
low_level_prompt = 'ABCDEFG'
tokenizer = PaligemmaTokenizer(max_len=200)
tokenized_prompt, tokenized_prompt_mask, token_ar_mask, token_loss_mask = tokenizer.tokenize_high_low_prompt(high_level_prompt, low_level_prompt)

# form a observation
data = {
    'image': img_dict,
    'image_mask': {key: jnp.ones(1, dtype=jnp.bool) for key in img_dict.keys()},
    'state': jnp.zeros((1, 32), dtype=jnp.float32),
    # 'state': None,
    'tokenized_prompt': jnp.stack([tokenized_prompt], axis=0),
    'tokenized_prompt_mask': jnp.stack([tokenized_prompt_mask], axis=0),
    'token_ar_mask': jnp.stack([token_ar_mask], axis=0),
    'token_loss_mask': jnp.stack([token_loss_mask], axis=0),
}
observation = Observation.from_dict(data)
rng = jax.random.key(42)
observation = _model.preprocess_observation(rng, observation, train=False, image_keys=list(observation.images.keys()))

### Step 3: Run one inference
# Clear memory before inference
gc.collect()
jax.clear_caches()

# Move to CPU for generation to avoid memory issues
with jax.default_device(jax.devices('cpu')[0]):
    output_tokens = model.sample_low_level_task(rng, observation, max_decoding_steps, PALIGEMMA_EOS_TOKEN, temperature, tokenizer)

### 20251005实验结果：完全乱输出，但是训练一下应该问题不大
### 20251006实验结果：0-shot 正常输出： move to table and then pick up black pen，之前不行是因为用了pi0的ckpt