import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["OPENPI_DATA_HOME"] = os.path.expanduser("~/.cache/openpi")

import time

import cv2
from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np

from openpi.models import model as _model
from openpi.models.model import Observation
from openpi.models.tokenizer import PaligemmaTokenizer
import openpi.shared.nnx_utils as nnx_utils
from openpi.training.config import get_config

PALIGEMMA_EOS_TOKEN = 1
max_decoding_steps = 25
temperature = 0.1

### Step 1: Initialize model and load pretrained params
print("debug line 0")
config = get_config("right_pi05_20")
model_rng = jax.random.key(0)
rng = jax.random.key(0)
print("debug line 0-0")
model = config.model.create(model_rng)

# Load pretrained params
print("debug line 0-1")
graphdef, state = nnx.split(model)
loader = config.weight_loader
params = nnx.state(model)
# Convert frozen params to bfloat16.
params = nnx_utils.state_map(params, config.freeze_filter, lambda p: p.replace(p.value.astype(jnp.bfloat16)))

print("denig line 0-2")
params_shape = params.to_pure_dict()
loaded_params = loader.load(params_shape)
state.replace_by_pure_dict(loaded_params)
model = nnx.merge(graphdef, state)

### Step 2: Construct an observation batch
# load 3 images from tmp_test as uint8 format
print("debug line 1")
img_share_path = os.path.dirname(os.path.abspath(__file__))
img_name_list = ["faceImg.png", "leftImg.png", "rightImg.png"]
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
high_level_prompt = "Pick up the flashcard on the table"
low_level_prompt = "ABCDEFG"
dummy_state = np.zeros(32, dtype=np.float32)  # Dummy state for tokenization
tokenizer = PaligemmaTokenizer(max_len=256, fast_tokenizer_path=None)
tokenized_prompt, tokenized_prompt_mask, token_ar_mask, token_loss_mask, subtask_region_mask, action_region_mask = tokenizer.tokenize_high_low_prompt(
    high_level_prompt, low_level_prompt, state=dummy_state
)

print("debug line 2")

# form a observation
data = {
    "image": img_dict,
    "image_mask": {key: jnp.ones(1, dtype=jnp.bool) for key in img_dict},
    "state": jnp.zeros((1, 32), dtype=jnp.float32),
    # 'state': None,
    "tokenized_prompt": jnp.stack([tokenized_prompt], axis=0),
    "tokenized_prompt_mask": jnp.stack([tokenized_prompt_mask], axis=0),
    "token_ar_mask": jnp.stack([token_ar_mask], axis=0).astype(jnp.int32),
    "token_loss_mask": jnp.stack([token_loss_mask], axis=0),
}
observation = Observation.from_dict(data)
rng = jax.random.key(42)
observation = _model.preprocess_observation(rng, observation, train=False, image_keys=list(observation.images.keys()))

# Set the low level task tokens to padding according to the loss mask (loss mask is the indication of low-level prompt)
# We move it from inside model to outside because the inside func need to be jittable
loss_mask = jnp.array(observation.token_loss_mask)
new_tokenized_prompt = observation.tokenized_prompt.at[loss_mask].set(0)
new_tokenized_prompt_mask = observation.tokenized_prompt_mask.at[loss_mask].set(False)
new_observation = _model.Observation(
    images=observation.images,
    image_masks=observation.image_masks,
    state=observation.state,
    tokenized_prompt=new_tokenized_prompt,
    tokenized_prompt_mask=new_tokenized_prompt_mask,
    token_ar_mask=observation.token_ar_mask,
    token_loss_mask=observation.token_loss_mask,
)
observation = _model.preprocess_observation(
    None, new_observation, train=False, image_keys=list(observation.images.keys())
)
observation = jax.tree.map(jax.device_put, observation)


### Step 3: Run one inference

# Jax Just-in-time compilation
# Make max_decoding_steps static for JIT to avoid tracer issues in jnp.pad
model.jit_sample_low_level_task = nnx_utils.module_jit(model.sample_low_level_task, static_argnums=(3,))
for _i in range(3):
    start_time = time.time()
    predicted_token, kv_cache, mask, ar_mask = model.jit_sample_low_level_task(
        rng, observation, max_decoding_steps, PALIGEMMA_EOS_TOKEN, temperature
    )
    for j in range(predicted_token.shape[0]):
        print("======================")
        print("\033[31m[PRED]\033[0m " + tokenizer.detokenize(np.array(predicted_token[j], dtype=np.int32)), flush=True)
        print(
            "\033[31m[MASK]\033[0m " + tokenizer.detokenize(np.array(data["tokenized_prompt"], dtype=np.int32)),
            flush=True,
        )
        print("======================")
    end_time = time.time()
    print(f"Time taken: {end_time - start_time} seconds")
    time.sleep(5)
# sampled_actions = model.sample_actions(rng, observation)

# action = jax.random.normal(rng, (1, 20, 32))
# output = model.compute_loss(rng, observation, action, train=True, real_action_dim=14)
# print(output)


### 20251005实验结果:完全乱输出,但是训练一下应该问题不大
### 20251006实验结果:0-shot 正常输出: move to table and then pick up black pen,之前不行是因为用了pi0的ckpt
### 20251010实验结果,0-shot 正常输出,jit加速后inference speed显著提升: A800上,第一次编译26s,之后稳定在1.4s左右
