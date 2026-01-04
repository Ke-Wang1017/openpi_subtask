from typing import Literal

import pytest
import torch
from torch import nn
from transformers import GemmaForCausalLM
from transformers import PaliGemmaForConditionalGeneration
from transformers.models.auto import CONFIG_MAPPING
from transformers.models.gemma import modeling_gemma


# Type alias for KV cache: list of (idx, k_cache, v_cache) tuples, one per layer
KVCache = list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]


class PaliGemmaWithExpertModel(nn.Module):
    def __init__(
        self,
        vlm_config,
        action_expert_config,
        use_adarms=None,
        precision: Literal["bfloat16", "float32"] = "bfloat16",
    ):
        if use_adarms is None:
            use_adarms = [False, False]
        super().__init__()

        vlm_config_hf = CONFIG_MAPPING["paligemma"]()
        vlm_config_hf._vocab_size = 257152  # noqa: SLF001
        vlm_config_hf.image_token_index = 257152
        vlm_config_hf.text_config.hidden_size = vlm_config.width
        vlm_config_hf.text_config.intermediate_size = vlm_config.mlp_dim
        vlm_config_hf.text_config.num_attention_heads = vlm_config.num_heads
        vlm_config_hf.text_config.head_dim = vlm_config.head_dim
        vlm_config_hf.text_config.num_hidden_layers = vlm_config.depth
        vlm_config_hf.text_config.num_key_value_heads = vlm_config.num_kv_heads
        vlm_config_hf.text_config.hidden_activation = "gelu_pytorch_tanh"
        vlm_config_hf.text_config.torch_dtype = "float32"
        vlm_config_hf.text_config.vocab_size = 257152
        vlm_config_hf.text_config.use_adarms = use_adarms[0]
        vlm_config_hf.text_config.adarms_cond_dim = vlm_config.width if use_adarms[0] else None
        vlm_config_hf.vision_config.intermediate_size = 4304
        vlm_config_hf.vision_config.projection_dim = 2048
        vlm_config_hf.vision_config.projector_hidden_act = "gelu_fast"
        vlm_config_hf.vision_config.torch_dtype = "float32"

        action_expert_config_hf = CONFIG_MAPPING["gemma"](
            head_dim=action_expert_config.head_dim,
            hidden_size=action_expert_config.width,
            intermediate_size=action_expert_config.mlp_dim,
            num_attention_heads=action_expert_config.num_heads,
            num_hidden_layers=action_expert_config.depth,
            num_key_value_heads=action_expert_config.num_kv_heads,
            vocab_size=257152,
            hidden_activation="gelu_pytorch_tanh",
            torch_dtype="float32",
            use_adarms=use_adarms[1],
            adarms_cond_dim=action_expert_config.width if use_adarms[1] else None,
        )

        self.paligemma = PaliGemmaForConditionalGeneration(config=vlm_config_hf)
        self.gemma_expert = GemmaForCausalLM(config=action_expert_config_hf)
        self.gemma_expert.model.embed_tokens = None

        self.to_bfloat16_for_selected_params(precision)

    def to_bfloat16_for_selected_params(self, precision: Literal["bfloat16", "float32"] = "bfloat16"):
        if precision == "bfloat16":
            self.to(dtype=torch.bfloat16)
        elif precision == "float32":
            self.to(dtype=torch.float32)
            return
        else:
            raise ValueError(f"Invalid precision: {precision}")

        params_to_keep_float32 = [
            "vision_tower.vision_model.embeddings.patch_embedding.weight",
            "vision_tower.vision_model.embeddings.patch_embedding.bias",
            "vision_tower.vision_model.embeddings.position_embedding.weight",
            "input_layernorm",
            "post_attention_layernorm",
            "model.norm",
        ]

        for name, param in self.named_parameters():
            if any(selector in name for selector in params_to_keep_float32):
                param.data = param.data.to(dtype=torch.float32)

    def embed_image(self, image: torch.Tensor):
        return self.paligemma.model.get_image_features(image)

    def embed_language_tokens(self, tokens: torch.Tensor):
        return self.paligemma.language_model.embed_tokens(tokens)

    def deembed(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Convert embeddings back to logits over vocabulary.

        This is the inverse operation of embed_language_tokens().
        Equivalent to JAX version: jnp.dot(embeddings, embedding_table.T)

        Args:
            embeddings: Tensor of shape (batch, seq_len, hidden_size)

        Returns:
            Logits tensor of shape (batch, seq_len, vocab_size)
        """
        # Matrix multiplication: embeddings @ embedding_table.T
        # embedding_table shape: (vocab_size, hidden_size)
        # Result shape: (batch, seq_len, vocab_size)
        return torch.matmul(embeddings, self.paligemma.language_model.embed_tokens.weight.T)

    def _init_cache(
        self,
        k: torch.Tensor,
        v: torch.Tensor,
        cache_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Initialize KV cache with padding to cache_size.

        Args:
            k: Key tensor of shape (batch, num_heads, seq_len, head_dim)
            v: Value tensor of shape (batch, num_heads, seq_len, head_dim)
            cache_size: Maximum cache size (total sequence length)

        Returns:
            Tuple of (idx, k_cache, v_cache) where:
                - idx: Current position index tensor of shape (batch,)
                - k_cache: Padded key cache of shape (batch, num_heads, cache_size, head_dim)
                - v_cache: Padded value cache of shape (batch, num_heads, cache_size, head_dim)
        """
        batch_size, num_heads, prefill_len, head_dim = k.shape

        # Create padded cache tensors
        k_cache = torch.zeros(
            batch_size, num_heads, cache_size, head_dim,
            dtype=k.dtype, device=k.device
        )
        v_cache = torch.zeros(
            batch_size, num_heads, cache_size, head_dim,
            dtype=v.dtype, device=v.device
        )

        # Copy initial k, v into cache
        k_cache[:, :, :prefill_len, :] = k
        v_cache[:, :, :prefill_len, :] = v

        # Track current index (position after prefill)
        idx = torch.full((batch_size,), prefill_len, dtype=torch.int64, device=k.device)

        return idx, k_cache, v_cache

    def _update_cache(
        self,
        k: torch.Tensor,
        v: torch.Tensor,
        idx: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Update KV cache with new key-value pairs (single token).

        Args:
            k: New key tensor of shape (batch, num_heads, 1, head_dim)
            v: New value tensor of shape (batch, num_heads, 1, head_dim)
            idx: Current position index tensor of shape (batch,)
            k_cache: Existing key cache of shape (batch, num_heads, cache_size, head_dim)
            v_cache: Existing value cache of shape (batch, num_heads, cache_size, head_dim)

        Returns:
            Tuple of (idx_new, k_cache, v_cache) with updated cache
        """
        assert k.shape[2] == 1, "Only support kv-cache updates of length 1"

        current_idx = idx[0].item()  # Assume same index for all batch elements

        # Update cache at current position using clone to avoid in-place modification
        k_cache = k_cache.clone()
        v_cache = v_cache.clone()
        k_cache[:, :, current_idx : current_idx + 1, :] = k
        v_cache[:, :, current_idx : current_idx + 1, :] = v

        # Increment index
        idx_new = idx + 1

        return idx_new, k_cache, v_cache

    def forward(
        self,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: list[torch.FloatTensor] | pytest.Cache | None = None,
        inputs_embeds: list[torch.FloatTensor] | None = None,
        use_cache: bool | None = None,
        adarms_cond: list[torch.Tensor] | None = None,
        kv_cache: KVCache | None = None,
    ):
        if adarms_cond is None:
            adarms_cond = [None, None]
        if inputs_embeds[1] is None:
            prefix_output = self.paligemma.language_model.forward(
                inputs_embeds=inputs_embeds[0],
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                adarms_cond=adarms_cond[0] if adarms_cond is not None else None,
            )
            prefix_past_key_values = prefix_output.past_key_values
            prefix_output = prefix_output.last_hidden_state
            suffix_output = None
            return [prefix_output, suffix_output], prefix_past_key_values, kv_cache
        elif inputs_embeds[0] is None:
            suffix_output = self.gemma_expert.model.forward(
                inputs_embeds=inputs_embeds[1],
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                adarms_cond=adarms_cond[1] if adarms_cond is not None else None,
            )
            suffix_output = suffix_output.last_hidden_state
            prefix_output = None
            prefix_past_key_values = None
            return [prefix_output, suffix_output], prefix_past_key_values, kv_cache
        else:
            models = [self.paligemma.language_model, self.gemma_expert.model]
            num_layers = self.paligemma.config.text_config.num_hidden_layers

            # Check if gradient checkpointing is enabled for any of the models
            use_gradient_checkpointing = (
                hasattr(self.gemma_expert.model, "gradient_checkpointing")
                and self.gemma_expert.model.gradient_checkpointing
                and self.training
            ) or (hasattr(self, "gradient_checkpointing") and self.gradient_checkpointing and self.training)

            # Force enable gradient checkpointing if we're in training mode and the model supports it
            if self.training and hasattr(self.gemma_expert.model, "gradient_checkpointing"):
                if not self.gemma_expert.model.gradient_checkpointing:
                    print("Forcing gradient checkpointing to be enabled for Gemma expert model")
                    self.gemma_expert.model.gradient_checkpointing = True
                use_gradient_checkpointing = True

            # Debug gradient checkpointing status
            if hasattr(self, "_debug_gc_printed") and not self._debug_gc_printed:
                print(f"Gemma expert model gradient checkpointing: {use_gradient_checkpointing}")
                print(f"Model training mode: {self.training}")
                print(
                    f"Gemma expert model has gradient_checkpointing attr: {hasattr(self.gemma_expert.model, 'gradient_checkpointing')}"
                )
                if hasattr(self.gemma_expert.model, "gradient_checkpointing"):
                    print(
                        f"Gemma expert model gradient_checkpointing value: {self.gemma_expert.model.gradient_checkpointing}"
                    )
                self._debug_gc_printed = True

            # Define the complete layer computation function for gradient checkpointing
            def compute_layer_complete(
                layer_idx, inputs_embeds, attention_mask, position_ids, adarms_cond, layer_kv_cache=None
            ):
                models = [self.paligemma.language_model, self.gemma_expert.model]

                query_states = []
                key_states = []
                value_states = []
                gates = []
                for i, hidden_states in enumerate(inputs_embeds):
                    layer = models[i].layers[layer_idx]
                    hidden_states, gate = layer.input_layernorm(hidden_states, cond=adarms_cond[i])  # noqa: PLW2901
                    gates.append(gate)

                    input_shape = hidden_states.shape[:-1]
                    hidden_shape = (*input_shape, -1, layer.self_attn.head_dim)
                    query_state = layer.self_attn.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
                    key_state = layer.self_attn.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
                    value_state = layer.self_attn.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

                    query_states.append(query_state)
                    key_states.append(key_state)
                    value_states.append(value_state)

                # Concatenate and process attention
                query_states = torch.cat(query_states, dim=2)
                key_states = torch.cat(key_states, dim=2)
                value_states = torch.cat(value_states, dim=2)

                dummy_tensor = torch.zeros(
                    query_states.shape[0],
                    query_states.shape[2],
                    query_states.shape[-1],
                    device=query_states.device,
                    dtype=query_states.dtype,
                )
                cos, sin = self.paligemma.model.language_model.rotary_emb(dummy_tensor, position_ids)
                query_states, key_states = modeling_gemma.apply_rotary_pos_emb(
                    query_states, key_states, cos, sin, unsqueeze_dim=1
                )

                # KV cache handling
                new_layer_kv_cache = None
                if layer_kv_cache is None:
                    # Initialize cache (first forward pass / prefill)
                    cache_size = attention_mask.shape[-1]
                    idx, k_cache, v_cache = self._init_cache(key_states, value_states, cache_size)
                    # Use full cached tensors for attention
                    key_states_for_attn = k_cache
                    value_states_for_attn = v_cache
                    new_layer_kv_cache = (idx, k_cache, v_cache)
                else:
                    idx, k_cache, v_cache = layer_kv_cache
                    seq_len = key_states.shape[2]
                    if seq_len == 1:
                        # Next token prediction: update cache
                        idx, k_cache, v_cache = self._update_cache(
                            key_states, value_states, idx, k_cache, v_cache
                        )
                        key_states_for_attn = k_cache
                        value_states_for_attn = v_cache
                        new_layer_kv_cache = (idx, k_cache, v_cache)
                    else:
                        # Action sampling: concatenate without updating cache
                        current_idx = idx[0].item()
                        key_states_for_attn = torch.cat(
                            [k_cache[:, :, :current_idx, :], key_states], dim=2
                        )
                        value_states_for_attn = torch.cat(
                            [v_cache[:, :, :current_idx, :], value_states], dim=2
                        )
                        # Keep original cache unchanged
                        new_layer_kv_cache = (idx, k_cache, v_cache)

                batch_size = query_states.shape[0]
                scaling = self.paligemma.language_model.layers[layer_idx].self_attn.scaling

                # Attention computation with cached key/values
                att_output, _ = modeling_gemma.eager_attention_forward(
                    self.paligemma.language_model.layers[layer_idx].self_attn,
                    query_states,
                    key_states_for_attn,
                    value_states_for_attn,
                    attention_mask,
                    scaling,
                )
                # Get head_dim from the current layer, not from the model
                head_dim = self.paligemma.language_model.layers[layer_idx].self_attn.head_dim
                att_output = att_output.reshape(batch_size, -1, 1 * 8 * head_dim)

                # Process layer outputs
                outputs_embeds = []
                start_pos = 0
                for i, hidden_states in enumerate(inputs_embeds):
                    layer = models[i].layers[layer_idx]
                    end_pos = start_pos + hidden_states.shape[1]

                    if att_output.dtype != layer.self_attn.o_proj.weight.dtype:
                        att_output = att_output.to(layer.self_attn.o_proj.weight.dtype)
                    out_emb = layer.self_attn.o_proj(att_output[:, start_pos:end_pos])

                    # first residual
                    out_emb = modeling_gemma._gated_residual(hidden_states, out_emb, gates[i])  # noqa: SLF001
                    after_first_residual = out_emb.clone()
                    out_emb, gate = layer.post_attention_layernorm(out_emb, cond=adarms_cond[i])
                    # Convert to bfloat16 if the next layer (mlp) uses bfloat16
                    if layer.mlp.up_proj.weight.dtype == torch.bfloat16:
                        out_emb = out_emb.to(dtype=torch.bfloat16)

                    out_emb = layer.mlp(out_emb)
                    # second residual
                    out_emb = modeling_gemma._gated_residual(after_first_residual, out_emb, gate)  # noqa: SLF001
                    outputs_embeds.append(out_emb)
                    start_pos = end_pos

                return outputs_embeds, new_layer_kv_cache

            # Initialize list for storing per-layer KV caches
            new_kv_cache: KVCache = []

            # Process all layers with gradient checkpointing if enabled
            for layer_idx in range(num_layers):
                # Get the layer's KV cache if available
                layer_kv_cache = kv_cache[layer_idx] if kv_cache is not None else None

                if use_gradient_checkpointing:
                    # Note: gradient checkpointing doesn't easily support returning multiple values
                    # For now, we skip KV cache when using gradient checkpointing during training
                    result = torch.utils.checkpoint.checkpoint(
                        compute_layer_complete,
                        layer_idx,
                        inputs_embeds,
                        attention_mask,
                        position_ids,
                        adarms_cond,
                        None,  # Don't use KV cache with gradient checkpointing
                        use_reentrant=False,
                        preserve_rng_state=False,
                    )
                    inputs_embeds, layer_cache = result
                else:
                    inputs_embeds, layer_cache = compute_layer_complete(
                        layer_idx, inputs_embeds, attention_mask, position_ids, adarms_cond, layer_kv_cache
                    )

                # Store the new layer cache
                if layer_cache is not None:
                    new_kv_cache.append(layer_cache)

            # final norm
            # Define final norm computation function for gradient checkpointing
            def compute_final_norms(inputs_embeds, adarms_cond):
                outputs_embeds = []
                for i, hidden_states in enumerate(inputs_embeds):
                    out_emb, _ = models[i].norm(hidden_states, cond=adarms_cond[i])
                    outputs_embeds.append(out_emb)
                return outputs_embeds

            # Apply gradient checkpointing to final norm if enabled
            if use_gradient_checkpointing:
                outputs_embeds = torch.utils.checkpoint.checkpoint(
                    compute_final_norms, inputs_embeds, adarms_cond, use_reentrant=False, preserve_rng_state=False
                )
            else:
                outputs_embeds = compute_final_norms(inputs_embeds, adarms_cond)

            prefix_output = outputs_embeds[0]
            suffix_output = outputs_embeds[1]
            prefix_past_key_values = None

            # Return the new KV cache if it was used
            if len(new_kv_cache) > 0:
                kv_cache = new_kv_cache

        return [prefix_output, suffix_output], prefix_past_key_values, kv_cache
