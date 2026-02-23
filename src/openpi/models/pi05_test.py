import dataclasses

import flax.nnx as nnx
import jax
import jax.numpy as jnp

import openpi.models.pi05_config as _pi05_config
import openpi.shared.nnx_utils as nnx_utils


def _state_l2_norm(state: nnx.State) -> jax.Array:
    squared_norms = []
    for leaf in state.flat_state().values():
        value = leaf.value if hasattr(leaf, "value") else leaf
        squared_norms.append(jnp.sum(jnp.square(value)))
    if not squared_norms:
        return jnp.array(0.0)
    return jnp.sqrt(jnp.sum(jnp.stack(squared_norms)))


def _compute_flow_grad_norms(config: _pi05_config.Pi05Config) -> tuple[jax.Array, jax.Array]:
    model = config.create(jax.random.key(0))

    obs, _ = config.fake_obs(batch_size=1), config.fake_act(batch_size=1)
    obs = obs.replace(  # avoid degenerate all-ones token inputs
        tokenized_prompt=jax.random.randint(jax.random.key(1), obs.tokenized_prompt.shape, minval=0, maxval=128)
    )
    actions = jax.random.normal(jax.random.key(2), (1, config.action_horizon, config.action_dim))

    def loss_fn(model):
        return jnp.mean(model.compute_loss(jax.random.key(3), obs, actions, train=True))

    diff_state = nnx.DiffState(0, nnx.Param)
    _, grads = nnx.value_and_grad(loss_fn, argnums=diff_state)(model)

    non_action_expert_filter = nnx.All(
        nnx_utils.PathRegex(".*llm.*"),
        nnx.Not(nnx_utils.PathRegex(".*llm.*_1.*")),
    )
    action_expert_filter = nnx_utils.PathRegex(".*llm.*_1.*")

    non_action_expert_grad_norm = _state_l2_norm(grads.filter(non_action_expert_filter))
    action_expert_grad_norm = _state_l2_norm(grads.filter(action_expert_filter))
    return non_action_expert_grad_norm, action_expert_grad_norm


def test_pi05_flow_prefix_stop_gradient():
    base_config = _pi05_config.Pi05Config(
        paligemma_variant="dummy",
        action_expert_variant="dummy",
        action_dim=8,
        action_horizon=4,
        max_token_len=16,
        subtask_loss_weight=0.0,
        fast_token_loss_weight=0.0,
        flow_matching_loss_weight=1.0,
        stop_gradient_flow_to_prefix=False,
    )
    insulated_config = dataclasses.replace(base_config, stop_gradient_flow_to_prefix=True)

    non_action_grad, action_grad = _compute_flow_grad_norms(base_config)
    insulated_non_action_grad, insulated_action_grad = _compute_flow_grad_norms(insulated_config)

    assert non_action_grad > 0
    assert insulated_non_action_grad == 0
    assert action_grad > 0
    assert insulated_action_grad > 0
