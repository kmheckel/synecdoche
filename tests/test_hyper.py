"""Tests for the NNX hypernetwork generators."""

import jax
import jax.numpy as jnp
import optax
import pytest
from flax import nnx

import synecdoche as syn


class MLP(nnx.Module):
    def __init__(self, rngs):
        self.l1 = nnx.Linear(16, 24, rngs=rngs)
        self.l2 = nnx.Linear(24, 8, rngs=rngs)

    def __call__(self, x):
        return self.l2(jax.nn.relu(self.l1(x)))


def _make(name, target):
    rngs = nnx.Rngs(1)
    return {
        "RandomProjection": lambda: syn.RandomProjection(target, 8, rngs=rngs),
        "DCT": lambda: syn.DCT(target, 8, rngs=rngs),
        "LowRank": lambda: syn.LowRank(target, 4, rngs=rngs),
        "MLPHyper": lambda: syn.MLPHyper(target, 8, 16, rngs=rngs),
    }[name]()


GENERATORS = ["RandomProjection", "DCT", "LowRank", "MLPHyper"]


@pytest.mark.parametrize("name", GENERATORS)
def test_generates_matching_structure_and_runs(name):
    """Each generator produces a pytree matching the target; the model runs on it."""
    model = MLP(nnx.Rngs(0))
    target = nnx.state(model, nnx.Param)
    hyper = _make(name, target)

    generated = hyper()
    # Same structure and per-leaf shapes as the target params.
    assert jax.tree_util.tree_structure(generated) == jax.tree_util.tree_structure(target)
    for g, t in zip(
        jax.tree_util.tree_leaves(generated),
        jax.tree_util.tree_leaves(target),
        strict=True,
    ):
        assert jnp.shape(g) == jnp.shape(t)
        assert bool(jnp.all(jnp.isfinite(g)))

    syn.apply_to(model, hyper)
    out = model(jnp.ones((4, 16)))
    assert out.shape == (4, 8)
    assert bool(jnp.all(jnp.isfinite(out)))


@pytest.mark.parametrize("name", GENERATORS)
def test_gradient_flows_to_hypernetwork(name):
    """The functional pattern lets grads reach the hypernetwork's params."""
    model = MLP(nnx.Rngs(0))
    target = nnx.state(model, nnx.Param)
    hyper = _make(name, target)
    apply = syn.functional(model)
    x = jax.random.normal(jax.random.PRNGKey(3), (4, 16))
    y = jax.random.normal(jax.random.PRNGKey(4), (4, 8))

    def loss(h):
        return jnp.mean((apply(h, x) - y) ** 2)

    grads = nnx.grad(loss)(hyper)
    gleaves = jax.tree_util.tree_leaves(grads)
    assert len(gleaves) == len(jax.tree_util.tree_leaves(nnx.state(hyper, nnx.Param)))
    assert all(bool(jnp.all(jnp.isfinite(jnp.asarray(g)))) for g in gleaves)


def test_training_reduces_loss():
    """A few gradient steps through generation actually fit a target output."""
    model = MLP(nnx.Rngs(0))
    target = nnx.state(model, nnx.Param)
    hyper = syn.MLPHyper(target, 8, 32, rngs=nnx.Rngs(1))
    apply = syn.functional(model)
    x = jax.random.normal(jax.random.PRNGKey(3), (8, 16))
    y = jax.random.normal(jax.random.PRNGKey(4), (8, 8))

    def loss(h):
        return jnp.mean((apply(h, x) - y) ** 2)

    opt = nnx.Optimizer(hyper, optax.adam(3e-2), wrt=nnx.Param)

    @nnx.jit
    def step(h, o):
        val, g = nnx.value_and_grad(loss)(h)
        o.update(h, g)
        return val

    losses = [float(step(hyper, opt)) for _ in range(60)]
    assert losses[-1] < losses[0]


def test_compression_ratio_and_param_count():
    """RandomProjection / DCT compress; param_count ignores frozen buffers."""
    model = MLP(nnx.Rngs(0))
    target = nnx.state(model, nnx.Param)
    tgt_n = syn.param_count(model)
    assert tgt_n == syn.param_count(target)  # module or pytree

    rp = syn.RandomProjection(target, 8, rngs=nnx.Rngs(1))
    # Only the embeddings are learnable; the fixed projection is a Constant buffer.
    assert syn.param_count(rp) == rp.num_layers * 8
    assert syn.compression_ratio(rp, model) < 1.0

    dct = syn.DCT(target, 8, rngs=nnx.Rngs(1))
    assert syn.compression_ratio(dct, model) < 1.0


def test_random_projection_buffer_not_trainable():
    """The random projection matrix is excluded from the trainable Param state."""
    model = MLP(nnx.Rngs(0))
    target = nnx.state(model, nnx.Param)
    rp = syn.RandomProjection(target, 8, rngs=nnx.Rngs(1))
    params = nnx.state(rp, nnx.Param)
    flat_keys = [jax.tree_util.keystr(k) for k, _ in jax.tree_util.tree_leaves_with_path(params)]
    assert any("embeddings" in k for k in flat_keys)
    assert not any("projection" in k for k in flat_keys)


def test_dynamic_hypernetwork_conditions_on_input():
    """Experimental input-conditioned hypernetwork generates and runs."""
    model = MLP(nnx.Rngs(0))
    target = nnx.state(model, nnx.Param)
    x = jax.random.normal(jax.random.PRNGKey(5), (6, 16))
    dh = syn.experimental.DynamicHypernetwork(
        target, in_features=16, embedding_dim=8, hidden_dim=16, rngs=nnx.Rngs(1)
    )
    generated = dh(x)
    assert jax.tree_util.tree_structure(generated) == jax.tree_util.tree_structure(target)
    nnx.update(model, generated)
    assert bool(jnp.all(jnp.isfinite(model(x))))
