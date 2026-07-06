"""Tests for the optional quax-backed lazy weights (skipped without the quax extra)."""

import jax
import jax.numpy as jnp
import pytest
from flax import nnx

quax = pytest.importorskip("quax")
lazy = pytest.importorskip("synecdoche.lazy")


def test_lowrank_lazy_matmul_matches_dense():
    """x @ W via quaxify equals x @ (l @ r), and stores only the factors."""
    w = lazy.LowRankWeight.init((32, 16), rank=4, rngs=nnx.Rngs(1))
    x = jax.random.normal(jax.random.PRNGKey(0), (5, 32))
    y = quax.quaxify(lambda W, xx: xx @ W)(w, x)
    assert y.shape == (5, 16)
    assert bool(jnp.allclose(y, x @ (w.a @ w.b), atol=1e-5))
    assert w.a.size + w.b.size < 32 * 16  # compressed storage


def test_lowrank_non_matmul_falls_back_to_materialise():
    """An op with no dispatch rule materialises correctly."""
    w = lazy.LowRankWeight.init((8, 8), rank=2, rngs=nnx.Rngs(2))
    total = quax.quaxify(lambda W: jnp.sum(W))(w)
    assert bool(jnp.allclose(total, jnp.sum(w.a @ w.b), atol=1e-4))


def test_dct_lazy_matmul_matches_dense():
    """DCT-coded weight decodes to the same matmul as its dense form."""
    d = lazy.DCTWeight.init((32, 16), num_coeffs=8, rngs=nnx.Rngs(3))
    dense = jax.scipy.fft.idct(d.coeffs, n=32 * 16, norm="ortho").reshape(32, 16)
    x = jax.random.normal(jax.random.PRNGKey(0), (5, 32))
    y = quax.quaxify(lambda W, xx: xx @ W)(d, x)
    assert bool(jnp.allclose(y, x @ dense, atol=1e-5))
    assert d.coeffs.size == 8  # far fewer than 32*16


def test_gradient_flows_to_lazy_factors():
    """Gradients reach the low-rank factors through a quaxified loss."""
    w = lazy.LowRankWeight.init((16, 16), rank=4, rngs=nnx.Rngs(4))
    x = jax.random.normal(jax.random.PRNGKey(0), (4, 16))

    def loss(w):
        return jnp.mean(quax.quaxify(lambda W, xx: xx @ W)(w, x) ** 2)

    g = jax.grad(loss)(w)
    assert bool(jnp.all(jnp.isfinite(g.a))) and bool(jnp.all(jnp.isfinite(g.b)))
