"""Helpers for measuring and applying hypernetworks."""

from __future__ import annotations

import jax
import jax.numpy as jnp
from flax import nnx

__all__ = [
    "apply_to",
    "compression_ratio",
    "functional",
    "materialize",
    "param_count",
]


def param_count(x) -> int:
    """Number of learnable parameters in an ``nnx.Module`` or a parameter pytree.

    For a module, counts its ``nnx.Param`` leaves only (frozen
    :class:`~synecdoche.hyper.Constant` buffers and generated weights are
    excluded), so it reflects what a gradient/ES optimiser actually touches.
    """
    if isinstance(x, nnx.Module):
        x = nnx.state(x, nnx.Param)
    return int(sum(jnp.size(leaf) for leaf in jax.tree_util.tree_leaves(x)))


def compression_ratio(hyper, target) -> float:
    """Learnable params of ``hyper`` divided by params of the ``target``.

    ``< 1`` means the hypernetwork describes the target with fewer free
    parameters. ``target`` may be a module or a param pytree.
    """
    return param_count(hyper) / param_count(target)


def materialize(hyper):
    """Return the generated target weight pytree (alias for ``hyper()``)."""
    return hyper()


def apply_to(model: nnx.Module, hyper) -> nnx.Module:
    """Generate weights from ``hyper`` and write them into ``model`` in place.

    Equivalent to ``nnx.update(model, hyper())``; returns ``model`` for chaining.
    The generated pytree must match ``nnx.state(model, nnx.Param)`` — i.e. ``hyper``
    was built from that same target.

    Eager use only. Inside ``nnx.jit`` / ``nnx.grad`` / ``nnx.vmap`` (including
    training and evolution loops), mutating an externally-created ``model`` crosses
    trace levels — use :func:`functional` instead.
    """
    nnx.update(model, hyper())
    return model


def functional(model: nnx.Module):
    """Build a transform-safe apply function that runs ``model`` on generated weights.

    Captures ``model``'s structure once and returns ``apply(hyper, *args, **kwargs)``
    which rebuilds the module from ``hyper()``'s generated weights (via
    ``nnx.merge``, no in-place mutation) and calls it. This is the pattern to use
    inside ``nnx.grad`` / ``nnx.jit`` / ES loops, so gradients flow to the
    hypernetwork's parameters and nothing crosses a trace boundary::

        apply = synecdoche.functional(model)
        def loss(hyper):
            return loss_fn(apply(hyper, x), y)
        grads = nnx.grad(loss)(hyper)          # grads w.r.t. the hypernetwork

    :param model: the target module ``hyper`` was built from.
    :return: ``apply(hyper, *args, **kwargs) -> model output``.
    """
    graphdef, _params, rest = nnx.split(model, nnx.Param, ...)

    def apply(hyper, *args, **kwargs):
        rebuilt = nnx.merge(graphdef, hyper(), rest)
        return rebuilt(*args, **kwargs)

    return apply
