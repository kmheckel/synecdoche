"""Experimental / input-conditioned hypernetworks — unstable API."""

from __future__ import annotations

import jax
import jax.numpy as jnp
from flax import nnx

from .hyper import HyperNetwork

__all__ = ["DynamicHypernetwork"]


class DynamicHypernetwork(HyperNetwork):
    """Input-conditioned hypernetwork: generate weights from a batch of inputs.

    The batch is averaged into a single context vector, concatenated with a
    learnable per-layer embedding, and decoded by a shared MLP into each layer's
    weights. Unlike the static generators in :mod:`synecdoche.hyper`, the produced
    weights depend on the data — a fast-weights / meta-learning style module.

    **Experimental:** the API and the batch-conditioning scheme may change.

    :param in_features: size of a single flattened input sample.
    :param embedding_dim: per-layer embedding width.
    :param hidden_dim: hidden width of the shared decoder MLP.
    """

    def __init__(
        self,
        target,
        in_features: int,
        embedding_dim: int,
        hidden_dim: int,
        *,
        rngs: nnx.Rngs,
    ):
        super().__init__(target)
        self.in_features = in_features
        self.embeddings = nnx.Param(
            nnx.initializers.truncated_normal(stddev=0.02)(
                rngs.params(), (self.num_layers, embedding_dim)
            )
        )
        self.fc1 = nnx.Linear(embedding_dim + in_features, hidden_dim, rngs=rngs)
        self.fc2 = nnx.Linear(hidden_dim, self.max_size, rngs=rngs)

    def __call__(self, x):
        """Generate the target weight pytree conditioned on a batch ``x``.

        :param x: a batch; averaged over axis 0 and flattened to ``in_features``.
        """
        context = jnp.mean(x, axis=0).reshape(self.in_features)  # shared context
        context = jnp.broadcast_to(context, (self.num_layers, self.in_features))
        inp = jnp.concatenate([self.embeddings[...], context], axis=1)
        matrix = self.fc2(jax.nn.relu(self.fc1(inp)))
        return self._assemble(matrix)
