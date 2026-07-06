"""HyperNetworks for Flax NNX: generate a target network's weights from a small set.

A *hypernetwork* substitutes the many weights of a target network with a smaller,
easier-to-manipulate set of parameters that *generate* them. That smaller set is
what you train (by gradient) or evolve (by ES) — the full weights are a
deterministic function of it. This is the literary *synecdoche*: a part standing
in for the whole.

Every generator here is a plain :class:`flax.nnx.Module`. Construct it from a
*template* of the target's parameter state and call it to get a matching
``nnx.State`` that drops straight into ``nnx.update``::

    model = MyModule(...)                       # any Flax NNX module
    target = nnx.state(model, nnx.Param)        # the template (structure + shapes)
    hyper = synecdoche.LowRank(target, rank=8, rngs=nnx.Rngs(0))
    nnx.update(model, hyper())                  # model now runs on generated weights

Because ``hyper()`` is differentiable, the surrogate/backprop gradient flows to the
hypernetwork's parameters; because those parameters are few, evolution strategies
become viable where they were not on the full weight space (variance ∝ dimension).

Generators differ only in how they map the learnable parameters to a
``(num_layers, max_size)`` matrix; the base class slices each row to its target
layer's size and reshapes. This "pad to the largest layer, then downselect" scheme
is simple and pytree-agnostic; it trades some generation FLOPs for not needing
per-layer heads.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from flax import nnx

__all__ = [
    "DCT",
    "Constant",
    "HyperNetwork",
    "LowRank",
    "MLPHyper",
    "RandomProjection",
]


class Constant(nnx.Variable):
    """A fixed, non-trainable buffer — part of the module's state but not a
    ``nnx.Param``, so it is excluded from gradient/evolution updates and from
    :func:`synecdoche.param_count`."""


def _target_spec(target):
    """Flatten a target ``Param`` pytree into ``(treedef, shapes, sizes)``."""
    leaves, treedef = jax.tree_util.tree_flatten(target)
    shapes = [tuple(jnp.shape(leaf)) for leaf in leaves]
    sizes = [int(jnp.size(leaf)) for leaf in leaves]
    return treedef, shapes, sizes


def _trunc(rngs, shape, stddev=0.02):
    return nnx.initializers.truncated_normal(stddev=stddev)(rngs.params(), shape)


class HyperNetwork(nnx.Module):
    """Base class: generate a target weight pytree from a small parameter set.

    :param target: the target network's ``nnx.Param`` state (``nnx.state(model,
        nnx.Param)``) — only its structure, shapes and sizes are captured, not its
        values. The generated output has exactly this pytree structure.

    Subclasses implement :meth:`_matrix` returning a ``(num_layers, >= max_size)``
    array; :meth:`__call__` slices and reshapes it back to the target structure.
    """

    def __init__(self, target):
        self._treedef, self._shapes, self._sizes = _target_spec(target)
        self.num_layers = len(self._sizes)
        self.max_size = max(self._sizes) if self._sizes else 0

    def _assemble(self, matrix):
        leaves = [
            matrix[i, : self._sizes[i]].reshape(self._shapes[i]) for i in range(self.num_layers)
        ]
        return jax.tree_util.tree_unflatten(self._treedef, leaves)

    def _matrix(self) -> jax.Array:  # pragma: no cover - abstract
        raise NotImplementedError

    def __call__(self):
        """Generate the target weight pytree (an ``nnx.State`` matching ``target``)."""
        return self._assemble(self._matrix())


class RandomProjection(HyperNetwork):
    """Fixed random projection of learnable per-layer embeddings.

    ``weights = embeddings @ P`` with ``P`` a fixed ``(embedding_dim, max_size)``
    random matrix. Only the ``(num_layers, embedding_dim)`` embeddings are
    learnable; ``P`` is a frozen basis (a random-feature *indirect encoding*).

    :param embedding_dim: latent width per layer (the compression knob).
    :param rademacher: use ±1 (Rademacher) instead of Gaussian projection entries.
    """

    def __init__(self, target, embedding_dim: int, *, rngs: nnx.Rngs, rademacher=False):
        super().__init__(target)
        self.embedding_dim = embedding_dim
        key = rngs.params()
        if rademacher:
            proj = jax.random.rademacher(key, (embedding_dim, self.max_size)).astype(jnp.float32)
        else:
            proj = jax.random.normal(key, (embedding_dim, self.max_size)) / jnp.sqrt(embedding_dim)
        self.projection = Constant(proj)
        self.embeddings = nnx.Param(_trunc(rngs, (self.num_layers, embedding_dim)))

    def _matrix(self):
        return self.embeddings[...] @ self.projection[...]


class DCT(HyperNetwork):
    """Inverse-DCT decoding of low-frequency coefficients (compressed weight search).

    Each layer's weights are the inverse discrete cosine transform of a short
    coefficient vector, upsampled to ``max_size`` — smooth weights described by a
    few frequencies. After Koutník, Gomez & Schmidhuber, *Evolving neural networks
    in compressed weight space*, GECCO 2010.

    :param embedding_dim: number of low-frequency coefficients per layer.
    """

    def __init__(self, target, embedding_dim: int, *, rngs: nnx.Rngs):
        super().__init__(target)
        self.embedding_dim = embedding_dim
        self.embeddings = nnx.Param(_trunc(rngs, (self.num_layers, embedding_dim)))

    def _matrix(self):
        # 1-D inverse DCT along the frequency axis, per layer -> (num_layers, max_size).
        # norm="ortho" keeps the output scale independent of max_size; without it the
        # coefficients (and their gradients) become vanishingly small as max_size grows.
        return jax.scipy.fft.idct(self.embeddings[...], n=self.max_size, axis=1, norm="ortho")


class LowRank(HyperNetwork):
    """Low-rank matrix factorisation of the stacked weight matrix.

    ``weights = left @ right`` with ``left`` ``(num_layers, rank)`` and ``right``
    ``(rank, max_size)``. Most efficient when ``num_layers`` and ``rank`` are both
    small relative to ``max_size``; the learnable count is ``rank·(num_layers +
    max_size)``.

    :param rank: factorisation rank (the compression knob).
    """

    def __init__(self, target, rank: int, *, rngs: nnx.Rngs):
        super().__init__(target)
        self.rank = rank
        self.left = nnx.Param(_trunc(rngs, (self.num_layers, rank)))
        self.right = nnx.Param(_trunc(rngs, (rank, self.max_size)))

    def _matrix(self):
        return self.left[...] @ self.right[...]


class MLPHyper(HyperNetwork):
    """Static hypernetwork: a shared MLP maps per-layer embeddings to weights.

    After Ha, Dai & Le, *HyperNetworks* (2016). A learnable embedding per layer is
    decoded by a shared two-layer MLP into that layer's flattened weights. More
    expressive than the linear generators, at a larger learnable-parameter count.

    :param embedding_dim: per-layer embedding width.
    :param hidden_dim: hidden width of the shared decoder MLP.
    """

    def __init__(self, target, embedding_dim: int, hidden_dim: int, *, rngs: nnx.Rngs):
        super().__init__(target)
        self.embeddings = nnx.Param(_trunc(rngs, (self.num_layers, embedding_dim)))
        self.fc1 = nnx.Linear(embedding_dim, hidden_dim, rngs=rngs)
        self.fc2 = nnx.Linear(hidden_dim, self.max_size, rngs=rngs)

    def _matrix(self):
        return self.fc2(jax.nn.relu(self.fc1(self.embeddings[...])))
