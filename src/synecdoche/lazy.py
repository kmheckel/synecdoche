"""Lazy, never-fully-materialised weights via quax multiple dispatch.

Optional module — requires the ``[quax]`` extra (``pip install synecdoche[quax]``).

Where :mod:`synecdoche.hyper` *generates* a full weight pytree from a compressed
representation, this module keeps the compression all the way into the computation:
a weight is a :class:`quax.ArrayValue` that stores only its factors and is decoded
lazily at the point of use. Following quax's LoRA example, a low-rank weight
computes ``x @ (l @ r)`` as ``(x @ l) @ r`` — the full ``(in, out)`` matrix never
materialises.

Wrap the forward pass in ``quax.quaxify`` and pass these arrays in place of dense
weights::

    import quax, synecdoche.lazy as sl
    w = sl.LowRankWeight.init((256, 256), rank=8, rngs=nnx.Rngs(0))  # 8*(256+256) params
    y = quax.quaxify(lambda W, x: x @ W)(w, x)                       # never forms 256x256
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.core
import jax.lax as lax
import quax
from flax import nnx

__all__ = ["DCTWeight", "LowRankWeight"]


class LowRankWeight(quax.ArrayValue):
    """A weight ``W = a @ b`` stored as its two low-rank factors, decoded lazily.

    ``a`` is ``(in, rank)`` and ``b`` is ``(rank, out)``. Under ``quax.quaxify`` a
    right-multiply ``x @ W`` is computed as ``(x @ a) @ b`` without ever forming the
    ``(in, out)`` product; any other operation falls back to materialising ``a @ b``.
    """

    a: jax.Array
    b: jax.Array

    def __init__(self, a: jax.Array, b: jax.Array):
        self.a = a
        self.b = b

    @classmethod
    def init(cls, shape, rank: int, *, rngs: nnx.Rngs, stddev: float = 0.02):
        """Create a random-initialised low-rank weight of ``shape = (in, out)``."""
        in_dim, out_dim = shape
        init = nnx.initializers.truncated_normal(stddev=stddev)
        return cls(init(rngs.params(), (in_dim, rank)), init(rngs.params(), (rank, out_dim)))

    def materialise(self):
        return self.a @ self.b

    def aval(self):
        return jax.core.ShapedArray((self.a.shape[0], self.b.shape[1]), self.a.dtype)


class DCTWeight(quax.ArrayValue):
    """A weight stored as low-frequency inverse-DCT coefficients, decoded lazily.

    ``coeffs`` is a 1-D vector of the leading DCT coefficients; the weight is
    ``idct(coeffs, n=size).reshape(shape)``. There is no cheaper-than-dense matmul
    for a DCT-coded matrix, so this materialises at use — but the *stored* form (and
    hence the checkpoint / evolution search space) is only ``len(coeffs)`` numbers.
    """

    coeffs: jax.Array
    target_shape: tuple = eqx.field(static=True)

    def __init__(self, coeffs: jax.Array, target_shape: tuple):
        self.coeffs = coeffs
        self.target_shape = tuple(target_shape)

    @classmethod
    def init(cls, shape, num_coeffs: int, *, rngs: nnx.Rngs, stddev: float = 0.02):
        """Create a random-initialised DCT-coded weight of ``shape = (in, out)``."""
        c = nnx.initializers.truncated_normal(stddev=stddev)(rngs.params(), (num_coeffs,))
        return cls(c, shape)

    def _size(self):
        n = 1
        for d in self.target_shape:
            n *= d
        return n

    def materialise(self):
        flat = jax.scipy.fft.idct(self.coeffs, n=self._size(), norm="ortho")
        return flat.reshape(self.target_shape)

    def aval(self):
        return jax.core.ShapedArray(self.target_shape, self.coeffs.dtype)


@quax.register(lax.dot_general_p)
def _lowrank_rhs(lhs, rhs: LowRankWeight, *, dimension_numbers, **kwargs):
    """Optimised ``x @ (l @ r) = (x @ l) @ r`` for the standard 2-D Linear case."""
    ((lhs_contract, rhs_contract), (lhs_batch, rhs_batch)) = dimension_numbers
    # Only the common case: contract the weight's input axis (0), no batching.
    if rhs_contract == (0,) and not lhs_batch and not rhs_batch:
        xa = lax.dot_general(lhs, rhs.a, ((lhs_contract, (0,)), ((), ())), **kwargs)
        return xa @ rhs.b
    return quax.quaxify(lax.dot_general)(lhs, rhs.materialise(), dimension_numbers, **kwargs)
