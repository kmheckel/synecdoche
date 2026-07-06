"""Synecdoche — HyperNetworks and compressed weight representations for Flax NNX.

*Synecdoche* is the figure of speech in which a part stands in for the whole. Here
a small set of learnable/evolvable parameters stands in for all the weights of a
larger network: you train or evolve the part, and it generates the whole.

Quick start::

    import synecdoche as syn
    from flax import nnx

    model = nnx.Linear(64, 64, rngs=nnx.Rngs(0))
    target = nnx.state(model, nnx.Param)

    hyper = syn.LowRank(target, rank=8, rngs=nnx.Rngs(1))
    syn.apply_to(model, hyper)                 # model runs on generated weights
    print(syn.compression_ratio(hyper, model)) # << 1

Static generators live in :mod:`synecdoche.hyper`; input-conditioned ones in
:mod:`synecdoche.experimental`. The optional :mod:`synecdoche.lazy` (needs the
``[quax]`` extra) provides weights that never materialise in full.
"""

from . import experimental, hyper
from .hyper import (
    DCT,
    Constant,
    HyperNetwork,
    LowRank,
    MLPHyper,
    RandomProjection,
)
from .utils import (
    apply_to,
    compression_ratio,
    functional,
    materialize,
    param_count,
)

__version__ = "0.2.0"

__all__ = [
    "DCT",
    "Constant",
    "HyperNetwork",
    "LowRank",
    "MLPHyper",
    "RandomProjection",
    "__version__",
    "apply_to",
    "compression_ratio",
    "experimental",
    "functional",
    "hyper",
    "materialize",
    "param_count",
]
