# 🏗️ Synecdoche

**HyperNetworks and compressed weight representations for JAX / Flax NNX.**

*Synecdoche* (si-NEK-duh-kee) is the figure of speech in which a part stands in for
the whole — "all hands on deck." Fittingly, this library lets a small, easy-to-
manipulate set of parameters stand in for **all** the weights of a much larger
network: you train or evolve the part, and it generates the whole.

```python
import synecdoche as syn
from flax import nnx

model = nnx.Linear(256, 256, rngs=nnx.Rngs(0))     # 65 792 parameters
target = nnx.state(model, nnx.Param)               # the template

hyper = syn.DCT(target, embedding_dim=32, rngs=nnx.Rngs(1))
syn.apply_to(model, hyper)                          # model now runs on generated weights
print(syn.compression_ratio(hyper, model))         # a much smaller description
```

## Why

A hypernetwork replaces a network's weights with a generator. That buys you three
things:

- **Compression** — describe many weights with few numbers.
- **Structure / inductive bias** — the generator constrains *what kinds* of weight
  matrices are reachable (smooth, low-rank, tied, …).
- **Tractable search** — evolution strategies and other black-box optimisers have
  variance that grows with the number of parameters, so they don't scale to full
  networks. Move the search into the hypernetwork's small latent space and they
  become viable again. (This is why synecdoche pairs naturally with spiking /
  non-differentiable networks trained by evolution.)

Everything is a plain `flax.nnx.Module`, so generated weights drop into any NNX
model, and the generator's parameters train by gradient **or** evolve by ES.

## Generators

Build any of these from a target's `nnx.state(model, nnx.Param)`; call it to get a
matching weight pytree.

| Generator | Idea | Learnable params |
|---|---|---|
| `RandomProjection` | fixed random basis · learnable embeddings (random-feature indirect encoding) | `num_layers × embedding_dim` |
| `DCT` | inverse-DCT of a few low-frequency coefficients — *compressed weight search* (Koutník, Gomez & Schmidhuber, 2010) | `num_layers × embedding_dim` |
| `LowRank` | `left @ right` factorisation of the stacked weights | `rank × (num_layers + max_size)` |
| `MLPHyper` | a shared MLP decodes per-layer embeddings (Ha, Dai & Le, *HyperNetworks*, 2016) | embeddings + MLP |
| `experimental.DynamicHypernetwork` | weights conditioned on the input batch (fast-weights / meta-learning) | embeddings + MLP |

`RandomProjection` and `DCT` are the strong compressors; `LowRank` / `MLPHyper` are
more expressive but compress only when the target's layers are large relative to
their number (the generators pad to the largest layer — see the docstrings).

## Training the compressed representation

Inside `nnx.jit` / `nnx.grad` / an ES loop, don't mutate the target model in place —
rebuild it functionally from generated weights with `syn.functional`:

```python
apply = syn.functional(model)                      # capture structure once

def loss_fn(hyper):
    return mse(apply(hyper, X), Y)                 # forward on generated weights

opt = nnx.Optimizer(hyper, optax.adam(3e-3), wrt=nnx.Param)
loss, grads = nnx.value_and_grad(loss_fn)(hyper)   # grads w.r.t. the hypernetwork
opt.update(hyper, grads)
```

See [`examples/compress_and_train.py`](examples/compress_and_train.py) for a runnable
end-to-end fit (16× compression, trains to convergence).

## Lazy weights (optional, via [quax](https://docs.kidger.site/quax/))

`pip install synecdoche[quax]` adds `synecdoche.lazy`: weights that keep their
compressed form **all the way into the computation** and never materialise in full,
using quax's multiple dispatch. A low-rank weight computes `x @ (a @ b)` as
`(x @ a) @ b` — the full matrix is never formed.

```python
import quax, synecdoche.lazy as sl

w = sl.LowRankWeight.init((1024, 1024), rank=8, rngs=nnx.Rngs(0))  # 8·2048 params, not 1M
y = quax.quaxify(lambda W, x: x @ W)(w, x)                          # never forms 1024×1024
```

## Install

```bash
pip install synecdoche          # core (jax + flax)
pip install synecdoche[quax]    # + lazy never-materialised weights
```

## Roadmap

- **Generative weight generators** — sampling weights from a learned distribution
  (diffusion / VAE over checkpoints, after *Neural Network Diffusion* and *G.pt*),
  the stochastic counterpart to the deterministic generators here.
- Per-layer output heads so `LowRank` / `MLPHyper` compress heterogeneous
  architectures without the pad-to-max-size overhead.
- First-class `spyx` adapter for evolving spiking networks in the compressed space.

## References

- Koutník, Gomez & Schmidhuber. *Evolving neural networks in compressed weight
  space.* GECCO 2010.
- Ha, Dai & Le. *HyperNetworks.* ICLR 2017.
- Kidger et al. *quax* — JAX + multiple dispatch for custom array types.
