"""Generate an MLP's weights from a tiny hypernetwork and train the compressed set.

Shows the core synecdoche loop: a target Flax NNX module never owns its own
weights — a hypernetwork generates them from a much smaller parameter set, and you
train *that*. Run::

    uv run python examples/compress_and_train.py
"""

import jax
import jax.numpy as jnp
import optax
from flax import nnx

import synecdoche as syn


class MLP(nnx.Module):
    def __init__(self, rngs):
        self.l1 = nnx.Linear(32, 64, rngs=rngs)
        self.l2 = nnx.Linear(64, 64, rngs=rngs)
        self.l3 = nnx.Linear(64, 1, rngs=rngs)

    def __call__(self, x):
        x = jax.nn.relu(self.l1(x))
        x = jax.nn.relu(self.l2(x))
        return self.l3(x)


def main():
    # A toy regression target.
    key = jax.random.PRNGKey(0)
    kx, kw = jax.random.split(key)
    X = jax.random.normal(kx, (256, 32))
    w_true = jax.random.normal(kw, (32, 1))
    Y = jnp.tanh(X @ w_true)

    model = MLP(nnx.Rngs(0))
    target = nnx.state(model, nnx.Param)
    print(f"target network: {syn.param_count(model)} parameters")

    # Describe all of those weights with a handful of DCT coefficients per layer.
    hyper = syn.DCT(target, embedding_dim=64, rngs=nnx.Rngs(1))
    print(
        f"hypernetwork:   {syn.param_count(hyper)} parameters "
        f"(compression ratio {syn.compression_ratio(hyper, model):.3f})"
    )

    # Train the hypernetwork's parameters; weights are regenerated each step.
    apply = syn.functional(model)  # transform-safe forward

    def loss_fn(h):
        return jnp.mean((apply(h, X) - Y) ** 2)

    opt = nnx.Optimizer(hyper, optax.adam(3e-3), wrt=nnx.Param)

    @nnx.jit
    def step(h, o):
        val, grads = nnx.value_and_grad(loss_fn)(h)
        o.update(h, grads)
        return val

    print("\ntraining the compressed representation:")
    for i in range(600):
        loss = float(step(hyper, opt))
        if i % 100 == 0:
            print(f"  step {i:>3}  mse={loss:.4f}")

    # Bake the learned weights back into the concrete model for inference.
    syn.apply_to(model, hyper)
    print(f"\nfinal mse (materialised model): {float(jnp.mean((model(X) - Y) ** 2)):.4f}")


if __name__ == "__main__":
    main()
