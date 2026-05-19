"""Measure the steady-state cost of one Morton argsort + reorder pass.

Compares against the steady-state cost of one P2G substep so we know whether
the sort is a 5%, 50%, or 200% overhead on top of the kernel work it's
supposed to coalesce.
"""

import os
os.environ['WANDB_MODE'] = 'disabled'

import time
import sys
import numpy as np
import jax
import jax.numpy as jnp

from mpm_jax.morton import morton_argsort


def time_argsort(n, G=64, warmup=3, n_iters=50):
    rng = np.random.RandomState(42)
    x = jnp.array(0.25 + 0.5 * rng.rand(n, 3), dtype=jnp.float32)
    v = jnp.array(rng.rand(n, 3), dtype=jnp.float32)
    C = jnp.zeros((n, 3, 3), dtype=jnp.float32)
    F = jnp.tile(jnp.eye(3, dtype=jnp.float32), (n, 1, 1))

    @jax.jit
    def sort_step(x, v, C, F):
        order = morton_argsort(x, float(G), G)
        return x[order], v[order], C[order], F[order]

    # Warmup.
    for _ in range(warmup):
        x2, v2, C2, F2 = sort_step(x, v, C, F)
    jax.block_until_ready(x2)

    # Timed.
    t0 = time.perf_counter()
    for _ in range(n_iters):
        x, v, C, F = sort_step(x, v, C, F)
    jax.block_until_ready(x)
    elapsed = time.perf_counter() - t0
    return elapsed / n_iters * 1000  # ms


for n in [50_000, 200_000, 1_000_000]:
    ms = time_argsort(n)
    print(f"N={n:>7d}: sort+reorder = {ms:.3f} ms/substep")
