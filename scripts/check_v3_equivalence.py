"""Order-invariant equivalence check: cuda_v3_inline vs cuda_v1_inline.

Run a short sim (N=2000, 3 frames) with both kernels and compare:
  - total kinetic energy   sum(0.5 * m * |v|^2)
  - center of mass         mean(x)
  - total momentum         sum(v)

These are invariant under particle relabeling, which is what cuda_v3_inline's
internal Morton sort does (the output state is in sorted order, not the
original order). Tolerances are loose because f32 + atomic-reduction order
differences accumulate, especially at N=2000 over 30 substeps.
"""

import os
os.environ['WANDB_MODE'] = 'disabled'

import numpy as np
import jax
import jax.numpy as jnp

from omegaconf import OmegaConf

from mpm_jax.solver import MPMState, make_params
from mpm_jax.constitutive import get_constitutive
from mpm_jax.boundary import build_boundary_fns
from mpm_jax.cuda.p2g_cuda import (
    build_jit_frame_inline,
    build_jit_frame_v3_inline,
)


def make_initial(n=2000, G=64):
    rng = np.random.RandomState(42)
    pts = 0.25 + 0.5 * rng.rand(n, 3)  # centred 0.5x0.5x0.5 cube around (0.5, ...)
    x = jnp.array(pts, dtype=jnp.float32)
    v = jnp.broadcast_to(jnp.array([0.0, 0.0, -1.0]), (n, 3)).astype(jnp.float32)
    C = jnp.zeros((n, 3, 3), dtype=jnp.float32)
    F = jnp.tile(jnp.eye(3, dtype=jnp.float32), (n, 1, 1))
    return MPMState(x=x, v=v, C=C, F=F), x.shape[0], G


def metrics(s, p_mass):
    x = np.asarray(s.x)
    v = np.asarray(s.v)
    ke = float(0.5 * p_mass * (v ** 2).sum())
    com = x.mean(axis=0).tolist()
    p = (p_mass * v).sum(axis=0).tolist()
    sx = np.sort(x.ravel())
    return {'ke': ke, 'com': com, 'p_total': p, 'sx_norm': float(np.linalg.norm(sx))}


def run(kernel_name, frames=3, steps_per_frame=10):
    state, n, G = make_initial()
    params = make_params(
        n_particles=n, num_grids=G, dt=3e-4,
        gravity=[0.0, 0.0, -9.8], rho=1000.0,
        clip_bound=0.5, damping=1.0,
        center=[0.5, 0.5, 0.5], size=[1.0, 1.0, 1.0],
    )
    g = jnp.arange(params.num_grids, dtype=jnp.float32)
    gx, gy, gz = jnp.meshgrid(g, g, g, indexing='ij')
    grid_x = jnp.stack([gx, gy, gz], axis=-1).reshape(-1, 3)

    bcs = [OmegaConf.create({
        'type': 'surface_collider',
        'point': [1.0, 1.0, 0.02],
        'normal': [0.0, 0.0, 1.0],
        'surface': 'sticky',
        'friction': 0.0,
        'start_time': 0.0,
        'end_time': 1e3,
    })]
    pre_fn, post_fn = build_boundary_fns(
        bcs, grid_x, params.dx,
        state.x, params.dt, params.p_mass,
    )

    elast_cfg = OmegaConf.create({
        'name': 'CorotatedElasticityJacobi',
        'E': 1e3,
        'nu': 0.2,
    })
    plast_cfg = OmegaConf.create({'name': 'IdentityPlasticity'})
    elasticity_fn = get_constitutive(elast_cfg)
    plasticity_fn = get_constitutive(plast_cfg)

    if kernel_name == 'v1_inline':
        jit_frame = build_jit_frame_inline(
            params, elasticity_fn, plasticity_fn, pre_fn, post_fn, steps_per_frame)
    elif kernel_name == 'v3_inline':
        jit_frame = build_jit_frame_v3_inline(
            params, elasticity_fn, plasticity_fn, pre_fn, post_fn, steps_per_frame)
    else:
        raise ValueError(kernel_name)

    for _ in range(frames):
        state = jit_frame(state)
    jax.block_until_ready(state.x)
    return state, params.p_mass


s1, m1 = run('v1_inline')
s3, m3 = run('v3_inline')
assert m1 == m3
print(f"p_mass = {m1}")

a = metrics(s1, m1)
b = metrics(s3, m3)
print(f"v1_inline: {a}")
print(f"v3_inline: {b}")

ke_rel = abs(a['ke'] - b['ke']) / max(1e-12, abs(a['ke']))
com_diff = np.linalg.norm(np.array(a['com']) - np.array(b['com']))
p_diff = np.linalg.norm(np.array(a['p_total']) - np.array(b['p_total']))
sx_rel = abs(a['sx_norm'] - b['sx_norm']) / max(1e-12, abs(a['sx_norm']))
print(f"KE relative diff = {ke_rel:.2e}")
print(f"COM L2 diff      = {com_diff:.2e}")
print(f"momentum L2 diff = {p_diff:.2e}")
print(f"sorted-x rel diff= {sx_rel:.2e}")

ok = ke_rel < 5e-3 and com_diff < 5e-3 and sx_rel < 5e-3
print('PASS' if ok else 'FAIL')
