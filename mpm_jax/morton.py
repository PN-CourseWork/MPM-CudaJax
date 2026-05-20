"""3D Morton (Z-order) coding for spatial sorting of MPM particles.

The intent: sorting particles by Morton code along the Z-order space-filling
curve clusters spatially close particles next to each other in memory. When
those particles are then processed by the same CUDA warp (32 lanes ->
32 consecutive after-sort particles), more warp lanes target the same
27-stencil grid node, so a warp-shuffle reduction (`__match_any_sync` +
`__shfl_xor_sync`) coalesces more atomics into one. See p2g_v3_inline.cu.

The bit-interleave handles up to 10 bits per axis (`num_grids` <= 1024),
which more than covers the project's typical G in [32, 256].
"""

import jax.numpy as jnp


def _spread_bits(x):
    """Spread the lowest 10 bits of ``x`` into bits 0, 3, 6, 9, ...

    The output sits in the low 30 bits of a uint32. The classic
    Magic-Number Method (cf. graphics gems / Hu 2018) — five mask-and-shift
    rounds map each bit ``i`` to position ``3*i``.
    """
    x = x & jnp.uint32(0x000003FF)              # keep 10 bits
    x = (x | (x << jnp.uint32(16))) & jnp.uint32(0x030000FF)
    x = (x | (x << jnp.uint32(8)))  & jnp.uint32(0x0300F00F)
    x = (x | (x << jnp.uint32(4)))  & jnp.uint32(0x030C30C3)
    x = (x | (x << jnp.uint32(2)))  & jnp.uint32(0x09249249)
    return x


def morton_code_3d(cx, cy, cz):
    """3D Morton code (Z-order curve) from per-axis cell coords.

    Each input must be uint32 in [0, 1023]. Output is a uint32 where bits
    are interleaved as ``... z2 y2 x2 z1 y1 x1 z0 y0 x0`` from MSB to LSB.
    """
    return _spread_bits(cz) | (_spread_bits(cy) << jnp.uint32(1)) | (_spread_bits(cx) << jnp.uint32(2))


def morton_argsort(x, inv_dx, num_grids):
    """Argsort particle indices by 3D Morton code of their cell.

    Args:
        x: (N, 3) float32 particle positions in [0, 1]
        inv_dx: scalar — 1 / grid spacing (= num_grids for unit-cube domain)
        num_grids: int — grid resolution along one axis

    Returns:
        (N,) int32 array of argsort indices.
    """
    cells = jnp.clip(jnp.floor(x * inv_dx).astype(jnp.int32), 0, num_grids - 1)
    cx = cells[:, 0].astype(jnp.uint32)
    cy = cells[:, 1].astype(jnp.uint32)
    cz = cells[:, 2].astype(jnp.uint32)
    codes = morton_code_3d(cx, cy, cz)
    return jnp.argsort(codes)
