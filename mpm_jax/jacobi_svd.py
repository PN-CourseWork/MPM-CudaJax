"""Branch-free Jacobi SVD for batched 3x3 matrices, written in JAX.

The motivation is fusion: `jnp.linalg.svd` lowers to a cuSOLVER call, which is
a host-dispatched external library routine. XLA cannot fuse downstream
producers (stress, weights, momentum) into a single kernel across that
boundary, so the `(N, 27, 3)` momentum intermediate has to materialise in
HBM before the scatter kernel can consume it. A pure-JAX Jacobi SVD has only
pointwise ops + tiny stack/index manipulations, which XLA *can* fuse — at
least in principle.

Implementation follows McAdams et al. 2011: form S = F^T F, run 4 sweeps of
3 Givens rotations (pairs 01, 02, 12) to diagonalise S, then recover
sigma = sqrt(diag(S)) and U = F V diag(1/sigma).

Matches `jnp.linalg.svd(F, full_matrices=False)` returning (U, sigma, Vh)
where Vh = V^T.
"""

import jax
import jax.numpy as jnp


def _givens_cs(a_pp, a_pq, a_qq, eps=1e-10):
    """Branch-free Givens coefficients zeroing a_pq in symmetric 2x2.

    Uses tau = (a_pp - a_qq) / (2 a_pq) — this is the correct sign for the
    rotation G = [[c, -s], [s, c]] that the symmetric update below uses.
    (The CUDA kernel `p2g_fused.cu` has tau = (a_qq - a_pp) / (2 a_pq),
    which is a sign bug; iteration still converges thanks to Jacobi
    robustness, but each rotation leaves residual off-diagonal — hence
    `test_cuda_fused_matches_jax`'s "looser tolerances" comment.)
    """
    small = jnp.abs(a_pq) < eps
    safe_pq = jnp.where(small, jnp.ones_like(a_pq), a_pq)
    tau = (a_pp - a_qq) / (2.0 * safe_pq)
    t = jnp.sign(tau) / (jnp.abs(tau) + jnp.sqrt(1.0 + tau * tau))
    c = jax.lax.rsqrt(1.0 + t * t)
    s = t * c
    c = jnp.where(small, jnp.ones_like(c), c)
    s = jnp.where(small, jnp.zeros_like(s), s)
    return c, s


def _rotate(S, V, p, q, r):
    """Apply one Jacobi rotation on the (p, q) pair to symmetric S; update V.

    S: (..., 3, 3) symmetric; V: (..., 3, 3); (p, q, r) is a permutation of (0,1,2).
    """
    Spp = S[..., p, p]
    Spq = S[..., p, q]
    Sqq = S[..., q, q]
    Spr = S[..., p, r]
    Sqr = S[..., q, r]

    c, s = _givens_cs(Spp, Spq, Sqq)

    new_Spp = c * c * Spp + 2.0 * c * s * Spq + s * s * Sqq
    new_Sqq = s * s * Spp - 2.0 * c * s * Spq + c * c * Sqq
    new_Spr = c * Spr + s * Sqr
    new_Sqr = -s * Spr + c * Sqr
    zero = jnp.zeros_like(Spq)

    S = S.at[..., p, p].set(new_Spp)
    S = S.at[..., q, q].set(new_Sqq)
    S = S.at[..., p, q].set(zero)
    S = S.at[..., q, p].set(zero)
    S = S.at[..., p, r].set(new_Spr)
    S = S.at[..., r, p].set(new_Spr)
    S = S.at[..., q, r].set(new_Sqr)
    S = S.at[..., r, q].set(new_Sqr)

    Vp = V[..., :, p]
    Vq = V[..., :, q]
    new_Vp = c[..., None] * Vp + s[..., None] * Vq
    new_Vq = -s[..., None] * Vp + c[..., None] * Vq

    V = V.at[..., :, p].set(new_Vp)
    V = V.at[..., :, q].set(new_Vq)
    return S, V


def jacobi_svd_3x3(F, num_sweeps=4):
    """SVD of batched 3x3 matrices via Jacobi iteration on F^T F.

    Parameters
    ----------
    F : (..., 3, 3) array
    num_sweeps : int
        Number of full sweeps (each sweep = 3 Givens rotations). 4 is enough
        for f32 convergence on physically reasonable deformation gradients
        (per McAdams 2011).

    Returns
    -------
    U : (..., 3, 3)
    sigma : (..., 3)
    Vh : (..., 3, 3)   (= V^T)
    """
    FtF = jnp.einsum("...ji,...jk->...ik", F, F)
    eye = jnp.eye(3, dtype=F.dtype)
    V = jnp.broadcast_to(eye, F.shape)
    S = FtF

    for _ in range(num_sweeps):
        S, V = _rotate(S, V, 0, 1, 2)
        S, V = _rotate(S, V, 0, 2, 1)
        S, V = _rotate(S, V, 1, 2, 0)

    sigma2 = jnp.diagonal(S, axis1=-2, axis2=-1)
    sigma = jnp.sqrt(jnp.maximum(sigma2, 0.0))

    FV = jnp.einsum("...ij,...jk->...ik", F, V)
    inv_sigma = jnp.where(sigma > 1e-10, 1.0 / sigma, jnp.zeros_like(sigma))
    U = FV * inv_sigma[..., None, :]

    Vh = jnp.swapaxes(V, -2, -1)
    return U, sigma, Vh
