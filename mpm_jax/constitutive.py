import math
import jax
import jax.numpy as jnp

from mpm_jax.jacobi_svd import jacobi_svd_3x3


def _lame_params(E, nu):
    mu = E / (2.0 * (1.0 + nu))
    la = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    return mu, la


def corotated_elasticity(E=2e6, nu=0.4):
    mu, la = _lame_params(E, nu)

    def compute_stress(F):
        U, sigma, Vh = jnp.linalg.svd(F, full_matrices=False)
        corotated = 2.0 * mu * (F - U @ Vh) @ jnp.swapaxes(F, -2, -1)
        J = jnp.prod(sigma, axis=-1).reshape(-1, 1, 1)
        I = jnp.eye(3, dtype=F.dtype)
        volume = la * J * (J - 1.0) * I
        return corotated + volume

    return compute_stress


def corotated_elasticity_jacobi(E=2e6, nu=0.4):
    """CorotatedElasticity with a pure-JAX Jacobi SVD (no cuSOLVER).

    Exists to test whether replacing `jnp.linalg.svd` (a host-dispatched
    cuSOLVER call) with a fusable pointwise SVD lets XLA fuse the whole
    stress -> weights -> momentum -> scatter chain into one kernel, avoiding
    the (N, 27, 3) materialisation. See `mpm_jax/jacobi_svd.py` for details.
    """
    mu, la = _lame_params(E, nu)

    def compute_stress(F):
        U, sigma, Vh = jacobi_svd_3x3(F)
        corotated = 2.0 * mu * (F - U @ Vh) @ jnp.swapaxes(F, -2, -1)
        J = jnp.prod(sigma, axis=-1).reshape(-1, 1, 1)
        I = jnp.eye(3, dtype=F.dtype)
        volume = la * J * (J - 1.0) * I
        return corotated + volume

    return compute_stress


def sigma_elasticity(E=2e6, nu=0.4):
    mu, la = _lame_params(E, nu)

    def compute_stress(F):
        U, sigma, Vh = jnp.linalg.svd(F, full_matrices=False)
        sigma = jnp.clip(sigma, 0.001)
        epsilon = jnp.log(sigma)
        trace = epsilon.sum(axis=-1, keepdims=True)
        tau = 2.0 * mu * epsilon + la * trace
        # Batched diag: (N, 3) -> (N, 3, 3)
        tau_diag = jax.vmap(jnp.diag)(tau)
        stress = U @ tau_diag @ jnp.swapaxes(U, -2, -1)
        return stress

    return compute_stress


def stvk_elasticity(E=2e6, nu=0.4):
    mu, la = _lame_params(E, nu)

    def compute_stress(F):
        U, sigma, Vh = jnp.linalg.svd(F, full_matrices=False)
        Ft = jnp.swapaxes(F, -2, -1)
        FtF = Ft @ F
        I = jnp.eye(3, dtype=F.dtype)
        E_strain = 0.5 * (FtF - I)
        stvk = 2.0 * mu * (F @ E_strain)
        J = jnp.prod(sigma, axis=-1).reshape(-1, 1, 1)
        volume = la * J * (J - 1.0) * I
        return stvk + volume

    return compute_stress


def fluid_elasticity(E=2e6, nu=0.4):
    _, la = _lame_params(E, nu)

    def compute_stress(F):
        U, sigma, Vh = jnp.linalg.svd(F, full_matrices=False)
        J = jnp.prod(sigma, axis=-1).reshape(-1, 1, 1)
        I = jnp.eye(3, dtype=F.dtype)
        return la * J * (J - 1.0) * I

    return compute_stress


def volume_elasticity(E=2e6, nu=0.4, mode="taichi"):
    mu, la = _lame_params(E, nu)

    def compute_stress(F):
        J = jnp.linalg.det(F).reshape(-1, 1, 1)
        I = jnp.eye(3, dtype=F.dtype)
        if mode == "ziran":
            kappa = 2.0 / 3.0 * mu + la
            gamma = 2
            return kappa * (J - 1.0 / jnp.power(J, gamma - 1)) * I
        else:  # taichi
            return la * J * (J - 1.0) * I

    return compute_stress


ELASTICITY = {
    "CorotatedElasticity": corotated_elasticity,
    "CorotatedElasticityJacobi": corotated_elasticity_jacobi,
    "SigmaElasticity": sigma_elasticity,
    "StVKElasticity": stvk_elasticity,
    "FluidElasticity": fluid_elasticity,
    "VolumeElasticity": volume_elasticity,
}


def identity_plasticity():
    def apply(F):
        return F
    return apply


def drucker_prager_plasticity(E=2e6, nu=0.4, friction_angle=25.0, cohesion=0.0):
    mu, la = _lame_params(E, nu)
    sin_phi = jnp.sin(jnp.deg2rad(friction_angle))
    alpha = math.sqrt(2.0 / 3.0) * 2.0 * float(sin_phi) / (3.0 - float(sin_phi))

    def apply(F):
        U, sigma, Vh = jnp.linalg.svd(F, full_matrices=False)
        sigma = jnp.clip(sigma, 0.05)
        epsilon = jnp.log(sigma)
        trace = epsilon.sum(axis=-1, keepdims=True)
        epsilon_hat = epsilon - trace / 3.0
        epsilon_hat_norm = jnp.clip(jnp.linalg.norm(epsilon_hat, axis=-1, keepdims=True), 1e-10)

        expand_epsilon = jnp.ones_like(epsilon) * cohesion
        shifted_trace = trace - cohesion * 3.0
        cond_yield = (shifted_trace < 0).reshape(-1, 1)

        delta_gamma = epsilon_hat_norm + (3.0 * la + 2.0 * mu) / (2.0 * mu) * shifted_trace * alpha
        compress_epsilon = epsilon - (jnp.clip(delta_gamma, 0.0) / epsilon_hat_norm) * epsilon_hat

        epsilon = jnp.where(cond_yield, compress_epsilon, expand_epsilon)
        diag_exp = jax.vmap(jnp.diag)(jnp.exp(epsilon))
        return U @ diag_exp @ Vh

    return apply


def von_mises_plasticity(E=2e6, nu=0.4, sigma_y=1e3):
    mu, _ = _lame_params(E, nu)

    def apply(F):
        U, sigma, Vh = jnp.linalg.svd(F, full_matrices=False)
        sigma = jnp.clip(sigma, 0.05)
        epsilon = jnp.log(sigma)
        trace = epsilon.sum(axis=-1, keepdims=True)
        epsilon_hat = epsilon - trace / 3.0
        epsilon_hat_norm = jnp.clip(jnp.linalg.norm(epsilon_hat, axis=-1, keepdims=True), 1e-10)

        delta_gamma = epsilon_hat_norm - sigma_y / (2.0 * mu)
        cond_yield = (delta_gamma > 0).reshape(-1, 1, 1)

        yield_epsilon = epsilon - (delta_gamma / epsilon_hat_norm) * epsilon_hat
        diag_exp = jax.vmap(jnp.diag)(jnp.exp(yield_epsilon))
        yield_F = U @ diag_exp @ Vh

        return jnp.where(cond_yield, yield_F, F)

    return apply


def sigma_plasticity():
    def apply(F):
        J = jnp.linalg.det(F)
        J = jnp.clip(J, 0.05, 1.2)
        Je_1_3 = jnp.power(J, 1.0 / 3.0).reshape(-1, 1)
        Je_diag = jnp.broadcast_to(Je_1_3, (F.shape[0], 3))
        return jax.vmap(jnp.diag)(Je_diag)

    return apply


PLASTICITY = {
    "IdentityPlasticity": identity_plasticity,
    "DruckerPragerPlasticity": drucker_prager_plasticity,
    "VonMisesPlasticity": von_mises_plasticity,
    "SigmaPlasticity": sigma_plasticity,
}

REGISTRY = {**ELASTICITY, **PLASTICITY}


def get_constitutive(cfg):
    params = {k: v for k, v in cfg.items() if k != "name"}
    return REGISTRY[cfg.name](**params)
