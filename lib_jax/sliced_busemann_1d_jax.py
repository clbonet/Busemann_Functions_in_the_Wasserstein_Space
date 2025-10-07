import jax
import jax.numpy as jnp

import jax.scipy.stats.norm as jnorm

from functools import partial
from .sliced_wasserstein import sliced_wasserstein_from_quantiles


def busemann_semi_gaussians_1d(rng, m0, s0, m1, s1, u_values, n_unif=100):
    """
        Compute the Busemann function for a 1D Gaussian geodesic ray and a
        discrete distribution

        Inputs:
        - m0: mean of mu_0 (tensor of size (1,))
        - s0: std of mu_0 (tensor of size (1,))
        - m1: mean of mu_1 (tensor of size (1,))
        - s1: std of mu_1 (tensor of size (1,))
        - u_values: tensor of size (n_distr, n)

        Output:
        - shape (n_distr,)
    """
    n_distr, n = u_values.shape

    percentiles = jnp.linspace(0, 1, n_unif)
    quantiles_u_values = jnp.percentile(u_values, percentiles*100, axis=1).T

    # unif = jax.random.uniform(rng, quantiles_u_values.shape)
    unif = jnp.clip(jax.random.uniform(rng, quantiles_u_values.shape), 1e-6, 1 - 1e-6)
    u_values_icdf = jax.vmap(jnp.interp, in_axes=(0, None, 0))(
        unif, percentiles, quantiles_u_values
    )

    m = jnp.mean(u_values_icdf, axis=-1) # shape (n_distr,)

    v_icdf = jnorm.ppf(unif)  # grid_1d)
    s = jnp.mean(u_values_icdf * v_icdf, axis=-1)

    return -(m1-m0)*(m.T-m0)-(s1-s0)*(s.T-s0)


@partial(jax.jit, static_argnums=[3, 4])
def busemann_sliced_semi_gaussian_1d(
    rng: jax.random.PRNGKey, Xs: jnp.ndarray, Xt: jnp.ndarray,
    n_projs: int = 50, p: float = 2, eps: float = 0):
    """
        Use a hierarchical hybrid projection

        Parameters
        ----------
        - Xs: (n, n_samples, d)
        - Xt: (m, m_samples, d)
        - num_projections: int
        - u_weights: (n,) (optional)
        - v_weights: (m,) (optional)
        - p: int (default 2)
        - eps: float (default 1)
    """
    d = Xs.shape[-1]

    master_key, key_directions, key_means, key_busemann, key_agg = jax.random.split(rng, num=5)

    ## projections samples
    # generate directions on the sphere
    theta = jax.random.normal(key_directions, (n_projs, d))
    theta = theta / jnp.linalg.norm(
        theta, axis=-1, keepdims=True
    )

    # Projections on R
    projs_Xs = jnp.einsum("cnd,ld->cnl", Xs, theta) # shape (n_distr, n_samples, n_projs)
    projs_Xt = jnp.einsum("cnd,ld->cnl", Xt, theta)

    Xps = jnp.moveaxis(projs_Xs, -1, 0) # shape (n_projs, n_distr, n_samples)
    Xpt = jnp.moveaxis(projs_Xt, -1, 0)

    m0 = jnp.zeros((n_projs, 1))
    s0 = eps * jnp.ones((n_projs, 1))

    # Unif on [-1,1]
    m1 = -1 + 2 * jax.random.uniform(key_means, (n_projs, 1))
    s1 = s0 + jnp.sqrt(1-m1**2) # such that W_2^2(\delta_0, \mu) = \int x^2 d\mu(x)=1
    # s1 = jnp.sqrt(jnp.clip(1 - m1**2, 1e-8))


    keys_busemann = jax.random.split(key_busemann, num=n_projs)
    busemann_vmap = jax.vmap(busemann_semi_gaussians_1d)

    projs_Xs_1d = busemann_vmap(keys_busemann, m0, s0, m1, s1, Xps) # shape (L, n)
    projs_Xt_1d = busemann_vmap(keys_busemann, m0, s0, m1, s1, Xpt) # shape (L, m)

    # Aggregate projections by Hierarchical hybrid projections
    psi = jax.random.normal(key_agg, (n_projs, 2)) 
    psi = psi / jnp.linalg.norm(psi, axis=-1, keepdims=True)

    full_projs_s = psi[:,0][:,None,None] * Xps + (psi[:,1][:,None] * projs_Xs_1d)[:,:,None]
    full_projs_t = psi[:,0][:,None,None] * Xpt + (psi[:,1][:,None] * projs_Xt_1d)[:,:,None]

    # compute quantiles of sliced distributions
    percentiles = jnp.linspace(0, 1, 100)
    # add weights in percentile to take into account non uniform distributions??
    quantiles_proj_source = jnp.percentile(full_projs_s.reshape(n_projs, -1), percentiles*100, axis=-1).T
    quantiles_proj_target = jnp.percentile(full_projs_t.reshape(n_projs, -1), percentiles*100, axis=-1).T

    # compute sliced wasserstein value
    return sliced_wasserstein_from_quantiles(
        quantiles_proj_source, quantiles_proj_target, percentiles, master_key, p
    )





def busemann1D(rng, u0_values, u1_values, w_values, n_unif=100):
    """
        Compute the Busemann function in 1D

        Input:
        - u0_values: tensor of size (n,1)
        - u1_values: tensor of size (m,1)
        - w_values: tensor of size (n_batch, k)
    """
    n_distr, n = w_values.shape

    percentiles = jnp.linspace(0, 1, n_unif)
    quantiles_u0_values = jnp.percentile(u0_values, percentiles*100, axis=1).T
    quantiles_u1_values = jnp.percentile(u1_values, percentiles*100, axis=1).T
    quantiles_w_values = jnp.percentile(w_values, percentiles*100, axis=1).T

    unif = jnp.clip(jax.random.uniform(rng, quantiles_u0_values.shape), 1e-6, 1 - 1e-6)
    u0_values_icdf = jax.vmap(jnp.interp, in_axes=(0, None, 0))(
        unif, percentiles, quantiles_u0_values
    )
    u1_values_icdf = jax.vmap(jnp.interp, in_axes=(0, None, 0))(
        unif, percentiles, quantiles_u1_values
    )

    w_values_icdf = jax.vmap(jnp.interp, in_axes=(0, None, 0))(
        jnp.repeat(unif, n_distr, 0), percentiles, quantiles_w_values
    )

    norm = jnp.sqrt(jnp.mean(jnp.square(u0_values_icdf - u1_values_icdf), axis=-1))
    return jnp.mean((u0_values_icdf-u1_values_icdf) * (w_values_icdf-u0_values_icdf), axis=-1) / norm


@partial(jax.jit, static_argnums=[3,4])
def busemann_sliced_1d(
    rng: jax.random.PRNGKey, Xs: jnp.ndarray, Xt: jnp.ndarray,
    n_projs: int = 50, p: float = 2):
    """
        Use a hierarchical hybrid projection

        Parameters
        ----------
        - Xs: (n, n_samples, d)
        - Xt: (m, m_samples, d)
        - num_projections: int
        - u_weights: (n,) (optional)
        - v_weights: (m,) (optional)
        - p: int (default 2)
    """
    n_distr, n_samples, d = Xs.shape

    master_key, key_directions, key_means, key_samples, key_busemann, key_agg = jax.random.split(rng, num=6)

    ## projections samples
    # generate directions on the sphere
    theta = jax.random.normal(key_directions, (n_projs, d))
    theta = theta / jnp.linalg.norm(
        theta, axis=-1, keepdims=True
    )

    # Projections on R
    projs_Xs = jnp.einsum("cnd,ld->cnl", Xs, theta) # shape (n_distr, n_samples, n_projs)
    projs_Xt = jnp.einsum("cnd,ld->cnl", Xt, theta)

    Xps = jnp.moveaxis(projs_Xs, -1, 0) # shape (n_projs, n_distr, n_samples)
    Xpt = jnp.moveaxis(projs_Xt, -1, 0)

    u0_values = jnp.zeros((n_projs,1,n_samples)) # Dirac initilization

    # Unif on [-1,1]
    m1 = -1 + 2 * jax.random.uniform(key_means, (n_projs, 1))
    s1 = jnp.sqrt(1-m1**2) # such that W_2^2(\delta_0, \mu) = \int x^2 d\mu(x)=1
    u1_values = m1[:,:,None] + s1[:,:,None] * jax.random.uniform(key_samples, (n_projs, 1, n_samples))


    keys_busemann = jax.random.split(key_busemann, num=n_projs)
    busemann_vmap = jax.vmap(busemann1D)

    projs_Xs_1d = busemann_vmap(keys_busemann, u0_values, u1_values, Xps) # shape (L, n)
    projs_Xt_1d = busemann_vmap(keys_busemann, u0_values, u1_values, Xpt) # shape (L, m)

    # Aggregate projections by Hierarchical hybrid projections
    psi = jax.random.normal(key_agg, (n_projs, 2)) 
    psi = psi / jnp.linalg.norm(psi, axis=-1, keepdims=True)

    full_projs_s = psi[:,0][:,None,None] * Xps + (psi[:,1][:,None] * projs_Xs_1d)[:,:,None]
    full_projs_t = psi[:,0][:,None,None] * Xpt + (psi[:,1][:,None] * projs_Xt_1d)[:,:,None]

    # compute quantiles of sliced distributions
    percentiles = jnp.linspace(0, 1, 100)
    # add weights in percentile to take into account non uniform distributions??
    quantiles_proj_source = jnp.percentile(full_projs_s.reshape(n_projs, -1), percentiles*100, axis=-1).T
    quantiles_proj_target = jnp.percentile(full_projs_t.reshape(n_projs, -1), percentiles*100, axis=-1).T

    # compute sliced wasserstein value
    return sliced_wasserstein_from_quantiles(
        quantiles_proj_source, quantiles_proj_target, percentiles, master_key, p
    )
