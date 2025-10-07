import jax
import jax.numpy as jnp

import jax.scipy.stats.norm as jnorm
import jax.scipy as jsp

from functools import partial
from .sliced_wasserstein import sliced_wasserstein_from_quantiles


def sample_zero_truncated_poisson(rng, rate_vector):
    # From https://github.com/hainn2803/s-OTDD/blob/main/otdd/pytorch/utils.py

    master_key, key_unif, key_poisson = jax.random.split(rng, num=3)

    u = jax.random.uniform(key_unif, rate_vector.shape) * (1 - jnp.exp(-rate_vector)) + jnp.exp(-rate_vector)
    t = -jnp.log(u)
    return 1 + jax.random.poisson(key_poisson, rate_vector - t)


@partial(jax.jit, static_argnums=[3, 4])
def sliced_otdd(
    rng: jax.random.PRNGKey, Xs: jnp.ndarray, Xt: jnp.ndarray,
    n_projs: int = 50, p: float = 2, k: int = 5
    ):
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
    d = Xs.shape[-1]

    master_key, key_directions, key_means, key_agg = jax.random.split(rng, num=4)

    # projections samples
    # generate directions on the sphere
    theta = jax.random.normal(key_directions, (n_projs, d))
    theta = theta / jnp.linalg.norm(
        theta, axis=-1, keepdims=True
    )

    # Projections on R
    projs_Xs = jnp.einsum("cnd,ld->cnl", Xs, theta)  # shape (n_distr, n_samples, n_projs)
    projs_Xt = jnp.einsum("cnd,ld->cnl", Xt, theta)

    Xps = jnp.moveaxis(projs_Xs, -1, 0)  # shape (n_projs, n_distr, n_samples)
    Xpt = jnp.moveaxis(projs_Xt, -1, 0)

    # Sample matrix of shape (L, k) of moments with parameters 1,2,3,4,5
    rates = jnp.tile(jnp.arange(k), (n_projs, 1))
    lambdas = sample_zero_truncated_poisson(key_means, rates)[:, None, None]
    lambdas_factorial = jnp.exp(jsp.special.gammaln(lambdas+1))

    # Mean over samples by class
    projs_labels_s = jnp.mean(jnp.expand_dims(Xps, axis=-1)**lambdas / lambdas_factorial, axis=2)
    projs_labels_t = jnp.mean(jnp.expand_dims(Xpt, axis=-1)**lambdas / lambdas_factorial, axis=2)

    # Aggregate projections by Hierarchical hybrid projections
    psi = jax.random.normal(key_agg, (n_projs, k+1))
    psi = psi / jnp.linalg.norm(psi, axis=-1, keepdims=True)

    full_projs_s = psi[:, 0][:, None, None] * Xps + jnp.einsum("lck, lk -> lc", projs_labels_s, psi[:, 1:])[:, :, None]
    full_projs_t = psi[:, 0][:, None, None] * Xpt + jnp.einsum("lck, lk -> lc", projs_labels_t, psi[:, 1:])[:, :, None]

    # compute quantiles of sliced distributions
    percentiles = jnp.linspace(0, 1, 100)
    # add weights in percentile to take into account non uniform distributions?
    quantiles_proj_source = jnp.percentile(full_projs_s.reshape(n_projs, -1),
                                           percentiles*100, axis=-1).T
    quantiles_proj_target = jnp.percentile(full_projs_t.reshape(n_projs, -1),
                                           percentiles*100, axis=-1).T

    # compute sliced wasserstein value
    return sliced_wasserstein_from_quantiles(
        quantiles_proj_source, quantiles_proj_target, percentiles, master_key, p
    )
