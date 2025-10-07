import jax
import jax.numpy as jnp

from functools import partial
from .sliced_wasserstein import sliced_wasserstein_from_quantiles


@jax.jit
def sqrtm(a):
    """
        square root of SPD matrix.

        Code from POT https://github.com/PythonOT/POT/blob/master/ot/backend.py
    """
    L, V = jnp.linalg.eigh(a)
    L = jnp.sqrt(jnp.clip(L, 1e-8))
    # Q[...] = V[...] @ diag(L[...])
    Q = jnp.einsum('...jk,...k->...jk', V, L)
    # R[...] = Q[...] @ V[...].T
    # return jnp.einsum('...jk,...kl->...jl', Q, jnp.transpose(V, (-1, -2)))
    return jnp.einsum('...jk,...kl->...jl', Q, jnp.swapaxes(V, -1, -2))


@jax.jit
def exp_bures(Sigma, S):
    r"""
    Exponential map in Bures-Wasserstein space at Sigma:

    .. math::
        \exp_\Sigma(S) = (I_d+S)\Sigma(I_d+S).

    Parameters
    ----------
    Sigma : array-like (d,d)
        SPD matrix
    S : array-like (d,d)
        Symmetric matrix

    Returns
    -------
    P : array-like (d,d)
        SPD matrix obtained as the exponential map of S at Sigma
    """
    d = S.shape[-1]
    Id = jnp.eye(d)
    C = Id + S

    return jnp.einsum("...ij,jk,...kl -> ...il", C, Sigma, C)


@jax.jit
def busemannGaussians(m0, Sigma0, m1, Sigma1, m, Sigma):
    """
        Compute the Busemann function on the Bures-Wasserstein space with (m1, Sigma1) and (m,Sigma) batched

        Inputs:
        - m0: mean mu_0, size (d,)
        - Sigma0: Covariance matrix mu_0, size (d, d)
        - m1: mean mu_1, size (L, d,)
        - Sigma1: Covariance matrix mu_1, size (L, d, d)
        - m: means of the data points, size (n_batch, d)
        - Sigma: covariances matrices of data points, size (n_batch, d, d)

        Output of shape (L, n_batch)
    """
    L = m1.shape[0]
    # n, d = m.shape
    d = m.shape[-1]

    diff_m1 = m1 - m0[None] # shape (L, d)
    diff_m = m - m0[None] # shape (n, d)

    ip_m = -jnp.sum(diff_m1[:,None] * diff_m[None], axis=-1) # shape (L, n)

    s012 = sqrtm(Sigma0)
    s0_12 = jnp.linalg.inv(s012)
    M0 = sqrtm(jnp.einsum("ij,...jk,kl->...il", s012, Sigma1, s012))
    T = jnp.einsum("ij,...jk,kl->...il", s0_12, M0, s0_12)

    C1 = jnp.einsum("ij,...jk->...ik", Sigma0, T-jnp.eye(d))

    s12 = sqrtm(Sigma)
    prod_S0_T = jnp.einsum("ij,...jk->...ik", Sigma0, T)
    C = Sigma0 - prod_S0_T - jnp.swapaxes(prod_S0_T, -1, -2) + Sigma1

    C12 = sqrtm(jnp.einsum("nij, mjk, nkl -> mnil", s12, C, s12))

    return ip_m + jnp.linalg.trace(C1)[:,None] - jnp.linalg.trace(C12)


def busemann_sliced_gaussian_features(
    rng: jax.random.PRNGKey, Xs: jnp.ndarray, Xt: jnp.ndarray,
    mus_Xs: jnp.ndarray, mus_Xt: jnp.ndarray, covs_Xs: jnp.ndarray, covs_Xt: jnp.ndarray,
    num_projections: int, p: int = 2, eps: float = 1): #, eps=1e-5):

    d = Xs.shape[-1]
    d_gaussian = mus_Xs.shape[-1]

    master_key, key_directions, key_means, key_spds_diag, key_spds_P, key_agg = jax.random.split(rng, num=6)

    # projections samples
    # generate directions on the sphere
    directions = jax.random.normal(key_directions, (num_projections, d))
    directions = directions / jnp.linalg.norm(
        directions, axis=-1, keepdims=True
    )

    # slice source and targets along the directions
    Xps = jnp.moveaxis(jnp.einsum("cnd,ld->cnl", Xs, directions), -1, 0)
    Xpt = jnp.moveaxis(jnp.einsum("cnd,ld->cnl", Xt, directions), -1, 0)

    m0, Sigma0 = jnp.zeros((d_gaussian,)), eps * jnp.eye(d_gaussian)

    # Sample direction means
    m_projs = jax.random.normal(key_means, (num_projections, d_gaussian))
    m_projs = m_projs / jnp.linalg.norm(m_projs, axis=-1, keepdims=True)

    # Sample covs
    # Not sure about this (not uniform, could sample on S_d(R) directly if don't care about rays), but 1st try
    theta = jnp.abs(jax.random.normal(key_spds_diag, (num_projections, d_gaussian)))
    D = theta[:, None] * jnp.eye(theta.shape[-1])

    Z = jax.random.normal(key_spds_P, (num_projections, d_gaussian, d_gaussian))
    Q, R = jnp.linalg.qr(Z)
    lambd = jnp.diagonal(R, axis1=-2, axis2=-1)
    lambd = lambd / jnp.abs(lambd)
    P = lambd[:, None] * Q

    # Normalize
    A = jnp.einsum("...ij,...jk,...lk->...il", P, D, P) # torch.matmul(P,torch.matmul(D, torch.transpose(P, -2, -1)))
    squared_norm_A = jnp.trace(jnp.einsum("...ij,jk,...kl->...il", A, Sigma0, A), axis1=-2, axis2=-1)
    norm = jnp.sqrt(squared_norm_A + jnp.linalg.norm(m_projs - m0[None], axis=-1)**2)

    # Normalize to have BW(mu0, mu1)=1
    S_projs = A / norm[:, None, None]
    m_projs = m_projs / norm[:, None]

    Sigma_projs = exp_bures(Sigma0, S_projs)

    projs_Xs_1d = busemannGaussians(m0, Sigma0, m_projs, Sigma_projs, mus_Xs, covs_Xs) # shape (L, n)
    projs_Xt_1d = busemannGaussians(m0, Sigma0, m_projs, Sigma_projs, mus_Xt, covs_Xt) # shape (L, m)

    # Aggregate projections by Hierarchical hybrid projections
    psi = jax.random.normal(key_agg, (num_projections, 2)) 
    psi = psi / jnp.linalg.norm(psi, axis=-1, keepdims=True)

    full_projs_s = psi[:, 0][:, None, None] * Xps + (psi[:, 1][:, None] * projs_Xs_1d)[:,:, None]
    full_projs_t = psi[:, 0][:, None, None] * Xpt + (psi[:, 1][:, None] * projs_Xt_1d)[:, :, None]

    # compute quantiles of sliced distributions
    percentiles = jnp.linspace(0, 1, 100)
    # add weights in percentile to take into account non uniform distributions??
    quantiles_proj_source = jnp.percentile(full_projs_s.reshape(num_projections, -1), percentiles*100, axis=-1).T
    quantiles_proj_target = jnp.percentile(full_projs_t.reshape(num_projections, -1), percentiles*100, axis=-1).T

    # compute sliced wasserstein value
    return sliced_wasserstein_from_quantiles(
        quantiles_proj_source, quantiles_proj_target, percentiles, master_key, p
    )


@partial(jax.jit, static_argnums=[3, 4, 5])
def busemann_sliced_gaussian_approx(
    Xs: jnp.ndarray, Xt: jnp.ndarray, rng: jax.random.PRNGKey,
    n_projs: int = 50, p: float = 2, eps: float = 1
):
    # Gaussian approximation
    _, ns, ds = Xs.shape
    _, nt, dt = Xt.shape

    mus_Xs, mus_Xt = jnp.mean(Xs, axis=1), jnp.mean(Xt, axis=1)

    Cs = Xs - mus_Xs[:, None]
    Ct = Xt - mus_Xt[:, None]
    covs_Xs = jnp.einsum("lnd, lnk -> ldk", Cs, Cs) / ns
    covs_Xt = jnp.einsum("lnd, lnk -> ldk", Ct, Ct) / nt

    # mus_Xs = jax.lax.stop_gradient(mus_Xs)
    # mus_Xt = jax.lax.stop_gradient(mus_Xt)
    # covs_Xs = jax.lax.stop_gradient(covs_Xs)
    # covs_Xt = jax.lax.stop_gradient(covs_Xt)

    return busemann_sliced_gaussian_features(rng, Xs, Xt, mus_Xs, mus_Xt, covs_Xs, covs_Xt, n_projs, p, eps)
