import jax
import ot

import jax.numpy as jnp
import numpy as np

from jax_tqdm import scan_tqdm
from functools import partial
from sklearn.decomposition import PCA

from .utils_bw import exp_bw, exp_bures, bures_wasserstein_batch, sqrtm
from .datasets import get_moments_from_dataset
from .utils_labels import get_labels_mmd_product
from .sliced_busemann_gaussian_jax import busemann_sliced_gaussian_features
from .sliced_wasserstein import sliced_wasserstein_from_quantiles


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

    n_classes, _, d = Xs.shape
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
    
    projs_Xs_1d = projs_Xs_1d.reshape(num_projections, n_classes, -1)
    projs_Xt_1d = projs_Xt_1d.reshape(num_projections, n_classes, -1)

    # Aggregate projections by Hierarchical hybrid projections
    psi = jax.random.normal(key_agg, (num_projections, 2)) 
    psi = psi / jnp.linalg.norm(psi, axis=-1, keepdims=True)
    
    full_projs_s = psi[:, 0][:, None, None] * Xps + (psi[:, 1][:, None, None] * projs_Xs_1d)
    full_projs_t = psi[:, 0][:, None, None] * Xpt + (psi[:, 1][:, None, None] * projs_Xt_1d)

    # compute quantiles of sliced distributions
    percentiles = jnp.linspace(0, 1, 100)
    # add weights in percentile to take into account non uniform distributions??
    quantiles_proj_source = jnp.percentile(full_projs_s.reshape(num_projections, -1), percentiles*100, axis=-1).T
    quantiles_proj_target = jnp.percentile(full_projs_t.reshape(num_projections, -1), percentiles*100, axis=-1).T

    # compute sliced wasserstein value
    return sliced_wasserstein_from_quantiles(
        quantiles_proj_source, quantiles_proj_target, percentiles, master_key, p
    )



@partial(jax.jit, static_argnums=[2, 4])
def wasserstein_gradient_descent_product(x0, x_tgt, target_value_and_grad, rng,
                                         n_epochs=101, lr=1, m=0, v0=None):
    @scan_tqdm(n_epochs)
    def step(carry, iter_num):
        xk, vk, key = carry
        master_key, subkey = jax.random.split(key)
        l, grad = target_value_and_grad(xk, x_tgt, subkey)

        v_x, v_mu, v_sigma = vk
        xk_x, xk_mu, xk_sigma = xk

        # Allows momentum
        v_x = grad[0] + m * v_x
        v_mu = grad[1] + m * v_mu
        # v_sigma = 2 * grad[2] + m * v_sigma

        xk_x = xk_x - lr * v_x
        xk_mu = xk_mu - lr * v_mu
        xk_sigma = exp_bw(xk_sigma, -2 * lr * grad[2])

        xk = (xk_x, xk_mu, xk_sigma)
        vk = (v_x, v_mu, v_sigma)

        return (xk, vk, master_key), l

    # Initial state
    if v0 is None:
        v0 = jnp.zeros_like(x0)

    # Use `lax.scan` to loop over epochs
    (xk, _, _), L_loss = jax.lax.scan(step, (x0, v0, rng), jnp.arange(n_epochs))

    return L_loss, xk


def swbg_flow(key_wgd, X_data_src, y_src, X_data_tgt, y_tgt,
              reduced_dim=2, lr=1, m=0.9, n_epochs=5000, n_projs=500):
    """
        Run the flow of SWBG on R^d x R^d' x S_d^{++}(R).
        The label distributions are embedded using PCA.

        Parameters
        ----------
        key_wgd: PRNGKey
        X_data_src: ndarray of shape (n_class, n_data_by_class, d)
        y_src: ndarray of shape (n_class, n_data_by_class)
        X_data_tgt: ndarray of shape (n_class, m_data_by_class, d)
        y_tgt: ndarray of shape (n_class, m_data_by_class)
        reduced_dim: int, dimension for PCA
        lr: float, step size
        m: float, momentum
        n_epochs: int, number of steps

        Outputs
        -------
        xk: ndarray of shape (10, n_data_by_class, reduced_dim)
        yk: ndarray of shape (10, n_data_by_class) (aligned with y_tgt)
        L_loss: ndarray of shape (n_epochs,), loss value
    """
    n_class, n_data_by_class, d = X_data_src.shape
    _, m_data_by_class, _ = X_data_tgt.shape

    X_concat = np.concatenate([X_data_src.reshape(-1, d), X_data_tgt.reshape(-1, d)], axis=0)
    pca = PCA(n_components=reduced_dim)
    X_concat_pca = pca.fit_transform(X_concat)

    X_data_src_emb = X_concat_pca[:n_class*n_data_by_class]
    X_data_tgt_emb = X_concat_pca[n_class*n_data_by_class:]
    X_data_src_emb = X_data_src_emb.reshape(n_class, n_data_by_class,
                                            reduced_dim)
    X_data_tgt_emb = X_data_tgt_emb.reshape(n_class, m_data_by_class,
                                            reduced_dim)

    _, _, mu_src, cov_src = get_moments_from_dataset(X_data_src_emb.reshape(-1, reduced_dim), y_src)
    mu_class_tgt, cov_class_tgt, mu_tgt, cov_tgt = get_moments_from_dataset(X_data_tgt_emb.reshape(-1, reduced_dim), y_tgt)

    if m_data_by_class == 1:
        cov_class_tgt = np.concatenate([np.eye(reduced_dim)[None] for k in range(n_class)], axis=0)
        cov_tgt = np.concatenate([np.eye(reduced_dim)[None] for k in range(n_class)], axis=0)


    # X_src = (X_data_src, mu_src.reshape(n_class, n_data_by_class, reduced_dim), cov_src.reshape(n_class, n_data_by_class, reduced_dim, reduced_dim))
    # X_tgt = (X_data_tgt, mu_tgt.reshape(n_class, m_data_by_class, reduced_dim), cov_tgt.reshape(n_class, m_data_by_class, reduced_dim, reduced_dim))
    
    X_src = (X_data_src, mu_src, cov_src)
    X_tgt = (X_data_tgt, mu_tgt, cov_tgt)

    def target_grad(x, y, key):
        def target(x, y):
            x, mu_x, sigma_x = x
            y, mu_y, sigma_y = y
            
            return busemann_sliced_gaussian_features(key, x, y, mu_x, mu_y, sigma_x, sigma_y, num_projections=n_projs, p=2)        
        
        return jax.value_and_grad(target)(x,y)

    Xs, mux, covx = X_src

    v0 = (jnp.zeros_like(X_src[0]), jnp.zeros_like(X_src[1]), jnp.zeros_like(X_src[2]))

    L_loss, tuple_xk = wasserstein_gradient_descent_product(
        X_src, X_tgt, target_grad, key_wgd, lr=lr, m=m,
        n_epochs=n_epochs, v0=v0
    )
        
    tuple_xk = (tuple_xk[0].reshape(-1, d), tuple_xk[1], tuple_xk[2])
    
    xk_labels = get_labels_mmd_product(mu_class_tgt, cov_class_tgt, y_tgt,
                                       tuple_xk)

    z = np.clip(tuple_xk[0], 0, 1)

    xk = np.zeros((n_class, n_data_by_class, z.shape[-1]))
    yk = np.zeros((n_class, n_data_by_class))
    for k in range(n_class):
        xk[k] = z[xk_labels == k]
        yk[k] = np.ones((len(xk[k]),)) * k

    return xk, yk, L_loss
