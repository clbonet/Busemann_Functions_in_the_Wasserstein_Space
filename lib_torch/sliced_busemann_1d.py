import torch

import numpy as np
import torch.nn.functional as F

from .emd1d import emd1D
from .utils_labels import pad_and_group_by_class


def busemann1D(u0_values, u1_values, w_values, u0_weights=None,
               u1_weights=None, w_weights=None, require_sort=True):
    """
        Compute the Busemann function in 1D

        Input:
        - u0_values: tensor of size (n,n0_distr)
        - u1_values: tensor of size (m,n1_distr)
        - w_values: tensor of size (n_distr, n_batch, k)
        - u0_weights
        - u1_weights
        - w_weights

        Output:
        - shape (n_distr, n_batch)
    """
    n_distr, n_batch, k = w_values.shape

    n = u0_values.shape[-1]
    m = u1_values.shape[-1]

    u0_values = u0_values.expand(n_distr, n_batch, n)
    u1_values = u1_values.expand(n_distr, n_batch, m)

    device = u0_values.device
    dtype = u0_values.dtype

    if u0_weights is None:
        u0_weights = torch.full(u0_values.shape, 1/n, dtype=dtype,
                                device=device)
    elif u0_weights.ndim != u0_values.ndim:
        u0_weights = torch.repeat_interleave(u0_weights[..., None],
                                             u0_values.shape[0], dim=-1).T

    if u1_weights is None:
        u1_weights = torch.full(u1_values.shape, 1/m, dtype=dtype,
                                device=device)
    elif u1_weights.ndim != u1_values.ndim:
        u1_weights = torch.repeat_interleave(u1_weights[..., None],
                                             u1_values.shape[0], dim=-1).T

    if w_weights is None:
        w_weights = torch.full(w_values.shape, 1/k, dtype=dtype, device=device)
    elif w_weights.ndim != w_values.ndim:
        w_weights = torch.repeat_interleave(w_weights[..., None],
                                            w_values.shape[0], dim=-1).T

    if require_sort:
        u0_values, u0_sorter = torch.sort(u0_values, -1)
        u1_values, u1_sorter = torch.sort(u1_values, -1)
        w_values, w_sorter = torch.sort(w_values, -1)

        u0_values = torch.gather(u0_values, -1, u0_sorter)
        u1_values = torch.gather(u1_values, -1, u1_sorter)
        w_values = torch.gather(w_values, -1, w_sorter)

        u0_weights = torch.gather(u0_weights, -1, u0_sorter)
        u1_weights = torch.gather(u1_weights, -1, u1_sorter)
        w_weights = torch.gather(w_weights, -1, w_sorter)

    u0_cdf = torch.cumsum(u0_weights, -1)
    u1_cdf = torch.cumsum(u1_weights, -1)
    w_cdf = torch.cumsum(w_weights, -1)

    cdf_axis, _ = torch.sort(torch.cat((u0_cdf, u1_cdf, w_cdf), -1), -1)

    u0_index = torch.searchsorted(u0_cdf, cdf_axis)
    u1_index = torch.searchsorted(u1_cdf, cdf_axis)
    w_index = torch.searchsorted(w_cdf, cdf_axis)

    u0_icdf = torch.gather(u0_values, -1, u0_index.clip(0, n-1))
    u1_icdf = torch.gather(u1_values, -1, u1_index.clip(0, m-1))
    w_icdf = torch.gather(w_values, -1, w_index.clip(0, k-1))

    cdf_axis = torch.nn.functional.pad(cdf_axis, (1, 0))
    delta = cdf_axis[..., 1:] - cdf_axis[..., :-1]

    norm = torch.sqrt(torch.sum(delta * torch.square(u0_icdf - u1_icdf), axis=-1))
    return torch.sum(delta * (u0_icdf-u1_icdf) * (w_icdf-u0_icdf), axis=-1) / norm


def busemann_sliced_1d(Xs, Xt, num_projections, u_weights=None, v_weights=None, p=2, features=True):
    """
        Use a hierarchical hybrid projection

        Parameters
        ----------
        - Xs: (n, n_samples, d)
        - Xt: (m, m_samples, d)
        - mus_Xs: (n, d)
        - mus_Xt: (m, d)
        - covs_Xs: (n, d, d)
        - covs_Xt: (m, d, d)
        - num_projections: int
        - u_weights: (n,) (optional)
        - v_weights: (m,) (optional)
        - p: int (default 2)
    """
    n_distr, n_samples, d = Xs.shape
    device, dtype = Xs.device, Xt.dtype

    theta = torch.randn(num_projections, d, dtype=dtype, device=device)
    theta = F.normalize(theta, p=2, dim=-1)

    # Projections on R
    projs_Xs = torch.einsum("cnd,ld->cnl", Xs, theta)  # shape (n_distr, n_samples, n_projs)
    projs_Xt = torch.einsum("cnd,ld->cnl", Xt, theta)

    Xps = torch.movedim(projs_Xs, -1, -2)  # shape (n_distr, n_projs, n_samples)
    Xpt = torch.movedim(projs_Xt, -1, -2)

    u0_values = torch.zeros((num_projections, 1), device=device, dtype=dtype)  # Dirac initilization

    # Unif on [-1,1]
    m1 = -1 + 2 * torch.rand(num_projections, 1, dtype=dtype, device=device)
    s1 = torch.sqrt(1-m1**2)  # such that W_2^2(\delta_0, \mu) = \int x^2 d\mu(x)=1
    u1_values = m1 + s1 * torch.randn(num_projections, n_samples, dtype=dtype, device=device)

    projs_Xs_1d = busemann1D(u0_values, u1_values, Xps, w_weights=u_weights).T # shape (L, n)
    projs_Xt_1d = busemann1D(u0_values, u1_values, Xpt, w_weights=v_weights).T # shape (L, m)

    # Aggregate projections by Hierarchical hybrid projections
    if features:
        Xps = torch.movedim(projs_Xs, -1, 0)  # shape (n_projs, n_distr, n_samples)
        Xpt = torch.movedim(projs_Xt, -1, 0)

        psi = torch.randn(num_projections, 2, dtype=dtype, device=device)
        psi = F.normalize(psi, p=2, dim=-1)

        full_projs_s = psi[:,0][:,None,None] * Xps + (psi[:,1][:,None] * projs_Xs_1d).unsqueeze(-1)
        full_projs_t = psi[:,0][:,None,None] * Xpt + (psi[:,1][:,None] * projs_Xt_1d).unsqueeze(-1)
    else:
        full_projs_s = projs_Xs_1d
        full_projs_t = projs_Xt_1d

    ot_1d = emd1D(full_projs_s.reshape(num_projections, -1),
                  full_projs_t.reshape(num_projections, -1), u_weights=u_weights, v_weights=v_weights, p=p)

    return torch.mean(ot_1d)    
