import torch

import numpy as np
import torch.nn.functional as F

from .emd1d import emd1D
from .utils_labels import pad_and_group_by_class
from .utils_projs import proj_support


def busemann_semi_gaussians_1d(m0, s0, m1, s1, u_values, u_weights=None):
    """
        Compute the Busemann function for a 1D Gaussian geodesic ray and a
        discrete distribution. It also works for s_0=0 (i.e. geodesic ray
        starting from a Dirac).

        Inputs:
        - m0: mean of mu_0 (tensor of size (n_batch, 1))
        - s0: std of mu_0 (tensor of size (n_batch, 1))
        - m1: mean of mu_1 (tensor of size (n_batch, 1))
        - s1: std of mu_1 (tensor of size (n_batch, 1))
        - u_values: tensor of size (n_distr, n_batch, n)
        - u_weights: corresponding weights, if None, uniform weights

        Output:
        - shape (n_batch, n_distr)
    """
    n_distr, n_batch, n = u_values.shape

    dtype = u_values.dtype
    device = u_values.device

    standard_distr = torch.distributions.Normal(torch.tensor([0.0],
                                                             device=device),
                                                torch.tensor([1.0],
                                                             device=device))

    if u_weights is None:
        u_weights = torch.full(u_values.shape, 1/n, dtype=dtype,
                               device=device)
    elif u_weights.ndim != u_values.ndim:
        u_weights = torch.repeat_interleave(u_weights[..., None],
                                            u_values.shape[0], dim=-1).T

    u_values, u_sorter = torch.sort(u_values, -1)
    u_weights = torch.gather(u_weights, -1, u_sorter)

    # u_weights = u_weights[..., u_sorter]
    u_cdf = torch.cumsum(u_weights, -1)

    m = torch.sum(u_values * u_weights, axis=-1)

    # Compute icdf of u and of standard Gaussian
    v_weights = torch.full(u_values.shape, 1/n, dtype=dtype, device=device)
    v_cdf = torch.cumsum(v_weights, -1)  # [:, None].expand(-1, n_batch).T

    cdf_axis, _ = torch.sort(torch.cat((u_cdf, v_cdf), -1), -1)

    u_index = torch.searchsorted(u_cdf, cdf_axis)

    u_icdf = torch.gather(u_values, -1, u_index.clip(0, n-1))
    v_icdf = standard_distr.icdf(cdf_axis)  # grid_1d)

    # For p=1, \phi^-1(p) = inf, discard these terms?
    v_icdf[torch.isinf(v_icdf)] = 0
    v_icdf[torch.isnan(v_icdf)] = 0

    cdf_axis = torch.nn.functional.pad(cdf_axis, (1, 0))
    delta = cdf_axis[..., 1:] - cdf_axis[..., :-1]

    s = torch.sum(delta * u_icdf * v_icdf, axis=-1)

    return -(m1-m0)*(m.T-m0)-(s1-s0)*(s.T-s0)


def proj_distr_Busemann(Xps, Xpt, num_projections, u_weights_class=None, v_weights_class=None, batch_projs=100000, eps=0):
    """
        Compute the Busemann functions on different geodesic rays sampled with
        \mu_0=N(0,eps^2) (accepts eps=0, i.e. mu0=\delta_0) and \mu_1 Gaussian with m_1 unif on [-1,1].

        Parameters
        ----------
        - Xps: ndarray shape (C_distr, n_projs, n_samples)
        - Xpt: ndarray shape (C_distr, n_projs, m_samples)
        - u_weights_class: (C_distr, n_samples)
        - v_weights_class: (C_distr, m_samples)
        - num_projections: int, number of projections
        - batch_projs: int, batch to compute Busemann function
        - eps: float (default 0): std of mu0

        Returns
        -------
        - projs_Xs_1d: ndarray shape (n_projs, C_distr)
        - projs_Xt_1d: ndarray shape (n_projs, C_distr)
    """
    device = Xps.device
    dtype = Xps.dtype

    # Sample geodesic rays
    # N(0, eps^2), eps=0 ok
    m0 = torch.zeros((num_projections, 1), device=device, dtype=dtype)
    s0 = eps * torch.ones((num_projections, 1), device=device, dtype=dtype)

    # Unif on [-1,1]
    m1 = -1 + 2 * torch.rand(num_projections, 1, dtype=dtype, device=device)
    s1 = s0 + torch.sqrt(1-m1**2)  # such that W_2^2(\mu_0, \mu_1) = 1

    if u_weights_class is not None:
        u_weights_class = u_weights_class.unsqueeze(1).expand(-1, Xps.shape[1], -1)
        v_weights_class = v_weights_class.unsqueeze(1).expand(-1, Xpt.shape[1], -1)

    if batch_projs >= num_projections:
        # compute Buseman function in 1d
        projs_Xs_1d = busemann_semi_gaussians_1d(m0, s0, m1, s1, Xps, u_weights=u_weights_class) # shape (num_projections, C_distr)
        projs_Xt_1d = busemann_semi_gaussians_1d(m0, s0, m1, s1, Xpt, u_weights=v_weights_class) # shape (num_projections, C_distr)
    else:
        n_chunks = num_projections // batch_projs + 1
        L_x = []
        L_y = []

        for n in range(n_chunks):
            n_projs = batch_projs if n<n_chunks-1 else (num_projections%batch_projs)

            proj_x = busemann_semi_gaussians_1d(
                m0[n*batch_projs:n*batch_projs+n_projs],
                s0[n*batch_projs:n*batch_projs+n_projs],
                m1[n*batch_projs:n*batch_projs+n_projs],
                s1[n*batch_projs:n*batch_projs+n_projs],
                Xps[:, n*batch_projs:n*batch_projs+n_projs],
                u_weights=u_weights_class
            )  # shape (num_projections, C_distr)

            proj_y = busemann_semi_gaussians_1d(
                m0[n*batch_projs:n*batch_projs+n_projs],
                s0[n*batch_projs:n*batch_projs+n_projs],
                m1[n*batch_projs:n*batch_projs+n_projs],
                s1[n*batch_projs:n*batch_projs+n_projs],
                Xpt[:, n*batch_projs:n*batch_projs+n_projs],
                u_weights=v_weights_class
            )  # shape (num_projections, C_distr)

            L_x.append(proj_x)
            L_y.append(proj_y)
    
        projs_Xs_1d = torch.cat(L_x, dim=0)
        projs_Xt_1d = torch.cat(L_y, dim=0)

    return projs_Xs_1d, projs_Xt_1d


def busemann_sliced_semi_gaussian_1d(Xs, Xt, num_projections, p=2, type_proj="linear", batch_projs=1000, features=True, eps=1):
    """
        Sliced Busemann projecting samples on R using a hierarchical hybrid projection
        to combine projections of the samples on R and of the distributions of the labels
        using the Busemann function. The geodesic rays are sampled with \mu_0=N(0,eps^2)
        (accepts eps=0, i.e. mu0=\delta_0) and \mu_1 Gaussian with m_1 unif on [-1,1].

        Parameters
        ----------
        - Xs: (n, n_samples, d)
        - Xt: (m, m_samples, d)
        - num_projections: int
        - p: int (default 2)
        - type_proj: str in "linear" or "conv"
        - batch_projs: int, batch projections for convolutions
        - features: bool (default True): if yes, project also features
        - eps: float (default 1): std of mu0, also works for eps=0, using mu0=delta_0
    """
    device, dtype = Xs.device, Xt.dtype

    # Projection support in 1D
    projs_Xs, projs_Xt = proj_support(
        Xs, Xt, num_projections, type_proj,
        batch_projs)  # shape (n_distr * n_samples, n_projs)

    Xps = torch.movedim(projs_Xs, -1, -2)  # shape (n_distr, n_projs, n_samples)
    Xpt = torch.movedim(projs_Xt, -1, -2)

    # Projection of distributions
    projs_Xs_1d, projs_Xt_1d = proj_distr_Busemann(
        Xps, Xpt, num_projections, eps=eps
    )  # shape (n_projs, n_distr)

    # Aggregate projections by Hierarchical hybrid projections
    if features:
        Xps = torch.movedim(projs_Xs, -1, 0)  # shape (n_projs, n_distr, n_samples)
        Xpt = torch.movedim(projs_Xt, -1, 0)

        psi = torch.randn(num_projections, 2, dtype=dtype, device=device)
        psi = F.normalize(psi, p=2, dim=-1)

        full_projs_s = psi[:, 0][:, None, None] * Xps + (psi[:, 1][:, None] * projs_Xs_1d).unsqueeze(-1)
        full_projs_t = psi[:, 0][:, None, None] * Xpt + (psi[:, 1][:, None] * projs_Xt_1d).unsqueeze(-1)
    else:
        full_projs_s = projs_Xs_1d
        full_projs_t = projs_Xt_1d

    ot_1d = emd1D(
        full_projs_s.reshape(num_projections, -1),
        full_projs_t.reshape(num_projections, -1),
        p=p
    )

    return torch.mean(ot_1d)


def busemann_sliced_semi_gaussian_1d_diff_labels(
        Xs, Xt, ys, yt, num_projections, p=2, type_proj="linear",
        batch_projs=1000, batch_w2=100000, eps=0
    ):
    """
        Sliced Busemann projecting samples on R using a hierarchical hybrid projection
        to combine projections of the samples on R and of the distributions of the labels
        using the Busemann function. The geodesic rays are sampled with \mu_0=N(0,eps^2)
        (accepts eps=0, i.e. mu0=\delta_0) and \mu_1 Gaussian with m_1 unif on [-1,1].

        Must be given the original datasets with labels, and thus accept distributions with
        different number of samples in each class.

        Parameters
        ----------
        - Xs: (n, d)
        - Xt: (m, d)
        - ys: (n,)
        - yt: (m,)
        - num_projections: int
        - p: int (default 2)
        - type_proj: "linear" or "conv"
        - batch_projs: int, batch projections for convolutions
        - batch_w2: int, batch 1d wasserstein distances
        - eps: float (default 0): std of mu0
    """
    device, dtype = Xs.device, Xt.dtype

    # Project the support in 1D
    projs_Xs, projs_Xt = proj_support(
        Xs, Xt, num_projections, type_proj, batch_projs
    )  # shape (n_distr * n_samples, n_projs)

    # Refactor to have Xps of shape (n_distrs, n_samples, n_projs), with
    # 0 weights for padding samples
    Xps, u_weights_class, u_weights = pad_and_group_by_class(projs_Xs, ys)  # Xps shape (n_distrs, n_samples, n_projs)
    Xpt, v_weights_class, v_weights = pad_and_group_by_class(projs_Xt, yt)

    Xps = torch.movedim(Xps, -1, -2)  # shape (n_distrs, n_projs, n_samples)
    Xpt = torch.movedim(Xpt, -1, -2)

    # Projection of distributions
    projs_Xs_1d, projs_Xt_1d = proj_distr_Busemann(
        Xps, Xpt, num_projections, u_weights_class, v_weights_class,
        batch_projs=batch_w2, eps=eps
    )  # shape (n_projs, n_distr)

    # Aggregate projections by Hierarchical hybrid projections
    Xps = torch.movedim(Xps, 1, 0)  # shape (n_projs, n_distr, n_samples)
    Xpt = torch.movedim(Xpt, 1, 0)

    psi = torch.randn(num_projections, 2, dtype=dtype, device=device)
    psi = F.normalize(psi, p=2, dim=-1)

    # Compute 1D Wasserstein distances
    if batch_w2 >= num_projections:
        full_projs_s = psi[:, 0][:, None, None] * Xps + (psi[:, 1][:, None] * projs_Xs_1d).unsqueeze(-1)
        full_projs_t = psi[:, 0][:, None, None] * Xpt + (psi[:, 1][:, None] * projs_Xt_1d).unsqueeze(-1)

        ot_1d = emd1D(
            full_projs_s.reshape(num_projections, -1),
            full_projs_t.reshape(num_projections, -1),
            u_weights=u_weights.reshape(-1),
            v_weights=v_weights.reshape(-1),
            p=p
        )
    else:
        n_chunks = num_projections // batch_w2 + 1

        L_w = []

        for n in range(n_chunks):
            n_projs = batch_w2 if n < n_chunks-1 else (num_projections%batch_w2)

            if n_projs > 0:
                cpt1_src = psi[n*batch_w2:n*batch_w2+n_projs,0][:,None,None] * Xps[n*batch_w2:n*batch_w2+n_projs]
                cpt2_src = psi[n*batch_w2:n*batch_w2+n_projs,1][:,None] * projs_Xs_1d[n*batch_w2:n*batch_w2+n_projs]

                cpt1_tgt = psi[n*batch_w2:n*batch_w2+n_projs,0][:,None,None] * Xpt[n*batch_w2:n*batch_w2+n_projs]
                cpt2_tgt = psi[n*batch_w2:n*batch_w2+n_projs,1][:,None] * projs_Xt_1d[n*batch_w2:n*batch_w2+n_projs]
                full_projs_s = cpt1_src + cpt2_src.unsqueeze(-1)
                full_projs_t = cpt1_tgt + cpt2_tgt.unsqueeze(-1)

                full_projs_src = full_projs_s.reshape(n_projs, -1)
                full_projs_tgt = full_projs_t.reshape(n_projs, -1)

                w = emd1D(
                    full_projs_src,
                    full_projs_tgt,
                    u_weights=u_weights.reshape(-1),
                    v_weights=v_weights.reshape(-1),
                    p=p
                )

                L_w.append(w)

        ot_1d = torch.cat(L_w, dim=0)

    return torch.mean(ot_1d)
