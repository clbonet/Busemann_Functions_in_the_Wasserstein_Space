import sys
import torch

import torch.nn.functional as F

sys.path.append("../")
from lib_torch.emd1d import emd1D


def busemann_gaussians_1d(m, s, m0, s0, m1, s1):
    """
        Compute the Busemann function between 1d Gaussian

        Parameters
        ----------
        - m: shape (L, n)
        - s: shape (L, n)
        - m0: shape (L, 1)
        - s0: shape (L, 1)
        - m1: shape (L, 1)
        - s1: shape (L, 1)
    """
    return -(m1-m0)*(m-m0)-(s1-s0)*(s-s0)


def sliced_1Dbusemann_GMM(mus_Xs, mus_Xt, covs_Xs, covs_Xt, num_projections, u_weights=None, v_weights=None, p=2, eps=1): #, eps=1e-5):
    """
        First project in 1D the Gaussian
        Then project it in R with the Busemann function.
        
        Parameters
        ----------
        - mus_Xs: (n, d)
        - mus_Xt: (m, d)
        - covs_Xs: (n, d, d)
        - covs_Xt: (m, d, d)
        - num_projections: int
        - u_weights: (n,) (optional)
        - v_weights: (m,) (optional)
        - p: int (default 2)
        - eps: float, initialization of Sigma0 = eps * I_d
    """
    d = mus_Xs.shape[1]
    device, dtype = mus_Xs.device, mus_Xt.dtype

    # Sample 1D projections
    theta = torch.randn(num_projections, d, dtype=dtype, device=device)
    theta = F.normalize(theta, p=2, dim=-1)

    # Projections of means on R
    projs_mus_Xs = torch.einsum("...nd,ld->...nl", mus_Xs, theta)  # shape (n_samples, n_projs)
    projs_mus_Xt = torch.einsum("...nd,ld->...nl", mus_Xt, theta)

    # Projections of covs on R
    projs_covs_Xs = torch.einsum("ld, ...ndd, ld->...nl", theta, covs_Xs, theta)
    projs_covs_Xt = torch.einsum("ld, ...ndd, ld->...nl", theta, covs_Xt, theta)


    # Sample projections 1D Gaussian rays
    m0 = torch.zeros((num_projections, 1), device=device, dtype=dtype)
    s0 = eps * torch.ones(num_projections, 1, device=device, dtype=dtype)

    # Unif on [-1,1]
    m1 = -1 + 2 * torch.rand(num_projections, 1, dtype=dtype, device=device)
    s1 = s0 + torch.sqrt(1-m1**2)  # such that W_2^2(\mu_0, \mu_1) = 1

    projs_1d_src = busemann_gaussians_1d(projs_mus_Xs.T, torch.sqrt(projs_covs_Xs).T, m0, s0, m1, s1) # shape (L, n)
    projs_1d_tgt = busemann_gaussians_1d(projs_mus_Xt.T, torch.sqrt(projs_covs_Xt).T, m0, s0, m1, s1) # shape (L, m)

    ot_1d = emd1D(projs_1d_src, projs_1d_tgt, u_weights=u_weights, v_weights=v_weights, p=p)
    return torch.mean(ot_1d)
