import torch

import torch.nn.functional as F

# from sklearn.decomposition import PCA
from torchdr import PCA, TSNE

from .utils_sqrtm import sqrtm
from .utils_bw import exp_bw, trace, dots
from .emd1d import emd1D
from .utils_labels import pad_and_group_by_class
from .utils_projs import proj_support


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
    n, d = m.shape

    diff_m1 = m1 - m0[None]  # shape (L, d)
    diff_m = m - m0[None]  # shape (n, d)

    ip_m = -torch.sum(diff_m1[:, None] * diff_m[None], axis=-1)  # shape (L, n)

    s012 = sqrtm(Sigma0)
    s0_12 = torch.linalg.inv(s012)
    M0 = sqrtm(dots(s012, Sigma1, s012))
    T = dots(s0_12, M0, s0_12)

    C1 = dots(Sigma0, T-torch.eye(d, device=T.device))

    s12 = sqrtm(Sigma)
    C = Sigma0 - dots(Sigma0, T) - dots(T, Sigma0) + Sigma1

    C12 = sqrtm(torch.einsum("nij, mjk, nkl -> mnil", s12, C, s12))

    if torch.any(torch.isnan(C12)):
        C12 = torch.nan_to_num(C12)

    return ip_m + trace(C1)[:, None] - trace(C12)


def busemann_sliced_gaussian(mus_Xs, mus_Xt, covs_Xs, covs_Xt, num_projections, u_weights=None, v_weights=None, p=2, eps=1): #, eps=1e-5):
    """
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

    m0 = torch.zeros((d,), device=device, dtype=dtype)
    Sigma0 = eps * torch.eye(d, device=device, dtype=dtype)

    # Sample direction means
    m_projs = torch.randn(num_projections, d, dtype=dtype, device=device)
    m_projs = F.normalize(m_projs, p=2, dim=-1)
    # m_projs = torch.zeros((num_projections, d), dtype=dtype, device=device)

    # Sample covs    
    theta = torch.randn(num_projections, d, dtype=dtype, device=device)
    theta = F.normalize(theta, p=2, dim=-1)
    theta = torch.abs(theta)

    D = theta[:, None] * torch.eye(theta.shape[-1], device=device, dtype=dtype)

    Z = torch.randn(num_projections, d, d, dtype=dtype, device=device)
    Q, R = torch.linalg.qr(Z)
    lambd = torch.diagonal(R, dim1=-2, dim2=-1)
    lambd = lambd / torch.abs(lambd)
    P = lambd[:, None] * Q

    # Normalize
    A = torch.matmul(P, torch.matmul(D, torch.transpose(P, -2, -1)))
    squared_norm = trace(dots(A, Sigma0, A)) + torch.linalg.norm(m_projs - m0[None], axis=-1)**2
    norm = squared_norm ** (1/2)

    # Normalize to have BW(mu0, mu1)=1
    S_projs = A / norm[:, None, None]
    m_projs = m_projs / norm[:, None]
    Sigma_projs = exp_bw(Sigma0, S_projs)

    projs_Xs_1d = busemannGaussians(m0, Sigma0, m_projs, Sigma_projs, mus_Xs, covs_Xs) # shape (L, n)
    projs_Xt_1d = busemannGaussians(m0, Sigma0, m_projs, Sigma_projs, mus_Xt, covs_Xt) # shape (L, m)

    ot_1d = emd1D(projs_Xs_1d, projs_Xt_1d, u_weights=u_weights, v_weights=v_weights, p=p)
    return torch.mean(ot_1d)


def busemann_sliced_gaussian_features(Xs, Xt, mus_Xs, mus_Xt, covs_Xs, covs_Xt, num_projections, u_weights=None, v_weights=None, p=2, eps=1): #, eps=1e-5):
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
        - eps: float, initialization of Sigma0 = eps * I_d
    """
    d = mus_Xs.shape[1]
    device, dtype = mus_Xs.device, mus_Xt.dtype

    theta = torch.randn(num_projections, d, dtype=dtype, device=device)
    theta = F.normalize(theta, p=2, dim=-1)

    # Projections on R
    Xps = torch.movedim(torch.einsum("cnd,ld->cnl", Xs, theta), -1, 0)
    Xpt = torch.movedim(torch.einsum("cnd,ld->cnl", Xt, theta), -1, 0)

    m0 = torch.zeros((d,), device=device, dtype=dtype)
    Sigma0 = eps * torch.eye(d, device=device, dtype=dtype)

    # Sample direction means
    m_projs = torch.randn(num_projections, d, dtype=dtype, device=device)
    m_projs = F.normalize(m_projs, p=2, dim=-1)
    # m_projs = torch.zeros((num_projections, d), dtype=dtype, device=device)

    # Sample covs
    # Not sure about this (not uniform, could sample on S_d(R) directly if don't care about rays), but 1st try
    theta = torch.abs(torch.randn(num_projections, d))
    # theta = torch.randn(num_projections, d)
    # theta = F.normalize(theta, p=2, dim=-1)

    D = theta[:, None] * torch.eye(theta.shape[-1], device=device, dtype=dtype)

    Z = torch.randn(num_projections, d, d, dtype=dtype, device=device)
    Q, R = torch.linalg.qr(Z)
    lambd = torch.diagonal(R, dim1=-2, dim2=-1)
    lambd = lambd / torch.abs(lambd)
    P = lambd[:, None] * Q

    # Normalize
    A = torch.matmul(P, torch.matmul(D, torch.transpose(P, -2, -1)))
    squared_norm = trace(dots(A, Sigma0, A)) + torch.linalg.norm(m_projs - m0[None], axis=-1)**2
    norm = squared_norm ** (1/2)

    # Normalize to have BW(mu0, mu1)=1
    S_projs = A / norm[:, None, None]
    m_projs = m_projs / norm[:, None]
    Sigma_projs = exp_bw(Sigma0, S_projs)

    projs_Xs_1d = busemannGaussians(m0, Sigma0, m_projs, Sigma_projs, mus_Xs, covs_Xs) # shape (L, n)
    projs_Xt_1d = busemannGaussians(m0, Sigma0, m_projs, Sigma_projs, mus_Xt, covs_Xt) # shape (L, m)

    # Aggregate projections by Hierarchical hybrid projections
    psi = torch.randn(num_projections, 2, dtype=dtype, device=device)
    psi = F.normalize(psi, p=2, dim=-1)

    full_projs_s = psi[:,0][:,None,None] * Xps + (psi[:,1][:,None] * projs_Xs_1d).unsqueeze(-1)
    full_projs_t = psi[:,0][:,None,None] * Xpt + (psi[:,1][:,None] * projs_Xt_1d).unsqueeze(-1)

    ot_1d = emd1D(full_projs_s.reshape(num_projections, -1),
                  full_projs_t.reshape(num_projections, -1), u_weights=u_weights, v_weights=v_weights, p=p)
    return torch.mean(ot_1d)


def busemann_sliced_gaussian_approx(Xs, Xt, num_projections, u_weights=None, v_weights=None, p=2, features=True, eps=1, do_pca=False, reduced_dim=2):
    """
        Parameters
        ----------
        Xs: shape (C, n, d)
        Xt: shape (C, n, d)
        num_projections: int
        features: bool, if True, use a hierarchical hybrid projection
        do_pca: bool, if True, compute an embedding in dimension 2, and work with the means and covs in the embedding space.
        reduced_dim: int, dim embedding
    """
    # Gaussian approximation
    ns_classes, ns, ds = Xs.shape
    nt_classes, nt, dt = Xt.shape

    if do_pca:
        # X_concat = np.concatenate([Xs.reshape(-1, ds), Xt.reshape(-1, dt)], axis=0)
        # pca = PCA(n_components=reduced_dim)
        # X_concat_pca = pca.fit_transform(X_concat)

        X_concat = torch([Xs.reshape(-1, ds), Xt.reshape(-1, dt)], axis=0)
        X_concat = X_concat - X_concat.mean(0)
        pca = PCA(n_components=reduced_dim)
        X_concat_pca = pca.fit_transform(X_concat)

        X_data_src_emb = X_concat_pca[:ns_classes*ns]
        X_data_tgt_emb = X_concat_pca[ns_classes*ns:]

        Xs = X_data_src_emb.reshape(ns_classes, ns, reduced_dim)
        Xt = X_data_tgt_emb.reshape(nt_classes, nt, reduced_dim)

    mus_Xs, mus_Xt = torch.mean(Xs, dim=1), torch.mean(Xt, dim=1)

    Cs = Xs - mus_Xs[:, None]
    Ct = Xt - mus_Xt[:, None]
    covs_Xs = torch.einsum("lnd, lnk -> ldk", Cs, Cs) / ns
    covs_Xt = torch.einsum("lnd, lnk -> ldk", Ct, Ct) / nt

    if features:
        return busemann_sliced_gaussian_features(Xs, Xt, mus_Xs, mus_Xt, covs_Xs, covs_Xt, num_projections, u_weights, v_weights, p, eps=eps)
    else:
        return busemann_sliced_gaussian(mus_Xs, mus_Xt, covs_Xs, covs_Xt, num_projections, u_weights, v_weights, p, eps=eps)


def busemann_sliced_gaussian_features_diff_labels(
        Xps, Xpt, u_weights, v_weights,
        mus_Xs, mus_Xt, covs_Xs, covs_Xt,
        num_projections, batch_w2=100000, p=2, eps=1
        ):  # , eps=1e-5):
    """
        Use a hierarchical hybrid projection

        Parameters
        ----------
        - Xs: (n, d)
        - Xt: (m, d)
        - u_weights: (n,)
        - v_weights: (m,)
        - mus_Xs: (n, d)
        - mus_Xt: (m, d)
        - covs_Xs: (n, d, d)
        - covs_Xt: (m, d, d)
        - num_projections: int
        - batch_w2: int, batch 1d wasserstein distances
        - p: int (default 2)
        - eps: float, initialization of Sigma0 = eps * I_d
    """
    d = mus_Xs.shape[1]
    device, dtype = mus_Xs.device, mus_Xt.dtype

    # sample geodesic rays
    m0 = torch.zeros((d,), device=device, dtype=dtype)
    Sigma0 = eps * torch.eye(d, device=device, dtype=dtype)

    # Sample direction means
    m_projs = torch.randn(num_projections, d, dtype=dtype, device=device)
    m_projs = F.normalize(m_projs, p=2, dim=-1)
    # m_projs = torch.zeros((num_projections, d), dtype=dtype, device=device)

    # Sample covs
    # Not sure about this (not uniform, could sample on S_d(R) directly if don't care about rays), but 1st try
    # theta = torch.abs(torch.randn(num_projections, d))

    theta = torch.randn(num_projections, d, dtype=dtype, device=device)
    theta = F.normalize(theta, p=2, dim=-1)
    theta = torch.abs(theta)

    D = theta[:, None] * torch.eye(theta.shape[-1], device=device, dtype=dtype)

    Z = torch.randn(num_projections, d, d, dtype=dtype, device=device)
    Q, R = torch.linalg.qr(Z)
    lambd = torch.diagonal(R, dim1=-2, dim2=-1)
    lambd = lambd / torch.abs(lambd)
    P = lambd[:, None] * Q

    # Normalize
    A = torch.matmul(P, torch.matmul(D, torch.transpose(P, -2, -1)))
    squared_norm = trace(dots(A, Sigma0, A)) + torch.linalg.norm(m_projs - m0[None], axis=-1)**2
    norm = squared_norm ** (1/2)

    # Normalize to have BW(mu0, mu1)=1
    S_projs = A / norm[:, None, None]
    m_projs = m_projs / norm[:, None]
    Sigma_projs = exp_bw(Sigma0, S_projs)

    if batch_w2 >= num_projections:
        projs_Xs_1d = busemannGaussians(m0, Sigma0, m_projs, Sigma_projs, mus_Xs, covs_Xs) # shape (num_projections, n_distr)
        projs_Xt_1d = busemannGaussians(m0, Sigma0, m_projs, Sigma_projs, mus_Xt, covs_Xt) # shape (num_projections, m_distr)
    else:
        n_chunks = num_projections // batch_w2 + 1
        L_x = []
        L_y = []

        for n in range(n_chunks):
            n_projs = batch_w2 if n < n_chunks-1 else (num_projections % batch_w2)

            if n_projs > 0:
                proj_x = busemannGaussians(
                    m0, Sigma0,
                    m_projs[n*batch_w2:n*batch_w2 + n_projs],
                    Sigma_projs[n*batch_w2:n*batch_w2 + n_projs],
                    mus_Xs, covs_Xs
                )  # shape (num_projections, n_distr)

                proj_y = busemannGaussians(
                    m0, Sigma0,
                    m_projs[n*batch_w2:n*batch_w2 + n_projs],
                    Sigma_projs[n*batch_w2:n*batch_w2 + n_projs],
                    mus_Xt, covs_Xt
                )  # shape (num_projections, m_distr)

                L_x.append(proj_x)
                L_y.append(proj_y)

        projs_Xs_1d = torch.cat(L_x, dim=0)
        projs_Xt_1d = torch.cat(L_y, dim=0)

    # Aggregate projections by Hierarchical hybrid projections
    psi = torch.randn(num_projections, 2, dtype=dtype, device=device)
    psi = F.normalize(psi, p=2, dim=-1)

    Xps = torch.movedim(Xps, 1, 0)  # shape (n_projs, n_distr, n_samples)
    Xpt = torch.movedim(Xpt, 1, 0)

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
            n_projs = batch_w2 if n<n_chunks-1 else (num_projections%batch_w2)

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


def embed_data(Xs, Xt, embedding, reduced_dim):
    """
        Parameters
        ----------
        - Xs: shape (ns, d)
        - Xt: shape (nt, d)
        - embedding: str, "pca" or "tsne"
        - reduced_dim: int
    """
    ns, nt = Xs.shape[0], Xt.shape[0]

    X_concat = torch.cat([Xs, Xt], axis=0)
    X_concat = X_concat - X_concat.mean(0)

    if embedding == "pca":
        pca = PCA(n_components=reduced_dim)
        X_concat_pca = pca.fit_transform(X_concat)

        X_data_src_emb = X_concat_pca[:ns]
        X_data_tgt_emb = X_concat_pca[ns:]

        Xs_emb = X_data_src_emb.reshape(ns, reduced_dim)
        Xt_emb = X_data_tgt_emb.reshape(nt, reduced_dim)
    elif embedding == "tsne":
        tsne = TSNE(n_components=reduced_dim, max_iter=100, optimizer="Adam", lr=1e-3)
        X_concat_tsne = tsne.fit_transform(X_concat)

        X_data_src_emb = X_concat_tsne[:ns]
        X_data_tgt_emb = X_concat_tsne[ns:]

        Xs_emb = X_data_src_emb.reshape(ns, reduced_dim)
        Xt_emb = X_data_tgt_emb.reshape(nt, reduced_dim)
    else:
        Xs_emb, Xt_emb = Xs, Xt

    return Xs_emb, Xt_emb


def busemann_sliced_gaussian_approx_diff_labels(
        Xs, Xt, ys, yt, num_projections,
        p=2, eps=1, embedding=None, reduced_dim=2,
        type_proj="linear", batch_projs=1000, batch_w2=100000
        ):
    """
        Parameters
        ----------
        - Xs: shape (n, d) or (n, c, d, d)
        - Xt: shape (n, d) or (n, c, d, d)
        - ys: (n,)
        - yt: (m,)
        - num_projections: int
        - embedding: str, either None, "pca" or "tsne", compute an embedding in dimension reduced_dim
        - means and covs in the embedding space.
        - reduced_dim: int, dim embedding
        - type_proj: str in "linear" or "conv"
        - batch_projs: int, batch projections for convolutions
        - batch_w2: int, batch 1d wasserstein distances
    """
    # Gaussian approximation
    ns = Xs.shape[0]
    nt = Xt.shape[0]

    # Projection in 1D
    projs_Xs, projs_Xt = proj_support(
        Xs, Xt, num_projections, type_proj, batch_projs
    )  # shape (n_distr * n_samples, n_projs)

    # Refactor to have Xps of shape (n_distrs, n_samples, n_projs), with
    # 0 weights for padding samples
    Xps, u_weights_class, u_weights = pad_and_group_by_class(projs_Xs, ys)  # Xps shape (n_distrs, n_samples, n_projs)
    Xpt, v_weights_class, v_weights = pad_and_group_by_class(projs_Xt, yt)

    Xps = torch.movedim(Xps, -1, -2)  # shape (n_distrs, n_projs, n_samples)
    Xpt = torch.movedim(Xpt, -1, -2)

    # Embed data using reduction dimension
    Xs_emb, Xt_emb = embed_data(
        Xs.reshape(ns, -1),
        Xt.reshape(nt, -1),
        embedding,
        reduced_dim
    )

    Xs_classes, u_weights_class, u_weights = pad_and_group_by_class(Xs_emb, ys)  # Xps shape (n_distrs, n_samples, reduced_dim)
    Xt_classes, v_weights_class, v_weights = pad_and_group_by_class(Xt_emb, yt)

    mus_Xs = torch.sum(u_weights_class[:, :, None] * Xs_classes, dim=1)
    mus_Xt = torch.sum(v_weights_class[:, :, None] * Xt_classes, dim=1)

    Cs = Xs_classes - mus_Xs[:, None]
    Ct = Xt_classes - mus_Xt[:, None]

    covs_Xs = torch.einsum("lnd, lnk -> ldk",
                           u_weights_class[:, :, None] * Cs, Cs)
    covs_Xt = torch.einsum("lnd, lnk -> ldk",
                           v_weights_class[:, :, None] * Ct, Ct)

    return busemann_sliced_gaussian_features_diff_labels(
        Xps, Xpt, u_weights, v_weights,
        mus_Xs, mus_Xt, covs_Xs, covs_Xt,
        num_projections, batch_w2, p=p, eps=eps
    )
