import torch
import torch.nn.functional as F

from .emd1d import emd1D
from .utils_labels import pad_and_group_by_class
from .utils_projs import proj_support


def sample_zero_truncated_poisson(rate_vector):
    # From https://github.com/hainn2803/s-OTDD/blob/main/otdd/pytorch/utils.py

    rate_vector = rate_vector.to(dtype=torch.float)
    u = torch.rand_like(rate_vector) * (1 - torch.exp(-rate_vector)) + torch.exp(-rate_vector)
    t = -torch.log(u)
    return 1 + torch.poisson(rate_vector - t)


def sliced_otdd(Xs, Xt, num_projections, k=5, p=2):
    """
        Parameters
        ----------
        Xs: shape (n_distr, n, d)
        Xt: shape (m_distr, n, d)
        num_projections: int
        k: int, parameter for moment transformation
        p: int
    """
    d = Xs.shape[-1]
    device, dtype = Xs.device, Xs.dtype

    theta = torch.randn(num_projections, d, dtype=dtype, device=device)
    theta = F.normalize(theta, p=2, dim=-1)

    # Projections on R
    Xps = torch.movedim(torch.einsum("cnd,ld->cnl", Xs, theta), -1, 0)
    Xpt = torch.movedim(torch.einsum("cnd,ld->cnl", Xt, theta), -1, 0)

    # Sample matrix of shape (L, k) of moments with parameters 1,2,3,4,5
    rates = torch.arange(k).unsqueeze(0).repeat(num_projections, 1)
    lambdas = sample_zero_truncated_poisson(rates)[:, None, None].to(device)
    lambdas_factorial = torch.lgamma(lambdas+1).exp()

    # Mean over samples by class
    projs_labels_s = torch.mean(Xps.unsqueeze(-1)**lambdas / lambdas_factorial,
                                axis=2)  # .detach()
    projs_labels_t = torch.mean(Xpt.unsqueeze(-1)**lambdas / lambdas_factorial,
                                axis=2)  # .detach()

    # Aggregate projections by Hierarchical hybrid projections
    psi = torch.randn(num_projections, k+1, dtype=dtype, device=device)
    psi = F.normalize(psi, p=2, dim=-1)

    full_projs_s = psi[:, 0][:, None, None] * Xps + torch.einsum("lck, lk -> lc", projs_labels_s, psi[:,1:])[:,:,None]
    full_projs_t = psi[:, 0][:, None, None] * Xpt + torch.einsum("lck, lk -> lc", projs_labels_t, psi[:,1:])[:,:,None]

    ot_1d = emd1D(
        full_projs_s.reshape(num_projections, -1),
        full_projs_t.reshape(num_projections, -1),
        p=p
    )

    return torch.mean(ot_1d)


def sliced_otdd_labels(Xs, Xt, ys, yt, num_projections, k=5, p=2, type_proj="linear", batch_w2=100000):
    """
        Can deal with classes with different number of samples

        Parameters
        ----------
        Xs: shape (n, d)
        Xt: shape (m, d)
        ys: shape (n,)
        yt: shape (m,)
        num_projections: int
        k: int, parameter for moment transformation
        p: int
        type_proj: "linear" or "conv"
    """
    d = Xs.shape[-1]
    device, dtype = Xs.device, Xs.dtype

    projs_Xs, projs_Xt = proj_support(Xs, Xt, num_projections, type_proj) # shape (n_distr * n_samples, n_projs)

    Xps, u_weights_class, u_weights = pad_and_group_by_class(projs_Xs, ys)  # Xps shape (n_distrs, n_samples, n_projs)
    Xpt, v_weights_class, v_weights = pad_and_group_by_class(projs_Xt, yt)

    Xps = torch.movedim(Xps, -1, 0) # shape (n_projs, n_distrs, n_samples)
    Xpt = torch.movedim(Xpt, -1, 0)


    # Sample matrix of shape (n_projs, k) of moments with parameters 1,2,3,4,5
    rates = torch.arange(k).unsqueeze(0).repeat(num_projections, 1)
    lambdas = sample_zero_truncated_poisson(rates)[:, None, None].to(device)
    lambdas_factorial = torch.lgamma(lambdas+1).exp()

    # Mean over samples by class
    projs_labels_s = torch.sum(u_weights_class[None,:,:,None] * Xps.unsqueeze(-1)**lambdas / lambdas_factorial, axis=2)  # .detach()
    projs_labels_t = torch.sum(v_weights_class[None,:,:,None] * Xpt.unsqueeze(-1)**lambdas / lambdas_factorial, axis=2)  # .detach()


    # Aggregate projections by Hierarchical hybrid projections
    psi = torch.randn(num_projections, k+1, dtype=dtype, device=device)
    psi = F.normalize(psi, p=2, dim=-1)
    

    # Compute 1D Wasserstein distances
    if batch_w2 >= num_projections:
        full_projs_s = psi[:, 0][:, None, None] * Xps + torch.einsum("lck, lk -> lc", projs_labels_s, psi[:,1:])[:,:,None]
        full_projs_t = psi[:, 0][:, None, None] * Xpt + torch.einsum("lck, lk -> lc", projs_labels_t, psi[:,1:])[:,:,None]

        ot_1d = emd1D(
            full_projs_s.reshape(num_projections, -1),
            full_projs_t.reshape(num_projections, -1),
            u_weights=u_weights.reshape(-1), v_weights=v_weights.reshape(-1),
            p=p
        )
        
    else:
        n_chunks = num_projections // batch_w2 + 1
        
        L_w = []

        for n in range(n_chunks):
            n_projs = batch_w2 if n<n_chunks-1 else (num_projections%batch_w2)
            
            if n_projs > 0:
                cpt1_src = psi[n*batch_w2:n*batch_w2+n_projs,0][:,None,None] * Xps[n*batch_w2:n*batch_w2+n_projs]
                cpt2_src = torch.einsum("lck, lk -> lc", projs_labels_s[n*batch_w2:n*batch_w2+n_projs], psi[n*batch_w2:n*batch_w2+n_projs,1:])[:,:,None]
                
                cpt1_tgt = psi[n*batch_w2:n*batch_w2+n_projs,0][:,None,None] * Xpt[n*batch_w2:n*batch_w2+n_projs]
                cpt2_tgt = torch.einsum("lck, lk -> lc", projs_labels_t[n*batch_w2:n*batch_w2+n_projs], psi[n*batch_w2:n*batch_w2+n_projs,1:])[:,:,None]
                
                full_projs_s = cpt1_src + cpt2_src
                full_projs_t = cpt1_tgt + cpt2_tgt
                
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
