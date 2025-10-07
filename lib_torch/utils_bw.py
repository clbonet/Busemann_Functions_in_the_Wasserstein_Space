import torch
import numpy as np

from tqdm import trange

from .utils_sqrtm import sqrtm


def dots(*args):
    r""" dots function for multiple matrix multiply """
    # return reduce(torch.matmul, args)

    if len(args) == 2:
        return torch.matmul(args[0], args[1])
    elif len(args) == 3:
        return torch.matmul(args[0], torch.matmul(args[1], args[2]))


def trace(C):
    return C.diagonal(offset=0, dim1=-1, dim2=-2).sum(-1)


def exp_bw(Sigma, S):
    """
        Exponential map Bures-Wasserstein space as Sigma: \exp_\Sigma(S)
    """
    d = S.shape[-1]
    Id = torch.eye(d, device=S.device)
    C = Id + S

    return dots(C, Sigma, C)


def logMapBW(m, s, m0, s0):
    d = len(m0)
    s012 = sqrtm(s0)
    s0_12 = torch.linalg.inv(s012)
    M = sqrtm(dots(s012, s, s012))
    T = dots(s0_12, M, s0_12)
    logMap_Sigma = T - torch.eye(d, device=T.device)
    return m - m0, logMap_Sigma


def proj_bw(mX, sX, m0, Sigma0):
    """
        Project X on {Sigma, BW_2^2(Sigma,Sigma0)=1}
    """
    X_Sigma0_12 = sqrtm(sX @ Sigma0)
    Sigma0_X_12 = sqrtm(Sigma0 @ sX)

    w = torch.sqrt(torch.trace(sX) + torch.trace(Sigma0)
                   - 2 * torch.trace(X_Sigma0_12)
                   + torch.linalg.norm(m0-mX)**2)

    gamma = w - 1

    m = (mX + gamma * m0) / (1+gamma)
    S = 1 / (1+gamma)**2 * sX + (gamma / (1+gamma))**2 * Sigma0 \
        + gamma / (1+gamma)**2 * (X_Sigma0_12 + Sigma0_X_12)
    return m, S


def bures_wasserstein_batch(m0, m1, s0, s1):
    """
        Parameters
        ----------
        m0: shape (n, d)
        m1: shape (m, d)
        s0: shape (n, d, d)
        s1: shape (m, d, d)

        Output:
        BW distance: shape (n, m)
    """
    n, d = m0.shape
    m = m1.shape[0]

    dist_m = torch.linalg.norm(m0[:, None]-m1[None], axis=-1)**2

    s12 = sqrtm(s0)
    C12 = sqrtm(torch.einsum("nij, mjk, nkl -> nmil", s12, s1, s12)
                .reshape(-1, d, d)).reshape(n, m, d, d)
    dist_b = trace(s0)[:, None] + trace(s1)[None] - 2 * trace(C12)

    output = torch.sqrt(dist_m + dist_b)
    return torch.nan_to_num(output, 0)


def get_off_diagonal_elements(M):
    return M[~torch.eye(*M.shape, dtype=torch.bool)].view(M.shape[0],
                                                          M.shape[1]-1)


def bures_wasserstein_barycenter(means, sigmas, weights=None, num_iter=1000,
                                 eps=1e-7, log=False):
    """
        Fixed point iteration taken algorithm taken from
        https://github.com/PythonOT/POT/blob/master/ot/gaussian.py
    """

    if weights is None:
        weights = torch.ones(sigmas.shape[0],
                             device=sigmas.device) / sigmas.shape[0]

    # Compute the mean barycenter
    mb = torch.sum(means * weights[:, None], axis=0)

    # Init the covariance barycenter
    Cb = torch.mean(sigmas * weights[:, None, None], axis=0)

    for it in range(num_iter):
        # fixed point update
        Cb12 = sqrtm(Cb)

        Cnew = sqrtm(Cb12 @ sigmas @ Cb12)
        Cnew *= weights[:, None, None]
        Cnew = torch.sum(Cnew, axis=0)

        # check convergence
        diff = torch.linalg.norm(Cb - Cnew)
        if diff <= eps:
            break
        Cb = Cnew

    if log:
        log = {}
        log['num_iter'] = it
        log['final_diff'] = diff
        return mb, Cb, log
    else:
        return mb, Cb


def bw_gd_barycenter(means, sigmas, num_iter=1000, lr=1, weights=None, log=False, bar=True, batch_size=None):
    device = means.device

    if weights is None:
        weights = torch.ones(sigmas.shape[0], device=device) / sigmas.shape[0]

    mb = torch.sum(means * weights[:, None], axis=0)

    sk = torch.sum(sigmas * weights[:, None, None], axis=0)
    Id = torch.eye(sigmas.shape[-1], device=device)

    L_loss = []

    if bar:
        pbar = trange(num_iter)
    else:
        pbar = range(num_iter)

    for e in pbar:
        sk12 = sqrtm(sk)
        sk_12 = torch.linalg.inv(sk12)

        if batch_size is not None:
            inds = np.random.choice(len(sigmas), batch_size,
                                    replace=True, p=weights.cpu().numpy())
            M = sqrtm(dots(sk12, sigmas[inds], sk12))
            grad_bw = Id - torch.mean(dots(sk_12, M, sk_12), axis=0)
        else:
            M = sqrtm(dots(sk12, sigmas, sk12))
            grad_bw = Id - torch.sum(dots(sk_12, M, sk_12) * weights[:, None, None], axis=0)

        sk = exp_bw(sk, - lr * grad_bw)

        if log:
            loss = bures_wasserstein_batch(mb[None], means, sk[None], sigmas)
            L_loss.append(torch.mean(loss).item())
            pbar.set_postfix_str(f"loss = {L_loss[-1]:.3f}")

    if log:
        return mb, sk, L_loss
    return mb, sk
