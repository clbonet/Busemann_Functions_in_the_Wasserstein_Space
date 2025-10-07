import torch
import sys

sys.path.append("../")
from lib_torch.otdd.distance import DatasetDistance
from lib_torch.otdd.sotdd import compute_pairwise_distance
from lib_torch.sliced_otdd import sliced_otdd_labels
from lib_torch.sliced_busemann_semi_gaussian_1d import busemann_sliced_semi_gaussian_1d_diff_labels
from lib_torch.sliced_busemann_gaussian import busemann_sliced_gaussian_approx_diff_labels


def compute_distance_datasets(method, dataloaders, distr_imgs, distr_labels,
                              kwargs_sotdd, n_projs=100, batch_projs=1000,
                              batch_w2=10000, embedding=None, reduced_dim=2,
                              device="cpu"):
    """
    Parameters
    ----------
    - method: str among ["otdd", "sotdd", "swb1d", "swbg"]
    - dataloaders: list of dataloaders
    - distr_imgs: list of full datasets
    - distr_labels: list of labels
    - kwargs_sotdd: dict, parameters for s-OTDD
    - n_projs: int, number of projections for sliced methods
    - batch_projs: int, batch size for convolutions projections
    - batch_w2: int, batch size to compute 1D wasserstein distances
    - embedding: std, either "pca" or "tsne", used for SWBG
    - reduced_dim: int, size of embedding space for SWBG

    Returns
    -------
    - pairwise_dist: tensor of shape (len(dataloaders), len(dataloaders))
    """

    if method == "otdd":
        # OTDD
        pairwise_dist = torch.zeros(len(dataloaders), len(dataloaders))

        for i in range(len(dataloaders)):
            for j in range(i+1, len(dataloaders)):
                dist = DatasetDistance(
                    dataloaders[i],
                    dataloaders[j],
                    inner_ot_method='exact',
                    debiased_loss=True,
                    p=2,
                    entreg=1e-3,
                    device=device,
                    verbose=False
                )
                d = dist.distance(maxsamples=None).item()
                pairwise_dist[i][j] = d
                pairwise_dist[j][i] = d

    elif method == "sotdd":
        # s-OTDD
        pairwise_dist = torch.zeros(len(dataloaders), len(dataloaders))

        list_pairwise_dist = compute_pairwise_distance(
            list_D=dataloaders,
            num_projections=n_projs,
            device=device,
            evaluate_time=True,
            **kwargs_sotdd
        )

        t = 0
        for i in range(len(dataloaders)):
            for j in range(i+1, len(dataloaders)):
                pairwise_dist[i, j] = list_pairwise_dist[t]
                pairwise_dist[j, i] = list_pairwise_dist[t]
                t += 1

    elif method == "my_sotdd":
        # s-OTDD v2
        pairwise_dist = torch.zeros(len(dataloaders), len(dataloaders))

        t = 0
        for i in range(len(dataloaders)):
            for j in range(i+1, len(dataloaders)):
                if kwargs_sotdd["num_channels"] == 1:
                    xi = distr_imgs[i].type(torch.float).reshape(len(distr_imgs[i]), kwargs_sotdd["dimension"]).to(device)
                    xj = distr_imgs[j].type(torch.float).reshape(len(distr_imgs[j]), kwargs_sotdd["dimension"]).to(device)
                    type_proj = "linear"
                else:
                    xi = distr_imgs[i].type(torch.float).to(device)
                    xj = distr_imgs[j].type(torch.float).to(device)
                    type_proj = "conv"

                sotdd = sliced_otdd_labels(
                    xi,
                    xj,
                    distr_labels[i].to(device),
                    distr_labels[j].to(device),
                    num_projections=n_projs,
                    type_proj=type_proj,
                    batch_w2=batch_w2
                )

                pairwise_dist[i, j] = sotdd.item()**(1/2)
                pairwise_dist[j, i] = sotdd.item()**(1/2)
                t += 1

    elif method == "swb1dg":
        # SW with Busemann projs (in 1D + Gaussian \mu_1)
        pairwise_dist = torch.zeros(len(dataloaders), len(dataloaders))

        t = 0
        for i in range(len(dataloaders)):
            for j in range(i+1, len(dataloaders)):
                if kwargs_sotdd["num_channels"] == 1:
                    xi = distr_imgs[i].type(torch.float).reshape(len(distr_imgs[i]), kwargs_sotdd["dimension"]).to(device)
                    xj = distr_imgs[j].type(torch.float).reshape(len(distr_imgs[j]), kwargs_sotdd["dimension"]).to(device)
                    type_proj = "linear"
                else:
                    xi = distr_imgs[i].type(torch.float).to(device)
                    xj = distr_imgs[j].type(torch.float).to(device)
                    type_proj = "conv"

                sotdd = busemann_sliced_semi_gaussian_1d_diff_labels(
                    xi,
                    xj,
                    distr_labels[i].to(device),
                    distr_labels[j].to(device),
                    num_projections=n_projs,
                    type_proj=type_proj,
                    batch_projs=batch_projs,
                    batch_w2=batch_w2
                )

                pairwise_dist[i, j] = sotdd.item()**(1/2)
                pairwise_dist[j, i] = sotdd.item()**(1/2)
                t += 1

    elif method == "swbg":
        # SW with Busemann projs (in 1D + Gaussian \mu_1)
        pairwise_dist = torch.zeros(len(dataloaders), len(dataloaders))

        t = 0
        for i in range(len(dataloaders)):
            for j in range(i+1, len(dataloaders)):
                if kwargs_sotdd["num_channels"] == 1:
                    xi = distr_imgs[i].type(torch.float).reshape(len(distr_imgs[i]), kwargs_sotdd["dimension"]).to(device)
                    xj = distr_imgs[j].type(torch.float).reshape(len(distr_imgs[j]), kwargs_sotdd["dimension"]).to(device)
                    type_proj = "linear"
                else:
                    xi = distr_imgs[i].type(torch.float).to(device)
                    xj = distr_imgs[j].type(torch.float).to(device)
                    type_proj = "conv"

                sotdd = busemann_sliced_gaussian_approx_diff_labels(
                    xi,
                    xj,
                    distr_labels[i].to(device),
                    distr_labels[j].to(device),
                    num_projections=n_projs,
                    type_proj=type_proj,
                    batch_projs=batch_projs,
                    batch_w2=batch_w2,
                    embedding=embedding,
                    reduced_dim=reduced_dim
                )

                pairwise_dist[i, j] = sotdd**(1/2)
                pairwise_dist[j, i] = sotdd**(1/2)
                t += 1

    else:
        print(method + " not implemented.")

    return pairwise_dist
