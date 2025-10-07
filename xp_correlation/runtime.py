## Code inspired from  https://github.com/hainn2803/s-OTDD/blob/main/correlation_mnist_experiment.py.

import torch
import os
import argparse
import time

import numpy as np

from utils_datasets import get_dataset, get_loaders
from utils_distances import compute_distance_datasets


parser = argparse.ArgumentParser()
parser.add_argument("--method", type=str, default="sotdd", help="Method to use")
parser.add_argument("--n_try", type=int, default=10, help="Number of tries")
parser.add_argument("--n_projs", type=int, default=100, help="Number of projections")
parser.add_argument("--batch_projs", type=int, default=100000, help="Batch size projections (for convolution projections)")
parser.add_argument("--batch_w2", type=int, default=100000, help="Batch size 1D W2")
parser.add_argument("--reduced_dim", type=int, default=2, help="Dimension embedding (for SWBG)")
parser.add_argument("--embedding", type=str, default="pca", help="Embedding to use for SWBG")
parser.add_argument("--dataset", type=str, default="MNIST", help="Dataset to use")
parser.add_argument("--path_data", type=str, default="~/torch_datasets",
                    help="Directory torch data")
parser.add_argument("--path_results", type=str, default="./results",
                    help="Directory to save the results")
args = parser.parse_args()


def main(str_dataset, path_data, device):
    dataset, transform, kwargs_sotdd = get_dataset(
        str_dataset,
        path_data,
        args.batch_w2
    )

    list_dataset_size = [5000 * (i + 1) for i in range(int(len(dataset) // 5000))]

    for dataset_size in list_dataset_size:
        L_times = []

        print(f"Dataset size: {dataset_size}")
        dataloaders, distr_imgs, distr_labels = get_loaders(
            dataset,
            np.arange(dataset_size),
            np.arange(dataset_size),
            transform,
            batch_size=dataset_size
        )

        for k in range(args.n_try):
            try:
                t0 = time.time()
                _ = compute_distance_datasets(
                    method=args.method,
                    dataloaders=dataloaders,
                    distr_imgs=distr_imgs,
                    distr_labels=distr_labels,
                    kwargs_sotdd=kwargs_sotdd,
                    n_projs=args.n_projs,
                    batch_projs=args.batch_projs,
                    batch_w2=args.batch_w2,
                    embedding=args.embedding,
                    reduced_dim=args.reduced_dim,
                    device=device
                )
                t1 = time.time() - t0
            except:
                t1 = np.inf
            L_times.append(t1)

        save_dir = args.path_results + f"/{str_dataset}_try_{args.n_try}_size_{dataset_size}"
        os.makedirs(save_dir, exist_ok=True)

        if args.method == "otdd":
            np.savetxt(f'{save_dir}/{args.method}_runtime.txt', L_times, delimiter=",")
        else:
            np.savetxt(f'{save_dir}/{args.method}_{args.n_projs}_runtime.txt', L_times, delimiter=",")


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    path_data = args.path_data
    dataset = args.dataset

    main(dataset, path_data, device)
