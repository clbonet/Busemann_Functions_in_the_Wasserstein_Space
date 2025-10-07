## Code inspired from  https://github.com/hainn2803/s-OTDD/blob/main/correlation_mnist_experiment.py.

import torch
import os
import argparse
import gc

import numpy as np

from utils_datasets import get_dataset, get_loaders
from utils_distances import compute_distance_datasets


parser = argparse.ArgumentParser()
parser.add_argument("--method", type=str, default="sotdd", help="Method to use")
parser.add_argument("--seed", type=int, default=42, help="seed")
parser.add_argument("--n_datasets", type=int, default=10, help="Number of datasets")
parser.add_argument("--n_projs", type=int, default=100, help="Number of projections")
parser.add_argument("--batch_projs", type=int, default=1000, help="Batch size projections (for convolution projections)")
parser.add_argument("--batch_w2", type=int, default=100000, help="Batch size 1D W2")
parser.add_argument("--reduced_dim", type=int, default=2, help="Dimension embedding (for SWBG)")
parser.add_argument("--embedding", type=str, default="pca", help="Embedding to use for SWBG")
parser.add_argument("--dataset", type=str, default="MNIST", help="Dataset to use")
parser.add_argument("--path_data", type=str, default="~/torch_datasets",
                    help="Directory torch data")
parser.add_argument("--path_results", type=str, default="./results",
                    help="Directory to save the results")
args = parser.parse_args()


def main(rng, str_dataset, path_data, device):
    dataset, transform, kwargs_sotdd = get_dataset(str_dataset, path_data)
        
    indices = np.arange(len(dataset))
    max_dataset_size = len(dataset) // 2
    print(f"Maximum number of datapoint for each dataset: {max_dataset_size}")
    
    list_dataset_size = rng.integers(low=5, high=11, size=args.n_datasets) * 1000
    
    for idx in range(len(list_dataset_size)):
        dataset_size = list_dataset_size[idx]
        
        save_dir = args.path_results + f"/{str_dataset}_seed_{idx}_size_{dataset_size}"
        os.makedirs(save_dir, exist_ok=True)
    
        shuffled_indices = rng.permutation(indices)

        print(f"Setting dataset to size of {dataset_size}..")
        idx1 = shuffled_indices[:dataset_size]
        idx2 = shuffled_indices[-dataset_size:]
    
        print(f"len(idx1): {len(idx1)}, len(idx2): {len(idx2)}")
        dataloaders, distr_imgs, distr_labels = get_loaders(dataset, idx1, idx2, transform)
        
        pairwise_dist = compute_distance_datasets(
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
        
        if args.method == "otdd":
            torch.save(pairwise_dist, f'{save_dir}/{args.method}_dist.pt')
        else:
            torch.save(pairwise_dist, f'{save_dir}/{args.method}_{args.n_projs}_dist.pt')
            
        # Cleanup to prevent memory leaks
        del dataloaders, distr_imgs, distr_labels, pairwise_dist
        gc.collect()
        torch.cuda.empty_cache()
        

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    path_data=args.path_data
    dataset = args.dataset

    rng = np.random.default_rng(args.seed)
    main(rng, dataset, path_data, device)
