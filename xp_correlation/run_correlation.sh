#!/bin/bash

correlation() {
    method=$1
    n_projs=$2
    dataset=$3

    python main_correlation_subdataset.py \
    --method  $method \
    --n_projs $n_projs \
    --n_datasets 2 \
    --path_data "~/torch_datasets" \
    --path_results "./results" \
    --dataset $dataset \
    --embedding "tsne" \
    --reduced_dim 10
}

for method in "sotdd" "swb1dg" "swbg"
do
    for n_projs in 10 50 100 500 1000 5000 10000
    do
        for dataset in "MNIST" "CIFAR10"
        do
            correlation $method $n_projs $dataset
        done
    done
done

correlation "otdd" 100 MNIST
correlation "otdd" 100 CIFAR10
