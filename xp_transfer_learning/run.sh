#!/bin/bash


transfer_learning() {
    n_epochs_wgd=$1
    n_projs=$2
    k_shot=$3
    method=$4

    python main_transfer_learning.py \
    --n_epochs_wgd $n_epochs_wgd \
    --src_dataset MNIST \
    --tgt_dataset FMNIST \
    --k_shot $k_shot \
    --ntry 3 \
    --n_epochs_nn 50  \
    --n_projs $n_projs \
    --path_data "~/torch_datasets" \
    --path_results "./results_tf_busemann/epochs_${n_epochs_wgd}" \
    --method $method
}


for method in "busemann_1d_semi_gaussian" "sliced_otdd"
do
    for n_epochs_wgd in 500 1000 10000 20000
    do
        for k_shot in 1 5 10 100 
        do
            for n_projs in 500 1000 5000 10000
            do
                echo "Running with n_epochs_wgd=$n_epochs_wgd, k_shot=$k_shot, n_projs=$n_projs"
                transfer_learning "$n_epochs_wgd" "$n_projs" "$k_shot" "$method"
            done
        done
    done
done
