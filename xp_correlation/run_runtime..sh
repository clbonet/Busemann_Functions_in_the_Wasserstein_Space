#!/bin/bash

runtime() {
	method=$1
    n_projs=$2

    python runtime.py \
    --method $method \
    --n_projs $n_projs \
    --path_data "$WORK/torch_datasets" \
    --path_results "$WORK/runtime_corr_MNIST" \
    --dataset MNIST \
    --n_try 100
}


runtime "otdd" 100

for n_projs in 1000
do
  	for method in "sotdd" "swb1dg" "swbg" "my_sotdd"
        do
          	runtime $method $n_projs
        done
done

