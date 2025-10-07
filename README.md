# Busemann Functions in the Wasserstein Space: Existence, Closed-Forms, and Applications to Slicing

This repository contains the code to replicate the experiments of the paper [Busemann Functions in the Wasserstein Space: Existence, Closed-Forms, and Applications to Slicing](https://arxiv.org/abs/2510.04579). In this paper, we study along which geodesics the Busemann function is defined on the Wasserstein space, and how to compute it in practice. Then, as it allows to project distributions on $\mathbb{R}$, we leverage it to define Sliced-Wasserstein distances on $\mathcal{P}_2\big(\mathbb{R}^d\times\mathcal{P}_2(\mathbb{R}^d)\big)$, $\mathcal{P}_2\big(\mathcal{P}_2(\mathbb{R}^d)\big)$ and $\mathcal{P}_2\big(\mathrm{BW}(\mathbb{R}^d)\big)$. We use these distances to compare and flow labeled datasets and Gaussian mixtures.

## Abstract

The Busemann function has recently found much interest in a variety of geometric machine learning problems, as it naturally defines projections onto geodesic rays of Riemannian manifolds and generalizes the notion of hyperplanes. As several sources of data can be conveniently modeled as probability distributions, it is natural to study this function in the Wasserstein space, which carries a rich formal Riemannian structure induced by Optimal Transport metrics. In this work, we investigate the existence and computation of Busemann functions in Wasserstein space, which admits geodesic rays. We establish closed-form expressions in two important cases: one-dimensional distributions and Gaussian measures. These results enable explicit projection schemes for probability distributions on $\mathbb{R}$, which in turn allow us to define novel Sliced-Wasserstein distances over Gaussian mixtures and labeled datasets. We demonstrate the efficiency of those original schemes on synthetic datasets as well as transfer learning problems.


## Citation

```
@article{bonet2025busemann,
  title={{Busemann Functions in the Wasserstein Space: Existence, Closed-Forms, and Applications to Slicing}},
  author={Bonet, Clément and Cazelles, Elsa and Drumetz, Lucas and Courty, Nicolas},
  journal={arXiv preprint arXiv:2510.04579},
  year={2025}
}
```

## Experiments
- Figure 1, Table 1 and figures of Appendix E.2 can be reproduced by using the code in the folder `xp_correlation`. The experiments can be reproduced by first running `./run_correlation.sh` and then the notebook `results_correlations_subdatasets.ipynb`.
- Figure 2 and 3 can be reproduced by running the code in the folder `xp_rings`.
- The Transfer Learning experiment (Table 2) is available in the folder `xp_transfer_learning`.
- The Gaussian mixture experiments (Appendix F) can be bound in `xp_gaussian_mixtures`.

## Credits
- The code of OTDD and SOTDD are taken from the repository [s-OTDD](https://github.com/hainn2803/s-OTDD), which also contains code from [otdd](https://github.com/microsoft/otdd/tree/main).
- For the Transfer Learning experiment, code was taken from the repository [Flowing_Datasets_with_WoW_Gradient_Flows](https://github.com/clbonet/Flowing_Datasets_with_WoW_Gradient_Flows).
- For Gaussian Mixture experiments, code were borrowed from the repository [sliced_OT_for_GMMs](https://github.com/MoePien/sliced_OT_for_GMMs).
- For the flow of Gaussian mixtures, we took some code of an example of the [POT](https://pythonot.github.io/auto_examples/gaussian_gmm/plot_GMM_flow.html#sphx-glr-auto-examples-gaussian-gmm-plot-gmm-flow-py) library.