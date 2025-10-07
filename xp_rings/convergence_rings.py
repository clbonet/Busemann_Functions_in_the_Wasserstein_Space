import argparse
import ot
import sys
import jax

import numpy as np
import jax.numpy as jnp

sys.path.append("../")
from lib_jax.sliced_busemann_1d_jax import busemann_sliced_1d
from lib_jax.sliced_otdd_jax import sliced_otdd
from lib_jax.sliced_busemann_gaussian_jax import busemann_sliced_gaussian_approx
from lib_jax.mmd import target_value_and_grad_riesz
from lib_jax.gd import wasserstein_gradient_descent_jit


parser = argparse.ArgumentParser()
parser.add_argument("--n_try", type=int, default=10,
                    help="Number of averaging")
parser.add_argument("--distance", type=str, default="sotdd",
                    help="Distance")
parser.add_argument("--seed", type=int, default=42,
                    help="Seed")
args = parser.parse_args()


def _generate_data(seed=42):
    """
        Modified from https://github.com/pierreglaser/kale-flow
    """
    N, r, _delta = 80, 0.3, 0.5

    rs = np.random.RandomState(seed)
    thetas = 2*np.pi*rs.rand(N+1)
    X = np.c_[r * np.cos(thetas), r * np.sin(thetas)][:-1]  # noqa
    for i in [1, 2]:
        X = np.r_[X, X[:N, :]-i*np.array([0, (2 + _delta) * r])]

    rs = np.random.RandomState(42)
    # Y = rs.randn(N*(2+1), 2) / 100 - np.array([r/np.sqrt(2), r/np.sqrt(2)])
    Y = rs.randn(N*(2+1), 2) / 100 - np.array([0, r])

    return X, Y



def get_data(seed):
    X, Y = _generate_data(seed)
    X = np.concatenate([X[:, 1][:, None], X[:, 0][:, None]], axis=-1)
    Y = np.concatenate([Y[:, 1][:, None], Y[:, 0][:, None]], axis=-1)

    X_data = np.zeros((3, 80, 2))
    X_data[0] = X[:80]
    X_data[1] = X[80:160]
    X_data[2] = X[160:]

    x0 = np.zeros((3, 80, 2))
    x0[0] = Y[:80]
    x0[1] = Y[80:160]
    x0[2] = Y[160:]

    return x0, X_data


def compute_wow(x, X_data):
    M = np.zeros((3, 3))

    for i in range(3):
        for j in range(3):
            a = jnp.ones(80) / 80
            b = jnp.ones(80) / 80
            C = ot.dist(x[i], X_data[j])
            w2 = ot.emd2(a, b, C)
            M[i, j] = w2

    M = jnp.array(M)
    a = jnp.ones(3) / 3
    b = jnp.ones(3) / 3
    return ot.emd2(a, b, M)


def wow_gradient_flows(rng, dist, x0, X_data):

    master_key, key = jax.random.split(rng, num=2)

    if dist == "swb1dg":
        def target_value_and_grad(x, y, rng, n_projs=500):
            n, n_samples, _ = x.shape

            def tgt_func(z):
                return busemann_sliced_1d(rng, z, y, n_projs)

            l, grad = jax.value_and_grad(tgt_func)(x)
            return l, n * n_samples * grad

        n_epochs = 1001
        L_loss, L_particles = wasserstein_gradient_descent_jit(
            x0, X_data, jax.jit(target_value_and_grad),
            key, n_epochs=n_epochs, lr=1)

    elif dist == "sotdd":
        def target_value_and_grad(x, y, rng, n_projs=500):
            n, n_samples, _ = x.shape

            def tgt_func(z):
                return sliced_otdd(rng, z, y, n_projs)

            l, grad = jax.value_and_grad(tgt_func)(x)
            return l, n * n_samples * grad

        n_epochs = 1001
        L_loss, L_particles = wasserstein_gradient_descent_jit(
            x0, X_data, jax.jit(target_value_and_grad),
            key, n_epochs=n_epochs, lr=1)

    elif dist == "swbg":
        def target_value_and_grad(x, y, rng, n_projs=500):
            n, n_samples, _ = x.shape

            def tgt_func(z):
                return busemann_sliced_gaussian_approx(z, y, rng, n_projs)
            l, grad = jax.value_and_grad(tgt_func)(x)
            return l, n * n_samples * grad

        n_epochs = 1001
        L_loss, L_particles = wasserstein_gradient_descent_jit(
            x0, X_data, jax.jit(target_value_and_grad),
            key, n_epochs=n_epochs, lr=1)

    elif dist == "mmdsw":
        def target_value_and_grad(x, y, rng, n_projs=500):
            def tgt_value_and_grad(x, y, key):
                return target_value_and_grad_riesz(x, y, key, r=1,
                                                   n_projs=n_projs)
            target_grad = jax.jit(tgt_value_and_grad)

            l, grad = target_grad(x, y, rng)
            return l, grad

        n_epochs = 5001
        L_loss, L_particles = wasserstein_gradient_descent_jit(
            x0, X_data, jax.jit(target_value_and_grad),
            key, n_epochs=n_epochs, lr=0.1)

    L_wow = []
    for e in range(n_epochs):
        wow = compute_wow(L_particles[e], jnp.array(X_data))
        L_wow.append(wow)

    return L_wow


if __name__ == "__main__":
    rng = np.random.default_rng(args.seed)
    seeds = rng.integers(low=0, high=10000, size=args.n_try)

    rng_jax = jax.random.PRNGKey(0)

    L_wow = []

    for k in range(args.n_try):
        x0, X_data = get_data(seeds[k])
        L = wow_gradient_flows(rng_jax, args.distance, jnp.array(x0), X_data)
        L_wow.append(L)

    np.savetxt(f"./results/convergence_{args.distance}.csv", np.array(L_wow),
               delimiter=",")
