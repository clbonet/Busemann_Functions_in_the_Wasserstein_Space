import numpy as np
import matplotlib.pyplot as plt


def _generate_data(seed=42):
    """
        From https://github.com/pierreglaser/kale-flow
    """
    N, r, _delta = 80, 0.3, 0.5
    
    X = np.c_[r * np.cos(np.linspace(0, 2 * np.pi, N + 1)), r * np.sin(np.linspace(0, 2 * np.pi, N + 1))][:-1]  # noqa
    for i in [1, 2]:
        X = np.r_[X, X[:N, :]-i*np.array([0, (2 + _delta) * r])]

    rs = np.random.RandomState(seed)
    # Y = rs.randn(N*(2+1), 2) / 100 - np.array([r/np.sqrt(2), r/np.sqrt(2)])
    Y = rs.randn(N*(2+1), 2) / 100 - np.array([0, r])

    return X, Y


def plot_rings(L, X, title=None):
    f, axs = plt.subplots(nrows=2, ncols=4, figsize=(15, 5))

    m = L[0].shape[0]
    
    # for i, k in enumerate([0, 20, 250, 500]):
    for i, k in enumerate([0, 2, 25, 50]):
        axs[0, i].scatter(X[:,0],X[:,1],label="Target")
        for j in range(m):
            axs[0, i].scatter(L[k][j,:,0],L[k][j,:,1])
        axs[0, i].set_title("Iter "+str(k))
        axs[0, i].axis("off")
        axs[0, i].set_xlim([-1.9, 0.4])
        axs[0, i].set_ylim([-0.8, 0.8])
        
    # for i, k in enumerate([1000, 200, 2500, 5000]):
    for i, k in enumerate([100, 200, 250, 500]):
        axs[1, i].scatter(X[:,0],X[:,1],label="Target")
        for j in range(m):
            axs[1, i].scatter(L[k][j,:,0],L[k][j,:,1])
        axs[1, i].set_title("Iter "+str(k))
        axs[1, i].axis("off")
        axs[1, i].set_xlim([-1.9, 0.4])
        axs[1, i].set_ylim([-0.8, 0.8])
        
    axs[0, 0].legend()

    if title is not None:
        plt.suptitle(title)
    plt.show()