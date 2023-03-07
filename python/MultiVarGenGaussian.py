# Copyright 2023 wngfra.
# SPDX-License-Identifier: Apache-2.0
import numpy as np
from scipy.stats import special_ortho_group


def random_scatter(m=1, p=2):
    M = np.random.rand(p, p)
    M = 0.5 * (M + M.T)
    return m * M / M.trace()


class MultiVarGenGaussian:
    """Multivariate Generalized Gaussian Distribution class."""

    def __init__(self, mu, Sigma, beta, p=2):
        self._p = p
        self._mu = mu
        self._beta = beta
        self.Sigma = Sigma

    @property
    def dim(self):
        return self._p

    @dim.setter
    def dim(self, val):
        self._p = val

    @property
    def mu(self):
        return self._mu

    @mu.setter
    def mu(self, val):
        self._mu = val

    @property
    def Sigma(self):
        return self._Sigma

    @Sigma.setter
    def Sigma(self, val):
        if val:
            self._Sigma = val
        else:
            self._Sigma = random_scatter(1, self._p)

    @property
    def beta(self):
        return self._beta

    @beta.setter
    def beta(self, val):
        self._beta = val

    def sample(self, n_samples=1000, noise_level=0):
        """Generative Model with Stochastic representation: $\mathbf{x} = \tau \Sigma^{1/2} \mathbf{u}$.
        1. $\mathbf{u} \in \mathbb{R}^2$ is a random vector sampled from a unit circle
        2. Scatter matrix $\Sigma = m \mathbf{M}$
        3. Shape parameter $\beta$
        4. $\tau^{2 \beta} \sim \Gamma(\frac{p}{2 \beta}, 2)$

        Args:
            n_samples (int, optional): _description_. Defaults to 1000.
            noise_level (int, optional): _description_. Defaults to 0.

        Returns:
            _type_: _description_
        """
        phi = 2 * np.pi * np.random.random(n_samples)
        u = np.array([np.cos(phi), np.sin(phi)]) + self._mu[:, np.newaxis]
        tau = np.power(
            np.random.gamma(0.5 * self._p / self._beta, 2, n_samples), 0.5 / self._beta
        )
        x = tau * np.matmul(np.sqrt(self._Sigma), u)
        R = special_ortho_group.rvs(self._p)
        x = np.matmul(R, x)
        if noise_level > 0:
            mu = self._mu[:, np.newaxis]
            sigma = np.std(x, axis=1)[:, np.newaxis]
            x += sigma * np.random.randn(self._p, x.shape[1]) + mu

        return x

    def pdf(self, X):
        d = X.shape[1]
        diff = X - self._mu
        exponent = -0.5 * np.sum(
            np.power(np.dot(diff, np.linalg.inv(self._Sigma)), 2), axis=1
        )
        exponent = np.power(1 + exponent / self._beta, -(self._beta + d) / 2)
        return np.prod(exponent)

    def log_likelihood(self, X):
        return np.sum(np.log(self.pdf(X)))


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    mvgg = MultiVarGenGaussian(mu=np.random.rand(2) * 10 - 5, Sigma=None, beta=0.5, p=2)
    xs = mvgg.sample(2000, 0.25)

    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    ax.set_aspect(1)
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    dots = ax.scatter(xs[0, :], xs[1, :])
    ax.set_xlim([xs[0, :].min(), xs[0, :].max()])
    ax.set_ylim([xs[1, :].min(), xs[1, :].max()])

    plt.suptitle("MGGD Generative Model")
    plt.show()
