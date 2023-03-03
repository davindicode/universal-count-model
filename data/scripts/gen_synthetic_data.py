import argparse
import pickle

import sys

import numpy as np

sys.path.append("../..")  # access to library

from neuroprob import utils
from neuroprob.likelihoods.discrete import gen_CMP


# models
class _bumps:
    @staticmethod
    def HDC_bumps(theta, A, invbeta, b, theta_0):
        """
        parameters have shape (neurons,)
        :return:
            rates of shape (..., neurons, eval_pts)
        """
        return (
            A[:, None]
            * np.exp(np.cos(theta[..., None, :] - theta_0[:, None]) / invbeta[:, None])
            + b[:, None]
        )


class hCMP_bumps(_bumps):
    """
    CMP with separate mu and nu parameter tuning curves
    """

    def __init__(self, rng, neurons):
        # rate tuning curves
        self.angle_0 = np.linspace(0, 2 * np.pi, neurons + 1)[:-1]
        self.beta = rng.uniform(size=(neurons,)) * 2.0 + 0.5
        self.rate_0 = rng.uniform(size=(neurons,)) * 4.0 + 2.0
        self.b = rng.uniform(size=(neurons,)) * 0.2

        # dispersion tuning curve
        self._angle_0 = rng.permutation(self.angle_0)
        self._beta = 0.3 * rng.uniform(size=(neurons,)) + 0.1
        self._rate_0 = rng.uniform(size=(neurons,)) * 0.5 + 0.6
        self._b = rng.uniform(size=(neurons,)) * 0.1

    def __call__(self, covariates, sample_bin):
        theta = covariates[..., 0]
        mu = _bumps.HDC_bumps(theta, self.rate_0, 1 / self.beta, self.b, self.angle_0)
        nu = _bumps.HDC_bumps(
            theta, self._rate_0, 1 / self._beta, self._b, self._angle_0
        )

        lamb = np.maximum(mu * sample_bin + 0.5 * (1 - 1 / nu), 0.0) ** (
            nu
        )  # threshold >= 0 as expression is approximate
        return lamb, nu  # (..., neurons, ts)


class IP_bumps(_bumps):
    """
    Poisson rates modulated by localized attention
    """

    def __init__(self, rng, neurons):
        # angular bumps
        self.angle_0 = np.linspace(0, 2 * np.pi, neurons + 1)[:-1]
        self.beta = rng.uniform(size=(neurons,)) * 2.6 + 0.4
        self.rate_0 = rng.uniform(size=(neurons,)) * 4.0 + 3.2
        self.b = rng.uniform(size=(neurons,)) * 0.1

        # attention
        self.mu = rng.normal(size=(neurons,))
        self.sigma = 0.6 * rng.uniform(size=(neurons,)) + 0.6
        self.A = 1.7 * rng.uniform(size=(neurons,))
        self.A_0 = np.ones(neurons) * 0.34

    def __call__(self, covariates):
        theta = covariates[..., 0]
        activity = _bumps.HDC_bumps(
            theta, self.rate_0, 1 / self.beta, self.b, self.angle_0
        )

        a = covariates[..., 1]
        x = (a[..., None, :] - self.mu[:, None]) / self.sigma[:, None]
        modulator = self.A[:, None] * np.exp(-(x**2)) + self.A_0[:, None]

        return activity * modulator  # (..., neurons, ts)


# GP trajectory
def random_walk(rng, track_samples):
    x_t = np.empty(track_samples)

    x_t[0] = 0
    rn = rng.normal(size=(track_samples,))
    for k in range(1, track_samples):
        x_t[k] = x_t[k - 1] + 0.5 * rn[k]
    return x_t


def stationary_GP_trajectories(rng, Tl, l, dt, jitter=1e-8):
    """
    generate smooth GP input
    """

    def rbf_kernel(x):
        dx = x[..., None] - x[..., None, :]  # (..., T, T)
        return np.exp(-0.5 * (dx**2))

    T = np.arange(Tl) * dt / l
    K = rbf_kernel(T)
    K.reshape(-1)[:: Tl + 1] += jitter

    L = np.linalg.cholesky(K)
    v = (L @ rng.normal(size=(Tl, 1)))[..., 0]
    return v


### main ###
def main():
    ### parser ###
    parser = argparse.ArgumentParser(
        usage="%(prog)s [options]",
        description="Generate synthetic count data.",
    )
    parser.add_argument(
        "-v", "--version", action="version", version=f"{parser.prog} version 1.0.0"
    )

    parser.add_argument("--seed", default=1, type=int)
    parser.add_argument("--savedir", default="../", type=str)

    args = parser.parse_args()

    # seed
    seed = args.seed
    rng = np.random.default_rng(seed)

    ### behaviour ###
    sample_bin = 0.1  # 100 ms
    Tl = 10000

    hd_t = random_walk(rng, Tl) % (2 * np.pi)

    l = 200.0 * sample_bin
    a_t = stationary_GP_trajectories(rng, Tl, l, sample_bin)

    ### sample activity ###
    savedir = args.savedir

    # heteroscedastic CMP
    neurons = 50

    covariates = hd_t[:, None]
    model = hCMP_bumps(rng, neurons)
    lamb, nu = model(covariates, sample_bin)
    syn_train = gen_CMP(rng, lamb[None], nu[None])[0, ...]

    # ground truth tuning
    steps = 100
    gt_covariates = np.linspace(0, 2 * np.pi, steps)[:, None]
    lamb, nu = model(gt_covariates, sample_bin)

    np.savez_compressed(
        savedir + "hCMP{}".format(seed),
        spktrain=syn_train,
        hd_t=hd_t,
        tbin=sample_bin,
        gt_covariates=gt_covariates,
        gt_lamb=lamb,
        gt_nu=nu,
    )

    # modulated Poisson
    neurons = 50

    covariates = np.stack([hd_t, a_t], axis=1)
    model = IP_bumps(rng, neurons)
    rate = model(covariates)
    syn_train = rng.poisson(rate * sample_bin).astype(float)

    # ground truth tuning
    steps = 100
    gt_covariates = np.stack(
        [0.0 * np.ones(steps), np.linspace(a_t.min(), a_t.max(), steps)],
        axis=1,
    )
    gt_rate = model(gt_covariates)

    np.savez_compressed(
        savedir + "modIP{}".format(seed),
        spktrain=syn_train,
        hd_t=hd_t,
        a_t=a_t,
        tbin=sample_bin,
        gt_covariates=gt_covariates,
        gt_rate=gt_rate,
    )


if __name__ == "__main__":
    main()
