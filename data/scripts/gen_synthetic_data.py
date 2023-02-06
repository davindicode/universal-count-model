import argparse
import sys
import pickle

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.parameter import Parameter

sys.path.append("../../lib")  # access to library

import neuroprob as nprb
from neuroprob import base, utils
from neuroprob.likelihoods.discrete import gen_CMP

dev = nprb.inference.get_device(gpu=0)



# models
class _bumps:
    @staticmethod
    def HDC_bumps(theta, A, invbeta, b, theta_0):
        """
        parameters have shape (neurons,)
        :return:
            rates of shape (..., neurons, eval_pts)
        """
        return A[:, None] * np.exp(
            (np.cos(theta[..., None, :] - theta_0[:, None]) - 1) / invbeta[:, None]) + b[:, None]


class hCMP_bumps(_bumps):
    """
    CMP with separate mu and nu parameter tuning curves
    """
    def __init__(self, rng, sample_bin, neurons):
        # rate tuning curves
        self.angle_0 = np.linspace(0, 2 * np.pi, neurons + 1)[:-1]
        self.beta = rng.uniform(size=(neurons,)) * 2.0 + 0.5
        self.rate_0 = rng.uniform(size=(neurons,)) * 4.0 + 4.0
        self.b = rng.uniform(size=(neurons,)) * 0.1

        # dispersion tuning curve
        self._angle_0 = rng.permutation(angle_0)
        self._beta = 0.6 * rng.uniform(size=(neurons,)) + 0.1
        self._rate_0 = rng.uniform(size=(neurons,)) * 0.5 + 0.5
        self._b = rng.uniform(size=(neurons,)) * 0.1
        
        self.sample_bin = sample_bin

    def __call__(self, covariates):
        theta = covariates[..., 0]
        rate = _bumps.HDC_bumps(theta, self.rate_0, 1/self.beta, self.b, self.angle_0)
        nu = _bumps.HDC_bumps(theta, self._rate_0, 1/self._beta, self._b, self._angle_0)
        
        mu = ((rate * self.sample_bin) ** (1 / nu) - 0.5 * (1 / nu - 1)) / self.sample_bin
        return mu, nu  # (..., neurons, ts)


class IP_bumps(_bumps):
    """
    Poisson rates modulated by localized attention
    """
    def __init__(self, rng, neurons):
        # angular bumps
        self.angle_0 = np.linspace(0, 2 * np.pi, neurons + 1)[:-1]
        self.beta = rng.uniform(size=(neurons,)) * 2.6 + 0.4
        self.rate_0 = rng.uniform(size=(neurons,)) * 4.0 + 3.0

        # attention
        self.mu = rng.normal(size=(neurons,))
        self.sigma = 0.6 * rng.uniform(size=(neurons,)) + 0.6
        self.A = 1.7 * rng.uniform(size=(neurons,))
        self.A_0 = np.ones(neurons) * 0.3

    def __call__(self, covariates):
        theta = covariates[..., 0]
        activity = _bumps.HDC_bumps(theta, A, invbeta, b, theta_0)
        
        a = covariates[..., 1]
        x = (a[..., None, :] - self.mu[:, None]) / self.sigma[:, None]
        modulator = self.A[:, None] * torch.exp(-(x**2)) + self.A_0[:, None]
        
        return activity * modulator  # (..., neurons, ts)



# GP trajectory
def rbf_kernel(x):
    return np.exp(-.5 * (x**2))


def stationary_GP_trajectories(Tl, dt, trials, tau_list, eps, kernel_func, jitter=1e-9):
    """
    generate smooth GP input
    """
    tau_list_ = tau_list*trials
    out_dims = len(tau_list_)
    
    l = np.array(tau_list_)[:, None]
    v = np.ones(out_dims)

    T = np.arange(Tl)[None, :]*dt / l
    dT = T[:, None, :] - T[..., None] # (tr, T, T)
    K = kernel_func(dT)
    K.reshape(out_dims, -1)[:, ::Tl+1] += jitter
    
    L = np.linalg.cholesky(K)
    v = (L @ eps[..., None])[..., 0]
    a_t = v.reshape(trials, -1, Tl)
    return a_t # trials, tau_arr, time




### main ###
def main():
    ### parser ###
    parser = argparse.ArgumentParser(usage="%(prog)s [OPTION] [FILE]...", 
                                     description="Generate synthetic count data.")
    parser.add_argument(
        "-v", "--version", action="version", version=f"{parser.prog} version 1.0.0"
    )

    parser.add_argument("--seed", default=123, type=int)
    parser.add_argument("--savedir", default="../", type=str)

    args = parser.parse_args()
    
    # seed
    seed = args.seed
    numpy.random.seed(seed)  # for calls to numpy.random in likelihood sample()
    rng = np.random.default_rng(seed)
    
    
    ### behaviour ###
    sample_bin = 0.1  # 100 ms
    track_samples = 10000

    hd_t = np.empty(track_samples)

    hd_t[0] = 0
    rn = rng.normal(size=(track_samples,))
    for k in range(1, track_samples):
        hd_t[k] = hd_t[k - 1] + 0.5 * rn[k]

    hd_t = hd_t % (2 * np.pi)

    # GP trajectory sample
    Tl = track_samples

    l = 200.0 * sample_bin * torch.ones((1, 1))
    v = torch.ones(1)
    kernel_tuples = [("variance", v), ("SE", "euclid", l)]

    with torch.no_grad():
        kernel, _ = template.create_kernel(kernel_tuples, "softplus", torch.double)

        T = torch.arange(Tl)[None, None, :, None] * sample_bin
        K = kernel(T, T)[0, 0]
        K.view(-1)[:: Tl + 1] += 1e-6

    K = np.array(K)
    L = np.linalg.cholesky(K)
    eps = rng.normal(size=(Tl,))
    a_t = L @ eps

    
    ### sample activity ###
    savedir = args.savedir
    
    # heteroscedastic CMP
    neurons = 50

    covariates = hd_t[:, None]
    model = hCMP_bumps(rng, sample_bin, neurons)
    mu, nu = model(covariates)
    syn_train = gen_CMP(mu, nu)

    trial = 0
    bin_size = 1
    tbin, resamples, rc_t, (rhd_t,) = utils.neural.bin_data(
        bin_size,
        sample_bin,
        syn_train[trial],
        track_samples,
        (np.unwrap(hd_t),),
        average_behav=True,
        binned=True,
    )
    rhd_t = rhd_t % (2 * np.pi)

    # ground truth tuning
    steps = 100
    covariates = np.linspace(0, 2*np.pi, steps)[:, None]
    mu, nu = model(covariates)

    np.savez_compressed(
        savedir + "hCMP_seed{}".format(seed), 
        spktrain=rc_t, 
        rhd_t=rhd_t, 
        tbin=tbin, 
        covariates=covariates, 
        mu=mu, 
        nu=nu, 
    )

    # modulated Poisson
    neurons = 50

    covariates = np.stack([hd_t, a_t], axis=1)
    model = IP_bumps(rng, neurons)
    rate = model(covariates)
    syn_train = rng.poisson(rate * sample_bin)

    trial = 0
    bin_size = 1
    tbin, resamples, rc_t, (rhd_t, ra_t) = utils.neural.bin_data(
        bin_size,
        sample_bin,
        syn_train[trial],
        track_samples,
        (np.unwrap(hd_t), a_t),
        average_behav=True,
        binned=True,
    )
    rhd_t = rhd_t % (2 * np.pi)

    # ground truth tuning
    steps = 100
    covariates = np.stack(
        [0.*np.ones(steps), np.linspace(X_loc[1].min(), X_loc[1].max(), steps)], 
        axis=1, 
    )
    gt_rate = model(covariates)
    
    np.savez_compressed(
        savedir + "IP_seed{}".format(seed), 
        spktrain=rc_t, 
        rhd_t=rhd_t, 
        ra_t=ra_t, 
        tbin=tbin, 
        covariates_z=covariates, 
        gt_rate = gt_rate, 
    )


if __name__ == "__main__":
    main()
