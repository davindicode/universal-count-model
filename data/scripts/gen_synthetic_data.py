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

dev = nprb.inference.get_device(gpu=0)



# components
class Poisson_attention_bumps(base._input_mapping):
    def __init__(
        self, neurons, mu, sigma, A, A_0, tens_type=torch.float, active_dims=None
    ):
        super().__init__(
            1, neurons, inv_link, tensor_type=tens_type, active_dims=active_dims
        )

        self.register_buffer(
            "mu", torch.tensor(mu, dtype=self.tensor_type).to(self.dummy.device)
        )
        self.register_buffer(
            "sigma", torch.tensor(sigma, dtype=self.tensor_type).to(self.dummy.device)
        )
        self.register_buffer(
            "A", torch.tensor(A, dtype=self.tensor_type).to(self.dummy.device)
        )
        self.register_buffer(
            "A_0", torch.tensor(A_0, dtype=self.tensor_type).to(self.dummy.device)
        )

    def compute_F(self, XZ):
        cov = XZ[..., self.active_dims]
        x = (cov[:, None, :, 0] - self.mu[None, :, None]) / self.sigma[None, :, None]
        return self.A[None, :, None] * torch.exp(-(x**2)) + self.A_0[None, :, None], 0


class mixture_composition(base._input_mapping):
    """
    Takes in identical base models to form a mixture model with custom functions.
    Does not support models with variational uncertainties, ignores those.
    """

    def __init__(self, input_dims, mappings, comp_func, inv_link="relu"):
        super().__init__(input_dims, mappings, inv_link)
        self.comp_func = (
            lambda x: (
                (x[0] * sample_bin) ** (1 / torch.exp(x[1]))
                - 0.5 * (1 / torch.exp(x[1]) - 1)
            )
            / sample_bin
        )

    def compute_F(self, XZ):
        """
        Note that the rates used for addition in the mixture model are evaluated as the
        quantities :math:`f(\mathbb{E}[a])`, different to :math:`\mathbb{E}[f(a)]` that is the
        posterior mean. The difference between these quantities is small when the posterior
        variance is small.

        """
        r_ = [base._inv_link[self.inv_link](m.compute_F(XZ)[0]) for m in self.mappings]
        return self.comp_func(r_), 0
    
    

# models
def CMP_bumps(sample_bin, track_samples, covariates, neurons, trials=1):
    """
    CMP with separate mu and nu parameter tuning curves
    """
    # von Mises fields
    angle_0 = np.linspace(0, 2 * np.pi, neurons + 1)[:-1]
    beta = np.random.rand(neurons) * 2.0 + 0.5
    rate_0 = np.random.rand(neurons) * 4.0 + 4.0
    w = np.stack(
        [np.log(rate_0), beta * np.cos(angle_0), beta * np.sin(angle_0)]
    ).T  # beta, phi_0 for theta modulation
    neurons = w.shape[0]

    vm_mu = vonMises_bumps(neurons)
    vm_mu.set_params(w)

    # dispersion tuning curve
    _angle_0 = np.random.permutation(
        angle_0
    )  # angle_0 + 0.4*np.random.randn(neurons)#np.random.permutation(angle_0)
    _beta = 0.6 * np.random.rand(neurons) + 0.1
    _rate_0 = np.random.rand(neurons) * 0.5 + 0.5
    w = np.stack(
        [np.log(_rate_0) + _beta, _beta * np.cos(_angle_0), _beta * np.sin(_angle_0)]
    ).T  # beta, phi_0 for theta modulation

    log_nu = vonMises_bumps(neurons)
    log_nu.set_params(w)

    # sum for mu input
    
    rate_model = mixture_composition(
        1, [log_mu, log_nu], comp_func, 
    )

    # CMP process output
    likelihood = nprb.likelihoods.COM_Poisson(
        sample_bin, neurons, "relu", dispersion_mapping=log_nu
    )

    input_group = nprb.inference.input_group()
    input_group.set_XZ(
        [covariates], track_samples, batch_size=track_samples, trials=trials
    )

    # NLL model
    glm = nprb.inference.VI_optimized(input_group, rate_model, likelihood)
    glm.validate_model(likelihood_set=False)
    return glm


def IP_bumps(sample_bin, track_samples, covariates, neurons, trials=1):
    """
    Poisson rates modulated by localized attention
    """
    # von Mises fields
    angle_0 = np.linspace(0, 2 * np.pi, neurons + 1)[:-1]
    beta = np.random.rand(neurons) * 2.6 + 0.4
    rate_0 = np.random.rand(neurons) * 4.0 + 3.0
    w = np.stack(
        [np.log(rate_0), beta * np.cos(angle_0), beta * np.sin(angle_0)]
    ).T  # beta, phi_0 for theta modulation
    neurons = w.shape[0]

    vm_rate = nprb.rate_models.vonMises_bumps(neurons)
    vm_rate.set_params(w)

    att_rate = attention_bumps(neurons, active_dims=[1])
    mu = np.random.randn(neurons)
    sigma = 0.6 * np.random.rand(neurons) + 0.6
    A = 1.7 * np.random.rand(neurons)
    A_0 = np.ones(neurons) * 0.3
    att_rate.set_params(mu, sigma, A, A_0)

    rate_model = product_model(2, [vm_rate, att_rate])

    input_group = nprb.inference.input_group()
    input_group.set_XZ(
        [covariates], track_samples, batch_size=track_samples, trials=trials
    )

    # Poisson process output
    likelihood = nprb.likelihoods.Poisson(sample_bin, neurons, "relu")
    
    # NLL model
    glm = nprb.inference.VI_optimized(input_group, rate_model, likelihood)
    glm.validate_model(likelihood_set=False)
    return glm


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
    rng = np.random.default_rng(seed)
    
    # Gaussian von Mises bump head direction model
    sample_bin = 0.1  # 100 ms
    track_samples = 10000
    trials = 1

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

    L = torch.cholesky(K)
    eps = torch.randn(Tl).double()
    v = L @ eps
    a_t = v.data.numpy()

    ### sample activity ###
    savedir = args.savedir
    
    # heteroscedastic CMP
    neurons = 50

    covariates = [hd_t[None, :, None].repeat(trials, axis=0)]
    model = hCMP_bumps(sample_bin, track_samples, covariates, neurons, trials=trials)
    model.to(dev)

    XZ, rate, _ = model.evaluate(0)
    syn_train = model.likelihood.sample(rate[0].cpu().numpy(), XZ=XZ)

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

    np.savez_compressed(savedir + "hCMP_seed{}".format(seed), spktrain=rc_t, rhd_t=rhd_t, tbin=tbin)
    torch.save({"model": model.state_dict()}, savedir + "hCMP_seed{}.pt".format(seed))

    # modulated Poisson
    neurons = 50

    covariates = [
        hd_t[None, :, None].repeat(trials, axis=0),
        a_t[None, :, None].repeat(trials, axis=0),
    ]
    model = IP_bumps(sample_bin, track_samples, covariates, neurons, trials=trials)
    model.to(dev)

    _, rate, _ = model.evaluate(0)
    syn_train = model.likelihood.sample(rate[0].cpu().numpy())

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

    np.savez_compressed(
        savedir + "IP_seed{}".format(seed), spktrain=rc_t, rhd_t=rhd_t, ra_t=ra_t, tbin=tbin
    )
    torch.save({"model": model.state_dict()}, savedir + "IP_seed{}.pt".format(seed))


if __name__ == "__main__":
    main()
