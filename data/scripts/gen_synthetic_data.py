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



sys.path.append("../../scripts")

import template



### mixture rate models ###
class _mappings(base._input_mapping):
    """ """

    def __init__(self, input_dims, mappings):
        """
        Additive fields, so exponential inverse link function.
        All models should have same the input and output structure.

        :params list models: list of base models, each initialized separately
        """
        self.maps = len(mappings)
        if self.maps < 2:
            raise ValueError("Need to have more than one component mapping")

        # covar_type = None # intially no uncertainty in model
        for m in range(len(mappings)):  # consistency check
            if m < len(mappings) - 1:  # consistency check
                if mappings[m].out_dims != mappings[m + 1].out_dims:
                    raise ValueError("Mappings do not match in output dimensions")
                if mappings[m].tensor_type != mappings[m + 1].tensor_type:
                    raise ValueError("Tensor types of mappings do not match")

        super().__init__(input_dims, mappings[0].out_dims, mappings[0].tensor_type)

        self.mappings = nn.ModuleList(mappings)

    def constrain(self):
        for m in self.mappings:
            m.constrain()

    def KL_prior(self):
        KL_prior = 0
        for m in self.mappings:
            KL_prior = KL_prior + m.KL_prior()
        return KL_prior


class product_model(_mappings):
    """
    Takes in identical base models to form a product model.
    """

    def __init__(self, input_dims, mappings):
        super().__init__(input_dims, mappings)

    def compute_F(self, XZ):
        """
        Note that the rates used for multiplication in the product model are evaluated as the
        quantities :math:`f(\mathbb{E}[a])`, different to :math:`\mathbb{E}[f(a)]` that is the
        posterior mean. The difference between these quantities is small when the posterior
        variance is small.

        The exact method would be to use MC sampling.

        :param torch.Tensor cov: input covariates of shape (sample, time, dim)
        """
        rate_ = []
        var_ = []
        for m in self.mappings:
            F_mu, F_var = m.compute_F(XZ)
            rate_.append(m.f(F_mu))
            if isinstance(F_var, numbers.Number):
                var_.append(0)
                continue
            var_.append(
                base._inv_link_deriv[self.inv_link](F_mu) ** 2 * F_var
            )  # delta method

        tot_var = 0
        rate_ = torch.stack(rate_, dim=0)
        for m, var in enumerate(var_):
            ind = torch.tensor([i for i in range(rate_.shape[0]) if i != m])
            if isinstance(var, numbers.Number) is False:
                tot_var = tot_var + (rate_[ind]).prod(dim=0) ** 2 * var

        return rate_.prod(dim=0), tot_var


class mixture_composition(_mappings):
    """
    Takes in identical base models to form a mixture model with custom functions.
    Does not support models with variational uncertainties, ignores those.
    """

    def __init__(self, input_dims, mappings, comp_func, inv_link="relu"):
        super().__init__(input_dims, mappings, inv_link)
        self.comp_func = comp_func

    def compute_F(self, XZ):
        """
        Note that the rates used for addition in the mixture model are evaluated as the
        quantities :math:`f(\mathbb{E}[a])`, different to :math:`\mathbb{E}[f(a)]` that is the
        posterior mean. The difference between these quantities is small when the posterior
        variance is small.

        """
        r_ = [base._inv_link[self.inv_link](m.compute_F(XZ)[0]) for m in self.mappings]
        return self.comp_func(r_), 0

    
    
class vonMises_bumps(nprb.mappings.custom_wrapper):
    """
    Angular (head direction/theta) variable GLM
    """

    def __init__(self, neurons, tensor_type=torch.float, active_dims=None):
        super().__init__(
            1, neurons, tensor_type=tensor_type, active_dims=active_dims
        )

    def set_params(self, w):
        self.w.data = w.type(self.tensor_type).to(self.dummy.device)

    def compute_F(self, XZ):
        XZ = self._XZ(XZ)
        theta = XZ[..., 0]
        g = torch.stack(
            (
                torch.ones_like(theta, device=theta.device),
                torch.cos(theta),
                torch.sin(theta),
            ),
            dim=-1,
        )
        return (g * self.w[None, :, None, :]).sum(-1), 0


class attention_bumps(nprb.mappings.custom_wrapper):
    def __init__(
        self, neurons, tens_type=torch.float, active_dims=None
    ):
        super().__init__(
            1, neurons, inv_link, tensor_type=tens_type, active_dims=active_dims
        )

    def set_params(self, mu, sigma, A, A_0):
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
    comp_func = (
        lambda x: (
            (x[0] * sample_bin) ** (1 / torch.exp(x[1]))
            - 0.5 * (1 / torch.exp(x[1]) - 1)
        )
        / sample_bin
    )
    rate_model = mixture_composition(
        1, [log_mu, log_nu], comp_func, 
    )

    # CMP process output
    likelihood = nprb.likelihoods.COM_Poisson(
        sample_bin, neurons, "relu", dispersion_mapping=log_nu
    )

    input_group = nprb.inference.input_group(1, [(None, None, None, 1)])
    input_group.set_XZ(
        covariates, track_samples, batch_size=track_samples, trials=trials
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

    input_group = nprb.inference.input_group(2, [(None, None, None, 1)] * 2)
    input_group.set_XZ(
        covariates, track_samples, batch_size=track_samples, trials=trials
    )

    # Poisson process output
    likelihood = nprb.likelihoods.Poisson(sample_bin, neurons, "relu")

    # NLL model
    glm = nprb.inference.VI_optimized(input_group, rate_model, likelihood)
    glm.validate_model(likelihood_set=False)
    return glm


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
    rng = np.random.default_rng()
    
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
    glm = CMP_bumps(sample_bin, track_samples, covariates, neurons, trials=trials)
    glm.to(dev)

    XZ, rate, _ = glm.evaluate(0)
    syn_train = glm.likelihood.sample(rate[0].cpu().numpy(), XZ=XZ)

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
    torch.save({"model": glm.state_dict()}, savedir + "hCMP_seed{}.pt".format(seed))

    # modulated Poisson
    neurons = 50

    covariates = [
        hd_t[None, :, None].repeat(trials, axis=0),
        a_t[None, :, None].repeat(trials, axis=0),
    ]
    glm = IP_bumps(sample_bin, track_samples, covariates, neurons, trials=trials)
    glm.to(dev)

    _, rate, _ = glm.evaluate(0)
    syn_train = glm.likelihood.sample(rate[0].cpu().numpy())

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
    torch.save({"model": glm.state_dict()}, savedir + "IP_seed{}.pt".format(seed))


if __name__ == "__main__":
    main()
