import sys

import matplotlib.pyplot as plt

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.parameter import Parameter

sys.path.append("../lib/")  # access to library


import neuroprob as nprb
from neuroprob import utils

dev = nprb.inference.get_device(gpu=0)

import pickle

import helper



### synthetic data ###

# parameterizations
def w_to_gaussian(w):
    """
    Get Gaussian and orthogonal theta parameterization from the GLM parameters.

    :param np.array w: input GLM parameters of shape (neurons, dims), dims labelling (w_1, w_x,
                       w_y, w_xx, w_yy, w_xy, w_cos, w_sin)
    """
    neurons = mu.shape[0]
    w_spat = w[:, 0:6]
    prec = np.empty((neurons, 3))  # xx, yy and xy/yx
    mu = np.empty((neurons, 2))  # x and y
    prec[:, 0] = -2 * w_spat[:, 3]
    prec[:, 1] = -2 * w_spat[:, 4]
    prec[:, 2] = -w_spat[:, 5]
    prec_mat = []
    for n in range(neurons):
        prec_mat.append([[prec[n, 0], prec[n, 2]], [prec[n, 2], prec[n, 1]]])
    prec_mat = np.array(prec_mat)
    denom = prec[:, 0] * prec[:, 1] - prec[:, 2] ** 2
    mu[:, 0] = (w_spat[:, 1] * prec[:, 1] - w_spat[:, 2] * prec[:, 2]) / denom
    mu[:, 1] = (w_spat[:, 2] * prec[:, 0] - w_spat[:, 1] * prec[:, 2]) / denom
    rate_0 = np.exp(
        w_spat[:, 0] + 0.5 * (mu * np.einsum("nij,nj->ni", prec_mat, mu)).sum(1)
    )

    w_theta = w[:, 6:]
    theta_0 = np.angle(w_theta[:, 0] + w_theta[:, 1] * 1j)
    beta = np.sqrt(w_theta[:, 0] ** 2 + w_theta[:, 1] ** 2)

    return mu, prec, rate_0, np.concatenate((beta[:, None], theta_0[:, None]), axis=1)


def gaussian_to_w(mu, prec, rate_0, theta_p):
    """
    Get GLM parameters from Gaussian and orthogonal theta parameterization

    :param np.array mu: mean of the Gaussian field of shape (neurons, 2)
    :param np.array prec: precision matrix elements xx, yy, and xy of shape (neurons, 3)
    :param np.array rate_0: rate amplitude of shape (neurons)
    :param np.array theta_p: theta modulation parameters beta and theta_0 of shape (neurons, 2)
    """
    neurons = mu.shape[0]
    prec_mat = []
    for n in range(neurons):
        prec_mat.append([[prec[n, 0], prec[n, 2]], [prec[n, 2], prec[n, 1]]])
    prec_mat = np.array(prec_mat)
    w = np.empty((neurons, 8))
    w[:, 0] = np.log(rate_0) - 0.5 * (mu * np.einsum("nij,nj->ni", prec_mat, mu)).sum(1)
    w[:, 1] = mu[:, 0] * prec[:, 0] + mu[:, 1] * prec[:, 2]
    w[:, 2] = mu[:, 1] * prec[:, 1] + mu[:, 0] * prec[:, 2]
    w[:, 3] = -0.5 * prec[:, 0]
    w[:, 4] = -0.5 * prec[:, 1]
    w[:, 5] = -prec[:, 2]
    w[:, 6] = theta_p[:, 0] * np.cos(theta_p[:, 1])
    w[:, 7] = theta_p[:, 0] * np.sin(theta_p[:, 1])
    return w


def w_to_vonmises(w):
    """
    :param np.array w: parameters of the GLM of shape (neurons, 3)
    """
    rate_0 = w[:, 0]
    theta_0 = np.angle(w[:, 1] + w[:, 2] * 1j)
    kappa = np.sqrt(w[:, 1] ** 2 + w[:, 2] ** 2)
    return rate_0, kappa, theta_0


def vonmises_to_w(rate_0, kappa, theta_0):
    """
    :param np.array rate_0: rate amplitude of shape (neurons)
    :param np.array theta_p: von Mises parameters kappa and theta_0 of shape (neurons, 2)
    """
    neurons = rate_0.shape[0]
    w = np.empty((neurons, 3))
    w[:, 0] = np.log(rate_0)
    w[:, 1] = kappa * np.cos(theta_0)
    w[:, 2] = kappa * np.sin(theta_0)
    return w


# GLM
class vonMises_GLM(nprb.mappings.GLM):
    """
    Angular (head direction/theta) variable GLM
    """

    def __init__(self, neurons, tensor_type=torch.float, active_dims=None):
        super().__init__(
            1, neurons, 3, tensor_type=tensor_type, active_dims=active_dims
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


class Gauss_GLM(nprb.mappings.GLM):
    """
    Quadratic GLM for position
    """

    def __init__(self, neurons, tensor_type=torch.float, active_dims=None):
        super().__init__(
            2, neurons, 6, tensor_type=tensor_type, active_dims=active_dims
        )

    def set_params(self, w):
        self.w.data = w.type(self.tensor_type).to(self.dummy.device)

    def compute_F(self, XZ):
        XZ = self._XZ(XZ)
        x = XZ[..., 0]
        y = XZ[..., 1]
        g = torch.stack(
            (torch.ones_like(x, device=x.device), x, y, x**2, y**2, x * y), dim=-1
        )
        return (g * self.w[None, :, None, :]).sum(-1), 0


class attention_bumps(nprb.mappings.custom_wrapper):
    def __init__(
        self, neurons, inv_link="relu", tens_type=torch.float, active_dims=None
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
def CMP_hdc(sample_bin, track_samples, covariates, neurons, trials=1):
    """
    CMP with separate mu and nu tuning curves
    """
    # Von Mises fields
    angle_0 = np.linspace(0, 2 * np.pi, neurons + 1)[:-1]
    beta = np.random.rand(neurons) * 2.0 + 0.5
    rate_0 = np.random.rand(neurons) * 4.0 + 4.0
    w = np.stack(
        [np.log(rate_0), beta * np.cos(angle_0), beta * np.sin(angle_0)]
    ).T  # beta, phi_0 for theta modulation
    neurons = w.shape[0]

    vm_rate = nprb.rate_models.vonMises_GLM(neurons, inv_link="exp")
    vm_rate.set_params(w)

    # Dispersion tuning curve
    _angle_0 = np.random.permutation(
        angle_0
    )  # angle_0 + 0.4*np.random.randn(neurons)#np.random.permutation(angle_0)
    _beta = 0.6 * np.random.rand(neurons) + 0.1
    _rate_0 = np.random.rand(neurons) * 0.5 + 0.5
    w = np.stack(
        [np.log(_rate_0) + _beta, _beta * np.cos(_angle_0), _beta * np.sin(_angle_0)]
    ).T  # beta, phi_0 for theta modulation

    vm_disp = nprb.rate_models.vonMises_GLM(neurons, inv_link="identity")
    vm_disp.set_params(w)

    # sum for mu input
    comp_func = (
        lambda x: (
            (x[0] * sample_bin) ** (1 / torch.exp(x[1]))
            - 0.5 * (1 / torch.exp(x[1]) - 1)
        )
        / sample_bin
    )
    rate_model = nprb.parametrics.mixture_composition(
        1, [vm_rate, vm_disp], comp_func, inv_link="softplus"
    )

    # CMP process output
    likelihood = nprb.likelihoods.COM_Poisson(
        sample_bin, neurons, "softplus", dispersion_mapping=vm_disp
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
    Poisson with spotlight attention
    """
    # Von Mises fields
    angle_0 = np.linspace(0, 2 * np.pi, neurons + 1)[:-1]
    beta = np.random.rand(neurons) * 2.6 + 0.4
    rate_0 = np.random.rand(neurons) * 4.0 + 3.0
    w = np.stack(
        [np.log(rate_0), beta * np.cos(angle_0), beta * np.sin(angle_0)]
    ).T  # beta, phi_0 for theta modulation
    neurons = w.shape[0]

    vm_rate = nprb.rate_models.vonMises_GLM(neurons, inv_link="exp")
    vm_rate.set_params(w)

    att_rate = attention_bumps(neurons, active_dims=[1])
    mu = np.random.randn(neurons)
    sigma = 0.6 * np.random.rand(neurons) + 0.6
    A = 1.7 * np.random.rand(neurons)
    A_0 = np.ones(neurons) * 0.3
    att_rate.set_params(mu, sigma, A, A_0)

    rate_model = nprb.parametrics.product_model(2, [vm_rate, att_rate], inv_link="relu")

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
    # Gaussian von Mises bump head direction model
    sample_bin = 0.1  # 100 ms
    track_samples = 10000
    trials = 1

    hd_t = np.empty(track_samples)

    hd_t[0] = 0
    rn = np.random.randn(track_samples)
    for k in range(1, track_samples):
        hd_t[k] = hd_t[k - 1] + 0.5 * rn[k]

    hd_t = hd_t % (2 * np.pi)


    # GP trajectory sample
    Tl = track_samples

    l = 200.0 * sample_bin * np.ones((1, 1))
    v = np.ones(1)
    kernel_tuples = [("variance", v), ("RBF", "euclid", l)]

    with torch.no_grad():
        kernel, _, _ = helper.create_kernel(kernel_tuples, "softplus", torch.double)

        T = torch.arange(Tl)[None, None, :, None] * sample_bin
        K = kernel(T, T)[0, 0]
        K.view(-1)[:: Tl + 1] += 1e-6

    L = torch.cholesky(K)
    eps = torch.randn(Tl).double()
    v = L @ eps
    a_t = v.data.numpy()


    ### sample activity ###

    # heteroscedastic CMP
    neurons = 50

    covariates = [hd_t[None, :, None].repeat(trials, axis=0)]
    glm = CMP_hdc(sample_bin, track_samples, covariates, neurons, trials=trials)
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


    np.savez_compressed("./data/CMPh_HDC", spktrain=rc_t, rhd_t=rhd_t, tbin=tbin)
    torch.save({"model": glm.state_dict()}, "./data/CMPh_HDC_model")


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


    np.savez_compressed("./data/IP_HDC", spktrain=rc_t, rhd_t=rhd_t, ra_t=ra_t, tbin=tbin)
    torch.save({"model": glm.state_dict()}, "./data/IP_HDC_model")

    
if __name__ == "__main__":
    main()
