import numbers

import numpy as np
import torch

from scipy.special import factorial
from torch.nn.parameter import Parameter

from .. import base, distributions as dist



def gen_IBP(intensity):
    """
    Sample of the Inhomogenous Bernoulli Process

    :param numpy.array intensity: intensity of the Bernoulli process at array index
    :returns: sample of binary variables with same shape as intensity
    :rtype: numpy.array
    """
    b = Bernoulli(torch.tensor(intensity))
    return b.sample().numpy()


def gen_CMP(mu, nu, max_rejections=1000):
    """
    Use rejection sampling to sample from the COM-Poisson count distribution. [1]

    References:

    [1] `Bayesian Inference, Model Selection and Likelihood Estimation using Fast Rejection
         Sampling: The Conway-Maxwell-Poisson Distribution`, Alan Benson, Nial Friel (2021)

    :param numpy.array rate: input rate of shape (..., time)
    :param float tbin: time bin size
    :param float eps: order of magnitude of P(N>1)/P(N<2) per dilated Bernoulli bin
    :param int max_count: maximum number of spike counts per bin possible
    :returns: inhomogeneous Poisson process sample
    :rtype: numpy.array
    """
    trials = mu.shape[0]
    neurons = mu.shape[1]
    Y = np.empty(mu.shape)

    for tr in range(trials):
        for n in range(neurons):
            mu_, nu_ = mu[tr, n, :], nu[tr, n, :]

            # Poisson
            k = 0
            left_bins = np.where(nu_ >= 1)[0]
            while len(left_bins) > 0:
                mu__, nu__ = mu_[left_bins], nu_[left_bins]
                y_dash = torch.poisson(torch.tensor(mu__)).numpy()
                _mu_ = np.floor(mu__)
                alpha = (
                    mu__ ** (y_dash - _mu_)
                    / scsps.factorial(y_dash)
                    * scsps.factorial(_mu_)
                ) ** (nu__ - 1)

                u = np.random.rand(*mu__.shape)
                selected = u <= alpha
                Y[tr, n, left_bins[selected]] = y_dash[selected]
                left_bins = left_bins[~selected]
                if k >= max_rejections:
                    raise ValueError("Maximum rejection steps exceeded")
                else:
                    k += 1

            # geometric
            k = 0
            left_bins = np.where(nu_ < 1)[0]
            while len(left_bins) > 0:
                mu__, nu__ = mu_[left_bins], nu_[left_bins]
                p = 2 * nu__ / (2 * mu__ * nu__ + 1 + nu__)
                u_0 = np.random.rand(*p.shape)

                y_dash = np.floor(np.log(u_0) / np.log(1 - p))
                a = np.floor(mu__ / (1 - p) ** (1 / nu__))
                alpha = (1 - p) ** (a - y_dash) * (
                    mu__ ** (y_dash - a) / scsps.factorial(y_dash) * scsps.factorial(a)
                ) ** nu__

                u = np.random.rand(*mu__.shape)
                selected = u <= alpha
                Y[tr, n, left_bins[selected]] = y_dash[selected]
                left_bins = left_bins[~selected]
                if k >= max_rejections:
                    raise ValueError("Maximum rejection steps exceeded")
                else:
                    k += 1

    return Y

    

class _count_model(base._likelihood):
    """
    Count likelihood base class.
    """

    def __init__(self, tbin, neurons, inv_link, tensor_type, strict_likelihood=True):
        super().__init__(tbin, neurons, neurons, inv_link, tensor_type)
        self.strict_likelihood = strict_likelihood
        self.dispersion_mapping = None

    def set_Y(self, spikes, batch_info):
        """
        Get all the activity into batches useable format for quick log-likelihood evaluation
        Tensor shapes: self.spikes (neuron_dim, batch_dim)

        tfact is the log of time_bin times the spike count
        lfact is the log (spike count)!
        """
        super().set_Y(spikes, batch_info)
        batch_edge, _, _ = self.batch_info

        self.lfact = []
        self.tfact = []
        self.totspik = []
        for b in range(self.batches):
            spikes = self.all_spikes[..., batch_edge[b] : batch_edge[b + 1]]
            self.totspik.append(spikes.sum(-1))
            self.tfact.append(spikes * torch.log(self.tbin.cpu()))
            self.lfact.append(torch.lgamma(spikes + 1.0))

    def KL_prior(self, importance_weighted=False):
        """ """
        if self.dispersion_mapping is not None:
            return self.dispersion_mapping.KL_prior(importance_weighted)
        else:
            return 0

    def sample_helper(self, h, b, neuron, samples):
        """
        NLL helper function for sample evaluation. Note that spikes is batched including history
        when the model uses history couplings, hence we sample the spike batches without the
        history segments from this function.
        """
        rates = self.f(h)  # watch out for underflow or overflow here
        batch_edge, _, _ = self.batch_info
        spikes = self.all_spikes[:, neuron, batch_edge[b] : batch_edge[b + 1]].to(
            self.tbin.device
        )
        if (
            self.trials != 1 and samples > 1 and self.trials < h.shape[0]
        ):  # cannot rely on broadcasting
            spikes = spikes.repeat(samples, 1, 1)  # MC x trials

        if self.inv_link == "exp":  # spike count times log rate
            n_l_rates = spikes * h
        else:
            n_l_rates = spikes * torch.log(rates + 1e-12)

        return rates, n_l_rates, spikes

    def sample_dispersion(self, XZ, samples, neuron):
        """
        Posterior predictive mean of the dispersion model.
        """
        if self.dispersion_mapping.MC_only:
            dh = self.dispersion_mapping.sample_F(XZ, samples)[:, neuron, :]
        else:
            disp, disp_var = self.dispersion_mapping.compute_F(XZ)
            dh = self.mc_gen(disp, disp_var, samples, neuron)

        return self.dispersion_mapping_f(dh)  # watch out for underflow or overflow here

    def objective(self, F_mu, F_var, XZ, b, neuron, samples=10, mode="MC"):
        """
        Computes the terms for variational expectation :math:`\mathbb{E}_{q(f)q(z)}[]`, which
        can be used to compute different likelihood objectives.
        The returned tensor will have sample dimension as MC over :math:`q(z)`, depending
        on the evaluation mode will be MC or GH or exact over the likelihood samples. This
        is all combined in to the same dimension to be summed over. The weights :math:`w_s`
        are the quadrature weights or equal weights for MC, with appropriate normalization.

        :param int samples: number of MC samples or GH points (exact will ignore and give 1)

        :returns: negative likelihood term of shape (samples, timesteps), sample weights (samples, 1
        :rtype: tuple of torch.tensors
        """
        if mode == "MC":
            h = self.mc_gen(F_mu, F_var, samples, neuron)  # h has only observed neurons
            rates, n_l_rates, spikes = self.sample_helper(h, b, neuron, samples)
            ws = torch.tensor(1.0 / rates.shape[0])
        elif mode == "GH":
            h, ws = self.gh_gen(F_mu, F_var, samples, neuron)
            rates, n_l_rates, spikes = self.sample_helper(h, b, neuron, samples)
            ws = ws[:, None]
        elif mode == "direct":  # no variational uncertainty
            rates, n_l_rates, spikes = self.sample_helper(
                F_mu[:, neuron, :], b, neuron, samples
            )
            ws = torch.tensor(1.0 / rates.shape[0])
        else:
            raise NotImplementedError

        if self.dispersion_mapping is None:
            disper_param = None
        else:  # MC sampling
            disper_param = self.sample_dispersion(XZ, samples, neuron)

        return self.nll(b, rates, n_l_rates, spikes, neuron, disper_param), ws


# Special cases
class Universal(base._likelihood):
    """
    Universal count distribution with finite cutoff at max_count.

    """

    def __init__(
        self, neurons, C, basis_mode, inv_link, max_count, shared_W=False, tensor_type=torch.float
    ):
        super().__init__(1.0, neurons * C, neurons, inv_link, tensor_type)  # dummy tbin
        self.K = max_count
        self.C = C
        self.neurons = neurons
        self.lsoftm = torch.nn.LogSoftmax(dim=-1)
        
        self.basis = self.get_basis(basis_mode)
        expand_C = torch.cat(
            [f_(torch.ones(1, self.C)) for f_ in self.basis], dim=-1
        ).shape[-1]  # size of expanded vector

        mapping_net = nprb.nn.Parallel_MLP(
            [], expand_C, (max_count + 1), channels, shared_W=shared_W, out=None
        )  # single linear mapping
        self.add_module("mapping_net", mapping_net)  # maps from NxC to NxK

    def set_Y(self, spikes, batch_info):
        """
        Get all the activity into batches useable format for fast log-likelihood evaluation.
        Batched spikes will be a list of tensors of shape (trials, neurons, time) with trials
        set to 1 if input has no trial dimension (e.g. continuous recording).

        :param np.ndarray spikes: becomes a list of [neuron_dim, batch_dim]
        :param int/list batch_size:
        :param int filter_len: history length of the GLM couplings (1 indicates no history coupling)
        """
        if self.K < spikes.max():
            raise ValueError("Maximum count is exceeded in the spike count data")
        super().set_Y(spikes, batch_info)

    def onehot_to_counts(self, onehot):
        """
        Convert one-hot vector representation of counts. Assumes the event dimension is the last.

        :param torch.Tensor onehot: one-hot vector representation of shape (..., event)
        :returns: spike counts
        :rtype: torch.tensor
        """
        counts = torch.zeros(*onehot.shape[:-1], device=onehot.device)
        inds = torch.where(onehot)
        counts[inds[:-1]] = inds[-1].float()
        return counts

    def counts_to_onehot(self, counts):
        """
        Convert counts to one-hot vector representation. Adds the event dimension at the end.

        :param torch.Tensor counts: spike counts of some tensor shape
        :param int max_counts: size of the event dimension (max_counts + 1)
        :returns: one-hot representation of shape (*counts.shape, event)
        :rtype: torch.tensor
        """
        km = self.K + 1
        onehot = torch.zeros(*counts.shape, km, device=counts.device)
        onehot_ = onehot.view(-1, km)
        g = onehot_.shape[0]
        onehot_[np.arange(g), counts.flatten()[np.arange(g)].long()] = 1
        return onehot_.view(*onehot.shape)
    
    def get_basis(basis_mode="el"):
        
        if basis_mode == "id":
            basis = (lambda x: x,)

        elif basis_mode == "el":  # element-wise exp-linear
            basis = (lambda x: x, lambda x: torch.exp(x))

        elif basis_mode == "eq":  # element-wise exp-quadratic
            basis = (lambda x: x, lambda x: x**2, lambda x: torch.exp(x))

        elif basis_mode == "ec":  # element-wise exp-cubic
            basis = (
                lambda x: x,
                lambda x: x**2,
                lambda x: x**3,
                lambda x: torch.exp(x),
            )

        elif basis_mode == "qd":  # exp and full quadratic

            def mix(x):
                C = x.shape[-1]
                out = torch.empty((*x.shape[:-1], C * (C - 1) // 2), dtype=x.dtype).to(
                    x.device
                )
                k = 0
                for c in range(1, C):
                    for c_ in range(c):
                        out[..., k] = x[..., c] * x[..., c_]
                        k += 1

                return out  # shape (..., C*(C-1)/2)

            basis = (
                lambda x: x,
                lambda x: x**2,
                lambda x: torch.exp(x),
                lambda x: mix(x),
            )

        else:
            raise ValueError("Invalid basis expansion")

        return basis
        
    def get_logp(self, F_mu, neuron):
        """
        Compute count probabilities from the rate model output.

        :param torch.Tensor F_mu: the F_mu product output of the rate model (samples and/or trials, F_dims, time)
        :returns: log probability tensor
        :rtype: tensor of shape (samples and/or trials, n, t, c)
        """
        T = F_mu.shape[-1]
        samples = F_mu.shape[0]
        
        inps = F_mu.permute(0, 2, 1).reshape(samples * T, -1)  # (samplesxtime, in_dimsxchannels)
        inps = inps.view(inps.shape[0], -1, self.C)
        inps = torch.cat([f_(inps) for f_ in self.basis], dim=-1)
        a = self.mapping_net(inps, neuron).view(out.shape[0], -1)  # # samplesxtime, NxK
        
        log_probs = self.lsoftm(a.view(samples, T, -1, self.K + 1).permute(0, 2, 1, 3))
        return log_probs

    def sample_helper(self, h, b, neuron, samples):
        """
        NLL helper function for sample evaluation.
        """
        batch_edge, _, _ = self.batch_info
        logp = self.get_logp(h, neuron)
        spikes = self.all_spikes[:, neuron, batch_edge[b] : batch_edge[b + 1]].to(
            self.tbin.device
        )
        if (
            self.trials != 1 and samples > 1 and self.trials < h.shape[0]
        ):  # cannot rely on broadcasting
            spikes = spikes.repeat(samples, 1, 1)
        tar = self.counts_to_onehot(spikes)

        return logp, tar

    def _neuron_to_F(self, neuron):
        """
        Access subset of neurons in expanded space.
        Note the F_dims here is equal to NxC flattened.
        """
        neuron = self._validate_neuron(neuron)
        if len(neuron) == self.neurons:
            F_dims = list(range(self.F_dims))
        else:  # access subset of neurons
            F_dims = list(
                np.concatenate(
                    [np.arange(n * self.C, (n + 1) * self.C) for n in neuron]
                )
            )

        return F_dims

    def objective(self, F_mu, F_var, XZ, b, neuron, samples=10, mode="MC"):
        """
        Computes the terms for variational expectation :math:`\mathbb{E}_{q(f)q(z)}[]`, which
        can be used to compute different likelihood objectives.
        The returned tensor will have sample dimension as MC over :math:`q(z)`, depending
        on the evaluation mode will be MC or GH or exact over the likelihood samples. This
        is all combined in to the same dimension to be summed over. The weights :math:`w_s`
        are the quadrature weights or equal weights for MC, with appropriate normalization.

        :param int samples: number of MC samples or GH points (exact will ignore and give 1)

        :returns: negative likelihood term of shape (samples, timesteps)
        """
        F_dims = self._neuron_to_F(neuron)

        if mode == "MC":
            h = self.mc_gen(
                F_mu, F_var, samples, F_dims
            )  # h has only observed neurons (from neuron)
            logp, tar = self.sample_helper(h, b, neuron, samples)
            ws = torch.tensor(1.0 / logp.shape[0])
        elif mode == "GH":
            h, ws = self.gh_gen(F_mu, F_var, samples, F_dims)
            logp, tar = self.sample_helper(h, b, neuron, samples)
            ws = ws[:, None]
        elif mode == "direct":
            rates, n_l_rates, spikes = self.sample_helper(
                F_mu[:, F_dims, :], b, neuron, samples
            )
            ws = torch.tensor(1.0 / rates.shape[0])
        else:
            raise NotImplementedError

        nll = -(tar * logp).sum(-1)
        return nll.sum(1), ws

    def sample(self, F_mu, neuron, XZ=None):
        """
        Sample from the categorical distribution.

        :param numpy.array log_probs: log count probabilities (trials, neuron, timestep, counts), no
                                      need to be normalized
        :returns: spike train of shape (trials, neuron, timesteps)
        :rtype: np.array
        """
        F_dims = self._neuron_to_F(neuron)
        log_probs = self.get_logp(
            torch.tensor(F_mu[:, F_dims, :], dtype=self.tensor_type)
        )
        c_dist = mdl.distributions.Categorical(logits=log_probs)
        cnt_prob = torch.exp(log_probs)
        return c_dist.sample().numpy()


# count distributions
class Bernoulli(_count_model):
    """
    Inhomogeneous Bernoulli likelihood, limits the count to binary trains.
    """

    def __init__(self, tbin, neurons, tensor_type=torch.float, strict_likelihood=True):
        inv_link = lambda x: torch.sigmoid(x) / tbin
        super().__init__(tbin, neurons, inv_link, tensor_type, strict_likelihood)

    def set_Y(self, spikes, batch_info):
        """
        Get all the activity into batches useable format for quick log-likelihood evaluation
        Tensor shapes: self.spikes (neuron_dim, batch_dim)

        tfact is the log of time_bin times the spike count
        """
        if spikes.max() > 1:
            raise ValueError("Only binary spike trains are accepted in set_Y() here")
        super().set_Y(spikes, batch_info)

    def get_saved_factors(self, b, neuron, spikes):
        """
        Get saved factors for proper likelihood values and perform broadcasting when needed.
        """
        if self.strict_likelihood:
            tfact = self.tfact[b][:, neuron, :].to(self.tbin.device)
            if (
                tfact.shape[0] != 1 and tfact.shape[0] < spikes.shape[0]
            ):  # cannot rely on broadcasting
                tfact = tfact.repeat(spikes.shape[0] // tfact.shape[0], 1, 1)
        else:
            tfact = 0
        return tfact

    def nll(self, b, rates, n_l_rates, spikes, neuron, disper_param=None):
        tfact = self.get_saved_factors(b, neuron, spikes)
        nll = -(n_l_rates + tfact + (1 - spikes) * torch.log(1 - rates * self.tbin))
        return nll.sum(1)

    def sample(self, rate, neuron=None, XZ=None):
        """
        Takes into account the quantization bias if we sample IPP with dilation factor.

        :param numpy.array rate: input rate of shape (trials, neuron, timestep)
        :returns: spike train of shape (trials, neuron, timesteps)
        :rtype: np.array
        """
        neuron = self._validate_neuron(neuron)
        return gen_IBP(rate[:, neuron, :] * self.tbin.item())


class Poisson(_count_model):
    """
    Poisson count likelihood.
    """

    def __init__(
        self, tbin, neurons, inv_link, tensor_type=torch.float, strict_likelihood=True
    ):
        super().__init__(tbin, neurons, inv_link, tensor_type, strict_likelihood)

    def get_saved_factors(self, b, neuron, spikes):
        """
        Get saved factors for proper likelihood values and perform broadcasting when needed.
        """
        if self.strict_likelihood:
            tfact = self.tfact[b][:, neuron, :].to(self.tbin.device)
            lfact = self.lfact[b][:, neuron, :].to(self.tbin.device)
            if (
                tfact.shape[0] != 1 and tfact.shape[0] < spikes.shape[0]
            ):  # cannot rely on broadcasting
                tfact = tfact.repeat(spikes.shape[0] // tfact.shape[0], 1, 1)
                lfact = lfact.repeat(spikes.shape[0] // lfact.shape[0], 1, 1)
        else:
            tfact, lfact = 0, 0
        return tfact, lfact

    def nll(self, b, rates, n_l_rates, spikes, neuron, disper_param=None):
        tfact, lfact = self.get_saved_factors(b, neuron, spikes)
        T = rates * self.tbin
        nll = -n_l_rates + T - tfact + lfact
        return nll.sum(1)

    def objective(self, F_mu, F_var, XZ, b, neuron, samples=10, mode="MC"):
        """
        The Poisson likelihood with the log Cox process has an exact variational likelihood term.
        """
        if self.inv_link == "exp" and F_var is not None:  # exact
            if (
                isinstance(F_var, numbers.Number) is False and len(F_var.shape) == 4
            ):  # diagonalize
                F_var = F_var.view(*F_var.shape[:2], -1)[:, :, :: F_var.shape[-1] + 1]

            batch_edge = self.batch_info[0]
            spikes = self.all_spikes[:, neuron, batch_edge[b] : batch_edge[b + 1]].to(
                self.tbin.device
            )
            n_l_rates = spikes * F_mu[:, neuron, :]

            tfact, lfact = self.get_saved_factors(b, neuron, spikes)
            T = self.f((F_mu + F_var / 2.0)[:, neuron, :]) * self.tbin
            nll = -n_l_rates + T - tfact + lfact

            ws = torch.tensor(1.0 / F_mu.shape[0])
            return (
                nll.sum(1),
                ws,
            )  # first dimension is summed over later (MC over Z), hence divide by shape[0]
        else:
            return super().objective(
                F_mu, F_var, XZ, b, neuron, samples=samples, mode=mode
            )

    def sample(self, rate, neuron=None, XZ=None):
        """
        Takes into account the quantization bias if we sample IPP with dilation factor.

        :param numpy.array rate: input rate of shape (trials, neuron, timestep)
        :returns: spike train of shape (trials, neuron, timesteps)
        :rtype: np.array
        """
        neuron = self._validate_neuron(neuron)
        return torch.poisson(
            torch.tensor(rate[:, neuron, :] * self.tbin.item())
        ).numpy()


class ZI_Poisson(_count_model):
    """
    Zero-inflated Poisson (ZIP) count likelihood. [1]

    References:

    [1] `Untethered firing fields and intermittent silences: Why grid‐cell discharge is so variable`,
        Johannes Nagele  Andreas V.M. Herz  Martin B. Stemmler

    """

    def __init__(
        self,
        tbin,
        neurons,
        inv_link,
        alpha,
        tensor_type=torch.float,
        strict_likelihood=True,
    ):
        super().__init__(tbin, neurons, inv_link, tensor_type, strict_likelihood)
        if alpha is not None:
            self.register_parameter("alpha", Parameter(alpha.type(self.tensor_type)))

    def constrain(self):
        if self.alpha is not None:
            self.alpha.data = torch.clamp(self.alpha.data, min=0.0, max=1.0)

    def get_saved_factors(self, b, neuron, spikes):
        """
        Get saved factors for proper likelihood values and perform broadcasting when needed.
        """
        tfact = self.tfact[b][:, neuron, :].to(self.tbin.device)
        lfact = self.lfact[b][:, neuron, :].to(self.tbin.device)
        if (
            tfact.shape[0] != 1 and tfact.shape[0] < spikes.shape[0]
        ):  # cannot rely on broadcasting
            tfact = tfact.repeat(spikes.shape[0] // tfact.shape[0], 1, 1)
            lfact = lfact.repeat(spikes.shape[0] // lfact.shape[0], 1, 1)

        return tfact, lfact

    def nll(self, b, rates, n_l_rates, spikes, neuron, disper_param=None):
        if disper_param is None:
            alpha_ = self.alpha.expand(1, self.neurons)[:, neuron, None]
        else:
            alpha_ = disper_param

        tfact, lfact = self.get_saved_factors(b, neuron, spikes)
        T = rates * self.tbin
        zero_spikes = spikes == 0  # mask
        nll_ = (
            -n_l_rates + T - tfact + lfact - torch.log(1.0 - alpha_)
        )  # -log (1-alpha)*p(N)
        p = torch.exp(-nll_)  # stable as nll > 0
        nll_0 = -torch.log(alpha_ + p)
        nll = zero_spikes * nll_0 + (~zero_spikes) * nll_
        return nll.sum(1)

    def sample(self, rate, neuron=None, XZ=None):
        """
        Sample from ZIP process.

        :param numpy.array rate: input rate of shape (trials, neuron, timestep)
        :returns: spike train of shape (trials, neuron, timesteps)
        :rtype: np.array
        """
        neuron = self._validate_neuron(neuron)
        rate_ = rate[:, neuron, :]

        if self.dispersion_mapping is None:
            alpha_ = (
                self.alpha[None, :, None]
                .expand(rate.shape[0], self.neurons, rate_.shape[-1])
                .data.cpu()
                .numpy()[:, neuron, :]
            )
        else:
            with torch.no_grad():
                alpha_ = (
                    self.sample_dispersion(XZ, rate.shape[0] // XZ.shape[0], neuron)
                    .cpu()
                    .numpy()
                )

        zero_mask = gen_IBP(alpha_)
        return (1.0 - zero_mask) * torch.poisson(
            torch.tensor(rate_ * self.tbin.item())
        ).numpy()


class hZI_Poisson(ZI_Poisson):
    """
    Heteroscedastic ZIP
    """

    def __init__(
        self,
        tbin,
        neurons,
        inv_link,
        dispersion_mapping,
        tensor_type=torch.float,
        strict_likelihood=True,
    ):
        super().__init__(tbin, neurons, inv_link, None, tensor_type, strict_likelihood)
        self.dispersion_mapping = dispersion_mapping
        self.dispersion_mapping_f = torch.sigmoid

    def constrain(self):
        return


class Negative_binomial(_count_model):
    """
    Gamma-Poisson mixture.

    :param np.ndarray r_inv: :math:`r^{-1}` parameter of the NB likelihood, if left to None this value is
                           expected to be provided by the heteroscedastic model in the inference class.
    """

    def __init__(
        self,
        tbin,
        neurons,
        inv_link,
        r_inv,
        tensor_type=torch.float,
        strict_likelihood=True,
    ):
        super().__init__(tbin, neurons, inv_link, tensor_type, strict_likelihood)
        if r_inv is not None:
            self.register_parameter("r_inv", Parameter(r_inv.type(self.tensor_type)))

    def constrain(self):
        if self.r_inv is not None:
            self.r_inv.data = torch.clamp(
                self.r_inv.data, min=0.0
            )  # effective r_inv > 1e-6 stabilized in NLL

    def get_saved_factors(self, b, neuron, spikes):
        """
        Get saved factors for proper likelihood values and perform broadcasting when needed.
        """
        if self.strict_likelihood:
            tfact = self.tfact[b][:, neuron, :].to(self.tbin.device)
            lfact = self.lfact[b][:, neuron, :].to(self.tbin.device)
            if (
                tfact.shape[0] != 1 and tfact.shape[0] < spikes.shape[0]
            ):  # cannot rely on broadcasting
                tfact = tfact.repeat(spikes.shape[0] // tfact.shape[0], 1, 1)
                lfact = lfact.repeat(spikes.shape[0] // lfact.shape[0], 1, 1)
        else:
            tfact, lfact = 0, 0
        return tfact, lfact

    def nll(self, b, rates, n_l_rates, spikes, neuron, disper_param=None):
        """
        The negative log likelihood function. Note that if disper_param is not None, it will use those values for
        the dispersion parameter rather than its own dispersion parameters.

        .. math::
            P(n|\lambda, r) = \frac{\lambda^n}{n!} \frac{\Gamma(r+n)}{\Gamma(r) \, (r+\lamba)^n} \left( 1 + \frac{\lambda}{r} \right)^{-r}

        where the mean is related to the conventional parameter :math:`\lambda = \frac{pr}{1-p}`

        For :math:`r \to \infty` we compute the likelihood as a correction to Poisson retaining :math:`\log{1 + O(r^{-1})}`.
        We parameterize the likelihood with :math:`r^{-1}`, as this allows one to reach the Poisson limit (:math:`r^{-1} \to 0`)
        using the series expansion.

        :param int b: batch index to evaluate
        :param torch.Tensor rates: rates of shape (trial, neuron, time)
        :param torch.Tensor n_l_rates: spikes*log(rates)
        :param torch.Tensor spikes: spike counts of shape (trial, neuron, time)
        :param list neuron: list of neuron indices to evaluate
        :param torch.Tensor disper_param: input for heteroscedastic NB likelihood of shape (trial, neuron, time),
                                          otherwise uses fixed :math:`r_{inv}`
        :returns: NLL of shape (trial, time)
        :rtype: torch.tensor
        """
        if disper_param is None:
            disper_param = self.r_inv.expand(1, self.neurons)[:, neuron, None]

        # when r becomes very large, parameterization in r becomes numerically unstable
        asymptotic_mask = disper_param < 1e-3
        r_ = 1.0 / (disper_param + asymptotic_mask)
        r_inv_ = disper_param  # use 1/r parameterization

        tfact, lfact = self.get_saved_factors(b, neuron, spikes)
        lambd = rates * self.tbin
        fac_lgamma = -torch.lgamma(r_ + spikes) + torch.lgamma(r_)
        fac_power = (spikes + r_) * torch.log(r_ + lambd) - r_ * torch.log(r_)

        nll_r = fac_power + fac_lgamma
        nll_r_inv = lambd + torch.log(
            1.0 + r_inv_ * (spikes**2 + 1.0 - spikes * (3 / 2 + lambd))
        )

        nll = -n_l_rates - tfact + lfact
        nll[asymptotic_mask] = nll[asymptotic_mask] + nll_r_inv[asymptotic_mask]
        nll[~asymptotic_mask] = nll[~asymptotic_mask] + nll_r[~asymptotic_mask]
        # nll = nll_r*(~asymptotic_mask) + nll_r_inv*asymptotic_mask
        return nll.sum(1)

    def sample(self, rate, neuron=None, XZ=None):
        """
        Sample from the Gamma-Poisson mixture.

        :param numpy.array rate: input rate of shape (trials, neuron, timestep)
        :param int max_count: maximum number of spike counts per time bin
        :returns: spike train of shape (trials, neuron, timesteps)
        :rtype: np.array
        """
        neuron = self._validate_neuron(neuron)
        rate_ = rate[:, neuron, :]

        if self.dispersion_mapping is None:
            r_ = (
                1.0
                / (
                    self.r_inv[None, :, None]
                    .expand(rate.shape[0], self.neurons, rate_.shape[-1])
                    .data.cpu()
                    .numpy()
                    + 1e-12
                )[:, neuron, :]
            )
        else:
            samples = rate.shape[0]
            with torch.no_grad():
                disp = self.sample_dispersion(XZ, rate.shape[0] // XZ.shape[0], neuron)
            r_ = 1.0 / (disp.cpu().numpy() + 1e-12)

        s = np.random.gamma(
            r_, rate_ * self.tbin.item() / r_
        )  # becomes delta around rate*tbin when r to infinity, cap at 1e12
        return torch.poisson(torch.tensor(s)).numpy()


class hNegative_binomial(Negative_binomial):
    """
    Heteroscedastic NB
    """

    def __init__(
        self,
        tbin,
        neurons,
        inv_link,
        dispersion_mapping,
        tensor_type=torch.float,
        strict_likelihood=True,
    ):
        super().__init__(tbin, neurons, inv_link, None, tensor_type, strict_likelihood)
        self.dispersion_mapping = dispersion_mapping
        self.dispersion_mapping_f = torch.nn.functional.softplus

    def constrain(self):
        return


class COM_Poisson(_count_model):
    """
    Conway-Maxwell-Poisson, as described in
    https://en.wikipedia.org/wiki/Conway%E2%80%93Maxwell%E2%80%93Poisson_distribution.

    """

    def __init__(
        self,
        tbin,
        neurons,
        inv_link,
        nu,
        tensor_type=torch.float,
        J=100,
        strict_likelihood=True,
    ):
        super().__init__(tbin, neurons, inv_link, tensor_type, strict_likelihood)
        if nu is not None:
            self.register_parameter("log_nu", Parameter(nu.type(self.tensor_type)))

        self.J = J
        self.register_buffer("powers", torch.arange(self.J + 1).type(self.tensor_type))
        self.register_buffer("jfact", torch.lgamma(self.powers + 1.0))

    def get_saved_factors(self, b, neuron, spikes):
        """
        Get saved factors for proper likelihood values and perform broadcasting when needed.
        """
        if self.strict_likelihood:
            tfact = self.tfact[b][:, neuron, :].to(self.tbin.device)
            if (
                tfact.shape[0] != 1 and tfact.shape[0] < spikes.shape[0]
            ):  # cannot rely on broadcasting
                tfact = tfact.repeat(spikes.shape[0] // tfact.shape[0], 1, 1)
        else:
            tfact = 0

        lfact = self.lfact[b][:, neuron, :].to(self.tbin.device)
        if (
            lfact.shape[0] != 1 and lfact.shape[0] < spikes.shape[0]
        ):  # cannot rely on broadcasting
            lfact = lfact.repeat(spikes.shape[0] // lfact.shape[0], 1, 1)

        return tfact, lfact

    def log_Z(self, log_lambda, nu):
        """
        Partition function.

        :param torch.Tensor lambd: lambda of shape (samples, neurons, timesteps)
        :param torch.Tensor nu: nu of shape (samples, neurons, timesteps)
        :returns: log Z of shape (samples, neurons, timesteps)
        :rtype: torch.tensor
        """
        # indx = torch.where((self.powers*lambd.max() - nu_.min()*self.j) < -1e1) # adaptive
        # if len(indx) == 0:
        #    indx = self.J+1
        log_Z_term = (
            self.powers[:, None, None, None] * log_lambda[None, ...]
            - nu[None, ...] * self.jfact[:, None, None, None]
        )
        return torch.logsumexp(log_Z_term, dim=0)

    def nll(self, b, rates, n_l_rates, spikes, neuron, disper_param=None):
        """
        :param int b: batch index to evaluate
        :param torch.Tensor rates: rates of shape (trial, neuron, time)
        :param torch.Tensor n_l_rates: spikes*log(rates)
        :param torch.Tensor spikes: spike counts of shape (neuron, time)
        :param list neuron: list of neuron indices to evaluate
        :param torch.Tensor disper_param: input for heteroscedastic NB likelihood of shape (trial, neuron, time),
                                          otherwise uses fixed :math:`\nu`
        :returns: NLL of shape (trial, time)
        :rtype: torch.tensor
        """
        if disper_param is None:
            nu_ = torch.exp(self.log_nu).expand(1, self.neurons)[:, neuron, None]
        else:
            nu_ = torch.exp(disper_param)  # nn.functional.softplus

        tfact, lfact = self.get_saved_factors(b, neuron, spikes)
        log_lambda = torch.log(rates * self.tbin + 1e-12)

        l_Z = self.log_Z(log_lambda, nu_)

        nll = -n_l_rates + l_Z - tfact + nu_ * lfact
        return nll.sum(1)

    def sample(self, rate, neuron=None, XZ=None):
        """
        Sample from the CMP distribution.

        :param numpy.array rate: input rate of shape (neuron, timestep)
        :returns: spike train of shape (trials, neuron, timesteps)
        :rtype: np.array
        """
        neuron = self._validate_neuron(neuron)
        mu_ = rate[:, neuron, :] * self.tbin.item()

        if self.dispersion_mapping is None:
            nu_ = (
                torch.exp(self.log_nu)[None, :, None]
                .expand(rate.shape[0], self.neurons, mu_.shape[-1])
                .data.cpu()
                .numpy()[:, neuron, :]
            )
        else:
            samples = rate.shape[0]
            with torch.no_grad():
                disp = self.sample_dispersion(XZ, rate.shape[0] // XZ.shape[0], neuron)
            nu_ = torch.exp(disp).cpu().numpy()

        return gen_CMP(mu_, nu_)


class hCOM_Poisson(COM_Poisson):
    """
    Heteroscedastic CMP
    """

    def __init__(
        self,
        tbin,
        neurons,
        inv_link,
        dispersion_mapping,
        tensor_type=torch.float,
        J=100,
        strict_likelihood=True,
    ):
        super().__init__(
            tbin, neurons, inv_link, None, tensor_type, J, strict_likelihood
        )
        self.dispersion_mapping = dispersion_mapping
        self.dispersion_mapping_f = lambda x: x

    def constrain(self):
        return
