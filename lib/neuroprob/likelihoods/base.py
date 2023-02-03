import numbers
import types
import numpy as np

import torch

from .. import distributions as dist
from ..base import _data_object, _link_functions, _inv_link_functions

_ll_modes = ["MC", "GH"]



class _likelihood(_data_object):
    """
    Likelihood base class.
    """

    def __init__(
        self, tbin, F_dims, neurons, inv_link, tensor_type=torch.float, mode="MC"
    ):
        """
        :param int F_dims: dimensionality of the inner quantity (equal to rate model output dimensions)
        :param string/LambdaType inv_link:
        :param dtype tensor_type: model tensor type for computation
        :param string mode: evaluation mode of the variational expectation, 'MC' and 'GH'
        """
        super().__init__()
        self.tensor_type = tensor_type
        self.register_buffer("tbin", torch.tensor(tbin, dtype=self.tensor_type))
        self.F_dims = F_dims
        self.neurons = neurons
        self.all_spikes = None  # label as data not set

        if mode in _ll_modes:
            self.mode = mode
        else:
            raise NotImplementedError("Evaluation method is not supported")

        if isinstance(inv_link, types.LambdaType):
            self.f = inv_link
            inv_link = "custom"
        elif _inv_link_functions.get(inv_link) is None:
            raise NotImplementedError("Link function is not supported")
        else:
            self.f = _inv_link_functions[inv_link]
        self.inv_link = inv_link

    def set_Y(self, spikes, batch_info):
        """
        Get all the activity into batches useable format for fast log-likelihood evaluation.
        Batched spikes will be a list of tensors of shape (trials, neurons, time) with trials
        set to 1 if input has no trial dimension (e.g. continuous recording).

        :param np.ndarray spikes: becomes a list of [neuron_dim, batch_dim]
        :param int/list batch_size:
        :param int filter_len: history length of the GLM couplings (1 indicates no history coupling)
        """
        if self.neurons != spikes.shape[-2]:
            raise ValueError("Spike data Y does not match neuron count of likelihood")

        if len(spikes.shape) == 2:  # add in trial dimension
            spikes = spikes[None, ...]

        self.setup_batching(batch_info, spikes.shape[-1], spikes.shape[0])
        self.all_spikes = spikes.type(self.tensor_type)

    def KL_prior(self, importance_weighted):
        """ """
        return 0

    def constrain(self):
        """
        Constrain parameters in optimization
        """
        return

    def mc_gen(self, q_mu, q_var, samples, neuron):
        """
        Diagonal covariance q_var due to separability in time for likelihood
        Function for generating Gaussian MC samples. No MC samples are drawn when the variance is 0.

        :param torch.Tensor q_mu: the mean of the MC distribution
        :param torch.Tensor q_var: the (co)variance, type (univariate, multivariate) deduced
                                   from tensor shape
        :param int samples: number of MC samples
        :returns: the samples
        :rtype: torch.tensor
        """
        q_mu = q_mu[:, neuron, :]
        if isinstance(q_var, numbers.Number):  # zero scalar
            h = (
                q_mu[None, ...]
                .expand(samples, *q_mu.shape)
                .reshape(-1, *q_mu.shape[1:])
                if samples > 1
                else q_mu
            )

        else:  # len(q_var.shape) == 3
            q_var = q_var[:, neuron, :]
            if samples == 1:  # no expanding
                h = dist.Rn_Normal(q_mu, q_var.sqrt())()
            else:  # shape is (ll_samplesxcov_samples, neurons, time)
                h = dist.Rn_Normal(q_mu, q_var.sqrt())((samples,)).view(
                    -1, *q_mu.shape[1:]
                )

        return h

    def gh_gen(self, q_mu, q_var, points, neuron):
        """
        Diagonal covariance q_var due to separability in time for likelihood
        Computes the Gauss-Hermite quadrature locations and weights.

        :param torch.Tensor q_mu: the mean of the MC distribution
        :param torch.Tensor q_var: the (co)variance, type (univariate, multivariate) deduced
                                   from tensor shape
        :param int points: number of quadrature points
        :returns: tuple of locations and weights tensors
        :rtype: tuple
        """
        locs, ws = np.polynomial.hermite.hermgauss(
            points
        )  # sample points and weights for quadrature
        locs = (
            torch.tensor(locs, dtype=self.tensor_type)
            .repeat_interleave(q_mu.shape[0], 0)
            .to(self.tbin.device)
        )
        ws = (
            torch.tensor(
                1
                / np.sqrt(np.pi)
                * ws
                / q_mu.shape[0],  # q_mu shape 0 is cov_samplesxtrials
                dtype=self.tensor_type,
            )
            .repeat_interleave(q_mu.shape[0], 0)
            .to(self.tbin.device)
        )

        q_mu = q_mu[:, neuron, :].repeat(
            points, 1, 1
        )  # fill sample dimension with GH quadrature points
        if isinstance(q_var, numbers.Number) is False:  # not the zero scalar
            q_var = q_var[:, neuron, :].repeat(points, 1, 1)
            h = torch.sqrt(2.0 * q_var) * locs[:, None, None] + q_mu

        else:
            h = q_mu

        return h, ws

    def _validate_neuron(self, neuron):
        """ """
        if neuron is None:
            neuron = np.arange(
                self.neurons
            )  # for spike coupling need to evaluate all units
        elif isinstance(neuron, list) is False and np.max(neuron) >= self.neurons:
            raise ValueError(
                "Accessing output dimensions beyond specified dimensions by model"
            )  # avoid illegal access
        return neuron

    def sample_rate(self, F_mu, F_var, trials, MC_samples=1):
        """
        returns: rate of shape (MC, trials, neurons, time)
        """
        with torch.no_grad():
            h = self.mc_gen(F_mu, F_var, MC_samples, list(range(F_mu.shape[1])))
            rate = self.f(h.view(-1, trials, *h.shape[1:]))
        return rate

    def nll(self, inner, inner_var, b, neuron):
        raise NotImplementedError

    def objective(self, F_mu, F_var, X, b, neuron, samples, mode):
        raise NotImplementedError

    def sample(self, rate, neuron=None, XZ=None):
        raise NotImplementedError