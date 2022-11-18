import math
import types

from numbers import Number

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

from tqdm.autonotebook import tqdm

from . import distributions
from .utils import signal

# import torch.autograd.profiler as profiler


# functions
def _expand_cov(cov):
    if len(cov.shape) == 1:  # expand arrays from (timesteps,)
        cov = cov[None, None, :, None]
    elif len(cov.shape) == 2:  # expand arrays (timesteps, dims)
        cov = cov[None, None, ...]
    elif len(cov.shape) == 3:
        cov = cov[None, ...]  # expand arrays (out, timesteps, dims)

    if len(cov.shape) != 4:  # trials, out, timesteps, dims
        raise ValueError(
            "Shape of input covariates at most trials x out x timesteps x dims"
        )

    return cov


# Link functions
_link_functions = {
    "exp": lambda x: torch.log(x),
    "softplus": lambda x: torch.where(
        x > 30, x, torch.log(torch.exp(x) - 1.0)
    ),  # | log(1+exp(30)) - 30 | < 1e-10, numerically stabilized
    "relu": lambda x: x,
    "identity": lambda x: x,
    "sigmoid": lambda x: torch.log(x) - torch.log(1.0 - x),
}

_inv_link_functions = {
    "exp": lambda x: torch.exp(x),
    "softplus": lambda x: F.softplus(x),
    "relu": lambda x: torch.clamp(x, min=0),
    "identity": lambda x: x,
    "sigmoid": lambda x: torch.sigmoid(x),
}

_ll_modes = ["MC", "GH"]


# GLM filters
class _filter(nn.Module):
    """
    GLM coupling filter base class.
    """

    def __init__(self, filter_len, conv_groups, tensor_type):
        """
        Filter length includes instantaneous part
        """
        super().__init__()
        self.conv_groups = conv_groups
        self.tensor_type = tensor_type
        if filter_len <= 0:
            raise ValueError("Filter length must be bigger than zero")
        self.filter_len = filter_len

    def forward(self):
        """
        Return filter values.
        """
        raise NotImplementedError

    def KL_prior(self, importance_weighted):
        """
        Prior of the filter model.
        """
        return 0

    def constrain(self):
        return


# observed
class _data_object(nn.Module):
    """
    Object that will take in data or generates data (leaf node objects in model graph).
    """

    def __init__(self):
        super().__init__()

    def setup_batching(self, batch_info, tsteps, trials):
        """
        :param int/list batch_info: batch size if scalar, else list of tuple (batch size, batch link), where the
                                 batch link is a boolean indicating continuity from previous batch
        :param int tsteps: total time steps of data to be batched
        :param int trials: total number of trials in the spike data
        """
        self.tsteps = tsteps
        self.trials = trials

        ### batching ###
        if type(batch_info) == list:  # not continuous data
            self.batches = len(batch_info)
            batch_size = [b["size"] for b in batch_info]
            batch_link = [b["linked"] for b in batch_info]
            batch_initial = [b["initial"] for b in batch_info]

        else:  # number
            batch_size = batch_info
            self.batches = int(np.ceil(self.tsteps / batch_size))
            n = self.batches - 1
            fin_bs = self.tsteps - n * batch_size
            batch_size = [batch_size] * n + [fin_bs]
            batch_link = [True] * self.batches
            batch_initial = [False] * self.batches

        batch_link[0] = False  # first batch
        batch_initial[0] = True
        batch_edge = list(np.cumsum(np.array([0] + batch_size)))
        self.batch_info = (batch_edge, batch_link, batch_initial)


# likelihoods
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
        if isinstance(q_var, Number):  # zero scalar
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
                h = distributions.Rn_Normal(q_mu, q_var.sqrt())()
            else:  # shape is (ll_samplesxcov_samples, neurons, time)
                h = distributions.Rn_Normal(q_mu, q_var.sqrt())((samples,)).view(
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
        if isinstance(q_var, Number) is False:  # not the zero scalar
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


# input objects
class _VI_object(nn.Module):
    """
    input VI objects
    """

    def __init__(self, dims, tensor_type):
        super().__init__()
        # self.register_buffer('dummy', torch.empty(0)) # keeping track of device
        self.dims = dims
        self.tensor_type = tensor_type

    def validate(self, tsteps, trials, batches):
        """ """
        raise NotImplementedError

    def sample(self, b, batch_info, samples, lv_input, importance_weighted):
        """
        :returns: tuple of (samples of :math:`q(z)`, KL terms)
        """
        raise NotImplementedError


### input output mapping models ###
class _input_mapping(nn.Module):
    """
    Input covariates to mean and covariance parameters. An input mapping consists of a mapping from input
    to inner and inner_var quantities.
    """

    def __init__(
        self,
        input_dims,
        out_dims,
        tensor_type=torch.float,
        active_dims=None,
        MC_only=False,
    ):
        """
        Constructor for the input mapping class.

        :param int neurons: the number of neurons or output dimensions of the model
        :param string inv_link: the name of the inverse link function
        :param int input_dims: the number of input dimensions in the covariates array of the model
        :param list VI_tuples: a list of variational inference tuples (prior, var_dist, topology, dims),
                               note that dims specifies the shape of the corresponding distribution and
                               is also the shape expected for regressors corresponding to this block as
                               (timesteps, dims). If dims=1, it is treated as a scalar hence X is then
                               of shape (timesteps,)
        """
        super().__init__()
        self.register_buffer("dummy", torch.empty(0))  # keeping track of device

        self.MC_only = MC_only  # default has VI-like output, not pure MC samples
        self.tensor_type = tensor_type
        self.out_dims = out_dims
        self.input_dims = input_dims

        if active_dims is None:
            active_dims = list(range(input_dims))
        elif len(active_dims) != input_dims:
            raise ValueError(
                "Active dimensions do not match expected number of input dimensions"
            )
        self.active_dims = active_dims  # dimensions to select from input covariates

        """if isinstance(inv_link, types.LambdaType):
            self.f = inv_link
            inv_link = 'custom'
        elif _inv_link_functions.get(inv_link) is None:
            raise NotImplementedError('Link function is not supported')
        else:
            self.f = _inv_link_functions[inv_link]
        self.inv_link = inv_link"""

    def compute_F(self, XZ):
        """
        Computes the diagonal posterior over :mathm:`F`, conditioned on the data. In most cases, this is amortized via
        learned weights/parameters of some approximate posterior. In ML/MAP settings the approximation is a
        delta distribution, meaning there is no variational uncertainty in the mapping.
        """
        raise NotImplementedError

    def sample_F(self, XZ, samples):
        """
        Samples from the full posterior (full covariance)

        Function for generating Gaussian MC samples. No MC samples are drawn when the variance is 0.

        :param torch.Tensor q_mu: the mean of the MC distribution
        :param torch.Tensor q_var: the (co)variance, type (univariate, multivariate) deduced
                                   from tensor shape
        :param int samples: number of MC samples per covariate sample
        :returns: the samples
        :rtype: torch.tensor
        """
        raise NotImplementedError

    def KL_prior(self, importance_weighted):
        """
        Prior on the model parameters as regularizer in the loss. Model parameters are integrated out
        approximately using the variational inference. This leads to Kullback-Leibler divergences in the
        overall objective, for MAP model parameters this reduces to the prior log probability.
        """
        return 0

    def constrain(self):
        """
        Constrain parameters in optimization.
        """
        return

    def _XZ(self, XZ):
        """
        Return XZ of shape (K, N, T, D)
        """
        if max(self.active_dims) >= XZ.shape[-1]:
            raise ValueError(
                "Active dimensions is outside input dimensionality provided"
            )
        return XZ[..., self.active_dims]

    def to_XZ(self, covariates, trials=1):
        """
        Convert covariates list input to tensors for input to mapping. Convenience function for rate
        evaluation functions and sampling functions.
        """
        cov_list = []
        timesteps = None
        out_dims = 1  # if all shared across output dimensions
        for cov_ in covariates:
            cov_ = _expand_cov(cov_.type(self.tensor_type))

            if cov_.shape[1] > 1:
                if out_dims == 1:
                    out_dims = cov_.shape[1]
                elif out_dims != cov_.shape[1]:
                    raise ValueError(
                        "Output dimensions in covariates dimensions are not consistent"
                    )

            if timesteps is not None:
                if timesteps != cov_.shape[2]:
                    raise ValueError(
                        "Time steps in covariates dimensions are not consistent"
                    )
            else:
                timesteps = cov_.shape[2]

            cov_list.append(cov_)

        XZ = (
            torch.cat(cov_list, dim=-1)
            .expand(trials, out_dims, timesteps, -1)
            .to(self.dummy.device)
        )
        return XZ
