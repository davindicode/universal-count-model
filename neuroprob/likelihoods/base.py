import numbers
import types

import numpy as np

import torch

from ..base import _data_object, _inv_link_functions, _link_functions
from ..distributions import Rn_Normal


_ll_modes = ["MC", "GH"]


def mc_gen(q_mu, q_var, samples, out_inds):
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
    q_mu = q_mu[:, out_inds, :]
    if isinstance(q_var, numbers.Number):  # zero scalar
        h = (
            q_mu[None, ...].expand(samples, *q_mu.shape).reshape(-1, *q_mu.shape[1:])
            if samples > 1
            else q_mu
        )

    else:  # len(q_var.shape) == 3
        q_var = q_var[:, out_inds, :]
        if samples == 1:  # no expanding
            h = Rn_Normal(q_mu, q_var.sqrt())()
        else:  # shape is (ll_samplesxcov_samples, out_dims, time)
            h = Rn_Normal(q_mu, q_var.sqrt())((samples,)).view(-1, *q_mu.shape[1:])

    return h


def gh_gen(q_mu, q_var, points, output_inds):
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
        torch.tensor(locs, dtype=q_mu.dtype)
        .repeat_interleave(q_mu.shape[0], 0)
        .to(q_mu.device)
    )
    ws = (
        torch.tensor(
            1
            / np.sqrt(np.pi)
            * ws
            / q_mu.shape[0],  # q_mu shape 0 is cov_samplesxtrials
            dtype=q_mu.dtype,
        )
        .repeat_interleave(q_mu.shape[0], 0)
        .to(q_mu.device)
    )

    q_mu = q_mu[:, output_inds, :].repeat(
        points, 1, 1
    )  # fill sample dimension with GH quadrature points
    if isinstance(q_var, numbers.Number) is False:  # not the zero scalar
        q_var = q_var[:, output_inds, :].repeat(points, 1, 1)
        h = torch.sqrt(2.0 * q_var) * locs[:, None, None] + q_mu

    else:
        h = q_mu

    return h, ws


class _likelihood(_data_object):
    """
    Likelihood base class.
    """

    def __init__(
        self, tbin, F_dims, out_dims, inv_link, tensor_type=torch.float, mode="MC"
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
        self.out_dims = out_dims
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

    def set_Y(self, observations, batch_info):
        """
        Get all the activity into batches useable format for fast log-likelihood evaluation.
        Batched spikes will be a list of tensors of shape (trials, out_dims, time) with trials
        set to 1 if input has no trial dimension (e.g. continuous recording).

        :param np.ndarray observations: observations shape (out_dims, ts) or (trials, out_dims, ts)
        :param int/list batch_info: batch size if scalar, else list of tuple (batch size, batch link), where the
                                    batch link is a boolean indicating continuity from previous batch
        """
        if self.out_dims != observations.shape[-2]:
            raise ValueError(
                "Observations array does not match output dimensions of likelihood"
            )

        if len(observations.shape) == 2:  # add in trial dimension
            observations = observations[None, ...]

        self.setup_batching(batch_info, observations.shape[-1], observations.shape[0])
        self.all_spikes = observations.type(self.tensor_type)

    def KL_prior(self):
        return 0

    def constrain(self):
        """
        Constrain parameters in optimization
        """
        return

    def _validate_out_inds(self, out_inds):
        """
        Check if accessed out_dims are within valid array bounds
        """
        if out_inds is None:
            out_inds = np.arange(
                self.out_dims
            )  # for spike coupling need to evaluate all units
        elif isinstance(out_inds, list) is False and np.max(out_inds) >= self.out_dims:
            raise ValueError(
                "Accessing output dimensions beyond specified dimensions by model"
            )  # avoid illegal access
        return out_inds

    def nll(self, inner, inner_var, b, out_inds):
        raise NotImplementedError

    def objective(self, F_mu, F_var, X, b, out_inds, samples, mode):
        raise NotImplementedError

    def sample(self, rate, out_inds=None, XZ=None, rng=None):
        raise NotImplementedError
