import numbers

import numpy as np
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter

from .. import base


class custom_wrapper(base._input_mapping):
    """
    Custom base class for rate models in general.
    """

    def compute_F(self, XZ):
        raise NotImplementedError


class GLM(base._input_mapping):
    """
    GLM rate model.
    """

    def __init__(
        self,
        input_dim,
        out_dims,
        w_len,
        bias=False,
        tensor_type=torch.float,
        active_dims=None,
    ):
        """
        :param int input_dims: total number of active input dimensions
        :param int out_dims: number of output dimensions
        :param int w_len: number of dimensions for the weights
        """
        super().__init__(input_dim, out_dims, tensor_type, active_dims)

        self.register_parameter(
            "w", Parameter(torch.zeros((out_dims, w_len), dtype=self.tensor_type))
        )
        if bias:
            self.register_parameter(
                "bias", Parameter(torch.zeros((out_dims), dtype=self.tensor_type))
            )
        else:
            self.bias = 0

    def set_params(self, w=None, bias=None):
        if w is not None:
            self.w.data = w.type(self.tensor_type).to(self.dummy.device)
        if bias is not None:
            self.bias.data = bias.type(self.tensor_type).to(self.dummy.device)

    def compute_F(self, XZ):
        """
        Default linear mapping

        :param torch.Tensor cov: covariates with shape (samples, timesteps, dims)
        :returns: inner product with shape (samples, neurons, timesteps)
        :rtype: torch.tensor
        """
        XZ = self._XZ(XZ)
        return (XZ * self.w[None, :, None, :]).sum(-1) + self.bias[None, :, None], 0

    def sample_F(self, XZ):
        return self.compute_F(XZ)[0]


class FFNN(base._input_mapping):
    """
    Feedforward artificial neural network model
    """

    def __init__(
        self,
        input_dim,
        out_dims,
        mu_ANN,
        sigma_ANN=None,
        tensor_type=torch.float,
        active_dims=None,
    ):
        """
        :param nn.Module mu_ANN: ANN parameterizing the mean function mapping
        :param nn.Module sigma_ANN: ANN paramterizing the standard deviation mapping if stochastic
        """
        super().__init__(input_dim, out_dims, tensor_type, active_dims)

        self.add_module("mu_ANN", mu_ANN)
        if sigma_ANN is not None:
            self.add_module("sigma_ANN", sigma_ANN)
        else:
            self.sigma_ANN = None

    def compute_F(self, XZ):
        """
        The input to the ANN will be of shape (samples*timesteps, dims).

        :param torch.Tensor cov: covariates with shape (samples, timesteps, dims)
        :returns: inner product with shape (samples, neurons, timesteps)
        :rtype: torch.tensor
        """
        XZ = self._XZ(XZ)
        incov = XZ.view(-1, XZ.shape[-1])
        post_mu = self.mu_ANN(incov).view(*XZ.shape[:2], -1).permute(0, 2, 1)
        if self.sigma_ANN is not None:
            post_var = self.sigma_ANN(incov).view(*XZ.shape[:2], -1).permute(0, 2, 1)
        else:
            post_var = 0

        return post_mu, post_var

    def sample_F(self, XZ):
        self.compute_F(XZ)[0]
