import numbers

import numpy as np
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter

from . import base


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
