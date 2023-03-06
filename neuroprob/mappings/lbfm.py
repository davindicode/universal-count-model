import numbers

import numpy as np
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter

from . import base



# linear algebra computations
def conditional_p_W_sites(site_locs, site_obs, site_cov):
    """
    Conditional distribution of weights on site parameters
    """
    return



class LBFM(base._input_mapping):
    """
    Linear basis function model
    """

    def __init__(
        self,
        input_dim,
        out_dims,
        w_dims,
        phi, 
        site_locs, 
        site_obs, 
        site_cov, 
        tensor_type=torch.float,
        active_dims=None,
    ):
        """
        :param int input_dims: total number of active input dimensions
        :param int out_dims: number of output dimensions
        :param int w_dim: number of dimensions for the weights
        :param torch.Tensor site_locs: site observation locations (out_dims, num_sites, in_dims)
        :param torch.Tensor site_obs: site observation values (out_dims, num_sites)
        :param torch.Tensor site_cov: site observation covariance (out_dims, num_sites, num_sites)
        """
        super().__init__(input_dim, out_dims, tensor_type, active_dims)

        self.register_parameter(
            "site_locs", Parameter(site_obs, dtype=self.tensor_type)
        )
        self.register_parameter(
            "site_obs", Parameter(site_obs, dtype=self.tensor_type)
        )
        self.register_parameter(
            "site_cov", Parameter(site_cov, dtype=self.tensor_type)
        )
        
        self.phi = phi  # maps (input_dims,) to (w_dims,)
        self.register_parameter(
            "w", Parameter(torch.zeros((out_dims, w_dims), dtype=self.tensor_type))
        )
        
    def constrain(self):
        if (
            self.u_scale_tril is not None
        ):  # constrain K to be PSD, L diagonals > 0 as needed for det K
            self.u_scale_tril.data = torch.tril(self.u_scale_tril.data)
            Nu = self.n_ind
            self.u_scale_tril.data[:, range(Nu), range(Nu)] = torch.clamp(
                self.u_scale_tril.data[:, range(Nu), range(Nu)], min=self.jitter
            )
            
    def KL_prior(self):
        return

    def compute_F(self, XZ):
        """
        Default linear mapping

        :param torch.Tensor cov: covariates with shape (samples, ts, dims)
        :returns: inner product with shape (samples, out_dims, ts)
        :rtype: torch.tensor
        """
        XZ = self._XZ(XZ)  # (samples, out_dims, ts, dims)
        phixz = self.phi(XZ)  # (samples, out_dims, ts, w_dims)
        
        self.site_obs 
        
        return (phixz * self.w[None, :, None, :]).sum(-1), 

    def sample_F(self, XZ, samples=1, eps=None):
        L = torch.linalg.cholesky(cov.double()).type(self.tensor_type)

        if samples > 1:  # expand
            XZ = XZ.repeat(samples, 1, 1, 1)
            L = L.repeat(samples)

        if eps is None:  # sample random vector
            eps = torch.randn(XZ.shape[:-1], dtype=self.tensor_type, device=cov.device)

        return loc + self.mean_function(XZ) + (L * eps[..., None, :]).sum(-1)

# class GLM(base._input_mapping):
#     """
#     generalized linear model
#     """

#     def __init__(
#         self,
#         input_dim,
#         out_dims,
#         w_len,
#         bias=False,
#         tensor_type=torch.float,
#         active_dims=None,
#     ):
#         """
#         :param int input_dims: total number of active input dimensions
#         :param int out_dims: number of output dimensions
#         :param int w_len: number of dimensions for the weights
#         """
#         super().__init__(input_dim, out_dims, tensor_type, active_dims)

#         self.register_parameter(
#             "w", Parameter(torch.zeros((out_dims, w_len), dtype=self.tensor_type))
#         )
#         if bias:
#             self.register_parameter(
#                 "bias", Parameter(torch.zeros((out_dims), dtype=self.tensor_type))
#             )
#         else:
#             self.bias = 0

#     def compute_F(self, XZ):
#         """
#         Linear mapping

#         :param torch.Tensor XZ: input covariates with shape (samples, ts, dims)
#         :returns:
#             inner product with shape (samples, out_dims, ts)
#         """
#         XZ = self._XZ(XZ)
#         return (XZ * self.w[None, :, None, :]).sum(-1) + self.bias[None, :, None], 0

#     def sample_F(self, XZ):
#         return self.compute_F(XZ)[0]
