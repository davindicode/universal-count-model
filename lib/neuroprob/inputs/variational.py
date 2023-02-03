import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

from .base import _variational

from .. import distributions as dist



class IndNormal(_variational):
    """
    Independent over time steps, normal variational distribution
    """

    def __init__(
        self, mu, std, topo, dims, tensor_type=torch.float
    ):
        super().__init__(tensor_type, mu.shape[0], dims)
        self.topo = topo
        self.lf = lambda x: F.softplus(x)
        self.lf_inv = lambda x: torch.where(x > 30, x, torch.log(torch.exp(x) - 1))

        ### variational ###
        if topo == "torus":
            self.variational = dist.Tn_Normal
        
        elif topo == "euclid":
            self.variational = dist.Rn_Normal
            
        else:
            raise NotImplementedError("Topology not supported.")

        if len(mu.shape) > 2:
            self.trials = mu.shape[-1]
        else:
            self.trials = 1

        self.register_parameter("mu", Parameter(mu.type(self.tensor_type)))  # mean

        self.register_parameter(
            "finv_std", Parameter(self.lf_inv(std.type(self.tensor_type)))
        )  # std

    def sample(self, t_lower, t_upper, offs, samples, net_input):
        """ """
        mu, std = self.eval_moments(t_lower, t_upper, net_input)
        vd = self.variational(mu, std)  # .to_event()
        v_samp = vd((samples,))  # samples, time, event_dims
        # nlog_q = self.nlog_q(vd, v_samp, offs)
        vt = vd.log_prob(v_samp)[:, offs:]
        nlog_q = -vt.sum(axis=tuple(range(1, len(vt.shape))))

        return v_samp, nlog_q

    def eval_moments(self, t_lower, t_upper, net_input):
        # (time, dims), (,trial) only if > 1
        # if self.trials == 1 and self.dims == 1: # standard one dimensional arrays, fast
        # else explicit event (and trial) dimension
        mu, std = self.mu[t_lower:t_upper, ...], self.lf(
            self.finv_std[t_lower:t_upper, ...]
        )  # time, (dims, trial)

        return mu, std


class Delta(_variational):
    """
    Independent over time steps, normal variational distribution
    """

    def __init__(self, mu, topo, dims, tensor_type=torch.float):
        super().__init__(tensor_type, mu.shape[0], dims)
        self.topo = topo

        ### variational ###
        self.variational = dist.Delta

        if len(mu.shape) > 2:
            self.trials = mu.shape[-1]
        else:
            self.trials = 1

        self.register_parameter("mu", Parameter(mu.type(self.tensor_type)))  # mean

    def sample(self, t_lower, t_upper, offs, samples, net_input):
        return

    def eval_moments(self, t_lower, t_upper, net_input):
        # (time, dims), (,trial) only if > 1
        # if self.trials == 1 and self.dims == 1: # standard one dimensional arrays, fast
        # else explicit event (and trial) dimension
        mu, std = self.mu[t_lower:t_upper, ...], self.lf(
            self.finv_std[t_lower:t_upper, ...]
        )  # time, (dims, trial)

        return mu, std