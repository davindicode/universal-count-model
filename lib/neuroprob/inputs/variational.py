import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

from .. import distributions as dist


class _variational(nn.Module):
    """ """

    def __init__(self, tensor_type, tsteps, dims):
        super().__init__()
        self.tensor_type = tensor_type
        self.dims = dims
        self.tsteps = tsteps

    def validate(self, tsteps, trials):
        raise NotImplementedError

    def eval_moments(self, t_lower, t_upper, net_input):
        raise NotImplementedError

    def sample(self, t_lower, t_upper, offs, samples, net_input):
        raise NotImplementedError


class IndNormal(_variational):
    """ """

    def __init__(
        self, mu, std, topo, dims, amortized=False, NF=None, tensor_type=torch.float
    ):
        super().__init__(tensor_type, mu.shape[0], dims)
        self.topo = topo
        self.lf = lambda x: F.softplus(x)
        self.lf_inv = lambda x: torch.where(x > 30, x, torch.log(torch.exp(x) - 1))

        ### variational ###
        self.amortized = amortized
        if topo == "torus":
            self.variational = dist.Tn_Normal
        elif topo == "sphere":
            if dims == 3:
                self.variational = dist.S2_VMF
            elif dims == 4:
                self.variational = dist.S3_Normal
        elif topo == "euclid":
            self.variational = dist.Rn_Normal
        else:
            raise NotImplementedError("Topology not supported.")

        if amortized:
            if (
                isinstance(v_mu, nn.Module) is False
                or isinstance(v_std, nn.Module) is False
            ):
                raise ValueError("Must provide a nn.Module object")
            self.add_module("mu", v_mu)
            self.add_module("finv_std", v_std)

        else:
            if len(mu.shape) > 2:
                self.trials = mu.shape[-1]
            else:
                self.trials = 1

            self.register_parameter("mu", Parameter(mu.type(self.tensor_type)))  # mean

            self.register_parameter(
                "finv_std", Parameter(self.lf_inv(std.type(self.tensor_type)))
            )  # std

            if NF is not None:
                if isinstance(NF, flows.FlowSequence) is False:
                    raise ValueError("Must provide a FlowSequence object")
                self.add_module("NF", NF.to(self.dummy.device))
            else:
                self.NF = None

    def validate(self, tsteps, trials):
        if self.amortized is False:
            if self.mu.shape[0] != tsteps:
                raise ValueError(
                    "Expected time steps do not match given initial latent standard deviations"
                )

    def sample(self, t_lower, t_upper, offs, samples, net_input):
        """ """
        mu, std = self.eval_moments(t_lower, t_upper, net_input)
        vd = self.variational(mu, std)  # .to_event()
        v_samp = vd((samples,))  # samples, time, event_dims
        # nlog_q = self.nlog_q(vd, v_samp, offs)
        vt = vd.log_prob(v_samp)[:, offs:]
        nlog_q = -vt.sum(axis=tuple(range(1, len(vt.shape))))

        if self.NF is not None:
            v_samp, log_jacob = self.NF(v_samp)
        else:
            log_jacob = 0

        return v_samp, nlog_q + log_jacob

    def eval_moments(self, t_lower, t_upper, net_input):
        if self.amortized:  # compute the variational parameters
            inp = net_input.t()[t_lower:t_upper, :].to(self.dummy.device)
            mu, std = self.mu(inp), self.lf(self.finv_std(inp))  # time, moment_dims
            if self.dims == 1:  # standard one dimensional arrays
                mu, std = mu[:, 0], std[:, 0]

        else:
            # (time, dims), (,trial) only if > 1
            # if self.trials == 1 and self.dims == 1: # standard one dimensional arrays, fast
            # else explicit event (and trial) dimension
            mu, std = self.mu[t_lower:t_upper, ...], self.lf(
                self.finv_std[t_lower:t_upper, ...]
            )  # time, (dims, trial)

        return mu, std


class Delta(_variational):
    """ """

    def __init__(self, mu, topo, dims, amortized=False, tensor_type=torch.float):
        super().__init__(tensor_type, mu.shape[0], dims)
        self.topo = topo

        ### variational ###
        self.amortized = amortized
        self.variational = dist.Delta

        if amortized:
            if isinstance(v_mu, nn.Module) is False:
                raise ValueError("Must provide a nn.Module object")
            self.add_module("v_mu", v_mu)

        else:
            if len(mu.shape) > 2:
                self.trials = mu.shape[-1]
            else:
                self.trials = 1

            self.register_parameter("mu", Parameter(mu.type(self.tensor_type)))  # mean

    def validate(self, tsteps, trials):
        if self.amortized is False:
            if self.mu.shape[0] != tsteps:
                raise ValueError(
                    "Expected time steps do not match given initial latent standard deviations"
                )

    def sample(self, t_lower, t_upper, offs, samples, net_input):
        return

    def eval_moments(self, t_lower, t_upper, net_input):
        if self.amortized:  # compute the variational parameters
            inp = net_input.t()[t_lower:t_upper, :].to(self.dummy.device)
            mu, std = self.mu(inp), self.lf(self.finv_std(inp))  # time, moment_dims
            if self.dims == 1:  # standard one dimensional arrays
                mu, std = mu[:, 0], std[:, 0]

        else:
            # (time, dims), (,trial) only if > 1
            # if self.trials == 1 and self.dims == 1: # standard one dimensional arrays, fast
            # else explicit event (and trial) dimension
            mu, std = self.mu[t_lower:t_upper, ...], self.lf(
                self.finv_std[t_lower:t_upper, ...]
            )  # time, (dims, trial)

        return mu, std


class Autoregressive(_variational):
    """ """

    def __init__(
        self,
        mu,
        std,
        topo,
        dims,
        latent_f="softplus",
        amortized=False,
        NF=None,
        tensor_type=torch.float,
    ):
        super().__init__(tensor_type, mu.shape[0], dims)
        self.topo = topo
