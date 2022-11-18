import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

from .. import distributions as dist


class _prior(nn.Module):
    """ """

    def __init__(self, p, tensor_type, dims):
        super().__init__()
        self.tensor_type = tensor_type
        self.AR_p = p
        self.dims = dims

    def validate(self, tsteps, trials):
        raise NotImplementedError

    def log_p(self, x, initial):
        raise NotImplementedError


class IndNormal(_prior):
    """ """

    def __init__(
        self,
        mu,
        std,
        topo,
        dims,
        tensor_type=torch.float,
        learn_mu=False,
        learn_std=False,
    ):
        super().__init__(0, tensor_type, dims)
        self.topo = topo
        self.lf = lambda x: F.softplus(x)
        self.lf_inv = lambda x: torch.where(x > 30, x, torch.log(torch.exp(x) - 1))

        ### prior ###
        if topo == "torus":
            self.prior_dist = dist.Tn_Normal
        elif topo == "SO(3)":
            if dims != 4:
                raise ValueError("Input dimensionality of this topology must be 4")
            self.prior_dist = dist.SO3_Normal
        elif topo == "sphere":
            if dims == 3:
                self.prior_dist = dist.S2_VMF
            elif dims == 4:
                self.prior_dist = dist.S3_Normal
            else:
                raise NotImplementedError("{}-sphere not supported.".format(dims - 1))
        elif topo == "euclid":
            self.prior_dist = dist.Rn_Normal
        else:
            raise NotImplementedError("Topology not supported.")

        if learn_mu:
            self.register_parameter("mu", Parameter(mu.type(self.tensor_type)))
        else:
            self.register_buffer("mu", mu.type(self.tensor_type))

        if learn_std:
            self.register_parameter(
                "finv_std", Parameter(self.lf_inv(std.type(self.tensor_type)))
            )
        else:
            self.register_buffer("finv_std", self.lf_inv(std.type(self.tensor_type)))

    def validate(self, tsteps, trials):
        if self.mu.shape[0] != tsteps or self.finv_std.shape[0] != tsteps:
            raise ValueError(
                "Expected time steps do not match given initial latent means"
            )

    def log_p(self, x, initial):
        """ """
        pd = self.prior_dist(self.mu, self.lf(self.finv_std))
        pt = pd.log_prob(x)
        return pt.sum(axis=tuple(range(1, len(pt.shape))))


class IndUniform(_prior):
    """ """

    def __init__(self, topo, dims, tensor_type=torch.float):
        super().__init__(0, tensor_type, dims)
        self.topo = topo

        if topo == "torus":
            self.prior_dist = dist.Tn_Uniform
        elif topo == "SO(3)":
            if dims != 4:
                raise ValueError("Input dimensionality of this topology must be 4")
            self.prior_dist = dist.SO3_Uniform
        elif topo == "sphere":
            self.prior_dist = dist.Sn_Uniform
        elif topo == "euclid":
            self.prior_dist = dist.Rn_Uniform
        else:
            raise NotImplementedError("Topology not supported.")

        if topo != "euclid":  # specify bounds of uniform domain in euclid
            if dims > 1:
                prior = [torch.zeros(dims), torch.zeros(dims)]
            else:
                prior = [0.0, 0.0]

        self.register_buffer(
            "mu", prior[0].type(self.tensor_type)
        )  # lower limit, used from dim
        self.register_buffer("std", prior[1].type(self.tensor_type))  # upper limit

    def validate(self, tsteps, trials):
        return

    def log_p(self, x, initial):
        pd = self.prior_dist(self.mu, self.std)
        pt = pd.log_prob(v_samp)
        return pt.sum(axis=tuple(range(1, len(pt.shape))))


### Autoregressive priors ###
class ARNormal(_prior):
    """
    Defined by Markov
    """

    def __init__(self, transition, topo, dims, p, tensor_type=torch.float):
        super().__init__(p, tensor_type, dims)
        self.topo = topo

        if topo == "torus":
            self.rw_dist = dist.Tn_Normal

        elif topo == "sphere":
            if dims == 3:  # mean vector length
                self.rw_dist = dist.S2_VMF
            elif dims == 4:
                self.rw_dist = dist.S3_Normal
            else:
                raise NotImplementedError("{}-sphere not supported.".format(dims - 1))

        elif topo == "euclid":
            self.rw_dist = dist.Rn_Normal

        else:
            raise NotImplementedError("Topology not supported.")

        self.add_module("transition", transition)

    def validate(self, tsteps, trials):
        # if self.mu.shape[0] != tsteps or self.finv_std.shape[0] != tsteps:
        #    raise ValueError('Expected time steps do not match given initial latent means')
        return

    def ini_pd(self, ini_x):
        raise NotImplementedError

    def log_p(self, x, initial):
        loc, std = self.transition(x[:, :-1])
        rd = self.rw_dist(loc, std)

        rt = rd.log_prob(x[:, self.AR_p :])
        rw_term = rt.sum(axis=tuple(range(1, len(rt.shape))))

        if initial is False:
            return rw_term
        else:  # add log prob of first initial time step
            prior_term = self.ini_pd(x[:, : self.AR_p])
            return rw_term + prior_term  # sum over time (and dims if > 1)


class AR1(ARNormal):
    """
    Stationary AR(1)
    """

    def __init__(
        self, loc, std, dims, learn_loc=False, learn_std=True, tensor_type=torch.float
    ):
        class transition_(nn.Module):
            def __init__(self, loc, std, learn_loc, learn_std, tensor_type):
                super().__init__()
                self.lf = lambda x: F.softplus(x)
                self.lf_inv = lambda x: torch.where(
                    x > 30, x, torch.log(torch.exp(x) - 1)
                )

                if learn_loc:
                    self.register_parameter("loc", Parameter(loc.type(tensor_type)))
                else:
                    self.register_buffer("loc", loc.type(tensor_type))

                if learn_std:
                    self.register_parameter(
                        "finv_std", Parameter(self.lf_inv(std.type(tensor_type)))
                    )
                else:
                    self.register_buffer("finv_std", self.lf_inv(std.type(tensor_type)))

            def forward(self, x):
                loc = torch.sigmoid(self.loc)
                std = self.lf(self.finv_std)
                std_ = std * torch.sqrt(1 - loc**2)
                return loc * x, std_

        transition = transition_(loc, std, learn_loc, learn_std, tensor_type)
        super().__init__(transition, "euclid", dims, 1, tensor_type)

    def ini_pd(self, ini_x):
        std = self.transition.lf(self.transition.finv_std)
        pd = dist.Rn_Normal(0.0, std)
        pt = pd.log_prob(ini_x)
        prior_term = pt.sum(axis=tuple(range(1, len(pt.shape))))
        return prior_term


class ARp(ARNormal):
    """
    Euclidean:
        z_t = sum_{k=1}^p loc_k * z_{t-k} + std * w_t
    """

    def __init__(
        self, loc, std, learn_loc=False, learn_std=True, tensor_type=torch.float
    ):
        super().__init__(transition, "euclid", dims, p, tensor_type)


class dAR1(ARNormal):
    """
    Torus and sphere (define on tangent space):
        Delta(z_t) = z_{t+1} - z_t = sum_{k=1}^{p-1} loc_k * Delta(z_{t-k}) + std*w_t

    if self.topo == 'euclid': # stationary linear
        rd = self.rw_dist(loc, std)
    elif self.topo == 'torus': # drift DS
        rd = self.rw_dist(loc, std)
    elif self.topo == 'sphere':
        rd = self.rw_dist(loc, std) # loc = x[:, :-1, :]
    """

    def __init__(
        self,
        loc,
        std,
        topo,
        dims,
        learn_loc=False,
        learn_std=True,
        tensor_type=torch.float,
    ):
        class transition_(nn.Module):
            def __init__(self, loc, std, learn_loc, learn_std, tensor_type):
                super().__init__()
                self.lf = lambda x: F.softplus(x)
                self.lf_inv = lambda x: torch.where(
                    x > 30, x, torch.log(torch.exp(x) - 1)
                )

                if learn_loc:
                    self.register_parameter("loc", Parameter(loc.type(tensor_type)))
                else:
                    self.register_buffer("loc", loc.type(tensor_type))

                if learn_std:
                    self.register_parameter(
                        "finv_std", Parameter(self.lf_inv(std.type(tensor_type)))
                    )
                else:
                    self.register_buffer("finv_std", self.lf_inv(std.type(tensor_type)))

            def forward(self, x):
                std = self.lf(self.finv_std)
                return x + self.loc, std

        transition = transition_(loc, std, learn_loc, learn_std, tensor_type)
        super().__init__(transition, topo, dims, 1, tensor_type)

        if topo == "torus":
            self.prior_dist = dist.Tn_Uniform

        elif topo == "sphere":
            self.prior_dist = dist.Sn_Uniform

    def ini_pd(self, ini_x):
        pd = self.prior_dist(self.transition.loc)  # dummy tensor
        pt = pd.log_prob(ini_x)
        prior_term = pt.sum(axis=tuple(range(1, len(pt.shape))))
        return prior_term
