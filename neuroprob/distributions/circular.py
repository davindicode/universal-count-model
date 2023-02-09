import math
from numbers import Number

import torch

import torch.distributions as dist
from torch.distributions import constraints
from torch.distributions.utils import _standard_normal, broadcast_all

from .base import TorchDistribution


class Tn_Normal(TorchDistribution):
    r"""
    Only need to implement rsample as it has priority over sample, which will copy it
    The forward pass of the base class is the sampling call
    Args:
        loc (float or Tensor): mean of the distribution (often referred to as mu)
        scale (float or Tensor): standard deviation of the distribution
            (often referred to as sigma)
    """
    arg_constraints = {"loc": constraints.real, "scale": constraints.positive}
    support = constraints.real
    has_rsample = True
    _mean_carrier_measure = 0

    @property
    def mean(self):
        return self.loc % (2 * math.pi)

    @property
    def stddev(self):
        return self.scale

    @property
    def variance(self):
        return self.stddev.pow(2)

    def __init__(self, loc, scale, validate_args=None, Ewald_terms=5):
        self.loc, self.scale = broadcast_all(loc, scale)
        if isinstance(loc, Number) and isinstance(scale, Number):
            batch_shape = torch.Size()
        else:
            batch_shape = self.loc.size()
        event_shape = torch.Size([1])
        # batch_shape, event_shape = self.loc.shape[:-1], self.loc.shape[-1:]
        super().__init__(batch_shape, validate_args=validate_args)
        self.ewald = torch.arange(-Ewald_terms, Ewald_terms + 1).to(self.loc.device)
        self.logTwoPi = math.log(2 * math.pi)

    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(Normal, _instance)
        batch_shape = torch.Size(batch_shape)
        new.loc = self.loc.expand(batch_shape)
        new.scale = self.scale.expand(batch_shape)
        super(Normal, new).__init__(batch_shape, validate_args=False)
        new._validate_args = self._validate_args
        return new

    def rsample(self, sample_shape=torch.Size()):
        shape = self._extended_shape(sample_shape)
        eps = _standard_normal(shape, dtype=self.loc.dtype, device=self.loc.device)
        return (self.loc + eps * self.scale) % (2 * math.pi)

    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)

        if isinstance(self.scale, Number):
            var = self.scale**2  # compute the variance
            log_scale = math.log(self.scale)
        else:  # add Ewald dimension
            var = (self.scale**2)[..., None]
            log_scale = torch.log(self.scale[..., None])

        cloc = self.loc % (2 * math.pi)
        loc = (
            cloc[..., None]
            + self.ewald.expand(*([1] * len(cloc.shape)), self.ewald.shape[0])
            * 2
            * math.pi
        )  # Ewald summation
        lprob = (
            -((value[..., None] - loc) ** 2) / (2 * var)
            - log_scale
            - 0.5 * self.logTwoPi
        )
        return torch.logsumexp(lprob, dim=-1)  # numerically stabilized

    def entropy(self, samples):
        r"""
        TODO: sum over event dimension as well, if it is there.

        Sample dimension is taken to be dim 0
        """
        max_ent = math.log(2 * math.pi)
        ent = -self.log_prob(samples).mean(0)
        ent[ent > max_ent] = max_ent
        return ent


class Tn_Uniform(TorchDistribution):
    r""" """
    arg_constraints = {}
    support = constraints.real
    has_rsample = True

    def __init__(self, loc, *args, validate_args=None):
        r"""
        :param torch.tensor loc: placeholder for determining the device
        """
        self.loc = loc
        batch_shape, event_shape = self.loc.shape[:-1], self.loc.shape[-1:]

        super().__init__(batch_shape, event_shape, validate_args=validate_args)
        self.logTwoPi = math.log(2 * math.pi)

    def rsample(self, sample_shape=torch.Size()):
        shape = self._extended_shape(sample_shape)
        return (
            torch.rand(shape, dtype=self.loc.dtype, device=self.loc.device)
            * 2
            * math.pi
        )

    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)

        return -torch.ones_like(value, device=self.loc.device) * self.logTwoPi
