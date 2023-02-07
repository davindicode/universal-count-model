import math

import torch

import torch.distributions as dist
from torch.distributions import constraints

from .base import TorchDistribution


### distributions ###
class Rn_Normal(TorchDistribution, dist.Normal):
    r""" """

    def __init__(self, loc, scale, validate_args=None):
        super().__init__(loc, scale, validate_args=validate_args)

    def entropy(self, samples):
        r"""
        Exact
        """
        return 0.5 + 0.5 * math.log(2 * math.pi) + torch.log(self.scale)


class Rn_Uniform(TorchDistribution, dist.Uniform):
    r""" """

    def __init__(self, low, high, validate_args=None):
        r"""
        Assumes float low and high, i.e. event_shape=(,)
        """
        super().__init__(low, high, validate_args=validate_args)
        self.lprob = math.log(high - low)

    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)

        # assert not (value < self.low).any() and not (value > self.high).any()
        value.data[value < self.low] = self.low
        value.data[value > self.high] = self.high

        return torch.ones_like(value, device=value.device) * self.lprob

    def entropy(self, samples):
        r"""
        Exact, identical to super().entropy()
        """
        return samples.new_ones(samples.shape[1]) * math.log(self.high - self.low)



class Delta(TorchDistribution):
    r"""
    Delta distribution
    """
    arg_constraints = {}
    support = constraints.real
    has_rsample = True

    def __init__(self, loc, *args, validate_args=None):
        r"""
        std is a place holder for specifying the distribution moments
        """
        self.loc = loc  # used for sample shape, type and device
        batch_shape, event_shape = self.loc.shape[:-1], self.loc.shape[-1:]
        super(Torus_Uniform, self).__init__(
            batch_shape, event_shape, validate_args=validate_args
        )

        self.max_ent = math.log(math.pi) - torch.lgamma(2.0)

    def rsample(self, sample_shape=torch.Size()):
        shape = self._extended_shape(sample_shape)
        return self.loc.expand(shape)

    def log_prob(self, value):
        r"""
        Return no contribution to log probability sum
        """
        return 0

    def entropy(self, samples):
        r""" """
        return samples.new_zeros(*samples.shape[1:])


