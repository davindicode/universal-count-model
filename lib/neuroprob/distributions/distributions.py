import math
from numbers import Number

import torch

import torch.distributions as dist
from torch.distributions import constraints
from torch.distributions.distribution import Distribution
from torch.distributions.utils import _standard_normal, broadcast_all, lazy_property


class TorchDistribution(Distribution):
    """
    Module to interface with PyTorch distributions. (Similar to Pyro distribution mixin)

    You should instead use `TorchDistribution` for new distribution classes.

    This is mainly useful for wrapping existing PyTorch distributions.
    Derived classes must first inherit from
    :class:`torch.distributions.distribution.Distribution` and then inherit
    from :class:`TorchDistributionMixin`.
    """

    def __call__(self, sample_shape=torch.Size()):
        """
        Samples a random value.

        This is reparameterized whenever possible, calling
        :meth:`~torch.distributions.distribution.Distribution.rsample` for
        reparameterized distributions and
        :meth:`~torch.distributions.distribution.Distribution.sample` for
        non-reparameterized distributions.

        :param sample_shape: the size of the iid batch to be drawn from the
            distribution.
        :type sample_shape: torch.Size
        :return: A random value or batch of random values (if parameters are
            batched). The shape of the result should be `self.shape()`.
        :rtype: torch.Tensor
        """
        return (
            self.rsample(sample_shape)
            if self.has_rsample
            else self.sample(sample_shape)
        )

    @property
    def event_dim(self):
        """
        :return: Number of dimensions of individual events.
        :rtype: int
        """
        return len(self.event_shape)

    def shape(self, sample_shape=torch.Size()):
        """
        The tensor shape of samples from this distribution.

        Samples are of shape::

          d.shape(sample_shape) == sample_shape + d.batch_shape + d.event_shape

        :param sample_shape: the size of the iid batch to be drawn from the
            distribution.
        :type sample_shape: torch.Size
        :return: Tensor shape of samples.
        :rtype: torch.Size
        """
        return sample_shape + self.batch_shape + self.event_shape


### Distributions ###
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


class Rn_MVN(TorchDistribution, dist.MultivariateNormal):
    pass


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


class Tn_MVN(TorchDistribution):  # Not scalable as ReLie sum grows exponentially
    """
    Args:
        loc (Tensor): mean of the distribution
        covariance_matrix (Tensor): positive-definite covariance matrix
        precision_matrix (Tensor): positive-definite precision matrix
        scale_tril (Tensor): lower-triangular factor of covariance, with positive-valued diagonal

    Note:
        Only one of :attr:`covariance_matrix` or :attr:`precision_matrix` or
        :attr:`scale_tril` can be specified.

        Using :attr:`scale_tril` will be more efficient: all computations internally
        are based on :attr:`scale_tril`. If :attr:`covariance_matrix` or
        :attr:`precision_matrix` is passed instead, it is only used to compute
        the corresponding lower triangular matrices using a Cholesky decomposition.
    """

    arg_constraints = {
        "loc": constraints.real_vector,
        "covariance_matrix": constraints.positive_definite,
        "precision_matrix": constraints.positive_definite,
        "scale_tril": constraints.lower_cholesky,
    }
    support = constraints.real
    has_rsample = True

    def __init__(
        self,
        loc,
        covariance_matrix=None,
        precision_matrix=None,
        scale_tril=None,
        validate_args=None,
    ):
        if loc.dim() < 1:
            raise ValueError("loc must be at least one-dimensional.")
        if (covariance_matrix is not None) + (scale_tril is not None) + (
            precision_matrix is not None
        ) != 1:
            raise ValueError(
                "Exactly one of covariance_matrix or precision_matrix or scale_tril may be specified."
            )

        loc_ = loc.unsqueeze(-1)  # temporarily add dim on right
        if scale_tril is not None:
            if scale_tril.dim() < 2:
                raise ValueError(
                    "scale_tril matrix must be at least two-dimensional, "
                    "with optional leading batch dimensions"
                )
            self.scale_tril, loc_ = torch.broadcast_tensors(scale_tril, loc_)
        elif covariance_matrix is not None:
            if covariance_matrix.dim() < 2:
                raise ValueError(
                    "covariance_matrix must be at least two-dimensional, "
                    "with optional leading batch dimensions"
                )
            self.covariance_matrix, loc_ = torch.broadcast_tensors(
                covariance_matrix, loc_
            )
        else:
            if precision_matrix.dim() < 2:
                raise ValueError(
                    "precision_matrix must be at least two-dimensional, "
                    "with optional leading batch dimensions"
                )
            self.precision_matrix, loc_ = torch.broadcast_tensors(
                precision_matrix, loc_
            )
        self.loc = loc_[..., 0]  # drop rightmost dim

        batch_shape, event_shape = self.loc.shape[:-1], self.loc.shape[-1:]
        super(Torus_Normal, self).__init__(
            batch_shape, event_shape, validate_args=validate_args
        )

        if scale_tril is not None:
            self._unbroadcasted_scale_tril = scale_tril
        elif covariance_matrix is not None:
            self._unbroadcasted_scale_tril = torch.cholesky(covariance_matrix)
        else:  # precision_matrix is not None
            self._unbroadcasted_scale_tril = _precision_to_scale_tril(precision_matrix)

    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(MultivariateNormal, _instance)
        batch_shape = torch.Size(batch_shape)
        loc_shape = batch_shape + self.event_shape
        cov_shape = batch_shape + self.event_shape + self.event_shape
        new.loc = self.loc.expand(loc_shape)
        new._unbroadcasted_scale_tril = self._unbroadcasted_scale_tril
        if "covariance_matrix" in self.__dict__:
            new.covariance_matrix = self.covariance_matrix.expand(cov_shape)
        if "scale_tril" in self.__dict__:
            new.scale_tril = self.scale_tril.expand(cov_shape)
        if "precision_matrix" in self.__dict__:
            new.precision_matrix = self.precision_matrix.expand(cov_shape)
        super(MultivariateNormal, new).__init__(
            batch_shape, self.event_shape, validate_args=False
        )
        new._validate_args = self._validate_args
        return new

    @lazy_property
    def scale_tril(self):
        return self._unbroadcasted_scale_tril.expand(
            self._batch_shape + self._event_shape + self._event_shape
        )

    @lazy_property
    def covariance_matrix(self):
        return torch.matmul(
            self._unbroadcasted_scale_tril,
            self._unbroadcasted_scale_tril.transpose(-1, -2),
        ).expand(self._batch_shape + self._event_shape + self._event_shape)

    @lazy_property
    def precision_matrix(self):
        identity = torch.eye(
            self.loc.size(-1), device=self.loc.device, dtype=self.loc.dtype
        )
        # TODO: use cholesky_inverse when its batching is supported
        return torch.cholesky_solve(identity, self._unbroadcasted_scale_tril).expand(
            self._batch_shape + self._event_shape + self._event_shape
        )

    @property
    def mean(self):
        return self.loc

    @property
    def variance(self):
        return (
            self._unbroadcasted_scale_tril.pow(2)
            .sum(-1)
            .expand(self._batch_shape + self._event_shape)
        )

    def rsample(self, sample_shape=torch.Size()):
        shape = self._extended_shape(sample_shape)
        eps = _standard_normal(shape, dtype=self.loc.dtype, device=self.loc.device)
        return self.loc + _batch_mv(self._unbroadcasted_scale_tril, eps)

    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)
        diff = value - self.loc
        M = _batch_mahalanobis(self._unbroadcasted_scale_tril, diff)
        half_log_det = (
            self._unbroadcasted_scale_tril.diagonal(dim1=-2, dim2=-1).log().sum(-1)
        )
        return -0.5 * (self._event_shape[0] * math.log(2 * math.pi) + M) - half_log_det


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


class S2_VMF(TorchDistribution):
    r"""
    Spherical von Mises distribution.
    This implementation combines the direction parameter and concentration
    parameter into a single combined parameter that contains both direction and
    magnitude. The ``value`` arg is represented in cartesian coordinates: it
    must be a normalized 3-vector that lies on the 2-sphere.
    See :class:`~pyro.distributions.VonMises` for a 2D polar coordinate cousin
    of this distribution.
    Currently only :meth:`log_prob` is implemented.
    :param torch.Tensor concentration: A combined location-and-concentration
        vector. The direction of this vector is the location, and its
        magnitude is the concentration.

    concentration is 1/scale**2
    """
    arg_constraints = {"scale": constraints.real}
    support = constraints.real  # TODO implement constraints.sphere or similar
    has_rsample = True

    def __init__(self, loc, scale, *args, validate_args=None):
        if loc.dim() < 1 or loc.shape[-1] != 3:
            raise ValueError(
                "Expected concentration to have rightmost dim 3, actual shape = {}".format(
                    loc.shape
                )
            )
        if not (torch.abs(loc.norm(2, -1) - 1) < 1e-6).all():
            raise ValueError(
                "direction vectors are not normalized, {:.2e}".format(
                    (loc.norm(2, -1) - 1).max()
                )
            )

        self.loc, scale = broadcast_all(loc, scale)
        self.scale = scale[..., 0]
        batch_shape, event_shape = self.loc.shape[:-1], self.loc.shape[-1:]
        super().__init__(batch_shape, event_shape, validate_args=validate_args)
        self.logTwoPi = math.log(2 * math.pi)

    def log_prob(self, value):
        if self._validate_args:
            if value.dim() < 1 or value.shape[-1] != 3:
                raise ValueError(
                    "Expected value to have rightmost dim 3, actual shape = {}".format(
                        value.shape
                    )
                )

            vn = value.data.norm(2, -1)
            if not (torch.abs(vn - 1) < 1e-6).all():
                value = value / vn[..., None]
                # raise ValueError('direction vectors are not normalized, {:.2e}'.format((value.norm(2, -1)-1).max()))

        concentration = 1 / self.scale**2
        log_normalizer = (
            concentration.log()
            - torch.log(1 - torch.exp(-2 * concentration))
            - self.logTwoPi
        )
        return (
            concentration * (self.loc * value).sum(-1) - concentration + log_normalizer
        )

    def expand(self, batch_shape):
        try:
            return super().expand(batch_shape)
        except NotImplementedError:
            validate_args = self.__dict__.get("_validate_args")
            loc = self.loc.expand(torch.Size(batch_shape) + (3,))
            scale = self.scale.expand(torch.Size(batch_shape))
            return type(self)(loc, concentration, validate_args=validate_args)

    def rsample(self, sample_shape=torch.Size()):
        r"""
        Rotates mean vector with rotation formula.
        """
        shape = self._extended_shape(sample_shape)
        u = torch.rand((*shape[:-1], 1), dtype=self.loc.dtype, device=self.loc.device)
        eps = _standard_normal(
            (*shape[:-1], 2), dtype=self.loc.dtype, device=self.loc.device
        )
        concentration = 1 / self.scale**2
        concentration = concentration.expand(sample_shape + concentration.shape)[
            ..., None
        ]

        W = 1 + 1.0 / concentration * torch.log(
            u + (1 - u) * torch.exp(-2 * concentration)
        )
        V = eps / eps.norm(2, -1)[..., None]
        x_ = torch.cat((V * torch.sqrt(1 - W**2), W), dim=-1)
        mu = self.loc

        R = torch.empty(
            *shape, shape[-1], dtype=self.scale.dtype, device=self.scale.device
        )
        offd = -mu[..., 0] * mu[..., 1] / (1 + mu[..., 2])
        ond_x = mu[..., 0] ** 2 / (1 + mu[..., 2])
        ond_y = mu[..., 1] ** 2 / (1 + mu[..., 2])
        R[..., 0, 0] = 1 - ond_x
        R[..., 0, 1] = offd
        R[..., 0, 2] = mu[..., 0]
        R[..., 1, 0] = offd
        R[..., 1, 1] = 1 - ond_y
        R[..., 1, 2] = mu[..., 1]
        R[..., 2, 0] = -mu[..., 0]
        R[..., 2, 1] = -mu[..., 1]
        R[..., 2, 2] = 1 - ond_x - ond_y

        x = torch.einsum("...ij,...j->...i", R, x_)
        x = x / x.norm(2, -1)[..., None]  # numerical normalization
        return x

    def entropy(self, samples):
        """
        Exact expression, numerically stable.

        :param torch.tensor samples: input samples of shape (samples, timestep, event_dims)
        """
        concentration = 1 / self.scale**2
        return samples.new_ones(*samples.shape[1:]) * (
            1
            - concentration * (1 + 1 / torch.tanh(concentration))
            + torch.log(1 - torch.exp(-2 * concentration))
            - torch.log(concentration)
            + self.logTwoPi
        )


class Sn_Uniform(TorchDistribution):
    """ """

    arg_constraints = {}
    support = constraints.real
    has_rsample = True

    def __init__(self, loc, *args, validate_args=None):
        r"""
        :param torch.tensor loc: placeholder
        """
        self.loc = loc  # used for sample shape, type and device
        batch_shape, event_shape = self.loc.shape[:-1], self.loc.shape[-1:]
        self.n = self.loc.shape[-1] - 1

        super().__init__(batch_shape, event_shape, validate_args=validate_args)
        self.logPi = math.log(math.pi)

    def rsample(self, sample_shape=torch.Size()):
        shape = self._extended_shape(sample_shape)
        R_ = torch.randn(shape, dtype=self.loc.dtype, device=self.loc.device)
        R2 = R_.norm(2, -1)[..., None]
        return R_ / R2

    def log_prob(self, value):
        """
        :param torch.tensor value: input samples of shape (samples, timestep, event_dims)
        """
        assert value.shape[-1] == self.n + 1
        return -torch.ones_like(value[0, :, 0], device=self.loc.device) * (
            (self.n / 2.0) * self.logPi - math.lgamma(self.n / 2.0 + 1.0)
        )

    def entropy(self, samples):
        """
        Exact expression, numerically stable.

        :param torch.tensor samples: input samples of shape (samples, timestep, event_dims)
        :returns: entropy of shape (timesteps,)
        :rtype: torch.tensor
        """
        assert samples.shape[-1] == self.n + 1
        return samples.new_ones(samples.shape[1]) * (
            (self.n / 2.0) * self.logPi - math.lgamma(self.n / 2.0 + 1.0)
        )


class Sn_Normal(TorchDistribution):
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
        """
        :param torch.tensor value: input samples of shape (sample, time/batch, event)
        :returns: log probability of shape (time/batch,)
        :rtype: torch.tensor
        """
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
        Sample dimension is taken to be dim 0
        """
        max_ent = math.log(2 * math.pi)
        ent = -self.log_prob(samples).mean(0)
        ent[ent > max_ent] = max_ent
        return ent


class SO3_Uniform(TorchDistribution):
    """ """

    arg_constraints = {}
    support = constraints.real
    has_rsample = True

    def __init__(self, loc, *args, validate_args=None):
        r"""
        :param torch.tensor loc: placeholder
        """
        self.loc = loc  # used for sample shape, type and device
        batch_shape, event_shape = self.loc.shape[:-1], self.loc.shape[-1:]
        assert self.loc.shape[-1] == 4

        super().__init__(batch_shape, event_shape, validate_args=validate_args)
        self.logPi = math.log(math.pi)

    def rsample(self, sample_shape=torch.Size()):
        shape = self._extended_shape(sample_shape)
        R_ = torch.randn(shape, dtype=self.loc.dtype, device=self.loc.device)
        R2 = R_.norm(2, -1)[..., None]
        return R_ / R2

    def log_prob(self, value):
        assert value.shape[-1] == 4
        return -torch.ones_like(value[..., 0], device=self.loc.device) * (
            (3.0 / 2.0) * self.logPi - math.lgamma(5 / 2.0) - math.log(2.0)
        )

    def entropy(self, value):
        assert value.shape[-1] == 4
        return value.new_ones(value.shape[1]) * (
            (3.0 / 2.0) * self.logPi - math.lgamma(5 / 2.0) - math.log(2.0)
        )


class SO3_Normal(TorchDistribution):
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
        Sample dimension is taken to be dim 0
        """
        max_ent = math.log(2 * math.pi)
        ent = -self.log_prob(samples).mean(0)
        ent[ent > max_ent] = max_ent
        return ent


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


class Multinomial(TorchDistribution, dist.Multinomial):
    """
    Multi-class binomial distribution, i.e. sum of Categorical distribution draws.
    """

    def __init__(self, total_count=1, probs=None, logits=None, validate_args=None):
        super().__init__(total_count, probs, logits, validate_args)


class Categorical(TorchDistribution, dist.Categorical):
    """
    Bernoulli equivalent over multi-class variables.
    """

    has_rsample = True

    def __init__(self, probs=None, logits=None, validate_args=None):
        super().__init__(probs=probs, logits=logits, validate_args=validate_args)

    def rsample(self, sample_shape):
        """
        Sample from the discrete distribution with probabilities :math:`\alpha` using the
            Gumbel-softmax distribution at zero temperature.

        :param numpy.array alpha: Probabilities of the discrete target distribution. The
            shape of the array dimensions is (samples, index) with index size :math:`n`.
        :returns: Samples from the Gumbel-softmax indexed from :math:`0` to :math:`n-1`
        :rtype: numpy.array
        """
        m = np.log(alpha) + np.random.gumbel(size=(alpha.shape))

        if torch.isinf(self.beta):
            return np.argmax(m, axis=1)
        else:  # finite temperature softmax
            return
