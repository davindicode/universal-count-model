import math

import torch
from torch.nn.parameter import Parameter

from ..base import safe_sqrt

from .base import Kernel


# stationary kernels
class Lengthscale(Kernel):
    """
    Base class for a family of covariance kernels which are functions of the
    distance :math:`|x-z|/l`, where :math:`l` is the length-scale parameter.

    :param torch.Tensor variance: variance parameter of shape (neurons)
    :param torch.Tensor lengthscale: length-scale parameter of shape (dims, neurons)
    """

    def __init__(
        self,
        input_dims,
        topology,
        lengthscale,
        track_dims=None,
        f="exp",
        tensor_type=torch.float,
    ):
        super().__init__(input_dims, track_dims, f, tensor_type)

        self.n = lengthscale.shape[1]  # separate lengthscale per output dimension
        assert (
            self.input_dims == lengthscale.shape[0]
        )  # lengthscale per input dimension

        if topology == "euclid":
            self.scaled_dist = self._scaled_dist
            self.square_scaled_dist = self._square_scaled_dist

        elif topology == "ring":
            self.scaled_dist = self._scaled_dist_ring
            self.square_scaled_dist = self._square_scaled_dist_ring

        else:
            raise NotImplementedError("Topology is not supported.")

        self._lengthscale = Parameter(
            self.lf_inv(lengthscale.type(tensor_type)).t()
        )  # N, D

    @property
    def lengthscale(self):
        return self.lf(self._lengthscale)[None, :, None, :]  # K, N, T, D

    @lengthscale.setter
    def lengthscale(self):
        self._lengthscale.data = self.lf_inv(lengthscale)

    def spectral_density(self):

        return

    ### metrics ###

    # Euclidean
    @staticmethod
    def _square_scaled_dist(lengthscale, X, Z):
        r"""
        Returns :math:`\|\frac{X-Z}{l}\|^2`.
        :param torch.Tensor X: input of shape (samples, neurons, points, dims)

        """
        scaled_X = X / lengthscale  # K, N, T, D
        scaled_Z = Z / lengthscale
        X2 = (scaled_X**2).sum(-1, keepdim=True)
        Z2 = (scaled_Z**2).sum(-1, keepdim=True)
        XZ = scaled_X.matmul(scaled_Z.permute(0, 1, 3, 2))
        r2 = X2 - 2 * XZ + Z2.permute(0, 1, 3, 2)
        return r2.clamp(min=0)

    @staticmethod
    def _scaled_dist(lengthscale, X, Z):
        r"""
        Returns :math:`\|\frac{X-Z}{l}\|`.
        """
        return safe_sqrt(Lengthscale._square_scaled_dist(lengthscale, X, Z))

    @staticmethod
    def _square_scaled_dist_ring(lengthscale, X, Z):
        r"""
        Returns :math:`\|\frac{X-Z}{l}\|^2`.
        """
        U = X[..., None, :] - Z[..., None, :].permute(0, 1, 3, 2, 4)

        return 2 * ((1 - torch.cos(U)) / (lengthscale[:, :, None, ...]) ** 2).sum(-1)

    @staticmethod
    def _scaled_dist_ring(lengthscale, X, Z):
        r"""
        Returns :math:`\|\frac{X-Z}{l}\|`.
        """
        return safe_sqrt(Lengthscale._square_scaled_dist_ring(lengthscale, X, Z))


class SquaredExponential(Lengthscale):
    r"""
    Implementation of Squared Exponential kernel:

        :math:`k(x,z) = \exp\left(-0.5 \times \frac{|x-z|^2}{l^2}\right).`

    .. note:: This kernel also has name `Squared Exponential` in literature.
    """

    def __init__(
        self,
        input_dims,
        lengthscale,
        track_dims=None,
        topology="euclid",
        f="exp",
        tensor_type=torch.float,
    ):
        super().__init__(input_dims, topology, lengthscale, track_dims, f, tensor_type)

    def forward(self, X, Z=None, diag=False):
        if diag:
            return X.new_ones(X.size(0), self.n, X.size(2))

        X, Z = self._XZ(X, Z)
        r2 = self.square_scaled_dist(self.lengthscale, X, Z)
        return torch.exp(-0.5 * r2)


class RationalQuadratic(Lengthscale):
    r"""
    Implementation of RationalQuadratic kernel:

        :math:`k(x, z) = \left(1 + 0.5 \times \frac{|x-z|^2}{\alpha l^2}
        \right)^{-\alpha}.`

    :param torch.Tensor scale_mixture: Scale mixture (:math:`\alpha`) parameter of this
        kernel. Should have size 1.
    """

    def __init__(
        self,
        input_dims,
        lengthscale,
        scale_mixture,
        track_dims=None,
        topology="euclid",
        f="exp",
        tensor_type=torch.float,
    ):
        super().__init__(input_dims, topology, lengthscale, track_dims, f, tensor_type)

        self._scale_mixture = Parameter(
            self.lf_inv(scale_mixture.type(tensor_type))
        )  # N

    @property
    def scale_mixture(self):
        return self.lf(self._scale_mixture)[None, :, None]  # K, N, T

    @scale_mixture.setter
    def scale_mixture(self):
        self._scale_mixture.data = self.lf_inv(scale_mixture)

    def forward(self, X, Z=None, diag=False):
        if diag:
            return X.new_ones(X.size(0), self.n, X.size(2))

        X, Z = self._XZ(X, Z)
        r2 = self.square_scaled_dist(self.lengthscale, X, Z)
        return (1 + (0.5 / self.scale_mixture) * r2).pow(-self.scale_mixture)


class Exponential(Lengthscale):
    r"""
    Implementation of Exponential kernel:

        :math:`k(x, z) = \sigma^2\exp\left(-\frac{|x-z|}{l}\right).`
    """

    def __init__(
        self,
        input_dims,
        lengthscale,
        track_dims=None,
        topology="euclid",
        f="exp",
        tensor_type=torch.float,
    ):
        super().__init__(input_dims, topology, lengthscale, track_dims, f, tensor_type)

    def forward(self, X, Z=None, diag=False):
        if diag:
            return X.new_ones(X.size(0), self.n, X.size(2))

        X, Z = self._XZ(X, Z)
        r = self.scaled_dist(self.lengthscale, X, Z)
        return torch.exp(-r)


class Matern32(Lengthscale):
    r"""
    Implementation of Matern32 kernel:

        :math:`k(x, z) = \sigma^2\left(1 + \sqrt{3} \times \frac{|x-z|}{l}\right)
        \exp\left(-\sqrt{3} \times \frac{|x-z|}{l}\right).`
    """

    def __init__(
        self,
        input_dims,
        lengthscale,
        track_dims=None,
        topology="euclid",
        f="exp",
        tensor_type=torch.float,
    ):
        super().__init__(input_dims, topology, lengthscale, track_dims, f, tensor_type)

    def forward(self, X, Z=None, diag=False):
        if diag:
            return X.new_ones(X.size(0), self.n, X.size(2))

        X, Z = self._XZ(X, Z)
        r = self.scaled_dist(self.lengthscale, X, Z)
        sqrt3_r = 3**0.5 * r
        return (1 + sqrt3_r) * torch.exp(-sqrt3_r)


class Matern52(Lengthscale):
    r"""
    Implementation of Matern52 kernel:

        :math:`k(x,z)=\sigma^2\left(1+\sqrt{5}\times\frac{|x-z|}{l}+\frac{5}{3}\times
        \frac{|x-z|^2}{l^2}\right)\exp\left(-\sqrt{5} \times \frac{|x-z|}{l}\right).`
    """

    def __init__(
        self,
        input_dims,
        lengthscale,
        track_dims=None,
        topology="euclid",
        f="exp",
        tensor_type=torch.float,
    ):
        super().__init__(input_dims, topology, lengthscale, track_dims, f, tensor_type)

    def forward(self, X, Z=None, diag=False):
        if diag:
            return X.new_ones(X.size(0), self.n, X.size(2))

        X, Z = self._XZ(X, Z)
        r2 = self.square_scaled_dist(self.lengthscale, X, Z)
        r = safe_sqrt(r2)
        sqrt5_r = 5**0.5 * r
        return (1 + sqrt5_r + (5 / 3) * r2) * torch.exp(-sqrt5_r)
