import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

from ..utils.signal import eye_like


# utils
def _torch_sqrt(x, eps=1e-12):
    """
    A convenient function to avoid the NaN gradient issue of :func:`torch.sqrt`
    at 0.
    """
    # Ref: https://github.com/pytorch/pytorch/issues/2421
    return (x + eps).sqrt()


class Kernel(nn.Module):
    """
    Base class for multi-lengthscale kernels used in Gaussian Processes.
    Inspired by pyro GP kernels.
    """

    def __init__(self, input_dims, track_dims=None, f="exp", tensor_type=torch.float):
        """
        :param int input_dims: number of input dimensions
        :param list track_dims: list of dimension indices used as input from X
        :param string f: inverse link function for positive constraints in hyperparameters
        """
        super().__init__()
        self.tensor_type = tensor_type

        if track_dims is None:
            track_dims = list(range(input_dims))
        elif input_dims != len(track_dims):
            raise ValueError(
                "Input size and the length of active dimensionals should be equal."
            )
        self.input_dims = input_dims
        self.track_dims = track_dims

        if f == "exp":
            self.lf = lambda x: torch.exp(x)
            self.lf_inv = lambda x: torch.log(x)
        elif f == "softplus":
            self.lf = lambda x: F.softplus(x)
            self.lf_inv = lambda x: torch.where(x > 30, x, torch.log(torch.exp(x) - 1))
        elif f == "relu":
            self.lf = lambda x: torch.clamp(x, min=0)
            self.lf_inv = lambda x: x
        else:
            raise NotImplementedError("Link function is not supported.")

    def forward(self, X, Z=None, diag=False):
        """
        Calculates covariance matrix of inputs on active dimensionals.
        """
        raise NotImplementedError

    def _slice_input(self, X):
        r"""
        Slices :math:`X` according to ``self.track_dims``.

        :param torch.Tensor X: input with shape (samples, neurons, timesteps, dimensions)
        :returns: a 2D slice of :math:`X`
        :rtype: torch.tensor
        """
        if X.dim() == 4:
            return X[..., self.track_dims]
        else:
            raise ValueError("Input X must be of shape (K x N x T x D).")

    def _XZ(self, X, Z=None):
        """
        Slice the input of :math:`X` and :math:`Z` into correct shapes.
        """
        if Z is None:
            Z = X

        X = self._slice_input(X)
        Z = self._slice_input(Z)
        if X.size(-1) != Z.size(-1):
            raise ValueError("Inputs must have the same number of features.")

        return X, Z


# kernel combinations
class Combination(Kernel):
    """
    Base class for kernels derived from a combination of kernels.

    :param Kernel kern0: First kernel to combine.
    :param kern1: Second kernel to combine.
    :type kern1: Kernel or numbers.Number
    """

    def __init__(self, kern0, kern1):
        if not isinstance(kern0, Kernel) or not isinstance(kern1, Kernel):
            raise TypeError(
                "The components of a combined kernel must be " "Kernel instances."
            )

        track_dims = set(kern0.track_dims) | set(
            kern1.track_dims
        )  # | OR operator, & is AND
        track_dims = sorted(track_dims)
        input_dims = max(track_dims) + 1  # cover all dimensions to it
        super().__init__(input_dims)

        self.kern0 = kern0
        self.kern1 = kern1


class Sum(Combination):
    """
    Returns a new kernel which acts like a sum/direct sum of two kernels.
    The second kernel can be a constant.
    """

    def forward(self, X, Z=None, diag=False):
        return self.kern0(X, Z, diag=diag) + self.kern1(X, Z, diag=diag)


class Product(Combination):
    """
    Returns a new kernel which acts like a product/tensor product of two kernels.
    The second kernel can be a constant.
    """

    def forward(self, X, Z=None, diag=False):
        return self.kern0(X, Z, diag=diag) * self.kern1(X, Z, diag=diag)


# deep kernel
class DeepKernel(Kernel):
    """
    Base class for kernels which map input through some parametric mapping, e.g. ANN.

    :param Kernel kern0: First kernel to combine.
    :param kern1: Second kernel to combine.
    :type kern1: Kernel or numbers.Number
    """

    def __init__(
        self, input_dims, kern, mapping, track_dims=None, tensor_type=torch.float
    ):
        if not isinstance(kern, Kernel):
            raise TypeError("Kernel instance needed for kernel parameter")

        super().__init__(input_dims, track_dims=track_dims, tensor_type=tensor_type)
        self.kern = kern
        self.map = mapping

    def forward(self, X, Z=None, diag=False):
        if Z is None:
            Z = X

        X = self._slice_input(X)
        Z = self._slice_input(Z)
        if X.size(-1) != Z.size(-1):
            raise ValueError("Inputs must have the same number of features.")

        return self.kern(self.map(X), self.map(Z), diag=diag)


# constant kernel
class Constant(Kernel):
    r"""
    Constant kernel (functions independent of input).
    """

    def __init__(self, variance, f="exp", tensor_type=torch.float):
        super().__init__(0, None, f, tensor_type)
        self._variance = Parameter(self.lf_inv(variance.type(tensor_type)))  # N

    @property
    def variance(self):
        return self.lf(self._variance)[None, :, None, None]  # K, N, T, T

    @variance.setter
    def variance(self):
        self._variance.data = self.lf_inv(variance)

    def forward(self, X, Z=None, diag=False):
        if Z is None:
            Z = X

        if diag:
            return self.variance[..., 0].expand(
                X.size(0), self._variance.shape[0], X.size(2)
            )  # K, N, T
        else:
            return self.variance.expand(
                X.size(0), self._variance.shape[0], X.size(2), Z.size(2)
            )  # K, N, T, T


### Full kernels
class DotProduct(Kernel):
    r"""
    Base class for kernels which are functions of :math:`x \cdot z`.
    """

    def __init__(self, input_dims, track_dims=None, f="exp", tensor_type=torch.float):
        super().__init__(input_dims, track_dims, f, tensor_type)

    def _dot_product(self, X, Z=None, diag=False):
        r"""
        Returns :math:`X \cdot Z`.
        """
        if diag:
            return (self._slice_input(X) ** 2).sum(-1)

        if Z is None:
            Z = X

        X = self._slice_input(X)
        Z = self._slice_input(Z)
        if X.size(-1) != Z.size(-1):
            raise ValueError("Inputs must have the same number of features.")

        return X.matmul(Z.permute(0, 1, 3, 2))


class Linear(DotProduct):
    r"""
    Implementation of Linear kernel:
        :math:`k(x, z) = \sigma^2 x \cdot z.`
    Doing Gaussian Process regression with linear kernel is equivalent to doing a
    linear regression.
    .. note:: Here we implement the homogeneous version. To use the inhomogeneous
        version, consider using :class:`Polynomial` kernel with ``degree=1`` or making
        a :class:`.Sum` with a :class:`.Constant` kernel.
    """

    def __init__(self, input_dims, track_dims=None, f="exp", tensor_type=torch.float):
        super().__init__(input_dims, track_dims, f, tensor_type)

    def forward(self, X, Z=None, diag=False):
        return self._dot_product(X, Z, diag)


class Polynomial(DotProduct):
    r"""
    Implementation of Polynomial kernel:
        :math:`k(x, z) = \sigma^2(\text{bias} + x \cdot z)^d.`
    :param torch.Tensor bias: Bias parameter of this kernel. Should be positive.
    :param int degree: Degree :math:`d` of the polynomial.
    """

    def __init__(
        self,
        input_dims,
        bias,
        degree=1,
        track_dims=None,
        f="exp",
        tensor_type=torch.float,
    ):
        super().__init__(input_dims, track_dims, f, tensor_type)
        self._bias = Parameter(self.lf_inv(bias.type(tensor_type)))  # N

        if not isinstance(degree, int) or degree < 1:
            raise ValueError(
                "Degree for Polynomial kernel should be a positive integer."
            )
        self.degree = degree

    @property
    def bias(self):
        return self.lf(self._bias)[None, :, None, None]  # K, N, T, T

    @bias.setter
    def bias(self):
        self._bias.data = self.lf_inv(bias)

    def forward(self, X, Z=None, diag=False):
        if diag:
            bias = self.bias[..., 0]
        else:
            bias = self.bias

        return (bias + self._dot_product(X, Z, diag)) ** self.degree


### non-stationary kernels
class Lengthscale(Kernel):
    """ """

    def __init__(
        self,
        input_dims,
        lengthscale,
        w,
        omega,
        Q,
        track_dims=None,
        topology="euclid",
        f="exp",
        tensor_type=torch.float,
    ):
        super().__init__(input_dims, topology, lengthscale, track_dims, f, tensor_type)


class DecayingSquaredExponential(Lengthscale):
    r"""
    Implementation of Decaying Squared Exponential kernel:

        :math:`k(x, z) = \exp\left(-0.5 \times \frac{|x-\beta|^2 + |z-\beta|^2} {l_{\beta}^2}\right) \,
        \exp\left(-0.5 \times \frac{|x-z|^2}{l^2}\right).`

    :param torch.Tensor scale_mixture: Scale mixture (:math:`\alpha`) parameter of this
        kernel. Should have size 1.
    """

    def __init__(
        self,
        input_dims,
        lengthscale,
        lengthscale_beta,
        beta,
        track_dims=None,
        f="exp",
        tensor_type=torch.float,
    ):
        super().__init__(input_dims, track_dims, f, tensor_type)
        assert (
            self.input_dims == lengthscale.shape[0]
        )  # lengthscale per input dimension

        self.beta = Parameter(beta.type(tensor_type).t())  # N, D
        self._lengthscale_beta = Parameter(
            self.lf_inv(lengthscale_beta.type(tensor_type)).t()[:, None, None, :]
        )  # N, K, T, D
        self._lengthscale = Parameter(
            self.lf_inv(lengthscale.type(tensor_type)).t()
        )  # N, D

    @property
    def lengthscale(self):
        return self.lf(self._lengthscale)[None, :, None, :]  # K, N, T, D

    @lengthscale.setter
    def lengthscale(self):
        self._lengthscale.data = self.lf_inv(lengthscale)

    @property
    def lengthscale_beta(self):
        return self.lf(self._lengthscale_beta)[:, None, None, :]  # N, K, T, D

    @lengthscale_beta.setter
    def lengthscale_beta(self):
        return self.lf(self._lengthscale_beta)[None, :, None, :]  # K, N, T, D

        self._lengthscale_beta.data = self.lf_inv(lengthscale_beta)

    def forward(self, X, Z=None, diag=False):
        X, Z = self._XZ(X, Z)

        if diag:
            return torch.exp(-(((X - self.beta) / self.lengthscale_beta) ** 2).sum(-1))

        scaled_X = X / self.lengthscale  # K, N, T, D
        scaled_Z = Z / self.lengthscale
        X2 = (scaled_X**2).sum(-1, keepdim=True)
        Z2 = (scaled_Z**2).sum(-1, keepdim=True)
        XZ = scaled_X.matmul(scaled_Z.permute(0, 1, 3, 2))
        r2 = X2 - 2 * XZ + Z2.permute(0, 1, 3, 2)

        return torch.exp(
            -0.5
            * (
                r2
                + (((X - self.beta) / self.lengthscale_beta) ** 2).sum(-1)[..., None]
                + (((Z - self.beta) / self.lengthscale_beta) ** 2).sum(-1)[..., None, :]
            )
        )


class SpectralMixture(Lengthscale):
    r"""
    Implementation of Radial Basis Function kernel:

        :math:`k(x,z) = \exp\left(-0.5 \times \frac{|x-z|^2}{l^2}\right).`

    .. note:: This kernel also has name `Squared Exponential` in literature.
    """

    def __init__(
        self,
        input_dims,
        lengthscale,
        w,
        omega,
        Q,
        track_dims=None,
        topology="euclid",
        f="exp",
        tensor_type=torch.float,
    ):
        super().__init__(input_dims, topology, lengthscale, track_dims, f, tensor_type)

    def _square_scaled_dist(self, X, Z=None):
        r"""
        Returns :math:`\|\frac{X-Z}{l}\|^2`.
        :param torch.Tensor X: input of shape (samples, neurons, points, dims)

        """
        X, Z = self._XZ(X, Z)

        scaled_X = X / self.lengthscale  # K, N, T, D
        scaled_Z = Z / self.lengthscale
        X2 = (scaled_X**2).sum(-1, keepdim=True)
        Z2 = (scaled_Z**2).sum(-1, keepdim=True)
        XZ = scaled_X.matmul(scaled_Z.permute(0, 1, 3, 2))
        r2 = X2 - 2 * XZ + Z2.permute(0, 1, 3, 2)
        return r2.clamp(min=0)

    def _scaled_dist(self, X, Z=None):
        r"""
        Returns :math:`\|\frac{X-Z}{l}\|`.
        """
        return _torch_sqrt(self._square_scaled_dist(X, Z))

    def forward(self, X, Z=None, diag=False):
        if diag:
            return X.new_ones(X.size(0), self.n, X.size(2))

        r2 = self.square_scaled_dist(X, Z)
        return (amp * torch.exp(-0.5 * r2) * torch.cos()).sum(-1)


# stationary kernels
class Isotropy(Kernel):
    """
    Base class for a family of isotropic covariance kernels which are functions of the
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
        elif topology == "torus_geodesic":
            self.scaled_dist = self._scaled_dist_Tn
            self.square_scaled_dist = self._square_scaled_dist_Tn
        elif topology == "torus":
            self.scaled_dist = self._scaled_dist_torus
            self.square_scaled_dist = self._square_scaled_dist_torus
        elif topology == "sphere":
            self.scaled_dist = self._scaled_dist_sphere
            self.square_scaled_dist = self._square_scaled_dist_sphere
            lengthscale = lengthscale[:1, :]  # dummy dimensions
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
        return _torch_sqrt(Isotropy._square_scaled_dist(lengthscale, X, Z))

    # Torus wrapped
    @staticmethod
    def _square_scaled_dist_Tn(lengthscale, X, Z):
        r"""
        Returns :math:`\|\frac{X-Z}{l}\|`.
        """
        U = (X[..., None, :] - Z[..., None, :].permute(0, 1, 3, 2, 4)) % (2 * math.pi)
        U[U > math.pi] -= 2 * math.pi
        a = 2.0
        c = -1.0
        U[U > a] = a + torch.cos(U[U > a]) - math.cos(a)  # + (U[U > a]-a)/c
        U[U < -a] = -a - torch.cos(U[U < -a]) + math.cos(a)  # + (U[U < -a]+a)/c
        # n = torch.arange(-3, 3, device=X.device)[:, None, None, None]*2*math.pi # not correct for multidimensions
        return (U / lengthscale[:, :, None, ...]).pow(2).sum(-1)

    @staticmethod
    def _scaled_dist_Tn(lengthscale, X, Z):
        r"""
        Returns :math:`\|\frac{X-Z}{l}\|^2`.
        """
        return _torch_sqrt(Isotropy._square_scaled_dist_Tn(lengthscale, X, Z))

    # Torus cosine
    @staticmethod
    def _square_scaled_dist_torus(lengthscale, X, Z):
        r"""
        Returns :math:`\|\frac{X-Z}{l}\|^2`.
        """
        U = X[..., None, :] - Z[..., None, :].permute(0, 1, 3, 2, 4)

        return 2 * ((1 - torch.cos(U)) / (lengthscale[:, :, None, ...]) ** 2).sum(-1)

    @staticmethod
    def _scaled_dist_torus(lengthscale, X, Z):
        r"""
        Returns :math:`\|\frac{X-Z}{l}\|`.
        """
        return _torch_sqrt(Isotropy._square_scaled_dist_torus(lengthscale, X, Z))

    # n-sphere S(n), input dimension is n
    @staticmethod
    def _square_scaled_dist_sphere(lengthscale, X, Z):
        r"""
        Returns :math:`\|\frac{X-Z}{l}\|^2`.
        """
        return 2 * (
            (1 - (X[..., None, :] * Z[..., None, :].permute(0, 1, 3, 2, 4)).sum(-1))
            / (lengthscale[..., None, 0]) ** 2
        )

    @staticmethod
    def _scaled_dist_sphere(lengthscale, X, Z):
        r"""
        Returns :math:`\|\frac{X-Z}{l}\|`.
        """
        return _torch_sqrt(Isotropy._square_scaled_dist_sphere(lengthscale, X, Z))


class SquaredExponential(Isotropy):
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


class RationalQuadratic(Isotropy):
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


class Exponential(Isotropy):
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


class Matern32(Isotropy):
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


class Matern52(Isotropy):
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
        r = _torch_sqrt(r2)
        sqrt5_r = 5**0.5 * r
        return (1 + sqrt5_r + (5 / 3) * r2) * torch.exp(-sqrt5_r)
