import torch
from torch.nn.parameter import Parameter

from .base import Kernel



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


