import torch
from torch.nn.parameter import Parameter

from .base import Kernel


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

        track_dims = set(kern0.track_dims) | set(kern1.track_dims)
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
