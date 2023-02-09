import torch
import torch.nn as nn
import torch.nn.functional as F


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
