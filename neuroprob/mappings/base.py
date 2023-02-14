import torch
import torch.nn as nn


def _expand_cov(cov):
    if len(cov.shape) == 1:  # expand arrays from (ts,)
        cov = cov[None, None, :, None]
    elif len(cov.shape) == 2:  # expand arrays (ts, dims)
        cov = cov[None, None, ...]
    elif len(cov.shape) == 3:
        cov = cov[None, ...]  # expand arrays (out, ts, dims)

    if len(cov.shape) != 4:  # trials, out, ts, dims
        raise ValueError(
            "Largest shape of input covariates is (trials, out_dims, ts, in_dims)"
        )

    return cov


class _input_mapping(nn.Module):
    """
    Input covariates to mean and covariance parameters. An input mapping consists of a mapping from input
    to inner and inner_var quantities.
    """

    def __init__(
        self,
        input_dims,
        out_dims,
        tensor_type=torch.float,
        active_dims=None,
        MC_only=False,
    ):
        """
        Constructor for the input mapping class.

        :param int input_dims: the number of input dimensions in the covariates array of the model
        :param int out_dims: the number of output dimensions of the model
        :param List active_dims: indices of dimensions in input selected by the model
        """
        super().__init__()
        self.register_buffer("dummy", torch.empty(0))  # keeping track of device

        self.MC_only = MC_only  # default has VI-like output, not pure MC samples
        self.tensor_type = tensor_type
        self.out_dims = out_dims
        self.input_dims = input_dims

        if active_dims is None:
            active_dims = list(range(input_dims))
        elif len(active_dims) != input_dims:
            raise ValueError(
                "Active dimensions do not match expected number of input dimensions"
            )
        self.active_dims = active_dims  # dimensions to select from input covariates

    def compute_F(self, XZ):
        """
        Computes the diagonal posterior over :mathm:`F`, conditioned on the data. In most cases, this is amortized via
        learned weights/parameters of some approximate posterior. In ML/MAP settings the approximation is a
        delta distribution, meaning there is no variational uncertainty in the mapping.
        """
        raise NotImplementedError

    def sample_F(self, XZ, samples):
        """
        Samples from the full posterior (full covariance)

        Function for generating Gaussian MC samples. No MC samples are drawn when the variance is 0.

        :param torch.Tensor q_mu: the mean of the MC distribution
        :param torch.Tensor q_var: the (co)variance, type (univariate, multivariate) deduced
                                   from tensor shape
        :param int samples: number of MC samples per covariate sample
        :returns: the samples
        :rtype: torch.tensor
        """
        raise NotImplementedError

    def KL_prior(self):
        """
        Prior on the model parameters as regularizer in the loss. Model parameters are integrated out
        approximately using the variational inference. This leads to Kullback-Leibler divergences in the
        overall objective, for MAP model parameters this reduces to the prior log probability.
        """
        return 0

    def constrain(self):
        """
        Constrain parameters in optimization.
        """
        return

    def _XZ(self, XZ):
        """
        Return XZ of shape (K, N, T, D)
        """
        if max(self.active_dims) >= XZ.shape[-1]:
            raise ValueError(
                "Active dimensions is outside input dimensionality provided"
            )
        return XZ[..., self.active_dims]

    def to_XZ(self, covariates_list, trials=1):
        """
        Convert covariates list of tensors to a single input tensors for mappings.
        """
        cov_list = []
        timesteps = None
        out_dims = 1  # if all shared across output dimensions
        for cov_ in covariates_list:
            cov_ = _expand_cov(cov_.type(self.tensor_type))

            if cov_.shape[1] > 1:
                if out_dims == 1:
                    out_dims = cov_.shape[1]
                elif out_dims != cov_.shape[1]:
                    raise ValueError(
                        "Output dimensions in covariates dimensions are not consistent"
                    )

            if timesteps is not None:
                if timesteps != cov_.shape[2]:
                    raise ValueError(
                        "Time steps in covariates dimensions are not consistent"
                    )
            else:
                timesteps = cov_.shape[2]

            cov_list.append(cov_)

        XZ = (
            torch.cat(cov_list, dim=-1)
            .expand(trials, out_dims, timesteps, -1)
            .to(self.dummy.device)
        )
        return XZ
