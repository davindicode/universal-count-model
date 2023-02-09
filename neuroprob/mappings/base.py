import torch
import torch.nn as nn


def _expand_cov(cov):
    if len(cov.shape) == 1:  # expand arrays from (timesteps,)
        cov = cov[None, None, :, None]
    elif len(cov.shape) == 2:  # expand arrays (timesteps, dims)
        cov = cov[None, None, ...]
    elif len(cov.shape) == 3:
        cov = cov[None, ...]  # expand arrays (out, timesteps, dims)

    if len(cov.shape) != 4:  # trials, out, timesteps, dims
        raise ValueError(
            "Shape of input covariates at most trials x out x timesteps x dims"
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

        :param int neurons: the number of neurons or output dimensions of the model
        :param string inv_link: the name of the inverse link function
        :param int input_dims: the number of input dimensions in the covariates array of the model
        :param list VI_tuples: a list of variational inference tuples (prior, var_dist, topology, dims),
                               note that dims specifies the shape of the corresponding distribution and
                               is also the shape expected for regressors corresponding to this block as
                               (timesteps, dims). If dims=1, it is treated as a scalar hence X is then
                               of shape (timesteps,)
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

        """if isinstance(inv_link, types.LambdaType):
            self.f = inv_link
            inv_link = 'custom'
        elif _inv_link_functions.get(inv_link) is None:
            raise NotImplementedError('Link function is not supported')
        else:
            self.f = _inv_link_functions[inv_link]
        self.inv_link = inv_link"""

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

    def to_XZ(self, covariates, trials=1):
        """
        Convert covariates list input to tensors for input to mapping. Convenience function for rate
        evaluation functions and sampling functions.
        """
        cov_list = []
        timesteps = None
        out_dims = 1  # if all shared across output dimensions
        for cov_ in covariates:
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
