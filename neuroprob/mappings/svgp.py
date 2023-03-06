from numbers import Number

import numpy as np
import torch
import torch.distributions as dist
import torch.nn as nn
from torch.nn.parameter import Parameter

from . import base



def conditional_p_F_U(
    X,
    X_u,
    kernel,
    u_loc,
    u_scale_tril,
    full_cov=False,
    compute_cov=True,
    whiten=False,
    jitter=1e-6,
):
    """
    Computes Gaussian conditional distirbution.

    :param torch.Tensor X: Input data to evaluate the posterior over
    :param torch.Tensor X_u: Input data to conditioned on, inducing points in sparse GP
    :param GP.kernels.Kernel kernel: A kernel module object
    :param torch.Tensor u_loc: mean of variational MVN distribution
    :param torch.Tensor u_scale_tril: lower triangular cholesky of covariance of variational MVN distribution
    :param bool full_cov: return the posterior covariance matrix or not
    :param bool compute_cov: compute the posterior covariance matrix or not
    :param bool whiten: whether to apply the whitening transform
    :param float jitter: size of small positive diagonal matrix to help stablize Cholesky decompositions
    :return:
        loc and covariance matrix (or variance vector)
    """
    N_u = X_u.size(-2)  # number of inducing, inducing points
    T = X.size(2)  # timesteps from KxNxTxD
    K = X.size(0)
    out_dims = u_loc.shape[0]

    Kff = kernel(X_u[None, ...])[0, ...].contiguous()
    Kff.data.view(Kff.shape[0], -1)[:, :: N_u + 1] += jitter  # add jitter to diagonal
    Lff = torch.linalg.cholesky(Kff)  # N, N_u, N_u

    Kfs = kernel(X_u[None, ...], X)  # K, N, N_u, T

    N_l = Kfs.shape[1]
    Kfs = Kfs.permute(1, 2, 0, 3).reshape(N_l, N_u, -1)  # N, N_u, KxT

    if N_l == 1:  # single lengthscale for all outputs
        # convert u_loc_shape from N, N_u to 1, N_u, N
        u_loc = u_loc.permute(-1, 0)[None, ...]

        # u_scale_tril N, N_u, N_u
        if u_scale_tril is not None:
            # convert u_scale_tril_shape from N, N_u, N_u to N_u, N_u, N, convert to 1 x 2D tensor for packing
            u_scale_tril = u_scale_tril.permute(-2, -1, 0).reshape(1, N_u, -1)

    else:  # multi-lengthscale
        # convert u_loc_shape to N, N_u, 1
        u_loc = u_loc[..., None]
        # u_scale_tril N, N_u, N_u

    if whiten:
        v_4D = u_loc[None, ...].repeat(K, 1, 1, 1)  # K, N, N_u, N_
        W = torch.linalg.solve_triangular(Lff, Kfs, upper=False)
        W = W.view(N_l, N_u, K, T).permute(2, 0, 3, 1)  # K, N, T, N_u
        if u_scale_tril is not None:
            S_4D = u_scale_tril[None, ...].repeat(K, 1, 1, 1)

    else:
        pack = torch.cat((u_loc, Kfs), dim=-1)  # N, N_u, L
        if u_scale_tril is not None:
            pack = torch.cat((pack, u_scale_tril), dim=-1)

        Lffinv_pack = torch.linalg.solve_triangular(Lff, pack, upper=False)
        v_4D = Lffinv_pack[None, :, :, : u_loc.size(-1)].repeat(
            K, 1, 1, 1
        )  # K, N, N_u, N_
        if u_scale_tril is not None:
            S_4D = Lffinv_pack[None, :, :, -u_scale_tril.size(-1) :].repeat(
                K, 1, 1, 1
            )  # K, N, N_u, N_u or N_xN_u

        W = Lffinv_pack[:, :, u_loc.size(-1) : u_loc.size(-1) + K * T]
        W = W.view(N_l, N_u, K, T).permute(2, 0, 3, 1)  # K, N, T, N_u

    if N_l == 1:
        loc = W.matmul(v_4D).permute(0, 3, 2, 1)[..., 0]  # K, N, T
    else:
        loc = W.matmul(v_4D)[..., 0]  # K, N, T

    if compute_cov is False:  # e.g. kernel ridge regression
        return loc, 0, Lff

    if full_cov:
        Kss = kernel(X)
        Qss = W.matmul(W.transpose(-2, -1))
        cov = Kss - Qss  # K, N, T, T
    else:
        Kssdiag = kernel(X, diag=True)
        Qssdiag = W.pow(2).sum(dim=-1)
        # due to numerical errors, clamp to avoid negative values
        cov = (Kssdiag - Qssdiag).clamp(min=0)  # K, N, T

    if u_scale_tril is not None:
        W_S = W.matmul(S_4D)  # K, N, T, N_xN_u
        if N_l == 1:
            W_S = W_S.view(K, 1, T, N_u, out_dims).permute(0, 4, 2, 3, 1)[..., 0]

        if full_cov:
            St_Wt = W_S.transpose(-2, -1)
            K = W_S.matmul(St_Wt)
            cov = cov + K
        else:
            Kdiag = W_S.pow(2).sum(dim=-1)
            cov = cov + Kdiag

    return loc, cov, Lff


class inducing_points(nn.Module):
    """
    Class to hold inducing points
    """

    def __init__(
        self, out_dims, inducing_points, MAP=False, tensor_type=torch.float, jitter=1e-6
    ):
        """
        :param int out_dims: number of output dimensions
        :param inducing_points: inducing point locations (out_dims, N_induc, dims)
        """
        super().__init__()
        self.tensor_type = tensor_type
        self.jitter = jitter

        self.Xu = Parameter(inducing_points.type(tensor_type))
        _, self.n_ind, self.input_dims = self.Xu.shape

        self.out_dims = out_dims

        u_loc = self.Xu.new_zeros((self.out_dims, self.n_ind))
        self.u_loc = Parameter(u_loc)

        if MAP:
            self.u_scale_tril = None
        else:
            identity = base.eye_like(self.Xu, self.n_ind)
            u_scale_tril = identity.repeat(self.out_dims, 1, 1)
            self.u_scale_tril = Parameter(u_scale_tril)

    def constrain(self):
        if (
            self.u_scale_tril is not None
        ):  # constrain K to be PSD, L diagonals > 0 as needed for det K
            self.u_scale_tril.data = torch.tril(self.u_scale_tril.data)
            Nu = self.n_ind
            self.u_scale_tril.data[:, range(Nu), range(Nu)] = torch.clamp(
                self.u_scale_tril.data[:, range(Nu), range(Nu)], min=self.jitter
            )

    def proximity_cost(self):
        """
        Proximity cost for inducing point locations, to improve numerical stability
        """
        Dmat = self.Xu[..., None, :] - self.Xu[:, None, ...]  # (outs, Nu, Nu, dims)

        dist_uu = torch.maximum(1e-2 - torch.abs(Dmat).sum(-1), torch.tensor(0.0))
        dist_uu[:, np.arange(self.n_ind), np.arange(self.n_ind)] = 0.0

        repulsion = dist_uu.sum()
        return 1e5 * repulsion


class SVGP(base._input_mapping):
    """
    Sparse Variational Gaussian Process model with covariates for regression and latent variables.
    Uses the variational approach following (Titsias 2009) and (Hensman et al. 2013)
    """

    def __init__(
        self,
        input_dims,
        out_dims,
        kernel,
        inducing_points,
        mean=0.0,
        learn_mean=False,
        MAP=False,
        whiten=False,
        compute_post_covar=True,
        tensor_type=torch.float,
        jitter=1e-6,
        active_dims=None,
        penalize_induc_proximity=True,
    ):
        """
        :param int out_dims: number of output dimensions of the GP, e.g. out_dims
        :param nn.Module inducing_points: initial inducing points with shape (out_dims, n_induc, input_dims)
        :param Kernel kernel: a tuple listing kernels, with content
                                     (kernel_type, topology, lengthscale, variance)
        :param Number/torch.Tensor/nn.Module mean: initial GP mean of shape (samples, out_dims, ts), or if nn.Module
                                        learnable function to compute the mean given input
        """
        super().__init__(input_dims, out_dims, tensor_type, active_dims)

        self.jitter = jitter
        self.whiten = whiten
        self.compute_post_covar = compute_post_covar  # False if doing kernel regression
        self.penalize_induc_proximity = penalize_induc_proximity

        ### GP mean ###
        if isinstance(mean, Number):
            if learn_mean:
                self.mean = Parameter(torch.tensor(mean, dtype=self.tensor_type))
            else:
                self.register_buffer("mean", torch.tensor(mean, dtype=self.tensor_type))
            self.mean_function = lambda x: self.mean

        elif isinstance(mean, torch.Tensor):
            if mean.shape != torch.Size([out_dims]):
                raise ValueError("Mean dimensions do not match output dimensions")
            if learn_mean:
                self.mean = Parameter(mean[None, :, None].type(self.tensor_type))
            else:
                self.register_buffer("mean", mean[None, :, None].type(self.tensor_type))
            self.mean_function = lambda x: self.mean

        elif isinstance(mean, nn.Module):
            self.add_module("mean_function", mean)
            if learn_mean is False:
                self.mean_function.requires_grad = False

        else:
            raise NotImplementedError("Mean type is not supported.")

        ### kernel ###
        if kernel.input_dims != self.input_dims:
            ValueError("Kernel dimensions do not match expected input dimensions")
        if kernel.tensor_type != self.tensor_type:
            ValueError("Kernel tensor type does not match model tensor type")
        self.add_module("kernel", kernel)

        ### inducing points ###
        if inducing_points.input_dims != input_dims:
            raise ValueError(
                "Inducing point dimensions do not match expected dimensions"
            )
        if inducing_points.out_dims != out_dims:
            raise ValueError("Inducing variable output dimensions do not match model")
        if kernel.tensor_type != self.tensor_type:
            ValueError("Inducing point tensor type does not match model tensor type")
        self.add_module("induc_pts", inducing_points)
        self.Luu = None  # needs to be computed in sample_F/compute_F

    def KL_prior(self):
        """
        Note self.Luu is computed/updated in compute_F or sample_F called before
        """
        if self.induc_pts.u_scale_tril is None:  # log p(u)
            zero_loc = self.induc_pts.Xu.new_zeros(self.induc_pts.u_loc.shape)
            if self.whiten:
                identity = base.eye_like(self.induc_pts.Xu, zero_loc.shape[1])[
                    None, ...
                ].repeat(zero_loc.shape[0], 1, 1)
                p = dist.MultivariateNormal(zero_loc, scale_tril=identity)
            else:  # loc (N, N_u), cov (N, N_u, N_u)
                p = dist.MultivariateNormal(zero_loc, scale_tril=self.Luu)

            kl = -p.log_prob(self.induc_pts.u_loc).sum()

        else:  # log p(u)/q(u)
            zero_loc = self.induc_pts.u_loc.new_zeros(self.induc_pts.u_loc.shape)
            if self.whiten:
                identity = base.eye_like(self.induc_pts.u_loc, zero_loc.shape[1])[
                    None, ...
                ].repeat(zero_loc.shape[0], 1, 1)
                p = dist.MultivariateNormal(
                    zero_loc, scale_tril=identity
                )  # .to_event(zero_loc.dim() - 1)
            else:  # loc (N, N_u), cov (N, N_u, N_u)
                p = dist.MultivariateNormal(
                    zero_loc, scale_tril=self.Luu
                )  # .to_event(zero_loc.dim() - 1)

            q = dist.MultivariateNormal(
                self.induc_pts.u_loc, scale_tril=self.induc_pts.u_scale_tril
            )  # .to_event(self.u_loc.dim()-1)
            kl = dist.kl.kl_divergence(q, p).sum()  # sum over out_dims
            if torch.isnan(kl).any():
                kl = 0.0
                print("Warning: sparse GP KL divergence is NaN, ignoring prior term.")

        if self.penalize_induc_proximity:
            kl += self.induc_pts.proximity_cost()

        return kl

    def constrain(self):
        self.induc_pts.constrain()

    def compute_F(self, XZ):
        """
        Computes moments of the posterior marginals and updating the cholesky matrix
        for the covariance over inducing point locations.

        :param XZ: input covariates with shape (samples, out_dims, ts, dims)
        :return:
            mean and diagonal covariance of the posterior (samples, out_dims, ts)
        """
        XZ = self._XZ(XZ)
        loc, var, self.Luu = _conditional_p_F_U(
            XZ,
            self.induc_pts.Xu,
            self.kernel,
            self.induc_pts.u_loc,
            self.induc_pts.u_scale_tril,
            compute_cov=self.compute_post_covar,
            full_cov=False,
            whiten=self.whiten,
            jitter=self.jitter,
        )

        return loc + self.mean_function(XZ), var

    def sample_F(self, XZ, samples=1, eps=None):
        """
        Samples from the predictive distribution (posterior over evaluation points)

        :param XZ: evaluation input covariates with shape (samples, out_dims, ts, dims)
        :return:
            joint samples of the posterior (samples, out_dims, ts)
        """
        XZ = self._XZ(XZ)

        loc, cov, self.Luu = _conditional_p_F_U(
            XZ,
            self.induc_pts.Xu,
            self.kernel,
            self.induc_pts.u_loc,
            self.induc_pts.u_scale_tril,
            compute_cov=True,
            full_cov=True,
            whiten=self.whiten,
            jitter=self.jitter,
        )
        cov.view(-1, cov.shape[-1] ** 2)[:, :: cov.shape[-1] + 1] += self.jitter

        L = torch.linalg.cholesky(cov.double()).type(self.tensor_type)

        if samples > 1:  # expand
            XZ = XZ.repeat(samples, 1, 1, 1)
            L = L.repeat(samples)

        if eps is None:  # sample random vector
            eps = torch.randn(XZ.shape[:-1], dtype=self.tensor_type, device=cov.device)

        return loc + self.mean_function(XZ) + (L * eps[..., None, :]).sum(-1)
