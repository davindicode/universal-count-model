from numbers import Number

import numpy as np
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter

from .. import base, distributions


class histogram(base._input_mapping):
    """
    Histogram rate model based on GLM framework.
    Has an identity link function with positivity constraint on the weights.
    Only supports regressor mode.
    """

    def __init__(
        self,
        bins_cov,
        out_dims,
        ini_rate=1.0,
        alpha=None,
        tensor_type=torch.float,
        active_dims=None,
    ):
        """
        The initial rate should not be zero, as that gives no gradient in the Poisson
        likelihood case

        :param tuple bins_cov: tuple of bin objects (np.linspace)
        :param int neurons: number of neurons in total
        :param float ini_rate: initial rate array
        :param float alpha: smoothness prior hyperparameter, None means no prior
        """
        super().__init__(len(bins_cov), out_dims, tensor_type, active_dims)
        ini = torch.tensor([ini_rate]).view(-1, *np.ones(len(bins_cov)).astype(int))
        self.register_parameter(
            "w",
            Parameter(
                ini
                * torch.ones(
                    (out_dims,) + tuple(len(bins) - 1 for bins in bins_cov),
                    dtype=self.tensor_type,
                )
            ),
        )
        if alpha is not None:
            self.register_buffer("alpha", torch.tensor(alpha, dtype=self.tensor_type))
        else:
            self.alpha = alpha
        self.bins_cov = bins_cov  # use PyTorch for integer indexing

    def set_params(self, w=None):
        if w is not None:
            self.w.data = torch.tensor(w, device=self.w.device, dtype=self.tensor_type)

    def compute_F(self, XZ):
        XZ = self._XZ(XZ)
        samples = XZ.shape[0]

        tg = []
        for k in range(self.input_dims):
            tg.append(torch.bucketize(XZ[..., k], self.bins_cov[k], right=True) - 1)

        XZ_ind = (
            torch.arange(samples)[:, None, None].expand(
                -1, self.out_dims, len(tg[0][b])
            ),
            torch.arange(self.out_dims)[None, :, None].expand(
                samples, -1, len(tg[0][b])
            ),
        ) + tuple(tg_[b][:, None, :].expand(samples, self.out_dims, -1) for tg_ in tg)

        return self.w[XZ_ind], 0

    def sample_F(self, XZ):
        return self.compute_F(XZ)[0]

    def KL_prior(self, importance_weighted):
        if self.alpha is None:
            return super().KL_prior(importance_weighted)

        smooth_prior = (
            self.alpha[0] * (self.w[:, 1:, ...] - self.w[:, :-1, ...]).pow(2).sum()
            + self.alpha[1] * (self.w[:, :, 1:, :] - self.w[:, :, :-1, :]).pow(2).sum()
            + self.alpha[2] * (self.w[:, ..., 1:] - self.w[:, ..., :-1]).pow(2).sum()
        )
        return -smooth_prior

    def constrain(self):
        self.w.data = torch.clamp(self.w.data, min=0)

    def set_unvisited_bins(self, ini_rate=1.0, unvis=np.nan):
        """
        Set the bins that have not been updated to unvisited value unvis.
        """
        self.w.data[self.w.data == ini_rate] = torch.tensor(unvis, device=self.w.device)


class spline(base._input_mapping):
    """
    Spline based mapping.
    """

    def __init__(
        self,
        bins_cov,
        out_dims,
        ini_rate=1.0,
        alpha=None,
        tensor_type=torch.float,
        active_dims=None,
    ):
        """
        The initial rate should not be zero, as that gives no gradient in the Poisson
        likelihood case

        :param tuple bins_cov: tuple of bin objects (np.linspace)
        :param int neurons: number of neurons in total
        :param float ini_rate: initial rate array
        :param float alpha: smoothness prior hyperparameter, None means no prior
        """
        super().__init__(len(bins_cov), out_dims, tensor_type, active_dims)


from numbers import Number

import numpy as np
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter

from .. import base, distributions
from ..utils.signal import eye_like


# linear algebra computations
# @torch.jit.script
def p_F_U(
    X,
    X_u,
    kernelobj,
    f_loc,
    f_scale_tril,
    full_cov=False,
    compute_cov=True,
    whiten=False,
    jitter=1e-6,
):
    r"""
    Computed Gaussian conditional distirbution.

    :param int out_dims: number of output dimensions
    :param torch.Tensor X: Input data to evaluate the posterior over
    :param torch.Tensor X_u: Input data to conditioned on, inducing points in sparse GP
    :param GP.kernels.Kernel kernel: A kernel module object
    :param torch.Tensor f_loc: Mean of :math:`q(f)`. In case ``f_scale_tril=None``,
        :math:`f_{loc} = f`
    :param torch.Tensor f_scale_tril: Lower triangular decomposition of covariance
        matrix of :math:`q(f)`'s
    :param torch.Tensor Lff: Lower triangular decomposition of :math:`kernel(X, X)`
        (optional)
    :param string cov_type: A flag to decide what form of covariance to compute
    :param bool whiten: A flag to tell if ``f_loc`` and ``f_scale_tril`` are
        already transformed by the inverse of ``Lff``
    :param float jitter: A small positive term which is added into the diagonal part of
        a covariance matrix to help stablize its Cholesky decomposition
    :returns: loc and covariance matrix (or variance) of :math:`p(f^*(X_{new}))`
    :rtype: tuple(torch.Tensor, torch.Tensor)
    """
    N_u = X_u.size(-2)  # number of inducing, inducing points
    T = X.size(2)  # timesteps from KxNxTxD
    K = X.size(0)
    out_dims = f_loc.shape[0]

    Kff = kernelobj(X_u[None, ...])[0, ...].contiguous()
    Kff.data.view(Kff.shape[0], -1)[:, :: N_u + 1] += jitter  # add jitter to diagonal
    Lff = torch.linalg.cholesky(Kff)  # N, N_u, N_u

    Kfs = kernelobj(X_u[None, ...], X)  # K, N, N_u, T

    N_l = Kfs.shape[1]
    Kfs = Kfs.permute(1, 2, 0, 3).reshape(N_l, N_u, -1)  # N, N_u, KxT

    if N_l == 1:  # single lengthscale for all outputs
        # convert f_loc_shape from N, N_u to 1, N_u, N
        f_loc = f_loc.permute(-1, 0)[None, ...]

        # f_scale_tril N, N_u, N_u
        if f_scale_tril is not None:
            # convert f_scale_tril_shape from N, N_u, N_u to N_u, N_u, N, convert to 1 x 2D tensor for packing
            f_scale_tril = f_scale_tril.permute(-2, -1, 0).reshape(1, N_u, -1)

    else:  # multi-lengthscale
        # convert f_loc_shape to N, N_u, 1
        f_loc = f_loc[..., None]
        # f_scale_tril N, N_u, N_u

    if whiten:
        v_4D = f_loc[None, ...].repeat(K, 1, 1, 1)  # K, N, N_u, N_
        W = Kfs.triangular_solve(Lff, upper=False)[0]
        W = W.view(N_l, N_u, K, T).permute(2, 0, 3, 1)  # K, N, T, N_u
        if f_scale_tril is not None:
            S_4D = f_scale_tril[None, ...].repeat(K, 1, 1, 1)

    else:
        pack = torch.cat((f_loc, Kfs), dim=-1)  # N, N_u, L
        if f_scale_tril is not None:
            pack = torch.cat((pack, f_scale_tril), dim=-1)

        Lffinv_pack = pack.triangular_solve(Lff, upper=False)[0]
        v_4D = Lffinv_pack[None, :, :, : f_loc.size(-1)].repeat(
            K, 1, 1, 1
        )  # K, N, N_u, N_
        if f_scale_tril is not None:
            S_4D = Lffinv_pack[None, :, :, -f_scale_tril.size(-1) :].repeat(
                K, 1, 1, 1
            )  # K, N, N_u, N_u or N_xN_u

        W = Lffinv_pack[:, :, f_loc.size(-1) : f_loc.size(-1) + K * T]
        W = W.view(N_l, N_u, K, T).permute(2, 0, 3, 1)  # K, N, T, N_u

    if N_l == 1:
        loc = W.matmul(v_4D).permute(0, 3, 2, 1)[..., 0]  # K, N, T
    else:
        loc = W.matmul(v_4D)[..., 0]  # K, N, T

    if compute_cov is False:  # e.g. kernel ridge regression
        return loc, 0, Lff

    if full_cov:
        Kss = kernelobj(X)
        Qss = W.matmul(W.transpose(-2, -1))
        cov = Kss - Qss  # K, N, T, T
    else:
        Kssdiag = kernelobj(X, diag=True)
        Qssdiag = W.pow(2).sum(dim=-1)
        # due to numerical errors, clamp to avoid negative values
        cov = (Kssdiag - Qssdiag).clamp(min=0)  # K, N, T

    if f_scale_tril is not None:
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


### GP models ###
class _GP(base._input_mapping):
    """ """

    def __init__(
        self, input_dims, out_dims, mean, learn_mean, tensor_type, active_dims
    ):
        super().__init__(input_dims, out_dims, tensor_type, active_dims)

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


class inducing_points(nn.Module):
    """ """

    def __init__(
        self, out_dims, inducing_points, constraints, MAP=False, tensor_type=torch.float
    ):
        """
        :param list constraints: list of tupes (dl, du, topo) where dl:du dimensions are constrained on topology topo
        """
        super().__init__()
        self.tensor_type = tensor_type
        self.Xu = Parameter(inducing_points.type(tensor_type))
        _, self.n_ind, self.input_dims = self.Xu.shape

        self.out_dims = out_dims
        self.constraints = constraints

        u_loc = self.Xu.new_zeros((self.out_dims, self.n_ind))
        self.u_loc = Parameter(u_loc)

        if MAP:
            self.u_scale_tril = None
        else:
            identity = eye_like(self.Xu, self.n_ind)
            u_scale_tril = identity.repeat(self.out_dims, 1, 1)
            self.u_scale_tril = Parameter(u_scale_tril)

    def constrain(self):
        # constrain topological inducing points of sphere
        for dl, du, topo in self.constraints:
            if topo == "sphere":
                L2 = self.Xu[..., dl:du].data.norm(2, -1)[..., None]
                self.Xu[..., k : k + n].data /= L2

        if (
            self.u_scale_tril is not None
        ):  # constrain K to be PSD, L diagonals > 0 as needed for det K
            self.u_scale_tril.data = torch.tril(self.u_scale_tril.data)
            Nu = self.n_ind
            self.u_scale_tril.data[:, range(Nu), range(Nu)] = torch.clamp(
                self.u_scale_tril.data[:, range(Nu), range(Nu)], min=1e-12
            )


class SVGP(_GP):
    """
    Sparse Variational Gaussian Process model with covariates for regression and latent variables.
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
        kernel_regression=False,
        tensor_type=torch.float,
        jitter=1e-6,
        active_dims=None,
    ):
        r"""
        Specify the kernel type with corresponding dimensions and arguments.
        
        .. math::
            [f, u] &\sim \mathcal{GP}(0, k([X, X_u], [X, X_u])),\\
            y & \sim p(y) = p(y \mid f) p(f),

        where :math:`p(y \mid f)` is the likelihood.

        We will use a variational approach in this model by approximating :math:`q(f,u)`
        to the posterior :math:`p(f,u \mid y)`. Precisely, :math:`q(f) = p(f\mid u)q(u)`,
        where :math:`q(u)` is a multivariate normal distribution with two parameters
        ``u_loc`` and ``u_scale_tril``, which will be learned during a variational
        inference process.

        The sparse model has :math:`\mathcal{O}(NM^2)` complexity for training,
        :math:`\mathcal{O}(M^3)` complexity for testing. Here, :math:`N` is the number
        of train inputs, :math:`M` is the number of inducing inputs. Size of
        variational parameters is :math:`\mathcal{O}(M^2)`.
        
        
        :param int out_dims: number of output dimensions of the GP, usually neurons
        :param nn.Module inducing_points: initial inducing points with shape (neurons, n_induc, input_dims)
        :param Kernel kernel: a tuple listing kernels, with content 
                                     (kernel_type, topology, lengthscale, variance)
        :param Number/torch.Tensor/nn.Module mean: initial GP mean of shape (samples, neurons, timesteps), or if nn.Module 
                                        learnable function to compute the mean given input
        """
        super().__init__(
            input_dims, out_dims, mean, learn_mean, tensor_type, active_dims
        )
        self.jitter = jitter
        self.whiten = whiten
        self.kernel_regression = kernel_regression

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

    def KL_prior(self, importance_weighted):
        """
        Ignores neuron, computes over all the output dimensions
        Note self.Luu is computed in compute_F or sample_F called before
        """
        if self.induc_pts.u_scale_tril is None:  # log p(u)
            zero_loc = self.induc_pts.Xu.new_zeros(self.induc_pts.u_loc.shape)
            if self.whiten:
                identity = eye_like(self.induc_pts.Xu, zero_loc.shape[1])[
                    None, ...
                ].repeat(zero_loc.shape[0], 1, 1)
                p = distributions.Rn_MVN(zero_loc, scale_tril=identity)
            else:  # loc (N, N_u), cov (N, N_u, N_u)
                p = distributions.Rn_MVN(zero_loc, scale_tril=self.Luu)

            return -p.log_prob(self.induc_pts.u_loc).sum()

        else:  # log p(u)/q(u)
            zero_loc = self.induc_pts.u_loc.new_zeros(self.induc_pts.u_loc.shape)
            if self.whiten:
                identity = eye_like(self.induc_pts.u_loc, zero_loc.shape[1])[
                    None, ...
                ].repeat(zero_loc.shape[0], 1, 1)
                p = distributions.Rn_MVN(
                    zero_loc, scale_tril=identity
                )  # .to_event(zero_loc.dim() - 1)
            else:  # loc (N, N_u), cov (N, N_u, N_u)
                p = distributions.Rn_MVN(
                    zero_loc, scale_tril=self.Luu
                )  # .to_event(zero_loc.dim() - 1)

            q = distributions.Rn_MVN(
                self.induc_pts.u_loc, scale_tril=self.induc_pts.u_scale_tril
            )  # .to_event(self.u_loc.dim()-1)
            kl = torch.distributions.kl.kl_divergence(q, p).sum()  # sum over neurons
            if torch.isnan(kl).any():
                kl = 0.0
                print("Warning: sparse GP prior is NaN, ignoring prior term.")
            return kl

    def constrain(self):
        self.induc_pts.constrain()

    def compute_F(self, XZ):
        """
        Computes moments of marginals :math:`q(f_i|\bm{u})` and also updating :math:`L_{uu}` matrix
        model call uses :math:`L_{uu}` for the MVN, call after this function

        Computes the mean and covariance matrix (or variance) of Gaussian Process
        posterior on a test input data :math:`X_{new}`:

        .. math:: p(f^* \mid X_{new}, X, y, k, X_u, u_{loc}, u_{scale\_tril})
            = \mathcal{N}(loc, cov).

        .. note:: Variational parameters ``u_loc``, ``u_scale_tril``, the
            inducing-point parameter ``Xu``, together with kernel's parameters have
            been learned.

        covariance_type is a flag to decide if we want to predict full covariance matrix or
        just variance.

        .. note:: The GP is centered around zero with fixed zero mean, but a learnable
            mean is added after computing the posterior to get the mapping mean.

        XZ # K, N, T, D
        X_u = self.Xu[None, ...] # K, N, T, D

        :param X: input regressors with shape (samples, timesteps, dims)
        :returns: loc and covariance matrix (or variance) of :math:`p(f^*(X_{new}))`
        :rtype: tuple(torch.Tensor, torch.Tensor), both of shape ()
        """
        XZ = self._XZ(XZ)
        loc, var, self.Luu = p_F_U(
            XZ,
            self.induc_pts.Xu,
            self.kernel,
            self.induc_pts.u_loc,
            self.induc_pts.u_scale_tril,
            compute_cov=~self.kernel_regression,
            full_cov=False,
            whiten=self.whiten,
            jitter=self.jitter,
        )

        return loc + self.mean_function(XZ), var

    def sample_F(self, XZ, samples=1, eps=None):
        """
        Samples from the variational posterior :math:`q(\bm{f}_*|\bm{u})`, which can be the predictive distribution
        """
        XZ = self._XZ(XZ)

        if XZ.shape[-1] < 1000:  # sample naive way
            loc, cov, self.Luu = p_F_U(
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
            L = cov.double().cholesky().type(self.tensor_type)

            if samples > 1:  # expand
                XZ = XZ.repeat(samples, 1, 1, 1)
                L = L.repeat(samples)

            if eps is None:  # sample random vector
                eps = torch.randn(
                    XZ.shape[:-1], dtype=self.tensor_type, device=cov.device
                )

            return loc + self.mean_function(XZ) + (L * eps[..., None, :]).sum(-1)

        else:  # decoupled sampling
            eps_f, eps_u = eps

            return loc + self.mean_function(XZ) + (L * eps[..., None, :]).sum(-1)
