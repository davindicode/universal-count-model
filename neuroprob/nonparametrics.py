import torch
import torch.nn as nn
from torch.nn.parameter import Parameter

import numpy as np
from numbers import Number

from . import distributions, base
from .GP import kernels, linalg
from .utils.signal import eye_like



class histogram(base._input_mapping):
    """
    Histogram rate model based on GLM framework.
    Has an identity link function with positivity constraint on the weights.
    Only supports regressor mode.
    """
    def __init__(self, bins_cov, out_dims, ini_rate=1.0, alpha=None, tens_type=torch.float, active_dims=None):
        """
        The initial rate should not be zero, as that gives no gradient in the Poisson 
        likelihood case
        
        :param tuple bins_cov: tuple of bin objects (np.linspace)
        :param int neurons: number of neurons in total
        :param float ini_rate: initial rate array
        :param float alpha: smoothness prior hyperparameter, None means no prior
        """
        super().__init__(len(bins_cov), out_dims, 'identity', [(None, None, None, 1)]*len(bins_cov), 
                         None, tens_type, active_dims)
        ini = torch.tensor([ini_rate]).view(-1, *np.ones(len(bins_cov)).astype(int))
        self.register_parameter('w', Parameter(ini*torch.ones((out_dims,) + \
                                               tuple(len(bins)-1 for bins in bins_cov), 
                                                              dtype=self.tensor_type)))
        if alpha is not None:
            self.register_buffer('alpha', torch.tensor(alpha, dtype=self.tensor_type))
        else:
            self.alpha = alpha
        self.bins_cov = bins_cov # use PyTorch for integer indexing
        
        
    def set_params(self, w=None):
        if w is not None:
            self.w.data = torch.tensor(w, device=self.w.device, dtype=self.tensor_type)
        
        
    def compute_F(self, XZ):
        XZ = self._XZ(XZ)
        samples = XZ.shape[0]
        
        tg = []
        for k in range(self.input_dims):
            tg.append(torch.bucketize(XZ[..., k], self.bins_cov[k], right=True)-1)

        XZ_ind = (torch.arange(samples)[:, None, None].expand(-1, self.out_dims, len(tg[0][b])), 
                  torch.arange(self.out_dims)[None, :, None].expand(samples, -1, len(tg[0][b])),) + \
                  tuple(tg_[b][:, None, :].expand(samples, self.out_dims, -1) for tg_ in tg)
 
        return self.w[XZ_ind], 0
    
    
    def KL_prior(self):
        if self.alpha is None:
            return super().KL_prior()
    
        smooth_prior = self.alpha[0]*(self.w[:, 1:, ...] - self.w[:, :-1, ...]).pow(2).sum() + \
            self.alpha[1]*(self.w[:, :, 1:, :] - self.w[:, :, :-1, :]).pow(2).sum() + \
            self.alpha[2]*(self.w[:, ..., 1:] - self.w[:, ..., :-1]).pow(2).sum()
        return -smooth_prior
    
    
    def constrain(self):
        self.w.data = torch.clamp(self.w.data, min=0)

        
    def set_unvisited_bins(self, ini_rate=1.0, unvis=np.nan):
        """
        Set the bins that have not been updated to unvisited value unvis.
        """
        self.w.data[self.w.data == ini_rate] = torch.tensor(unvis, device=self.w.device)



# GP models
_gp_modes = ['sparse', 'sparse_MAP', 'full']#, 'grid']
_inversion_methods = ['cholesky']#, 'CG']



class Gaussian_process(base._input_mapping):
    """
    A Variational Gaussian Process model with covariates for regression and latent variables.
    """
    def __init__(self, input_dims, out_dims, kernel_tuples, mean=0.0, learn_mean=False, inv_link='exp', kern_f='exp', 
                 covariance_type='diagonal', whiten=False, inversion_method='cholesky', gp_mode='sparse', 
                 inducing_points=None, variational_f=None, tens_type=torch.float, jitter=1e-6, active_dims=None):
        r"""
        Specify the kernel type with corresponding dimensions and arguments.
        Note that using shared_kernel_params=True uses Pyro's kernel module, which sets 
        hyperparameters directly without passing through a nonlinearity function.
        
        .. math::
            [f, u] &\sim \mathcal{GP}(0, k([X, X_u], [X, X_u])),\\
            y & \sim p(y) = p(y \mid f) p(f),

        where :math:`p(y \mid f)` is the likelihood.

        We will use a variational approach in this model by approximating :math:`q(f,u)`
        to the posterior :math:`p(f,u \mid y)`. Precisely, :math:`q(f) = p(f\mid u)q(u)`,
        where :math:`q(u)` is a multivariate normal distribution with two parameters
        ``u_loc`` and ``u_scale_tril``, which will be learned during a variational
        inference process.

        .. note:: The sparse model has :math:`\mathcal{O}(NM^2)` complexity for training,
            :math:`\mathcal{O}(M^3)` complexity for testing. Here, :math:`N` is the number
            of train inputs, :math:`M` is the number of inducing inputs. Size of
            variational parameters is :math:`\mathcal{O}(M^2)`.
        
        
        :param int out_dims: number of output dimensions of the GP, usually neurons
        :param np.array inducing_points: initial inducing points with shape (neurons, n_induc, input_dims)
        :param tuples kernel_tuples: a tuple listing kernels, with content 
                                     (kernel_type, topology, lengthscale, variance)
        :param tuples prior_tuples: a tuple listing prior distribution, with content 
                                    (kernel_type, topology, lengthscale, variance)
        :param tuples variational_types: a tuple listing variational distributions, with content 
                                         (kernel_type, topology, lengthscale, variance)
        :param np.array/nn.Module mean: initial GP mean of shape (samples, neurons, timesteps), or if nn.Module 
                                        learnable function to compute the mean given input
        :param string inv_ink: inverse link function name
        """
        super().__init__(input_dims, out_dims, inv_link, covariance_type, tens_type, active_dims)
        
        self.jitter = jitter
        #self.shared_kernel_params = shared_kernel_params

        ### GP options ###
        if inversion_method in _inversion_methods:
            self.inversion_method = inversion_method
        else:
            raise NotImplementedError('Inversion method is not supported.')
            
        if gp_mode in _gp_modes:
            self.gp_mode = gp_mode
        else:
            raise NotImplementedError('GP type is not supported.')
        
        ### kernel ###
        kernel, track_dims, constrain_dims = kernels.create_kernel(kernel_tuples, kern_f, self.tensor_type)
        if track_dims != self.input_dims:
            ValueError('Kernel dimensions do not match expected input dimensions')
        self.constrain_dims = constrain_dims
        self.add_module("kernel", kernel)
        
        ### GP mean ###
        if isinstance(mean, Number):
            if learn_mean:
                self.mean = Parameter(torch.tensor(mean, dtype=self.tensor_type))
            else:
                self.register_buffer('mean', torch.tensor(mean, dtype=self.tensor_type))
            self.mean_function = lambda x: self.mean
            
        elif isinstance(mean, np.ndarray):
            if mean.shape != torch.Size([out_dims]):
                raise ValueError('Mean dimensions do not match output dimensions')
            if learn_mean:
                self.mean = Parameter(torch.tensor(mean[None, :, None], dtype=self.tensor_type))
            else:
                self.register_buffer('mean', torch.tensor(mean[None, :, None], dtype=self.tensor_type))
            self.mean_function = lambda x: self.mean
            
        elif isinstance(mean, nn.Module):
            self.add_module("mean_function", mean)
            if learn_mean is False:
                self.mean_function.requires_grad = False
                
        else:
            raise NotImplementedError('Mean type is not supported.')
        
        ### Approximate GP setup ###
        if self.gp_mode == 'sparse' or self.gp_mode == 'sparse_MAP': # inducing points
            if inducing_points.shape[-1] != input_dims:
                raise ValueError('Inducing point dimensions do not match expected dimensions')
            inducing_points = torch.tensor(inducing_points, dtype=self.tensor_type)
            self.n_inducing_points = inducing_points.size(-2)
            
            self.whiten = whiten
            self.Xu = Parameter(inducing_points)

            u_loc = self.Xu.new_zeros((self.out_dims, self.n_inducing_points))
            self.u_loc = Parameter(u_loc)

            if self.covariance_type is not None and self.gp_mode != 'sparse_MAP':
                identity = eye_like(self.Xu, self.n_inducing_points)
                u_scale_tril = identity.repeat(self.out_dims, 1, 1)
                self.u_scale_tril = Parameter(u_scale_tril)
                #PyroParam(u_scale_tril, constraints.lower_cholesky)
                #self.t_lc = torch.distributions.transforms.LowerCholeskyTransform(cache_size=0)
            else:
                self.u_scale_tril = None
                
        else: # variational GP
            f_loc = torch.tensor(variational_f[0], dtype=self.tensor_type)
            f_scale_tril = torch.tensor(variational_f[1], dtype=self.tensor_type)
            self.f_loc = Parameter(f_loc)
            self.f_scale_tril = Parameter(f_scale_tril)
        
            
    def KL_prior(self):
        """
        Ignores neuron, computes over all the output dimensions, suits coupled models
        """
        if self.gp_mode == 'sparse_MAP': # log p(u)
            zero_loc = self.Xu.new_zeros(self.u_loc.shape)
            if self.whiten:
                identity = eye_like(self.Xu, zero_loc.shape[1])[None, ...].repeat(zero_loc.shape[0], 1, 1)
                p = distributions.Rn_MVN(zero_loc, scale_tril=identity)
            else: # loc (N, N_u), cov (N, N_u, N_u)
                p = distributions.Rn_MVN(zero_loc, scale_tril=self.Luu)
                
            return p.log_prob(self.u_loc).sum()
        
        elif self.gp_mode == 'sparse': # log p(u)/q(u)
            if self.inversion_method == 'cholesky':
                zero_loc = self.u_loc.new_zeros(self.u_loc.shape)
                if self.whiten:
                    identity = eye_like(self.u_loc, zero_loc.shape[1])[None, ...].repeat(zero_loc.shape[0], 1, 1)
                    p = distributions.Rn_MVN(zero_loc, scale_tril=identity)#.to_event(zero_loc.dim() - 1)
                else: # loc (N, N_u), cov (N, N_u, N_u)
                    p = distributions.Rn_MVN(zero_loc, scale_tril=self.Luu)#.to_event(zero_loc.dim() - 1)

                q = distributions.Rn_MVN(self.u_loc, scale_tril=self.u_scale_tril)#.to_event(self.u_loc.dim()-1)
                kl = torch.distributions.kl.kl_divergence(q, p).sum() # sum over neurons
                if torch.isnan(kl).any():
                    kl = 0.
                    print('Warning: sparse GP prior is NaN, ignoring prior term.')
                return -kl
            else:
                return -kl_MVN
            
            #if self.Xu_collide:
        #    self.Xu
            
        else: # log p(f)/q(f)
            zero_loc = self.f_loc.new_zeros(self.f_loc.shape)
            p = distributions.Rn_MVN(zero_loc, scale_tril=self.Lff)
            q = distributions.Rn_MVN(self.f_loc, scale_tril=self.f_scale_tril)
            return torch.distributions.kl.kl_divergence(q, p).sum() # sum over neurons
    
    
    def constrain(self):
        # constrain topological inducing points of sphere
        for k, n in self.constrain_dims:
            L2 = self.Xu[..., k:k+n].data.norm(2, -1)[..., None]
            self.Xu[..., k:k+n].data /= L2
        
        if self.u_scale_tril is not None: # constrain K to be PSD, L diagonals > 0 as needed for det K
            self.u_scale_tril.data = torch.tril(self.u_scale_tril.data)
            Nu = self.u_scale_tril.shape[-1]
            self.u_scale_tril.data[:, np.arange(Nu), np.arange(Nu)] = \
                torch.clamp(self.u_scale_tril.data[:, np.arange(Nu), np.arange(Nu)], min=1e-12)
            
        
    def compute_F(self, XZ):
        """
        Computes :math:`p(f(x)|u)` and also updating :math:`L_{uu}` matrix
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
        
        
        :param X: input regressors with shape (samples, timesteps, dims)
        :returns: loc and covariance matrix (or variance) of :math:`p(f^*(X_{new}))`
        :rtype: tuple(torch.Tensor, torch.Tensor)
        """
        #if self.shared_kernel_params:
        #    loc, cov, self.Luu = linalg.pFU_shared(self.out_dims, X, self.Xu, self.kernel, self.u_loc, self.u_scale_tril,
        #                       cov_type=self.covariance_type, whiten=self.whiten, jitter=self.jitter)
        #else:
        XZ = self._XZ(XZ)
        if self.gp_mode == 'sparse' or self.gp_mode == 'sparse_MAP':
            loc, cov, self.Luu = linalg.p_F_U(self.out_dims, XZ, self.Xu, self.kernel, self.u_loc, self.u_scale_tril,
                                                 cov_type=self.covariance_type, whiten=self.whiten, jitter=self.jitter)
        else:
            loc, cov, self.Lff = linalg.p_F_U(self.out_dims, XZ, self.Xf, self.kernel, self.f_loc, self.f_scale_tril, 
                                                 cov_type=self.covariance_type, whiten=False, jitter=self.jitter)
        
        return loc + self.mean_function(XZ), cov