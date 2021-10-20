import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F

from numbers import Number
import types

import math
import numpy as np


from . import distributions
#from .NF import flows
from .utils import signal

from tqdm.autonotebook import tqdm

#import torch.autograd.profiler as profiler





# Link functions
_link_functions = {
    'exp': lambda x : torch.log(x),
    'softplus': lambda x : torch.where(x > 30, x, torch.log(torch.exp(x) - 1.)), #| log(1+exp(30)) - 30 | < 1e-10, numerically stabilized
    'relu': lambda x : x,
    'identity': lambda x : x, 
    'sigmoid': lambda x: torch.log(x)-torch.log(1.-x)
}

_inv_link_functions = {
    'exp': lambda x : torch.exp(x),
    'softplus': lambda x : F.softplus(x), 
    'relu': lambda x : torch.clamp(x, min=0),
    'identity': lambda x : x, 
    'sigmoid': lambda x: torch.sigmoid(x)
}

_ll_modes = ['MC', 'GH']



# GLM filters
class _filter(nn.Module):
    """
    GLM coupling filter base class.
    """
    def __init__(self, history, conv_groups, tens_type):
        super().__init__()
        self.conv_groups = conv_groups
        self.tensor_type = tens_type
        if history <= 0:
            raise ValueError('History length must be bigger than zero')
        self.history_len = history
        
        
    def forward(self):
        """
        Return filter values.
        """
        raise NotImplementedError
        
        
    def KL_prior(self):
        """
        Prior of the filter model.
        """
        return 0
        
        
    def constrain(self):
        return
    
    
    
# observed
class _data_object(nn.Module):
    """
    Object that will take in data or generates data (leaf node objects in model graph).
    """
    def __init__(self):
        super().__init__()
        
    def setup_batching(self, batches, tsteps, trials):
        """
        :param int/list batches: batch size if scalar, else list of tuple (batch size, batch link), where the 
                                 batch link is a boolean indicating continuity from previous batch
        :param int tsteps: total time steps of data to be batched
        :param int trials: total number of trials in the spike data
        """
        self.tsteps = tsteps
        self.trials = trials
        
        ### batching ###
        if type(batches) == list: # not continuous data
            self.batches = len(batches)
            batch_size = [b[0] for b in batches]
            self.batch_link = [b[1] for b in batches]
        else: # number
            batch_size = batches
            self.batches = int(np.ceil(self.tsteps/batch_size))
            n = self.batches-1
            fin_bs = self.tsteps - n*batch_size
            batch_size = [batch_size]*n + [fin_bs]
            self.batch_link = [True for b in batch_size]
            
        self.batch_link[0] = False # first batch
        self.batch_edge = np.cumsum(np.array([0]+batch_size))

    

# likelihoods
class _likelihood(_data_object):
    """
    Likelihood base class.
    """
    def __init__(self, tbin, F_dims, neurons, inv_link, tensor_type=torch.float, mode='MC', jitter=1e-6):
        """
        :param int F_dims: dimensionality of the inner quantity (equal to rate model output dimensions)
        :param string/LambdaType inv_link:
        :param dtype tensor_type: model tensor type for computation
        :param string mode: evaluation mode of the variational expectation, 'MC' and 'GH'
        """
        super().__init__()
        self.tensor_type = tensor_type
        self.register_buffer('tbin', torch.tensor(tbin, dtype=self.tensor_type))
        self.F_dims = F_dims
        self.neurons = neurons
        self.spikes = None # label as data not set
        
        self.jitter = jitter
        if mode in _ll_modes:
            self.mode = mode
        else:
            raise NotImplementedError('Evaluation method is not supported')
            
        if isinstance(inv_link, types.LambdaType):
            self.f = inv_link
            inv_link = 'custom'
        elif _inv_link_functions.get(inv_link) is None:
            raise NotImplementedError('Link function is not supported')
        else:
            self.f = _inv_link_functions[inv_link]
        self.inv_link = inv_link
        
        
    def set_Y(self, spikes, batch_size, filter_len=1):
        """
        Get all the activity into batches useable format for fast log-likelihood evaluation. 
        Batched spikes will be a list of tensors of shape (trials, neurons, time) with trials 
        set to 1 if input has no trial dimension (e.g. continuous recording).
        
        :param np.array spikes: becomes a list of [neuron_dim, batch_dim]
        :param int/list batch_size: 
        :param int filter_len: history length of the GLM couplings (1 indicates no history coupling)
        """
        if self.neurons != spikes.shape[-2]:
            raise ValueError('Spike data Y does not match neuron count of likelihood')
            
        if len(spikes.shape) == 2: # add in trial dimension
            spikes = spikes[None, ...]
            
        self.setup_batching(batch_size, spikes.shape[-1], spikes.shape[0])
        self.filter_len = filter_len
        
        self.spikes = [] # list of spike trains i.e. arrays of counts
        self.all_spikes = torch.tensor(spikes, dtype=self.tensor_type)
        
        for b in range(self.batches):
            if self.batch_link[b]: # include history of previous batch
                self.spikes.append(self.all_spikes[..., self.batch_edge[b]-filter_len+1:self.batch_edge[b+1]])
            else:
                self.spikes.append(self.all_spikes[..., self.batch_edge[b]:self.batch_edge[b+1]])
        
        
    def KL_prior(self):
        """
        """
        return 0
    
        
    def constrain(self):
        """
        Constrain parameters in optimization
        """
        return
    
    
    def mc_gen(self, q_mu, q_var, samples, neuron):
        """
        Function for generating Gaussian MC samples. No MC samples are drawn when the variance is 0.
        
        :param torch.tensor q_mu: the mean of the MC distribution
        :param torch.tensor q_var: the (co)variance, type (univariate, multivariate) deduced 
                                   from tensor shape
        :param int samples: number of MC samples
        :returns: the samples
        :rtype: torch.tensor
        """
        q_mu = q_mu[:, neuron, :]
        if isinstance(q_var, Number): # zero scalar
            h = q_mu[None, ...].expand(samples, *q_mu.shape).reshape(-1, *q_mu.shape[1:]) if samples > 1 else q_mu
            
        elif len(q_var.shape) == 3:
            q_var = q_var[:, neuron, :]
            if samples == 1: # no expanding
                h = distributions.Rn_Normal(q_mu, q_var.sqrt())()
            else: # shape is (ll_samplesxcov_samples, neurons, time)
                h = distributions.Rn_Normal(q_mu, q_var.sqrt())((samples,)).view(-1, *q_mu.shape[1:])
                
        else: # full covariance
            q_var = q_var[:, neuron, ...]
            q_var.view(*q_var.shape[:2], -1)[:, :, ::q_var.shape[-1]+1] += self.jitter
            h = distributions.Rn_MVN(q_mu, covariance_matrix=q_var)((samples,)).view(-1, *q_mu.shape[1:])
            
        return h
    
    
    def gh_gen(self, q_mu, q_var, points, neuron):
        """
        Computes the Gauss-Hermite quadrature locations and weights.
        
        :param torch.tensor q_mu: the mean of the MC distribution
        :param torch.tensor q_var: the (co)variance, type (univariate, multivariate) deduced 
                                   from tensor shape
        :param int points: number of quadrature points
        :returns: tuple of locations and weights tensors
        :rtype: tuple
        """
        locs, ws = np.polynomial.hermite.hermgauss(points) # sample points and weights for quadrature
        locs = torch.tensor(locs, dtype=self.tensor_type).repeat_interleave(q_mu.shape[0], 0).to(self.tbin.device)
        ws = torch.tensor(1 / np.sqrt(np.pi) * ws / q_mu.shape[0], # q_mu shape 0 is cov_samplesxtrials
                          dtype=self.tensor_type).repeat_interleave(q_mu.shape[0], 0).to(self.tbin.device)
        
        q_mu = q_mu[:, neuron, :].repeat(points, 1, 1) # fill sample dimension with GH quadrature points
        if isinstance(q_var, Number) is False:
            if len(q_mu.shape) > 3: # turn into diagonal
                q_var = q_var.view(*q_var.shape[:2], -1)[:, :, ::q_var.shape[-1]+1]
            q_var = q_var[:, neuron, :].repeat(points, 1, 1)
            h = torch.sqrt(2.*q_var) * locs[:, None, None] + q_mu
            
        else:
            h = q_mu
            
        return h, ws
    
    
    def _validate_neuron(self, neuron):
        """
        """
        if neuron is None:
            neuron = np.arange(self.neurons) # for spike coupling need to evaluate all units
        elif isinstance(neuron, list) is False and np.max(neuron) >= self.neurons:
            raise ValueError('Accessing output dimensions beyond specified dimensions by model')  # avoid illegal access
        return neuron
    
    
    def sample_rate(self, F_mu, F_var, trials, MC_samples=1):
        """
        """
        with torch.no_grad():
            h = self.mc_gen(F_mu, F_var, MC_samples, list(range(self.F_dims)))
            rate = self.f(h.view(-1, trials, *h.shape[1:]))
        return rate
    
    
    def nll(self, inner, inner_var, b, neuron):
        raise NotImplementedError
        
        
    def objective(self, F_mu, F_var, X, b, neuron, samples, mode):
        raise NotImplementedError
        
        
    def sample(self, rate, neuron=None, XZ=None):
        raise NotImplementedError

        
        
# input models
class _input(_data_object):
    """
    with priors and variational distributions attached for VI inference. 
    To set up the input mapping, one has to call *preprocess()* to initialize the input :math:`X` and latent 
    :math:`Z` variables.
    
    Allows priors and SVI for latent variables, and regressors when using observed variables.
    """
    def __init__(self, dims, VI_tuples, tensor_type, latent_f, stimulus_filter):
        super().__init__()
        self.register_buffer('dummy', torch.empty(0)) # keeping track of device
        self.tensor_type = tensor_type
        
        if _inv_link_functions.get(latent_f) is None:
            raise NotImplementedError("Latent link function is not supported.")
        self.lf = _inv_link_functions[latent_f]
        self.lf_inv = _link_functions[latent_f]
        
        self.dims = dims
        self.set_VI(VI_tuples)
        
        self.XZ = None # label as data not set
        if stimulus_filter is not None:
            self.add_module('stimulus_filter', stimulus_filter)
        else:
            self.stimulus_filter = None
            
            
    def _validate_priors(self):
        for cnt, vi in enumerate(self.VI_tuples):
            p, v, topo, dims = vi
            
            if p == 'mapping' or p == 'RW_mapping':
                obj = getattr(self, 'p_inputs_{}'.format(cnt))
                
                if obj.tsteps != (self.tsteps if p == 'mapping' else self.tsteps-1):
                    raise ValueError('Time steps of mapping input does not match expected time steps')
                if obj.trials != self.trials:
                    raise ValueError('Trial count of mapping input does not match expected trial count')
            
            
    def _set_priors(self, topo, cnt, dims, p, prior):
        """
        """
        # priors
        if p == 'Normal':
            if topo == 'torus':
                self.prior_dist.append(distributions.Tn_Normal)
            elif topo == 'SO(3)':
                if dims != 4:
                    raise ValueError('Input dimensionality of this topology must be 4')
                self.prior_dist.append(distributions.SO3_Normal)
            elif topo == 'sphere':
                if dims == 3:
                    self.prior_dist.append(distributions.S2_VMF)
                elif dims == 4:
                    self.prior_dist.append(distributions.S3_Normal)
                else:
                    raise NotImplementedError("{}-sphere not supported.".format(dims-1))
            elif topo == 'euclid':
                self.prior_dist.append(distributions.Rn_Normal)
            else:
                raise NotImplementedError("Topology not supported.")
            self.rw_dist.append(None)

        elif p == 'MVN':
            if topo == 'euclid':
                self.variational.append(distributions.Rn_MVN)
            elif topo == 'torus':
                self.variational.append(distributions.Tn_MVN)
            else:
                raise NotImplementedError("Topology not supported.")

        elif p == 'Uniform':
            if topo == 'torus':
                self.prior_dist.append(distributions.Tn_Uniform)
            elif topo == 'SO(3)':
                if dims != 4:
                    raise ValueError('Input dimensionality of this topology must be 4')
                self.prior_dist.append(distributions.SO3_Uniform)
            elif topo == 'sphere':
                self.prior_dist.append(distributions.Sn_Uniform)
            elif topo == 'euclid':
                self.prior_dist.append(distributions.Rn_Uniform)
            else:
                raise NotImplementedError("Topology not supported.")
            self.rw_dist.append(None)

        elif p == 'RW':
            self.AR_p = 1

            if topo == 'torus':
                self.prior_dist.append(distributions.Tn_Uniform)
                self.rw_dist.append(distributions.Tn_Normal)
            elif topo == 'sphere':
                self.prior_dist.append(distributions.Sn_Uniform)
                if dims == 3: # mean vector length
                    self.rw_dist.append(distributions.S2_VMF)
                elif dims == 4:
                    self.rw_dist.append(distributions.S3_Normal)
                else:
                    raise NotImplementedError("{}-sphere not supported.".format(dims-1))
            elif topo == 'euclid':
                self.prior_dist.append(distributions.Rn_Normal)
                self.rw_dist.append(distributions.Rn_Normal)
            else:
                raise NotImplementedError("Topology not supported.")

        elif p == 'RW_mapping':
            self.AR_p = 1

            if topo == 'torus':
                self.prior_dist.append(distributions.Tn_Uniform)
            elif topo == 'euclid':
                self.prior_dist.append(distributions.Rn_Normal)
            else:
                raise NotImplementedError("Topology not supported.")
            self.rw_dist.append(None)

        elif p == 'mapping' or p is None:
            self.prior_dist.append(None)
            self.rw_dist.append(None)

        else:
            raise NotImplementedError("Prior distribution not supported.")
             
        # set parameters or modules
        if p == 'mapping' or p == 'RW_mapping':
            if prior[-1].covariance_type != 'diagonal': # needs to output Gaussian distribution
                raise ValueError('')
            if prior[-1].out_dims != dims:
                raise ValueError('Output dimensions of mapping do not match expected dimensions')

            self.add_module('p_inputs_{}'.format(cnt), prior[-2].to(self.dummy.device))
            self.add_module('p_mapping_{}'.format(cnt), prior[-1].to(self.dummy.device))

        if p == 'mapping' or self.prior_dist[cnt] is None:
            return
        elif p == 'Uniform':
            if topo is not 'euclid': # specify bounds of uniform domain in euclid
                if dims > 1:
                    prior = [np.zeros(dims), np.zeros(dims)]
                else:
                    prior = [0.0, 0.0]

            self.register_buffer('p_mu_{}'.format(cnt), torch.tensor(prior[0], 
                                 dtype=self.tensor_type).to(self.dummy.device)) # lower limit, used from dim
            self.register_buffer('p_std_{}'.format(cnt), torch.tensor(prior[1], 
                                 dtype=self.tensor_type).to(self.dummy.device)) # upper limit
        else: # standard loc std formulation
            if prior[2] is True:
                self.register_parameter('p_mu_{}'.format(cnt), Parameter(torch.tensor(prior[0],
                                                                                      dtype=self.tensor_type).to(self.dummy.device)))
            else:
                self.register_buffer('p_mu_{}'.format(cnt), torch.tensor(prior[0], 
                                                                         dtype=self.tensor_type).to(self.dummy.device))

            if prior[3] is True:
                self.register_parameter('p_std_{}'.format(cnt), Parameter(self.lf_inv(torch.tensor(prior[1], 
                                                                                       dtype=self.tensor_type).to(self.dummy.device))))
            else:
                self.register_buffer('p_std_{}'.format(cnt), self.lf_inv(torch.tensor(prior[1], 
                                                                          dtype=self.tensor_type).to(self.dummy.device)))
        
    
    def set_VI(self, VI_tuples):
        """
        Set prior and variational distribution pairs, check dimensionality with model dimensionality.
        
        :param list VI_tuples: list of tuples
        """
        self.prior_dist = []
        self.rw_dist = []
        self.variational = []
        self.VI_tuples = []
        
        self.AR_p = 0 # default
        for cnt, vi in enumerate(VI_tuples):
            p_, v, topo, dims = vi
            
            if p_ is None:
                p, prior_params = None, None
            else:
                p, prior_params = p_
                
            self.VI_tuples.append((p, v, topo, dims))
            
            # prior distributions
            self._set_priors(topo, cnt, dims, p, prior_params)
                
            # variational distributions
            if v == 'Normal' or v == 'VAE' or v == 'NF':
                if topo == 'torus':
                    self.variational.append(distributions.Tn_Normal)
                elif topo == 'sphere':
                    if dims == 3:
                        self.variational.append(distributions.S2_VMF)
                    elif dims == 4:
                        self.variational.append(distributions.S3_Normal)
                elif topo == 'euclid':
                    self.variational.append(distributions.Rn_Normal)
                else:
                    raise NotImplementedError("Topology not supported.")
                    
            elif v == 'MVN':
                if topo == 'euclid':
                    self.variational.append(distributions.Rn_MVN)
                elif topo == 'torus':
                    self.variational.append(distributions.Tn_MVN)
                else:
                    raise NotImplementedError("Topology not supported.")
    
            elif v == 'Delta':
                self.variational.append(distributions.Delta)
                    
            elif v is None:
                self.variational.append(None)
                
            else:
                raise NotImplementedError("Variational distribution not supported.")
        
        if self.dims != sum([d[3] for d in self.VI_tuples]):
            raise ValueError('Input dimensionality of model does not match total dimensions in VI blocks')
        
        
    def set_XZ(self, input_data, timesteps, batch_size, trials=1, filter_len=1):
        """
        Preprocesses input data for training. Batches the input data and takes care of setting 
        priors on latent dimensions, as well as initializing the SVI framework. Modules and Parameters 
        are moved to the current device.
        
        The priors are specified by setting the variables `p_mu_{}` and `p_std_{}`. In the case of a GP 
        prior ('mapping'), the mapping module is placed in the variable `p_mapping_{}` with its input 
        group in the variable `p_inputs_{}`. The differential mapping prior ('RW_mapping') requires 
        specification of both the GP module and prior distributions.
        
        The latent variables are initialized by specifying the moments of the variational distribution 
        through the variables `lv_mu_{}` and `lv_std_{}`, representing the mean and the scale. The mean 
        can be higher dimensional, the scale value is one-dimensional.
        
        The random walk prior ('RW') in the torus has a learnable offset and standard deviation for the 
        transition distribution. The euclidean version is the linear dynamical system, with p_mu being 
        the decay factor, p_std the stationary standard deviation. p_mu < 1 to remain bounded.
        
        Batching AR processes or GLM with history takes into account the overlap of batches when continuous 
        i.e. batches are linked temporally.
        
        :param np.array input_data: input array of observed regressors of shape (timesteps,) or (timesteps, 
                                    dims) when the event shape of this block is bigger than 1, or 
                                    input array of latent initialization as a list of [prior_tuple, var_tupe] 
                                    with (p_mu, p_std, learn_mu, learn_std, GP_module) as prior_tuple and 
                                    var_tuple (mean, std)
        :param int/list batch_size: batch size to use, if list this indicates batches separated temporally
        """
        if len(input_data) != len(self.VI_tuples):
            raise ValueError('Input dimensionalities do not match VI block structure')
        
        self.setup_batching(batch_size, timesteps, trials)
        self._validate_priors()
        self.filter_len = filter_len
        if self.stimulus_filter is not None and self.stimulus_filter.history_len != filter_len:
            raise ValueError('Stimulus filter length and input filtering length do not match')
        
        ### read the input ###
        cov_split = []
        self.regressor_mode = True # if covariates is all regressor values, pack into array later
        for cnt, vi in enumerate(self.VI_tuples):
            p, v, topo, dims = vi
            cov = input_data[cnt]
            
            if isinstance(cov, list): # check if variable is a regressor or a latent (list type)
                self.regressor_mode = False
                if p is None:
                    raise ValueError('Prior is not set for latent variable')
                if v is None:
                    raise ValueError('Variational type is not set for latent variable')
                    
                if v == 'VAE':
                    if isinstance(cov[0], nn.Module) is False:
                        raise ValueError('Input object must be of type torch.nn.Module')
                    self.regressor_mode = False
                    self.add_module('lv_mu_{}'.format(cnt), cov[0].to(self.dummy.device))
                    self.add_module('lv_std_{}'.format(cnt), cov[1].to(self.dummy.device))

                else: # initialize variational parameters
                    self.regressor_mode = False
                    
                    if cov[0].shape[0] != self.tsteps:
                        raise ValueError('Expected time steps do not match given initial latent means')
                    self.register_parameter(
                        'lv_mu_{}'.format(cnt), 
                        Parameter(torch.tensor(cov[0], dtype=self.tensor_type).to(self.dummy.device))
                    ) # mean
                    
                    if v is not 'Delta':
                        if cov[1].shape[0] != self.tsteps:
                            raise ValueError('Expected time steps do not match given initial latent standard deviations')
                        self.register_parameter(
                            'lv_std_{}'.format(cnt), 
                            Parameter(self.lf_inv(torch.tensor(cov[1], dtype=self.tensor_type).to(self.dummy.device)))
                        ) # std
                        
                        if v == 'NF':
                            if isinstance(cov[2], flows.FlowSequence) is False:
                                raise ValueError('Must provide a FlowSequence object')
                            self.add_module('lv_NF_{}'.format(cnt), cov[2].to(self.dummy.device))
                        
                cov_split.append(None)
                
            else: # allow to specify priors on observed variables
                cov = torch.tensor(cov, dtype=self.tensor_type)
                if len(cov.shape) == 1: # expand arrays from (timesteps,)
                    cov = cov[None, :, None]
                elif len(cov.shape) == 2: # expand arrays (timesteps, dims)
                    cov = cov[None, ...]
                
                if cov.shape[0] != self.trials:
                    raise ValueError('Trial count does not match trial count in covariates')
                if cov.shape[-2] != self.tsteps:
                    raise ValueError('Expected time steps do not match given covariates')
                if len(cov.shape) != 3: # trials, timesteps, dims
                    raise ValueError('Shape of input covariates at most trials x timesteps x dims')
                
                cov_split_ = []
                for b in range(self.batches):
                    if self.batch_link[b]:
                        cov_split_.append(cov[:, self.batch_edge[b]-self.filter_len+1:self.batch_edge[b+1], :])
                    else:
                        cov_split_.append(cov[:, self.batch_edge[b]:self.batch_edge[b+1], :])
                cov_split.append(cov_split_)
            
        ### assign input to storage ###
        if self.regressor_mode: # turn into compact tensor arrays
            self.XZ = []
            for b in range(self.batches): # final shape (trials, time, dims)
                self.XZ.append(torch.cat(tuple(cov[b] for cov in cov_split), dim=-1)) 
        else:
            self.XZ = cov_split
            
            
    def constrain(self):
        """
        """
        return
    
    
    def _get_offset(self, b):
        offs = self.filter_len-1 if b > 0 else 0 # offset due to history 
        if self.batch_link[b]: # overlapping points from neighbouring batches
            offs += self.AR_p
        return offs
    
    
    def _VAE(self, k, net_input):
        """
        """
        mu_net = getattr(self, 'lv_mu_{}'.format(k))
        std_net = getattr(self, 'lv_std_{}'.format(k))
        
        offs = self._get_offset(b)
        X_loc_ = net_input[self.batch_edge[b]-offs:self.batch_edge[b+1], :]
            
        mu, std = mu_net(X_loc_), self.lf(std_net(X_loc_)) # time, moment_dims
        if dims == 1: # standard one dimensional arrays
            mu, std = mu[:, 0], self.lf(std[:, 0])
        
        return mu, std
    
    
    def _latents(self, b, k, dims, topo):
        """
        """
        l = getattr(self, 'lv_mu_{}'.format(k))
        if topo == 'sphere': # normalize 
            l = l/l.data.norm(2, -1)[..., None]
        cc = torch.cat((l, getattr(self, 'lv_std_{}'.format(k))), dim=1) # (time, moment_dim), (,trial) only if > 1
        
        offs = self._get_offset(b)
        X_loc_ = cc[self.batch_edge[b]-offs:self.batch_edge[b+1], :]
            
        if self.trials == 1 and dims == 1: # standard one dimensional arrays, fast
            mu, std = X_loc_[:, 0], self.lf(X_loc_[:, 1])
        else: # explicit event (and trial) dimension
            mu, std = X_loc_[:, :dims], self.lf(X_loc_[:, dims:])
        
        return mu, std
    
    
    def _mapping(self, batch, k, dims, net_input, entropy):
        """
        """
        mapping_in = getattr(self, 'p_inputs_{}'.format(k))
        mapping = getattr(self, 'p_mapping_{}'.format(k))
        t_, log_prior, nlog_q, KL_prior_m = mapping_in.sample_XZ(batch, 1, net_input=net_input, 
                                                                entropy=entropy) # samples, timesteps, dims
        KL_prior = KL_prior_m + mapping.KL_prior()

        f_loc, f_var = mapping.compute_F(t_) # samples, output, timesteps
        f_loc, f_var = f_loc.permute(0, 2, 1), f_var.permute(0, 2, 1) # K, timesteps, output
        if self.trials == 1 and dims == 1: # convert to scalar event dimension
            f_loc, f_var = f_loc[..., 0], f_var[..., 0]
            
        return f_loc, f_var, log_prior, nlog_q, KL_prior
    
    
    def _nlog_q(self, vd, v_samp, entropy, start=0):
        """
        """
        if entropy:
            return vd.entropy(v_samp)[start:].sum()
        else:
            vt = vd.log_prob(v_samp)[:, start:]
            return -vt.sum(axis=tuple(range(1,len(vt.shape))))
        
        
    def _log_p(self, pd, v_samp):
        """
        """
        pt = pd.log_prob(v_samp)
        return pt.sum(axis=tuple(range(1,len(pt.shape))))
    
    
    def _time_slice(self, XZ):
        """
        """
        if self.stimulus_filter is not None:
            _XZ = self.stimulus_filter(XZ.permute(0, 2, 1))[0].permute(0, 2, 1) # ignore filter variance
            KL_prior = self.stimulus_filter.KL_prior()
        else:
            _XZ = XZ[:, self.filter_len-1:, :] # covariates has initial history part excluded
            KL_prior = 0
            
        return _XZ, KL_prior
    
    
    def _XZ(self, XZ, samples):
        """
        Expand XZ to standard shape
        """
        trials, bs, inpd = XZ.shape
        
        if trials != self.trials:
            raise ValueError('Trial number in input does not match expected trial number')
            
        if trials > 1: # cannot rely on broadcasting
            _XZ = XZ[None, ...].repeat(samples, 1, 1, 1).view(-1, bs, inpd).to(self.dummy.device)
        else:
            _XZ = XZ.expand(samples, bs, inpd).to(self.dummy.device)
            
        return _XZ
    
    
    def prior_moments(self, k):
        """
        """
        loc = getattr(self, 'p_mu_{}'.format(k))
        std = getattr(self, 'p_std_{}'.format(k))
        return loc, self.lf(std)
            
    
    def sample_XZ(self, b, samples, net_input=None, entropy=False):
        """
        Draw samples from the covariate distribution, provides an implementation of SVI.
        In [1] we amortise the variational parameters with a recognition network. Note the input 
        to this network is the final output, i.e. all latents in each layer of a deep GP are mapped 
        from the final output layer.
        
        History in GLMs is incorporated through the offset before the batch_edge values indicating the 
        batching timepoints. AR(1) structure is incorporated by appending the first timestep of the next 
        batch into the current one, which is then used for computing the Markovian prior.
        
        Note that in LVM + GLM, the first filter_len-1 number of LVs is not inferred.
        
        Note when input is trialled, we can only have MC samples flag equal to trials (one per trial).
        
        :param int b: the batch ID to sample from
        :param int samples: the number of samples to take
        :param torch.tensor net_input: direct input for VAE amortization of shape (dims, time)
        :returns: covariates sample of shape (samples, timesteps, dims), log_prior
        :rtype: tuple
        
        References:
        
        [1] `Variational auto-encoded deep Gaussian Processes`,
        Zhenwen Dai, Andreas Damianou, Javier Gonzalez & Neil Lawrence
        
        """
        if self.regressor_mode: # regressor mode, no lv, trial blocks are repeated
            XZ, kl_stim = self._time_slice(self._XZ(self.XZ[b], samples))
            dummy = torch.zeros(samples*self.trials).to(self.dummy.device)
            return XZ, dummy, 0 if entropy else dummy, kl_stim
            
        ### LVM ###
        log_prior = 0
        nlog_q = 0
        KL_prior = 0
        cov = []
            
        for k, vi in enumerate(self.VI_tuples):
            p, v, topo, dims = vi
            
            if p is None or isinstance(self.XZ[k], list): # regressor variable
                cov.append(self._XZ(self.XZ[k][b], samples))
                continue
                
            ### continuous latent variables ###      
            if v == 'VAE': # compute the variational parameters
                mu, std = self._VAE(k, net_input.t().to(self.dummy.device))
            else:
                mu, std = self._latents(b, k, dims, topo)       

            vd = self.variational[k](mu, std)#.to_event()
            v_samp = vd((samples,)) # samples, time, event_dims
            if v == 'NF':
                NF_net = getattr(self, 'lv_NF_{}'.format(k))
                v_samp, log_jacob = NF_net(v_samp)

            ### prior setup ###            
            if self.AR_p == 1: # AR(1) structure
                
                if p == 'RW_mapping':
                    loc, std = self.prior_moments(k)
                    pd = self.prior_dist[k](loc, std)
                    
                    f_loc, f_var, log_p_m, nlog_q_m, kl_p = self._mapping(b, k, dims, net_input, entropy)
                    
                    log_prior = log_prior + log_p_m
                    nlog_q = nlog_q + nlog_q_m
                    KL_prior = KL_prior + kl_p
                    
                    #if GP.cov_type == 'full':
                    #    rd = distributions.Rn_MVN(f_loc, covariance_matrix=f_var)
                    rd = distributions.Rn_Normal(f_loc, f_var.sqrt())
                    
                    dv = v_samp[:, 1:]-v_samp[:, :-1]
                    if topo == 'torus': # geodesic distance
                        dv = dv % (2*np.pi)
                        dv[dv > np.pi] = dv[dv > np.pi] - 2*np.pi
                    
                    rw_term = self._log_p(rd, dv)

                elif p == 'RW':                    
                    loc, std = self.prior_moments(k)

                    if topo == 'euclid': # stationary linear 
                        loc =  torch.sigmoid(loc)
                        rd = self.rw_dist[k](loc*v_samp[:, :-1], std*torch.sqrt(1-loc**2))
                        pd = self.prior_dist[k](0., std)
                    elif topo == 'torus': # drift DS
                        rd = self.rw_dist[k](v_samp[:, :-1]+loc, std)
                        pd = self.prior_dist[k](loc)
                    elif topo == 'sphere':
                        rd = self.rw_dist[k](v_samp[:, :-1, :], std)
                        pd = self.prior_dist[k](loc)#.to_event()

                    rw_term = self._log_p(rd, v_samp[:, 1:])
                                 
                if self.batch_link[b]:
                    prior_term = rw_term
                else: # add log prob of first initial time step
                    prior_term = rw_term + self._log_p(pd, v_samp[:, 0:1]) # sum over time (and dims if > 1)
                    
                start = self.AR_p if self.batch_link[b] else 0 # avoid overlap from AR structure
                # RW prior case, don't double count edge parameter, v_samp contains extra overlap when batch_link True
                var_term = self._nlog_q(vd, v_samp, entropy, start)
                v_samp_ = v_samp[:, start:]
                    
            else:
                if p == 'mapping':
                    f_loc, f_var, log_p_m, nlog_q_m, kl_p = self._mapping(b, k, dims, net_input, entropy)
                    
                    log_prior = log_prior + log_p_m
                    nlog_q = nlog_q + nlog_q_m
                    KL_prior = KL_prior + kl_p
                        
                    #if GP.cov_type == 'full':
                    #    pd = distributions.Rn_MVN(f_loc[0], covariance_matrix=f_var[0])
                    pd = distributions.Rn_Normal(f_loc, f_var.sqrt())

                else:
                    loc, std = self.prior_moments(k)
                    pd = self.prior_dist[k](loc, std)#.to_event()
                    
                prior_term = self._log_p(pd, v_samp)
                var_term = self._nlog_q(vd, v_samp, entropy)
                v_samp_ = v_samp
            
            ### compute log probability terms ###
            fac = self.tsteps / v_samp_.shape[1] # subsampling in batched mode
            log_prior = log_prior + fac*prior_term
            nlog_q = nlog_q + fac*var_term
            if len(v_samp_.shape) == 2: # one-dimensional latent variables expanded
                v_samp_ = v_samp_[..., None]
            cov.append(v_samp_)

        XZ, kl_stim = self._time_slice(torch.cat(cov, dim=-1))
        return XZ, log_prior, nlog_q, KL_prior+kl_stim
    
    
    def eval_XZ(self, net_input=None):
        """
        Evaluate the (latent) input variables. The latent variable structure is defined by the input 
        structure to the variational distributions specified.
        
        :param torch.tensor net_input: the input to the VAE for fast inference of shape (time, moment)
        :returns: a list with elements of shape (timesteps,) for input variables, if separate trial runs, 
                 we get additional first list dimension over runs/trials
        :rtype: list
        """
        X_loc, X_std = [], []
        
        ### retrieve/compute the input variables ###
        if self.regressor_mode:
            c = torch.cat(tuple(cov for cov in self.XZ), dim=1).permute(2, 1, 0)
            X_loc = list(c.cpu().numpy()) # dims x (timesteps, samples)
            X_std = list(np.zeros_like(c.cpu().numpy()))
            
        else:  
            with torch.no_grad():
                for k, vi in enumerate(self.VI_tuples):
                    p, v, topo, dims = vi
                    
                    if self.XZ[k] is not None:
                        c = torch.cat(tuple(cov for cov in self.XZ[k]), dim=1)
                        c = torch.cat((c, torch.zeros_like(c)), dim=-1).permute(1, 2, 0) # timesteps, moment_dim, samples
                    elif v == 'VAE':
                        input = net_input.t().to(self.dummy.device)
                        mu = getattr(self, 'lv_mu_{}'.format(k))(input)
                        std = self.lf(getattr(self, 'lv_std_{}'.format(k))(input))
                        c = torch.cat((mu, std), dim=-1) # (timesteps, moment_dim)
                    else:
                        if topo == 'sphere': # normalize
                            l = getattr(self, 'lv_mu_{}'.format(k))
                            l.data /= l.data.norm(2, -1)[..., None]
                        c = torch.cat((getattr(self, 'lv_mu_{}'.format(k)), # (time, moment_dim)
                                       self.lf(getattr(self, 'lv_std_{}'.format(k)))), dim=1)

                    if dims == 1:
                        X_loc.append(c[:, 0].cpu().numpy())
                        X_std.append(c[:, 1].cpu().numpy())
                    else:
                        X_loc.append(c[:, :dims].cpu().numpy())
                        X_std.append(c[:, dims:].cpu().numpy())
        
        ### deal with continuity of data ###        
        X_loc_, X_std_ = [], []
        for k, vi in enumerate(self.VI_tuples):
            p, v, topo, dims = vi
            
            if v == 'VAE': # no batching
                X_loc_.append(X_loc[k])
                X_std_.append(X_std[k])
                continue

            _X_loc, _X_std = [], []
            for b in range(self.batches):
                if self.batch_link[b] is False:
                    if b > 0: # safe current data
                        _X_loc.append(np.concatenate(X_loc__, axis=0))
                        _X_std.append(np.concatenate(X_std__, axis=0))
                    X_loc__, X_std__ = [], []
                    
                X_loc__.append(X_loc[k][self.batch_edge[b]:self.batch_edge[b+1]])
                X_std__.append(X_std[k][self.batch_edge[b]:self.batch_edge[b+1]])
                
            # final segment or whole segment
            _X_loc.append(np.concatenate(X_loc__, axis=0))
            _X_std.append(np.concatenate(X_std__, axis=0))
            
            if len(_X_loc) > 1: # multiple segments
                X_loc_.append(_X_loc)
                X_std_.append(_X_std)
            else: # continuous segment
                X_loc_.append(_X_loc[0])
                X_std_.append(_X_std[0])

        return X_loc_, X_std_



# input output mapping models
_covariance_types = [None, 'full', 'diagonal']



class _input_mapping(nn.Module):
    """
    Input covariates to mean and covariance parameters. An input mapping consists of a mapping from input 
    to inner and inner_var quantities.
    """
    def __init__(self, input_dims, out_dims, inv_link, covariance_type=None, 
                 tensor_type=torch.float, active_dims=None):
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
        self.register_buffer('dummy', torch.empty(0)) # keeping track of device
        
        self.tensor_type = tensor_type
        self.out_dims = out_dims
        self.input_dims = input_dims
        
        if active_dims is None:
            active_dims = list(range(input_dims))
        elif len(active_dims) != input_dims:
            raise ValueError('Active dimensions do not match expected number of input dimensions')
        self.active_dims = active_dims # dimensions to select from input covariates
        
        if covariance_type in _covariance_types:
            self.covariance_type = covariance_type
        else:
            raise ValueError('Covariance type of mapping is not recognized')
        
        if isinstance(inv_link, types.LambdaType):
            self.f = inv_link
            inv_link = 'custom'
        elif _inv_link_functions.get(inv_link) is None:
            raise NotImplementedError('Link function is not supported')
        else:
            self.f = _inv_link_functions[inv_link]
        self.inv_link = inv_link
    
        
    def compute_F(self, XZ):
        """
        Computes the posterior over :mathm:`F`, conditioned on the data. In most cases, this is amortized via 
        learned weights/parameters of some approximate posterior. In ML/MAP settings the approximation is a 
        delta distribution, meaning there is no variational uncertainty in the mapping.
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
        if max(self.active_dims) >= XZ.shape[-1]:
            raise ValueError('Active dimensions is outside input dimensionality provided')
        return XZ[..., self.active_dims]
    
    
    def to_XZ(self, covariates, trials=1):
        """
        Convert covariates list input to tensors for input to mapping. Convenience function for rate 
        evaluation functions and sampling functions.
        """
        cov_list = []
        timesteps = None
        for cov_ in covariates:
            if len(cov_.shape) == 1: # expand arrays from (timesteps,)
                cov_ = cov_[None, :, None]
            elif len(cov_.shape) == 2: # expand arrays (timesteps, dims)
                cov_ = cov_[None, ...]
            if len(cov_.shape) != 3:
                raise ValueError('Covariates are of invalid shape')

            if timesteps is not None:
                if timesteps != cov_.shape[1]:
                    raise ValueError('Time steps in covariates dimensions are not consistent')
            else:
                timesteps = cov_.shape[1]

            cov_list.append(torch.tensor(cov_, device=self.dummy.device,
                                         dtype=self.tensor_type))
        XZ = torch.cat(cov_list, dim=-1).expand(trials, timesteps, -1)
        return XZ
    
    
    def _samples(self, covariates, neuron, n_samp, full_cov, trials):
        """
        Evaluates samples from the variational posterior for the rate.
        
        :param bool full_cov: flag if True allows to draw function samples, False allows faster uncertainty
        """
        XZ = self.to_XZ(covariates, trials) # samples, timesteps, dims
        
        with torch.no_grad(): # compute predictive mean and variance
            mean, var = self.compute_F(XZ)
            if isinstance(var, Number): # deterministic mapping
                samples = self.f(mean)
            else:
                mean, var = mean[:, neuron, :], var[:, neuron, :]

                if full_cov:
                    var.view(trials, len(neuron), -1)[:, :, ::var.shape[-1]+1] += self.jitter
                    samples = self.f(distributions.Rn_MVN(mean, covariance_matrix=var)((n_samp,)))
                else:
                    samples = self.f(distributions.Rn_Normal(mean, var.sqrt())((n_samp,)))
            
        return samples
        
        
    def eval_rate(self, covariates, neuron, mode='mean', percentiles=[0.05, 0.95], n_samp=1000, trials=1):
        """
        Evaluate the rate of the model as function of input covariates.
        
        :param np.array covariates: input covariates to evaluate rate over with shape (dims, timesteps)
        :param list neuron: neuron indices over which to evaluate rate
        :param string mode: what to evaluate, `mean` gives posterior mean (ML/MAP models this is the point 
                            estimate equivalently), `posterior` gives the percentile lines, `tuning` gives 
                            individual tuning curve samples from the variational posterior
        :param list percentiles: which percentiles to evaluate in `posterior` mode in addition to the mean
        :param int n_samp: the number of MC samples to use for evaluating posterior statistics
        :returns: output rate to [neurons, steps] or [steps] depending on neuron array or scalar
        :rtype: np.array
        """
        if np.max(neuron) >= self.out_dims: # avoid illegal access
            raise ValueError('Accessing output dimensions beyond specified dimensions by model')
        if ((mode == 'mean') or (mode == 'posterior') or (mode == 'tuning')) is False:
            raise ValueError('Rate evaluation mode is not recognized')
        
        if self.covariance_type is None: # no variational uncertainty
            if mode != 'mean':
                raise ValueError('Mapping has no variational uncertainty, hence must evaluate in mean mode')
            with torch.no_grad(): # will set regressor_mode to True
                XZ = self.to_XZ(covariates, trials)
                F_mu, _ = self.compute_F(XZ)

            return self.f(F_mu[:, neuron, :]).data.cpu().numpy()
     
        samples = self._samples(covariates, neuron, n_samp, False, trials)
        mean = samples.mean(0).cpu()
        if mode == 'mean':
            return mean.numpy()
        elif mode == 'posterior':
            lower, upper = signal.percentiles_from_samples(samples.cpu().float(), percentiles)
            return lower.numpy(), mean.numpy(), upper.numpy()
        else:
            samples = self._samples(covariates, neuron, n_samp, True, trials)
            return samples.cpu().numpy()