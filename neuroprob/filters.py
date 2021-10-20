import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F

import numpy as np
from numbers import Number

from . import base
from . import distributions



### filters ###
class raised_cosine_bumps(base._filter):
    """
    Raised cosine basis, takes the form of 
    
    .. math:: p(f^* \mid X_{new}, X, y, k, X_u, u_{loc}, u_{scale\_tril})
            = \mathcal{N}(loc, cov).
            
    """
    def __init__(self, a, c, phi, w, timesteps, learnable=[True, True, True, True], tens_type=torch.float):
        """
        Raised cosine basis as used in the literature.
        
        .. note::
        
            Stimulus history object needs to assert (w_u.shape[-1] == 1 or w_u.shape[-1] == self.rate_model.out_dims)
            self.register_parameter('w_u', Parameter(torch.tensor(w_u).float())) # (basis, 1, s_len) or (basis, neurons, s_len)
            w_h # (basis, neurons, neurons)
            w_h # (basis, neurons) for self-coupling only with conv_groups = neurons
            self.register_parameter('phi_u', Parameter(torch.tensor(phi_u).float()))
            
            conv_groups = 1 # all-to-all couplings
            conv_groups = neurons # only self-couplings
        
        :param nn.Parameter w: component weights of shape (basis, post, pre)
        :param nn.Parameter phi: basis function parameters of shape (basis, post, pre)
        
        """
        if w.shape != phi.shape:
            raise ValueError('Parameters w and phi must match in shape')
        if len(w.shape) == 2:
            w = w[..., None]
            phi = phi[..., None]
        else:
            if len(w.shape) != 3:
                raise ValueError('Parameters w must have 3 array axes')
            
        conv_groups = phi.shape[1]//phi.shape[2]
        super().__init__(timesteps, conv_groups=conv_groups, tens_type=tens_type)

        if learnable[3]:
            self.register_parameter('w', Parameter(torch.tensor(w, dtype=self.tensor_type)))
        else:
            self.register_buffer('w', torch.tensor(w, dtype=self.tensor_type))
        if learnable[2]:
            self.register_parameter('phi', Parameter(torch.tensor(phi, dtype=self.tensor_type)))
        else:
            self.register_buffer('phi', torch.tensor(phi, dtype=self.tensor_type))
        if learnable[0]:
            self.register_parameter('a', Parameter(torch.tensor(a, dtype=self.tensor_type)))
        else:
            self.a = a
        if learnable[1]:
            self.register_parameter('c', Parameter(torch.tensor(c, dtype=self.tensor_type)))
        else:
            self.c = c


    def compute_filter(self):
        """
        :returns: filter of shape (post, pre, timesteps)
        :rtype: torch.tensor
        """
        t_ = torch.arange(self.history_len, device=self.w.device, dtype=self.tensor_type)
        t = self.history_len-t_
        A = torch.clamp(self.a*torch.log(t+self.c)[None, None, None, :] - \
                        self.phi[..., None], min=-np.pi, max=np.pi) # (basis, post, pre, timesteps)
        return (self.w[..., None]*.5*(torch.cos(A) + 1.)).sum(0)
    
    
    def forward(self, input, stimulus=None):
        """
        Introduces the spike coupling by convolution with the spike train, no padding and left removal 
        for causal convolutions.
        
        :param torch.tensor input: input spiketrain or covariates with shape (trials, neurons, timesteps) 
                                   or (samples, neurons, timesteps)
        :returns: filtered input of shape (trials, neurons, timesteps)
        """
        h_ = self.compute_filter()
        return F.conv1d(input, h_, groups=self.conv_groups), 0
    
    
    
class hetero_raised_cosine_bumps(base._filter):
    """
    Raised cosine basis, takes the form of 
    
    .. math:: p(f^* \mid X_{new}, X, y, k, X_u, u_{loc}, u_{scale\_tril})
            = \mathcal{N}(loc, cov).
            
    """
    def __init__(self, a, c, phi, timesteps, hetero_model, learnable=[True, True, True], inner_loop_bs=100, 
                 tens_type=torch.float):
        """
        Raised cosine basis as used in the literature, with basis function amplitudes :math:`w` given 
        by a Gaussian process.
        
        :param nn.Parameter phi: basis function parameters of shape (basis, pre, post)
        
        """
        if len(phi.shape) == 2:
            phi = phi[..., None]
        else:
            if len(phi.shape) != 3:
                raise ValueError('Parameters w must have 3 array axes')
            
        conv_groups = phi.shape[1]//phi.shape[2]
        super().__init__(timesteps, conv_groups=conv_groups, tens_type=tens_type)

        if learnable[2]:
            self.register_parameter('phi', Parameter(torch.tensor(phi, dtype=self.tensor_type)))
        else:
            self.register_buffer('phi', torch.tensor(phi, dtype=self.tensor_type))
        if learnable[0]:
            self.register_parameter('a', Parameter(torch.tensor(a, dtype=self.tensor_type)))
        else:
            self.a = a
        if learnable[1]:
            self.register_parameter('c', Parameter(torch.tensor(c, dtype=self.tensor_type)))
        else:
            self.c = c
        
        self.inner_bs = inner_loop_bs                 
        self.add_module('hetero_model', hetero_model)

        
    def compute_filter(self, stim):
        """
        :param torch.tensor stim: input covariates to the GP of shape (sample x timesteps x dims)
        :returns: filter of shape (post, pre, timesteps)
        :rtype: torch.tensor
        """
        F_mu, F_var = self.hetero_model.compute_F(stim) # K, neurons, T

        w = F_mu.permute(0, 2, 1).reshape(-1, *self.phi.shape) # KxT, basis, post, pre 
        t_ = torch.arange(self.history_len, device=self.phi.device, dtype=self.tensor_type)
        t = self.history_len-t_
        A = torch.clamp(self.a*torch.log(t+self.c)[None, None, None, :] - \
                        self.phi[..., None], min=-np.pi, max=np.pi) # (basis, post, pre, timesteps)
        
        filt_mean = (w[..., None]*.5*(torch.cos(A[None, ...]) + 1.)).sum(1) # KxT, post, pre, timesteps
        if isinstance(v_, Number) is False:
            w_var = F_var.permute(0, 2, 1).reshape(-1, *self.phi.shape)
            filt_var = (w_var[..., None]*.5*(torch.cos(A[None, ...]) + 1.)).sum(1)
        else:
            filt_var= 0
            
        return filt_mean, filt_var
    
    
    def KL_prior(self):
        return self.hetero_model.KL_prior()
    
    
    def forward(self, input, stimulus):
        """
        Introduces stimulus-dependent raised cosine basis. The basis function parameters are drawn from a 
        GP that depends on the stimulus values at that given time
        
        :param torch.tensor input: input spiketrain or covariates with shape (trials, neurons, timesteps) 
                                   or (samples, neurons, timesteps)
        :param torch.tensor stimulus: input stimulus of shape (trials, timesteps, dims)
        :returns: filtered input of shape (trials, neurons, timesteps)
        """
        assert stimulus is not None
        assert input.shape[0] == 1
        #assert stimulus.shape[1] == input.shape[-1]-self.history_len+1
        
        input_unfold = input.unfold(-1, self.history_len, 1) # samples, neurons, timesteps, fold_dim
        stim_ = stimulus[:, self.history_len:, :] # samples, timesteps, dims
        K = stim_.shape[0]
        T = input_unfold.shape[-2]
        
        inner_batches = np.ceil(T / self.inner_bs).astype(int)
        a_ = []
        a_var_ = []
        for b in range(inner_batches): # inner loop batching
            stim_in = stim_[:, b*self.inner_bs:(b+1)*self.inner_bs, :]
            input_in = input_unfold[..., b*self.inner_bs:(b+1)*self.inner_bs, :]
            
            h_, v_ = self.compute_filter(stim_in) # KxT, out, in, D_fold
            if h_.shape[1] == 1: # output dims
                a = (input_in*h_[:, 0, ...].view(K, -1, *h_.shape[-2:]).permute(0, 2, 1, 3)) # K, N, T, D_fold
                a_.append(a.sum(-1)) # K, N, T
            else: # neurons
                a = (input_in[..., None, :]*h_.view(K, -1, *h_.shape[-3:]).permute(0, 2, 1, 3, 4)) # K, N, T, n_in, D_fold
                a_.append(a.sum(-1).sum(-1)) # K, N, T
                
        filt_var = 0 if len(a_var_) == 0 else torch.cat(a_var_, dim=-1)
        return torch.cat(a_, dim=-1), filt_var

    
    
class filter_model(base._filter):
    """
    Nonparametric GLM coupling filters. Is equivalent to multi-output GP time series. [1]

    References:
    
    [1] `Non-parametric generalized linear model`,
        Matthew Dowling, Yuan Zhao, Il Memming Park (2020)
        
    """
    def __init__(self, out_dim, in_dim, timesteps, tbin, filter_model, 
                 num_induc=6, tens_type=torch.float):
        """
        :param int out_dim: the number of output dimensions per input dimension (1 or neurons)
        :param int in_dim: the number of input dimensions for the overall filter (neurons)
        """
        conv_groups = in_dim//out_dim
        super().__init__(timesteps, conv_groups=conv_groups, tens_type=tens_type)
        self.out_dim = out_dim
        self.in_dim = in_dim

        assert filter_model.out_dims == in_dim*out_dim                                        
        self.add_module('filter_model', filter_model)
        self.register_buffer('cov', torch.arange(timesteps)[None, :, None]*tbin)
        
    def compute_filter(self):
        """
        :returns: filter mean of shape (n_in, n_out, timesteps), filter variance of same shape
        :rtype: tuple of torch.tensor
        """
        F_mu, F_var = self.filter_model.compute_F(self.cov) # (K=1 for time series), neurons, timesteps
        if isinstance(F_var, Number) is False:
            F_var = F_var[0].view(self.in_dim, self.out_dim, -1)
        return F_mu[0].view(self.in_dim, self.out_dim, -1), F_var
    
    def KL_prior(self):
        return self.filter_model.KL_prior()
    
    def forward(self, input, stimulus=None):
        """
        Introduces the spike coupling by convolution with the spike train, no padding and left removal 
        for causal convolutions.
        
        :param torch.tensor input: input spiketrain or covariates with shape (trials, neurons, timesteps) 
                                   or (samples, neurons, timesteps)
        :returns: filtered input of shape (trials, neurons, timesteps)
        """
        h_, v_ = self.compute_filter()
        mean_conv = F.conv1d(input, h_, groups=self.conv_groups)
        
        if isinstance(v_, Number) is False:
            var_conv = F.conv1d(input, v_, groups=self.conv_groups)
        else:
            var_conv = 0
            
        return mean_conv, var_conv



class hetero_filter_model(base._filter):
    """
    Nonparametric stimulus-dependent GLM coupling filters.
        
    """
    def __init__(self, out_dim, in_dim, timesteps, tbin, filter_model, 
                 inner_loop_bs=10, tens_type=torch.float, mode='unfold'):
        """
        :param int out_dim: the number of output dimensions per input dimension (1 or neurons)
        :param int in_dim: the number of input dimensions for the overall filter (neurons)
        """
        conv_groups = in_dim//out_dim
        super().__init__(timesteps, conv_groups=conv_groups, tens_type=tens_type)
        self.out_dim = out_dim
        self.in_dim = in_dim
        self.inner_bs = inner_loop_bs
        
        assert (mode == 'unfold') or (mode == 'repeat')
        self.mode = mode
                             
        assert filter_model.out_dims == in_dim*out_dim
        self.add_module('filter_model', filter_model)
        self.register_buffer('cov', torch.arange(timesteps)[None, :, None]*tbin)
        
    def compute_filter(self, stim):
        """
        The sample dimension is the label index for all nonzero spike bins.
        
        :param torch.tensor stim: the stimulus variable at all nonzero bins of shape (samples, timestep, dims), note 
                                  the sample dimension includes the MC samples as well as the history shifts (KxT)
        :returns: filter mean of shape (samples, n_out, n_in, timesteps), filter variance of same shape
        :rtype: tuple of torch.tensor
        """
        cov = torch.cat((self.cov.expand(stim.shape[0], *self.cov.shape[1:]), stim), dim=-1) # sample x timesteps x dims
        F_mu, F_var = self.filter_model.compute_F(cov) # samples, neurons, timesteps
        if isinstance(F_var, Number) is False:
            F_var = F_var.view(stim.shape[0], self.out_dim, self.in_dim, -1)
        return F_mu.view(stim.shape[0], self.out_dim, self.in_dim, -1), F_var
    
    def KL_prior(self):
        return self.filter_model.KL_prior()
    
    def forward(self, input, stimulus):
        """
        The two modes of stimulus coupling are `unfold` and `repeat`, the first evaluates the filter GP at 
        stimulus values at the instantaneous time, whereas the latter uses the stimulus value at the time 
        one wants to evaluate the conditional rate expanded across the history.
        
        :param torch.tensor input: input spiketrain or covariates with shape (trials, neurons, timesteps) 
                                   or (samples, neurons, timesteps)
        :param torch.tensor stimulus: input stimulus of shape (trials, timesteps, dims)
        :returns: filtered input of shape (trials, neurons, timesteps)
        """
        assert stimulus is not None
        assert input.shape[0] == 1
        #assert stimulus.shape[1] == input.shape[-1]-self.history_len+1
        
        input_unfold = input.unfold(-1, self.history_len, 1) # samples, neurons, timesteps, fold_dim
        if self.mode == 'unfold':
            stim_unfold = stimulus[:, :-1, :].unfold(1, self.history_len, 1) # samples, timesteps, dims, fold_dim
        else: # repeat
            stim_ = stimulus[:, self.history_len:, :]
            stim_unfold = stim_[..., None].expand(*stim_.shape, self.history_len) # samples, timesteps, dims, fold_dim
            
        K = stim_unfold.shape[0]
        T = input_unfold.shape[-2]
        
        inner_batches = np.ceil(T / self.inner_bs).astype(int)
        a_ = []
        a_var_ = []
        for b in range(inner_batches): # inner loop batching
            stim_in = stim_unfold[:, b*self.inner_bs:(b+1)*self.inner_bs, ...].reshape(-1, *stim_unfold.shape[-2:]) # KxT, D, D_fold
            input_in = input_unfold[..., b*self.inner_bs:(b+1)*self.inner_bs, :]
            
            h_, v_ = self.compute_filter(stim_in.permute(0, 2, 1)) # KxT, out, in, D_fold      
                
            if h_.shape[1] == 1: # output dims
                a = (input_in*h_[:, 0, ...].view(K, -1, *h_.shape[-2:]).permute(0, 2, 1, 3)) # K, N, T, D_fold
                a_.append(a.sum(-1)) # K, N, T
                if isinstance(v_, Number) is False:
                    a_var = (input_in*v_[:, 0, ...].view(K, -1, *v_.shape[-2:]).permute(0, 2, 1, 3))
                    a_var_.append(a_var.sum(-1)) # K, N, T
                
            else: # neurons
                a = (input_in[..., None, :]*h_.view(K, -1, *h_.shape[-3:]).permute(0, 2, 1, 3, 4)) # K, N, T, n_in, D_fold
                a_.append(a.sum(-1).sum(-1)) # K, N, T
                if isinstance(v_, Number) is False:
                    a_var = (input_in[..., None, :]*v_.view(K, -1, *v_.shape[-3:]).permute(0, 2, 1, 3, 4))
                    a_var_.append(a_var.sum(-1).sum(-1)) # K, N, T
                
        filt_var = 0 if len(a_var_) == 0 else torch.cat(a_var_, dim=-1)
        return torch.cat(a_, dim=-1), filt_var

    
    
### wrappers ###
class filtered_likelihood(nn.Module):
    """
    Wrapper for base._likelihood classes with filters.
    """
    def __init__(self, likelihood, filter):
        super().__init__()
        self.add_module('likelihood', likelihood)
        self.add_module('filter', filter)
        
        if self.filter.tensor_type != self.likelihood.tensor_type:
            raise ValueError('Filter and likelihood tensor types do not match')
        self.tensor_type = self.likelihood.tensor_type
        self.spikes = None
        
        self.inv_link = self.likelihood.inv_link
        self.F_dims = self.likelihood.F_dims
        self.neurons = self.likelihood.neurons
        
                  
    def KL_prior(self):
        return self.filter.KL_prior() + self.likelihood.KL_prior()
    
    
    def set_Y(self, spikes, batch_size, filter_len=1):
        self.likelihood.set_Y(spikes, batch_size, filter_len)
        if self.likelihood.filter_len != self.filter.history_len+1: # history excludes instantaneous part
            raise ValueError('Likelihood filtering length does not match filter history length')
        self.filter_len = self.likelihood.filter_len
        self.batches = self.likelihood.batches
        self.trials = self.likelihood.trials
        self.tsteps = self.likelihood.tsteps
        
                  
    def constrain(self):
        """
        Constrain parameters in optimization
        """
        self.likelihood.constrain()
        self.filter.constrain()
    
    
    def mc_gen(self, q_mu, q_var, samples, neuron):
        return self.likelihood.mc_gen(q_mu, q_var, samples, neuron)
        
        
    def gh_gen(self, q_mu, q_var, points, neuron):
        return self.likelihood.gh_gen(q_mu, q_var, points, neuron)
    
    
    def _validate_neuron(self, neuron):
        return self.likelihood._validate_neuron(neuron)
    
    
    def sample_rate(self, F_mu, F_var, trials, MC_samples=1):
        return self.likelihood.sample_rate(F_mu, F_var, trials, MC_samples)
        
        
    def objective(self, F_mu, F_var, XZ, b, neuron, samples=10, mode='MC'):
        """
        spike coupling filter
        """
        spk = self.likelihood.spikes[b].to(self.likelihood.tbin.device)
        spk_filt, spk_var = self.spike_filter(spk[..., :-1], XZ) # trials, neurons, timesteps
        mean = F_mu+spk_filt
        variance = F_var+spk_var
        return self.likelihood.objective(mean, variance, XZ, b, neuron, samples, mode)
    
        
    def filtered_rate(self, F_mu, F_var, unobs_neuron, trials, MC_samples=1):
        """
        Evaluate the instantaneous rate after spike coupling, with unobserved neurons not contributing 
        to the filtered population rate.
        """
        unobs_neuron = self.likelihood._validate_neuron(unobs_neuron)
        spk = self.likelihood.spikes[b].to(self.likelihood.tbin.device)

        with torch.no_grad():
            hist, hist_var = self.spike_filter(spk[..., :-1], XZ)
            hist[:, unobs_neuron, :] = 0 # mask
            hist_var[:, unobs_neuron, :] = 0 # mask
            h = self.mc_gen(F_mu + hist, F_var + hist_var, MC_samples, torch.arange(self.neurons))
            intensity = self.likelihood.f(h.view(-1, trials, *h.shape[1:]))
            
        return intensity
        
        
    def sample(self, rate, neuron=None, XZ=None):
        """
        Assumes all neurons outside neuron are observed for spike filtering.
        """
        neuron = self.likelihood._validate_neuron(neuron)
        #ini_train = 
        

        steps = rate.shape[1]
        spikes = []
        spiketrain = torch.empty((*ini_train.shape[:2], self.history_len), device=self.likelihood.tbin.device)

        iterator = tqdm(range(steps), leave=False) # AR sampling
        rate = []
        for t in iterator:
            if t == 0:
                spiketrain[..., :-1] = torch.tensor(ini_train, device=self.likelihood.tbin.device)
            else:
                spiketrain[..., :-2] = spiketrain[..., 1:-1].clone() # shift in time
                spiketrain[..., -2] = torch.tensor(spikes[-1], device=self.likelihood.tbin.device)

            with torch.no_grad(): # spiketrain last time element is dummy, [:-1] used
                hist, hist_var = self.spike_filter(spiketrain, cov_[:, t:t+self.history_len, :])

            rate_ = self.likelihood.f(F_mu[..., t].view(MC_samples, -1, F_mu.shape[1]) + \
                                      hist[None, ..., 0]).mean(0).cpu().numpy() # (trials, neuron)

            ddisp = disp_f(disp.data)[..., t:t+1].view(MC_samples, -1, disp.shape[1], 
                                                       1).mean(0).cpu().numpy() if disp is not None else None
            if obs_spktrn is None:
                spikes.append(self.likelihood.sample(rate_[..., None], n_, XZ=XZ)[..., 0])
                #spikes.append(point_process.gen_IBP(1. - np.exp(-rate_*self.likelihood.tbin.item())))
            else: # condition on observed spike train partially
                spikes.append(obs_spktrn[..., t])
                spikes[-1][:, neuron] = self.likelihood.sample(rate_[..., None], neuron, XZ=XZ)[..., 0]
                #spikes[-1][:, neuron] = point_process.gen_IBP(1. - np.exp(-rate_[:, neuron]*self.likelihood.tbin.item()))
            rate.append(rate_)

        rate = np.stack(rate, axis=-1) # trials, neurons, timesteps
        spktrain = np.transpose(np.array(spikes), (1, 2, 0)) # trials, neurons, timesteps
        
        return spktrain, rate