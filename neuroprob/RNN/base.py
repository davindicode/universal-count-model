### imports ###
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter

import numpy as np



### network classes ###
class Recurrent(nn.Module):
    """
    Base class for recurrent modules
    
    :param bool noisy: include noise? Learnable if 
    :param bool bias: include learnable neuron biases?
    :param np.array delay: delay matrix (integer steps) of shape (neurons, neurons)
    :param bool adjoint: 
    :param string tensor_type: 
    """
    def __init__(self, neuron_groups, delta_t, noise_tau, nonlin, noisy, bias, delay, adjoint, 
                 learn_tau, learn_noise, learn_noisetau, tensor_type):
        """
        """
        super().__init__()
        self.tensor_type = tensor_type
        
        self.num_neurons = neuron_groups[:, 0].type(torch.IntTensor)
        self.tot_neurons = torch.sum(self.num_neurons).item()
        
        self.register_buffer('d_t', torch.tensor(delta_t, dtype=self.tensor_type))
        
        self.learn_noise = learn_noise
        self.learn_noisetau = learn_noisetau
        if self.learn_noisetau:
            self.register_parameter('_ntau', Parameter(torch.log(torch.tensor(noise_tau/delta_t, 
                                                                                     dtype=self.tensor_type))))
        else:
            if noise_tau == 0:
                self.n_decay = 0
                self.register_buffer('div_sqdt', torch.tensor(1 / np.sqrt(delta_t), dtype=self.tensor_type))
            else:
                self.register_buffer('n_sqdt_tau', torch.tensor(np.sqrt(delta_t) / noise_tau, dtype=self.tensor_type))
                self.register_buffer('n_decay', torch.tensor(np.exp(-delta_t / noise_tau), dtype=self.tensor_type))
        
        self.learn_tau = learn_tau
        if self.learn_tau: # learn timescale, separate per neuron
            _tau = torch.empty((1, self.tot_neurons), dtype=self.tensor_type)
            p_sz = 0
            cnt = 0
            for sz in self.num_neurons:
                _tau[:, p_sz : p_sz + sz] = neuron_groups[cnt, 1] / delta_t
                p_sz += sz
                cnt += 1
            self.num_groups = cnt
            self.register_parameter('_tau', Parameter(torch.log(_tau)))
            
        else: # precompute fixed neuron timescale values
            dt_tau = torch.empty((1, self.tot_neurons), dtype=self.tensor_type)
            decay = torch.empty((1, self.tot_neurons), dtype=self.tensor_type)
            p_sz = 0
            cnt = 0
            for sz in self.num_neurons:
                decay[:, p_sz : p_sz + sz] = np.exp(-delta_t / neuron_groups[cnt, 1])
                dt_tau[:, p_sz : p_sz + sz] = 1 - decay[:, p_sz : p_sz + sz] # integral approximation
                p_sz += sz
                cnt += 1
            self.num_groups = cnt

            self.register_buffer('decay', decay)
            self.register_buffer('dt_tau', dt_tau)
        
        self.noisy = noisy
        self.F = nonlin
        
        if bias is not None:
            self.register_parameter('b', Parameter(torch.tensor(bias[None, :], dtype=self.tensor_type)))
        else:
            self.b = 0
            
        if delay is not None:
            self.register_buffer('delay', torch.tensor(delay, dtype=torch.long))
            
        if adjoint:
            class ode_func(nn.Module):
                def forward(self, t, h):
                    I = self.F(h)
                    h = self.decay * self.h + self.dt_tau * (torch.matmul(self.I, self.w.t()) + u_ext + \
                                                      self.eta + self.b)
                    return h
                
    def reset_net(self, h, hist_id=None, gain_params=None):
        """
        After each optimization step, refresh the computational graph and network state/parameters.
        Sets the (trial, neurons) shape of neuron variables.
        """
        #h = torch.tensor(init_h, device=self.d_t.device, dtype=self.tensor_type)
        if len(h.shape) == 3:
            self.hist_len = h.shape[0]
            trials = h.shape[1]
            self.hist_id = hist_id
            self.h = h[hist_id]
            
            # range in last dimension as presynaptic ji convention
            self.neur_access = torch.arange(self.tot_neurons, device=self.d_t.device, 
                                            dtype=torch.long).expand(self.tot_neurons, trials, self.tot_neurons)
            self.trial_access = torch.arange(trials, device=self.d_t.device, 
                                             dtype=torch.long)[None, :, None].expand(self.tot_neurons, trials, self.tot_neurons)
        else:
            self.h = h
        
        self.I = self.F(h, gain_params)
        if self.noisy:
            self.eta = torch.zeros_like(h, device=self.d_t.device, dtype=self.tensor_type)
        else:
            self.eta = 0
            
    def precompute(self):
        """
        Noise here is of shape (trials, neurons) similar to membrane potentials :math:`h`.
        """
        if self.noisy:
            if self.learn_noisetau:
                noise_tau = torch.exp(self._ntau)
                self.n_sqdt_tau = 1. / (noise_tau * torch.sqrt(self.d_t))
                self.n_decay = torch.exp(-1. / noise_tau)
            
            noise = torch.randn((*self.h.shape), device=self.d_t.device)
            if self.n_decay == 0:
                self.eta = self.div_sqdt * torch.matmul(noise, self.L_n.t())
            else:
                self.eta = self.n_decay * self.eta + self.n_sqdt_tau * torch.matmul(noise, self.L_n.t())
                
        if self.learn_tau:
            tau = torch.exp(self._tau)
            self.decay = torch.exp(-1./tau)
            self.dt_tau = 1 - self.decay
        
    def forward(self, u_ext, gain_params=None):
        r"""
        Euler integration step with exact exponential integration.
        This is the integral formulation with inhomogeneous terms from recurrent interactions 
        and external input, or the Green function approach to the ODE/SDE 
        
        .. math::
            \dot{u}(t) = &\exp{-t/\tau} u(0) + \int_0^t \exp{-|t-t'|/\tau} \, [Wu(t') + b(t') + \xi(t')] \, \mathrm{d}t'
        
        dt_tau is defined as :math:`1 - \exp{-\Delta t / \tau}` as we approximate the integral 
        with a finite interval. We approximate the integrand inhomogeneous term with its value 
        at 0, then the order of approximation is :math:`O((\frac{\Delta t}{\tau})^2)`. We repeat 
        this for each time step with interval :math:`\Delta t`.
        
        :param torch.tensor u_ext: input to network with shape (trials, neurons)
        """
        self.precompute()
        self.I = self.F(self.h, gain_params)
        self.h = self.decay * self.h + self.dt_tau * (torch.matmul(self.I, self.w.t()) + u_ext + \
                                                      self.eta + self.b)
        
    def forward_hist(self, u_ext, gain_params=None):
        """
        Euler integration step
        :param torch.tensor u_ext: input to network with shape (trials, neurons)
        """
        self.precompute()
        if self.hist_id == self.hist_len-1: # write to next time step
            self.hist_id = 0
        else:
            self.hist_id += 1
        
        rel_tau = (self.hist_id - self.delay) % self.hist_len
        self.I[self.hist_id] = self.F(self.h, gain_params)
        self.h = self.decay * self.h + self.dt_tau * (
            (self.I[(rel_tau[:, None, :], self.trial_access, self.neur_access)]*self.w[:, None, :]).sum(-1).t()\
             + u_ext + self.eta + self.b) # (hist_time, trials, neurons)
        
    def solve_ode(self):
        """
        Noiseless adaptive step size solver.
        """
        true_y = odeint(ode_func(), true_y0, t, method='dopri5')
        return 
    
    def plasticity(self, post_act, pre_act, eta, p=2, norm_f=1.0):
        """
        :math:`L_p`-norm Hebbian normalized plasticity, multiplicative normalization.
        Total synaptic weights on dendrites seem to be conserved [1].
        
        :param torch.tensor post_act: post-synaptic activity of shape (trial, neuron)
        :param torch.tensor pre_act: pre-synaptic activity of shape (trial, neuron)
        
        References:
        
        [1] `Conservation of total synaptic weight through balanced synaptic depression and potentiation`,
            Sébastien Royer & Denis Paré
            
        """
        d = self.w + eta*post_act[..., None]*pre_act[:, None, :]
        self.w = norm_f * d / (d**p).sum(-1)**(1/p)
        
        

class FeedForward(nn.Module):
    r"""
    Base class for feed-forward modules
    """
    def __init__(self, nonlin, bias, tensor_type):
        super().__init__()
        self.tensor_type = tensor_type
        
        self.F = nonlin
        
        if bias is not None:
            self.register_parameter('b', Parameter(torch.tensor(bias[None, :], dtype=self.tensor_type)))
        else:
            self.b = 0
        
    def forward(self, h, gain_params=None):
        return torch.matmul(self.F(h, gain_params), self.w.t()) + self.b