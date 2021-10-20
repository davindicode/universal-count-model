### imports ###
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter

import numpy as np

from .base import Recurrent



class Free(Recurrent):
    r"""
    Free network structure
    """
    def __init__(self, neuron_groups, delta_t, noise_tau, nonlin, w_ij, learn_tau=False, learn_noise=False, learn_noisetau=False, 
                 noisy=True, L_n=None, connection_clip=None, bias=None, adjoint=False, 
                 delay=None, tensor_type=torch.float):
        r"""
        w_ij is used to set the excitatory/inhibitory nature of neurons, None will lead to random weights and E/I
        connection_clip is used to clamp weights to zero, i.e. remove synaptic connections
        
        :param np.array neuron_groups: neuron type data of shape (2, [neurons, power, tau_membrane])
        :param float delta_t: time step of Euler integration simulation
        :param 
        :param torch.tensor L_n: Cholesky noise matrix, None means it is learned
        :param
        """
        super().__init__(neuron_groups, delta_t, noise_tau, nonlin, noisy, bias, delay, 
                         adjoint, learn_tau, learn_noise, learn_noisetau, tensor_type)
    
        # connectivity
        self.register_parameter('w', Parameter(torch.tensor(w_ij, dtype=self.tensor_type)))
        
        if connection_clip is None:
            connection_clip = np.zeros((self.tot_neurons, self.tot_neurons))
        self.register_buffer('connection_clip', torch.tensor(connection_clip, dtype=torch.bool))
        
        # generate the boolean indicator for E/I weights
        self.dale = torch.zeros(self.tot_neurons).type(torch.bool)
        for j in range(self.tot_neurons):
            if torch.sum(self.w[:,j]) > 0:
                 self.dale[j] = True
                    
        # noise
        if self.noisy:
            if self.learn_noise:
                self.register_parameter('L_n', Parameter(torch.tensor(L_n, dtype=self.tensor_type)))
            else:
                self.register_buffer('L_n', torch.tensor(L_n, dtype=self.tensor_type))
    
    def constraints(self, dale=True):
        self.w.data[self.connection_clip] = 0.
        
        if self.noisy and self.L_n.requires_grad is True:
            self.L_n.data = torch.tril(self.L_n.data)
        
        if dale: # enforce Dale's law, set .data to value, otherwise we get rid of leaf node! Autograd complains
            for j in range(self.tot_neurons):
                if self.dale[j]: # E
                    self.w.data[self.w.data[:,j] < 0., j] = 0.
                else: # I
                    self.w.data[self.w.data[:,j] > 0., j] = 0.
        
        
        
class Ring_EI(Recurrent):
    """
    Imposes ring symmetry on the network
    """
    def __init__(self, neuron_groups, delta_t, noise_tau, nonlin, adjoint=False, 
                 learn_tau=False, learn_noise=False, learn_noisetau=False, noisy=True, 
                 w_A=[0.1, 0.1, 0.1, 0.1, 0.1, 0.1], w_d=[1.0, 1.0, 1.0, 1.0], 
                 n_A=[1.0, 1.0, 0.0], n_d=1.0, connection_clip=None, bias=None,
                 delay=None, tensor_type=torch.float):
        """
        connection_clip is used to clamp weights to zero, i.e. remove synaptic connections
        """
        super().__init__(neuron_groups, delta_t, noise_tau, nonlin, noisy, bias, delay, 
                         adjoint, learn_tau, learn_noise, learn_noisetau, tensor_type)
        
        # w_A: A_EE, A_II, A_EI, A_IE, self connections sA_E, sA_I
        self.register_parameter('w_A', Parameter(torch.tensor(w_A, dtype=self.tensor_type)))
        self.register_parameter('w_d', Parameter(torch.tensor(w_d, dtype=self.tensor_type)))
        
        if connection_clip is None:
            connection_clip = torch.zeros(6).type(torch.bool)
        self.register_buffer('connection_clip', connection_clip)
        
        dphi_E = 2*np.pi / self.num_neurons[0]
        dphi_I = 2*np.pi / self.num_neurons[1]
        diff_EE = torch.zeros((self.num_neurons[0], self.num_neurons[0]), dtype=self.tensor_type)
        diff_II = torch.zeros((self.num_neurons[1], self.num_neurons[1]), dtype=self.tensor_type)
        diff_EI = torch.zeros((self.num_neurons[0], self.num_neurons[1]), dtype=self.tensor_type)
        for k in range(self.num_neurons[0]): # EE
            for l in range(k):
                diff_EE[k,l] = dphi_E*(k-l)
                diff_EE[l,k] = diff_EE[k,l]
                
        for k in range(self.num_neurons[1]): # II
            for l in range(k):
                diff_II[k,l] = dphi_E*(k-l)
                diff_II[l,k] = diff_II[k,l]
           
        for k in range(self.num_neurons[0]): # EI
            for l in range(self.num_neurons[1]):
                diff_EI[k,l] = dphi_E*k - dphi_I*l
                
        self.register_buffer('diff_EE', diff_EE)
        self.register_buffer('diff_II', diff_II)
        self.register_buffer('diff_EI', diff_EI) # note diff_IE = -diff_EI.T
        
        if self.noisy:
            if self.learn_noise: # n_A: sigma_E, sigma_I, rho
                self.register_parameter('n_A', Parameter(torch.tensor(n_A, dtype=self.tensor_type)))
                self.register_parameter('n_d', Parameter(torch.tensor(n_d, dtype=self.tensor_type)))
            else:
                self.register_buffer('n_A', torch.tensor(n_A, dtype=self.tensor_type))
                self.register_buffer('n_d', torch.tensor(n_d, dtype=self.tensor_type))
                
    def reset_net(self, init_h, sep_self=False, gain_params=None):
        """
        Construct all tensors for current time step
        """
        super().reset_net(init_h, gain_params)
        
        # construct the weight matrix
        EE = self.w_A[0] * torch.exp((torch.cos(self.diff_EE) - 1)/self.w_d[0]**2)
        II = -self.w_A[1] * torch.exp((torch.cos(self.diff_II) - 1)/self.w_d[1]**2)
        EI = -self.w_A[2] * torch.exp((torch.cos(self.diff_EI) - 1)/self.w_d[2]**2)
        IE = self.w_A[3] * torch.exp((torch.cos(self.diff_EI.t()) - 1)/self.w_d[3]**2)
        self.w = torch.cat((torch.cat((EE, EI), dim=1), torch.cat((IE, II), dim=1)), dim=0)
        if sep_self:
            self.w[range(self.num_neurons[0]), range(self.num_neurons[0])] = self.w_A[4]
            self.w[range(self.num_neurons[0], self.num_neurons[1]), 
                   range(self.num_neurons[0], self.num_neurons[1])] = -self.w_A[5]

        if self.noisy:
            f = self.n_A[2]*self.n_A[1]*self.n_A[0]
            # construct the noise matrix
            EI = self.n_A[0]**2 * torch.exp((torch.cos(self.diff_EE) - 1)/self.n_d**2)
            II = self.n_A[1]**2 * torch.exp((torch.cos(self.diff_II) - 1)/self.n_d**2)
            EI = f * torch.exp((torch.cos(self.diff_EI) - 1)/self.n_d**2)
            n = torch.cat((torch.cat((EE, EI), dim=1), torch.cat((EI.t(), II), dim=1)), dim=0)
            self.L_n = torch.cholesky(n)
            
    def constraints(self):
        self.w_A.data[self.connection_clip] = 0.0