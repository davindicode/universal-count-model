### imports ###
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter

import numpy as np

from .base import FeedForward



class Free(FeedForward):
    r"""
    Free network structure
    """   
    def __init__(self, w_ij, nonlin, connection_clip=None, 
                 bias=None, tensor_type=torch.float):
        r"""
        w_ij is used to set the excitatory/inhibitory nature of neurons, None will lead to random weights and E/I
        connection_clip is used to clamp weights to zero, i.e. remove synaptic connections
        """
        super().__init__(nonlin, bias, tensor_type=tensor_type)
    
        # connectivity
        self.neurons_in = w_ij.shape[1]
        self.neurons_out = w_ij.shape[0]
        self.register_parameter('w', Parameter(torch.tensor(w_ij, dtype=self.tensor_type)))
        
        if connection_clip is None:
            connection_clip = np.zeros((self.neurons_out, self.neurons_in))
        self.register_buffer('connection_clip', torch.tensor(connection_clip, dtype=torch.bool))
        
        # generate the boolean indicator for E/I weights
        self.dale = torch.zeros(self.neurons_in).type(torch.bool)
        for j in range(self.neurons_in):
            if torch.sum(self.w[:,j]) > 0:
                 self.dale[j] = True
    
    def constraints(self, dale=True):
        self.w.data[self.connection_clip] = 0.
        
        if dale: # enforce Dale's law, set .data to value, otherwise we get rid of leaf node! Autograd complains
            for j in range(self.tot_neurons):
                if self.dale[j]: # E
                    self.w.data[self.w.data[:,j] < 0., j] = 0.
                else: # I
                    self.w.data[self.w.data[:,j] > 0., j] = 0.