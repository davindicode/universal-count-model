import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np

from tqdm.autonotebook import tqdm



class rate_decoder(nn.Module):
    """
    Multi-layer perceptron class based decoder. The rate decoder takes a potentially 
    learnable kernel that it uses to convolve the spike train with for obtaining a 
    scalar rate value per neuron.
    """
    def __init__(self, mu_net, sigma_net, x_dist, kernel, learn_kernel=False):
        super().__init__()
        self.add_module('mu_net', mu_net)
        self.add_module('sigma_net', sigma_net)
        self.x_dist = x_dist
        if learn_kernel:
            self.register_parameter('kernel', nn.Parameter(kernel))
        else:
            self.register_buffer('kernel', kernel)
            
            
    def moments(self, input):
        """
        Input of shape (time, neurons, hist_len)
        """
        rates = (input*self.kernel[None, ...]).sum(-1) # time, neurons
        X_mu = self.mu_net(rates) # time, neurons as argument
        X_sigma = self.sigma_net(rates)
        return X_mu, X_sigma

    
    def forward(self, input):
        """
        Input of shape (time, neurons, hist_len)
        """
        pred_dist = self.x_dist(*self.moments(input))
        return pred_dist
    
    
    
class temporal_decoder(nn.Module):
    """
    Multi-layer perceptron class based decoder. The input is directly mapped to the 
    predictive distribution over stimuli. Depending on the complexity of the mapping, 
    this can capture non-trivial spatio-temporal correlations relevant to decoding.
    """
    def __init__(self, mu_net, sigma_net, x_dist):
        super().__init__()
        self.add_module('mu_net', mu_net)
        self.add_module('sigma_net', sigma_net)
        self.x_dist = x_dist
        
        
    def moments(self, input):
        """
        Input of shape (time, neurons, hist_len)
        """
        inp = input.reshape(input.shape[0], -1)
        X_mu = self.mu_net(inp) # time, neurons as argument
        X_sigma = self.sigma_net(inp)
        return X_mu, X_sigma
    

    def forward(self, input):
        """
        Input of shape (time, neurons, hist_len)
        """
        pred_dist = self.x_dist(*self.moments(input))
        return pred_dist



def fit_decoder(decoding_triplets, batches, timesteps, optimizer, max_epochs, loss_margin=0.0, 
                margin_epochs=10, scheduler=None, sch_st=None, dev='cpu'):
    """
    Fit the optimal probabilistic decoder with cross entropy loss.
    Optimizes a list of decoders for different dimensions/topologies.
    """
    loss_tracker = []
    minloss = np.inf
    cnt = 0
        
    iterator = tqdm(range(max_epochs))
    for epoch in iterator:
        sloss = 0
        for b in range(batches):
            optimizer.zero_grad()
            
            loss = 0
            for trp in decoding_triplets:
                decoder, input_batched, target_batched = trp
                ib = input_batched[b]
                tb = target_batched[b]
                
                batch_size = tb.shape[0]
                x_dist = decoder(ib.to(dev))

                subsample_fac = timesteps/batch_size
                loss_ = -x_dist.log_prob(tb.to(dev)).sum() * subsample_fac
                
                loss += loss_
                sloss += loss_.item()/batches
                
            loss.backward()
            optimizer.step()

        iterator.set_postfix(loss=sloss)
        loss_tracker.append(sloss)

        if scheduler is not None and epoch % sch_st == sch_st-1:
            scheduler.step()
            
        if sloss <= minloss + loss_margin:
            cnt = 0
        else:
            cnt += 1

        if sloss < minloss:
            minloss = sloss

        if cnt > margin_epochs:
            print("\nStopped at epoch {}.".format(epoch+1))
            break

    return loss_tracker

