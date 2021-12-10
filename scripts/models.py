import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F
import torch.optim as optim

import numpy as np

import sys
sys.path.append("..") # access to library

    

import neuroprob as mdl
from neuroprob import utils


 

    
### networks ###
def hyper_params(basis_mode='ew'):
    enc_layers = [50, 50, 100]
    if basis_mode == 'ew': # element-wise
        basis = (lambda x: x, lambda x: torch.exp(x))
        
    elif basis_mode == 'qd': # quadratic
        def mix(x):
            N = x.shape[-1]
            out = torch.empty((*x.shape[:-1], N*(N-1)//2), dtype=x.dtype).to(x.device)
            k = 0
            for n in range(1, N):
                for n_ in range(n):
                    out[..., k] = x[..., n]*x[..., n_]
                    k += 1
                
            return out
        
        basis = (lambda x: x, lambda x: x**2, lambda x: torch.exp(x), lambda x: mix(x))
    folds = 10
    
    return enc_layers, basis



class net(nn.Module):
    def __init__(self, C, basis, max_count, channels, shared_W=False):
        super().__init__()
        self.basis = basis
        self.C = C
        expand_C = torch.cat([f_(torch.ones(1, self.C)) for f_ in self.basis], dim=-1).shape[-1]
        
        mnet = utils.pytorch.Parallel_MLP([], expand_C, (max_count+1), channels, shared_W=shared_W, 
                                        nonlin=utils.pytorch.Siren(), out=None)
        self.add_module('mnet', mnet)
        
        
    def forward(self, input, neuron):
        """
        :param torch.tensor input: input of shape (samplesxtime, channelsxin_dims)
        """
        input = input.view(input.shape[0], -1, self.C)
        input = torch.cat([f_(input) for f_ in self.basis], dim=-1)
        out = self.mnet(input, neuron)
        return out.view(out.shape[0], -1) # t, NxK



class enc_model(nn.Module):
    """
    Multi-layer perceptron class
    """
    def __init__(self, layers, angle_dims, euclid_dims, hist_len, out_dims):
        """
        Assumes angular dimensions to be ordered first in the input of shape dimensionsxhist_len.
        dim_fill is hist_len or hist_len*neurons when parallel neurons
        """
        super().__init__()
        self.angle_dims = angle_dims
        self.in_dims = 2*angle_dims + euclid_dims
        net = utils.pytorch.MLP(layers, self.in_dims, out_dims, nonlin=utils.pytorch.Siren(), out=None)
        self.add_module('net', net)

    def forward(self, input):
        """
        Input of shape (samplesxtime, dims)
        """
        embed = torch.cat((torch.cos(input[:, :self.angle_dims]), 
                           torch.sin(input[:, :self.angle_dims]), 
                           input[:, self.angle_dims:]), dim=-1)
        return self.net(embed)
    
    

### setup of the probabilistic model ###
def cov_used(mode, behav_tuple):
    """
    Create the used covariates list for different models
    """
    resamples = behav_tuple[0].shape[0]
    hd_t, w_t, s_t, x_t, y_t, time_t = behav_tuple
    
    if mode == 'hd':
        covariates = [hd_t]
        
    elif mode == 'w':
        covariates = [w_t]
        
    elif mode == 'hd_w' or mode == 'hdTw':
        covariates = [hd_t, w_t]
        
    elif mode == 'hd_w_s' or mode == 'hd_wTs':
        covariates = [hd_t, w_t, s_t]
        
    elif mode == 'hd_w_s_t':
        covariates = [hd_t, w_t, s_t, time_t]
        
    elif mode == 'hd_w_s_t_R1':
        covariates = [
            hd_t, w_t, s_t, time_t, 
            [np.random.randn(resamples, 1)*0.1, np.ones((resamples, 1))*0.01]
        ]
        
    elif mode == 'hd_w_s_pos_t':
        covariates = [hd_t, w_t, s_t, x_t, y_t, time_t]
        
    elif mode == 'hd_w_s_pos_t_R1':
        covariates = [
            hd_t, w_t, s_t, x_t, y_t, time_t, 
            [np.random.randn(resamples, 1)*0.1, np.ones((resamples, 1))*0.01]
        ]
        
    elif mode == 'hd_w_s_pos_t_R2':
        covariates = [
            hd_t, w_t, s_t, x_t, y_t, time_t, 
            [np.random.randn(resamples, 2)*0.1, np.ones((resamples, 2))*0.01]
        ]
        
    elif mode == 'hd_w_s_pos_t_R3':
        covariates = [
            hd_t, w_t, s_t, x_t, y_t, time_t, 
            [np.random.randn(resamples, 3)*0.1, np.ones((resamples, 3))*0.01]
        ]
        
    elif mode == 'hd_w_s_pos_t_R4':
        covariates = [
            hd_t, w_t, s_t, x_t, y_t, time_t, 
            [np.random.randn(resamples, 4)*0.1, np.ones((resamples, 4))*0.01]
        ]
        
    elif mode == 'hd_w_s_pos' or mode == 'hd_w_sTpos' or mode == 'hd_wTsTpos':
        covariates = [hd_t, w_t, s_t, x_t, y_t]
        
    elif mode == 'hd_t':
        covariates = [hd_t, time_t]
        
    elif mode == 'hd_w_t':
        covariates = [hd_t, w_t, time_t]
        
    elif mode == 'hdxR1':
        covariates = [
            hd_t, 
            [np.random.randn(resamples, 1)*0.1, np.ones((resamples, 1))*0.01]
        ]
        
    elif mode == 'T1xR1':   
        covariates = [
            [np.random.rand(resamples, 1)*2*np.pi, np.ones((resamples, 1))*0.01], 
            [np.random.randn(resamples, 1)*0.1, np.ones((resamples, 1))*0.01]
        ]
        
    elif mode == 'T1xt':
        covariates = [
            [np.random.rand(resamples, 1)*2*np.pi, np.ones((resamples, 1))*0.01], 
            time_t
        ]

    elif mode == 'T1': #hd_t[:, None] np.random.rand(resamples, 1)*2*np.pi
        covariates = [
            [np.random.rand(resamples, 1)*2*np.pi, np.ones((resamples, 1))*0.1]
        ]
        
    elif mode == 'R1':
        covariates = [
            [np.random.randn(resamples, 1)*0.1, np.ones((resamples, 1))*0.01]
        ]
        
    elif mode == 'R2':
        covariates = [
            [np.random.randn(resamples, 2)*0.1, np.ones((resamples, 2))*0.01]
        ]

    elif mode is None:
        covariates = []
        
    else:
        raise ValueError
        
    return covariates



def kernel_used(mode, behav_tuple, num_induc, outdims, var=1.):
    hd_t, w_t, s_t, x_t, y_t, time_t = behav_tuple
    
    left_x = x_t.min()
    right_x = x_t.max()
    bottom_y = y_t.min()
    top_y = y_t.max()
    
    l = 10.*np.ones(outdims)
    l_w = w_t.std()*np.ones(outdims)
    l_ang = 5.*np.ones(outdims)
    l_s = 10.*np.ones(outdims)
    v = var*np.ones(outdims)
    l_time = time_t.max()/2.*np.ones(outdims)
    l_one = np.ones(outdims)
    
    if mode == 'hd':
        ind_list = [np.linspace(0, 2*np.pi, num_induc+1)[:-1]]
        kernel_tuples = [('variance', v), 
              ('RBF', 'torus', np.array([l_ang]))]
        
    elif mode == 'w':
        ind_list = [np.linspace(-w_t.std(), w_t.std(), num_induc)]
        kernel_tuples = [('variance', v), 
              ('RBF', 'euclid', np.array([l_w]))]

    elif mode == 'hd_w':
        ind_list = [np.linspace(0, 2*np.pi, num_induc+1)[:-1], 
                    np.random.randn(num_induc)*w_t.std()]
        kernel_tuples = [('variance', v), 
              ('RBF', 'torus', np.array([l_ang])), 
              ('RBF', 'euclid', np.array([l_w]))]
        
        
    elif mode == 'hd_w_s':
        ind_list = [np.linspace(0, 2*np.pi, num_induc+1)[:-1], 
                    np.random.randn(num_induc)*w_t.std(), 
                    np.random.uniform(0, s_t.std(), size=(num_induc,))]
        kernel_tuples = [('variance', v), 
              ('RBF', 'torus', np.array([l_ang])), 
              ('RBF', 'euclid', np.array([l_w, l_s]))]
        
    elif mode == 'hd_w_s_t':
        ind_list = [np.linspace(0, 2*np.pi, num_induc+1)[:-1], 
                    np.random.randn(num_induc)*w_t.std(), 
                    np.random.uniform(0, s_t.std(), size=(num_induc,)), 
                    np.linspace(0, time_t.max(), num_induc)]
        kernel_tuples = [('variance', v), 
              ('RBF', 'torus', np.array([l_ang])), 
              ('RBF', 'euclid', np.array([l_w, l_s, l_time]))]
        
    elif mode == 'hd_w_s_t_R1gp' or mode == 'hd_w_s_t_R1':
        ind_list = [np.linspace(0, 2*np.pi, num_induc+1)[:-1], 
                    np.random.randn(num_induc)*w_t.std(), 
                    np.random.uniform(0, s_t.std(), size=(num_induc,)), 
                    np.linspace(0, time_t.max(), num_induc), 
                    np.random.randn(num_induc)]
        kernel_tuples = [('variance', v), 
                         ('RBF', 'torus', np.array([l_ang])), 
                         ('RBF', 'euclid', np.array([l_w, l_s, l_time])), 
                         ('linear', 'euclid', np.array([l_one]))]

    elif mode == 'hd_w_s_pos':
        ind_list = [np.linspace(0, 2*np.pi, num_induc+1)[:-1], 
                    np.random.randn(num_induc)*w_t.std(), 
                    np.random.uniform(0, s_t.std(), size=(num_induc,)), 
                    np.random.uniform(left_x, right_x, size=(num_induc,)), 
                    np.random.uniform(bottom_y, top_y, size=(num_induc,))]
        kernel_tuples = [('variance', v), 
              ('RBF', 'torus', np.array([l_ang])), 
              ('RBF', 'euclid', np.array([l_w, l_s, l, l]))]
        
    elif mode == 'hd_w_s_pos_t':
        ind_list = [np.linspace(0, 2*np.pi, num_induc+1)[:-1], 
                    np.random.randn(num_induc)*w_t.std(), 
                    np.random.uniform(0, s_t.std(), size=(num_induc,)), 
                    np.random.uniform(left_x, right_x, size=(num_induc,)), 
                    np.random.uniform(bottom_y, top_y, size=(num_induc,)), 
                    np.linspace(0, time_t.max(), num_induc)]
        kernel_tuples = [('variance', v), 
              ('RBF', 'torus', np.array([l_ang])), 
              ('RBF', 'euclid', np.array([l_w, l_s, l, l, l_time]))]
        
    elif mode == 'hd_w_s_pos_t_R1':
        ind_list = [np.linspace(0, 2*np.pi, num_induc+1)[:-1], 
                    np.random.randn(num_induc)*w_t.std(), 
                    np.random.uniform(0, s_t.std(), size=(num_induc,)), 
                    np.random.uniform(left_x, right_x, size=(num_induc,)), 
                    np.random.uniform(bottom_y, top_y, size=(num_induc,)), 
                    np.linspace(0, time_t.max(), num_induc), 
                    np.random.randn(num_induc)]
        kernel_tuples = [('variance', v), 
              ('RBF', 'torus', np.array([l_ang])), 
              ('RBF', 'euclid', np.array([l_w, l_s, l, l, l_time, l_one]))]
        
    elif mode == 'hd_w_s_pos_t_R2':
        ind_list = [np.linspace(0, 2*np.pi, num_induc+1)[:-1], 
                    np.random.randn(num_induc)*w_t.std(), 
                    np.random.uniform(0, s_t.std(), size=(num_induc,)), 
                    np.random.uniform(left_x, right_x, size=(num_induc,)), 
                    np.random.uniform(bottom_y, top_y, size=(num_induc,)), 
                    np.linspace(0, time_t.max(), num_induc), 
                    np.random.randn(num_induc), 
                    np.random.randn(num_induc)]
        kernel_tuples = [('variance', v), 
              ('RBF', 'torus', np.array([l_ang])), 
              ('RBF', 'euclid', np.array([l_w, l_s, l, l, l_time, l_one, l_one]))]
        
    elif mode == 'hd_w_s_pos_t_R3':
        ind_list = [np.linspace(0, 2*np.pi, num_induc+1)[:-1], 
                    np.random.randn(num_induc)*w_t.std(), 
                    np.random.uniform(0, s_t.std(), size=(num_induc,)), 
                    np.random.uniform(left_x, right_x, size=(num_induc,)), 
                    np.random.uniform(bottom_y, top_y, size=(num_induc,)), 
                    np.linspace(0, time_t.max(), num_induc), 
                    np.random.randn(num_induc), 
                    np.random.randn(num_induc), 
                    np.random.randn(num_induc)]
        kernel_tuples = [('variance', v), 
              ('RBF', 'torus', np.array([l_ang])), 
              ('RBF', 'euclid', np.array([l_w, l_s, l, l, l_time, l_one, l_one, l_one]))]
        
    elif mode == 'hd_w_s_pos_t_R4':
        ind_list = [np.linspace(0, 2*np.pi, num_induc+1)[:-1], 
                    np.random.randn(num_induc)*w_t.std(), 
                    np.random.uniform(0, s_t.std(), size=(num_induc,)), 
                    np.random.uniform(left_x, right_x, size=(num_induc,)), 
                    np.random.uniform(bottom_y, top_y, size=(num_induc,)), 
                    np.linspace(0, time_t.max(), num_induc), 
                    np.random.randn(num_induc), 
                    np.random.randn(num_induc), 
                    np.random.randn(num_induc), 
                    np.random.randn(num_induc)]
        kernel_tuples = [('variance', v), 
              ('RBF', 'torus', np.array([l_ang])), 
              ('RBF', 'euclid', np.array([l_w, l_s, l, l, l_time, l_one, l_one, l_one, l_one]))]
        
    elif mode == 'hd_t':
        ind_list = [np.linspace(0, 2*np.pi, num_induc+1)[:-1], 
                    np.linspace(0, time_t.max(), num_induc)]
        kernel_tuples = [('variance', v), 
              ('RBF', 'torus', np.array([l_ang])), 
              ('RBF', 'euclid', np.array([l_time]))]
        
    elif mode == 'hd_w_t':
        ind_list = [np.linspace(0, 2*np.pi, num_induc+1)[:-1], 
                    np.random.randn(num_induc)*w_t.std(), 
                    np.linspace(0, time_t.max(), num_induc)]
        kernel_tuples = [('variance', v), 
              ('RBF', 'torus', np.array([l_ang])), 
              ('RBF', 'euclid', np.array([l_w, l_time]))]
        
    elif mode == 'hdxR1':
        ind_list = [np.linspace(0, 2*np.pi, num_induc+1)[:-1], 
                    np.random.randn(num_induc)]
        kernel_tuples = [('variance', v), 
              ('RBF', 'torus', np.array([l_ang])), 
              ('RBF', 'euclid', np.array([l_one]))]
        
    elif mode == 'T1xR1':
        ind_list = [np.linspace(0, 2*np.pi, num_induc+1)[:-1], 
                    np.random.randn(num_induc)]
        l = 10.*np.ones((1, outdims))
        kernel_tuples = [('variance', v), ('RBF', 'torus', l), ('RBF', 'euclid', np.array([l_one]))]

    elif mode == 'T1':
        ind_list = [np.linspace(0, 2*np.pi, num_induc+1)[:-1]]
        l = 10.*np.ones((1, outdims))
        kernel_tuples = [('variance', v), ('RBF', 'torus', l)]
        
    elif mode == 'T1xt':
        ind_list = [np.linspace(0, 2*np.pi, num_induc+1)[:-1], 
                    np.linspace(0, time_t.max(), num_induc)]
        l = 10.*np.ones((1, outdims))
        kernel_tuples = [('variance', v), ('RBF', 'euclid', np.array([l_time])), ('RBF', 'torus', l)]
        
    elif mode == 'R1':
        ind_list = [np.linspace(-1, 1, num_induc)]
        l = 1.*np.ones((1, outdims))
        kernel_tuples = [('variance', v), ('RBF', 'euclid', l)]
        
    elif mode == 'R2':
        ind_list = [np.linspace(-1, 1, num_induc), 
                    np.random.randn(num_induc)]
        l = 1.*np.ones((2, outdims))
        kernel_tuples = [('variance', v), ('RBF', 'euclid', l)]
        
    elif mode is None: # for GP filters
        ind_list = []
        kernel_tuples = [('variance', v)]

    else:
        raise ValueError
        
    return kernel_tuples, ind_list



def get_VI_blocks(mode, behav_tuple):
    resamples = behav_tuple[0].shape[0]
    hd_t, w_t, s_t, x_t, y_t, time_t = behav_tuple
    covariates = cov_used(mode, behav_tuple)
    
    ang_dims = 0
    if mode == 'hd':
        VI_tuples = [(None, None, None, 1)]*len(covariates)
        ang_dims = 1
        
    elif mode == 'w':
        VI_tuples = [(None, None, None, 1)]*len(covariates)

    elif mode == 'hd_w' or mode == 'hdTw' or mode == 'hd_w_s' \
            or mode == 'hd_wTs' or mode == 'hd_w_s_pos' \
            or mode == 'hd_w_sTpos' or mode == 'hd_wTsTpos' \
            or mode == 'hd_t' or mode == 'hd_w_t'or mode == 'hd_w_s_t' \
            or mode == 'hd_w_s_pos_t':
        VI_tuples = [(None, None, None, 1)]*len(covariates)
        ang_dims = 1
        
    elif mode == 'hd_w_s_t_R1' or mode == 'hd_w_s_pos_t_R1':
        VI_tuples = [(None, None, None, 1)]*(len(covariates)-1) + \
                    [(['RW', (4.0, 1.0, True, False)], 'Normal', 'euclid', 1)]
        ang_dims = 1
        
    elif mode == 'hd_w_s_pos_t_R2':
        VI_tuples = [(None, None, None, 1)]*(len(covariates)-1) + \
                    [(['RW', (np.array([4.0]*2), np.array([1.0]*2), True, False)], 'Normal', 'euclid', 2)]
        ang_dims = 1
        
    elif mode == 'hd_w_s_pos_t_R3':
        VI_tuples = [(None, None, None, 1)]*(len(covariates)-1) + \
                    [(['RW', (np.array([4.0]*3), np.array([1.0]*3), True, False)], 'Normal', 'euclid', 3)]
        ang_dims = 1
        
    elif mode == 'hd_w_s_pos_t_R4':
        VI_tuples = [(None, None, None, 1)]*(len(covariates)-1) + \
                    [(['RW', (np.array([4.0]*4), np.array([1.0]*4), True, False)], 'Normal', 'euclid', 4)]
        ang_dims = 1
        
    elif mode == 'hdxR1':
        VI_tuples = [(None, None, None, 1), 
                     (['RW', (4.0, 1.0, True, False)], 'Normal', 'euclid', 1)]
        ang_dims = 1
        
    elif mode == 'T1xR1':
        VI_tuples = [(['RW', (0.0, 4.0, True, True)], 'Normal', 'torus', 1), 
                     (['RW', (4.0, 1.0, True, False)], 'Normal', 'euclid', 1)]
        ang_dims = 1

    elif mode == 'T1':
        VI_tuples = [(['RW', (0.0, 4.0, True, True)], 'Normal', 'torus', 1)]
        ang_dims = 1
        
    elif mode == 'T1xt':
        VI_tuples = [(['RW', (0.0, 4.0, True, True)], 'Normal', 'torus', 1), 
                     (None, None, None, 1)]
        ang_dims = 1
        
    elif mode == 'R1':
        VI_tuples = [(['RW', (4.0, 1.0, True, False)], 'Normal', 'euclid', 1)]
        
    elif mode == 'R2':
        VI_tuples = [(['RW', (np.array([4.0, 4.0]), np.array([1.0, 1.0]), True, False)], 'Normal', 'euclid', 2)]
        
    elif mode is None:
        VI_tuples = []
        ind_list = []

    else:
        raise ValueError
        
    in_dims = sum([d[3] for d in VI_tuples])
    return VI_tuples, in_dims, ang_dims



### assembly of the probabilistic model ###
def GP_params(ind_list, kernel_tuples, num_induc, neurons, inv_link, jitter, mean_func, filter_data, 
              learn_mean=True):
    """
    Create the GP object.
    """
    inducing_points = np.array(ind_list).T[None, ...].repeat(neurons, axis=0)
    inpd = inducing_points.shape[-1]

    gp_mapping = mdl.nonparametrics.Gaussian_process(inpd, neurons, kernel_tuples,
                                                     inv_link=inv_link, jitter=jitter, 
                                                     whiten=True, inducing_points=inducing_points, 
                                                     mean=mean_func, learn_mean=learn_mean)
        
    return gp_mapping




def likelihood_params(ll_mode, mode, behav_tuple, num_induc, inner_dims, inv_link, tbin, jitter, 
                      J, cutoff, neurons, mapping_net, C):
    """
    Create the likelihood object.
    """
    if mode is not None:
        kernel_tuples_, ind_list = kernel_used(mode, behav_tuple, num_induc, inner_dims)
            
    if ll_mode =='hZIP':
        inv_link_hetero = 'sigmoid'
    elif ll_mode =='hCMP':
        inv_link_hetero = 'identity'
    elif ll_mode =='hNB':
        inv_link_hetero = 'softplus'
    else:
        inv_link_hetero = None
            
    if inv_link_hetero is not None:
        mean_func = np.zeros((inner_dims))
        kt, ind_list = kernel_used(mode, behav_tuple, num_induc, inner_dims)
        gp_lvms = GP_params(ind_list, kt, num_induc, neurons, inv_link, jitter, mean_func, None, 
                            learn_mean=True)
    else:
        gp_lvms = None
    
    inv_link_hetero = None
    if ll_mode == 'IBP':
        likelihood = mdl.likelihoods.Bernoulli(tbin, inner_dims, inv_link)
        
    elif ll_mode == 'IP':
        likelihood = mdl.likelihoods.Poisson(tbin, inner_dims, inv_link)
        
    elif ll_mode == 'ZIP' or ll_mode =='hZIP':
        alpha = .1*np.ones(inner_dims)
        likelihood = mdl.likelihoods.ZI_Poisson(tbin, inner_dims, inv_link, alpha, dispersion_mapping=gp_lvms)
        #inv_link_hetero = lambda x: torch.sigmoid(x)/tbin
        
    elif ll_mode == 'NB' or ll_mode =='hNB':
        r_inv = 10.*np.ones(inner_dims)
        likelihood = mdl.likelihoods.Negative_binomial(tbin, inner_dims, inv_link, r_inv, dispersion_mapping=gp_lvms)
        
    elif ll_mode == 'CMP' or ll_mode =='hCMP':
        log_nu = np.zeros(inner_dims)
        likelihood = mdl.likelihoods.COM_Poisson(tbin, inner_dims, inv_link, log_nu, J=J, dispersion_mapping=gp_lvms)
        
    elif ll_mode == 'IG': # renewal process
        shape = np.ones(inner_dims)
        likelihood = mdl.likelihoods.Gamma(tbin, inner_dims, inv_link, shape, allow_duplicate=False)
        
    elif ll_mode == 'IIG': # renewal process
        mu_t = np.ones(inner_dims)
        likelihood = mdl.likelihoods.invGaussian(tbin, inner_dims, inv_link, mu_t, allow_duplicate=False)
        
    elif ll_mode == 'LN': # renewal process
        sigma_t = np.ones(inner_dims)
        likelihood = mdl.likelihoods.logNormal(tbin, inner_dims, inv_link, sigma_t, allow_duplicate=False)
        
    elif ll_mode == 'U':
        likelihood = mdl.likelihoods.Universal(inner_dims//C, C, inv_link, cutoff, mapping_net)
    else:
        raise NotImplementedError
        
    return likelihood



def ANN_params(tot_dims, enc_layers, angle_dims, neurons, inv_link):
    """
    full ANN model
    """
    hist_len = 1
    
    mu_ANN = enc_model(enc_layers, angle_dims, tot_dims-angle_dims, hist_len, neurons)
    mapping = mdl.parametrics.ANN(tot_dims, neurons, inv_link, mu_ANN, sigma_ANN=None, tens_type=torch.float)
    return mapping



def set_model(max_count, mtype, mode, ll_mode, behav_tuple, neurons, tbin, rc_t, num_induc, 
              inv_link='exp', jitter=1e-4, batch_size=10000, hist_len=19, filter_props=None, J=100, mapping_net=None, 
              C=1, enc_layers=None, var=1., gpp_num_induc=64, gpp_kernel='RBF'):
    """
    Assemble the encoding model.
    C is the number of GPs per neuron, or `channels'
    """
    inner_dims = neurons*C
    
    assert neurons == rc_t.shape[-2]
    resamples = rc_t.shape[-1]
    cutoff = max_count
    
    mean = 0 if ll_mode == 'U' else np.zeros((inner_dims)) # not learnable vs learnable
    if ll_mode == 'IBP': # overwrite
        inv_link = lambda x: torch.sigmoid(x)/tbin
    
    VI_tuples, in_dims, angle_dims = get_VI_blocks(mode, behav_tuple)
    covariates = cov_used(mode, behav_tuple)
    
    input_group = mdl.inference.input_group(in_dims, VI_tuples)
    input_group.set_XZ(covariates, resamples, batch_size=batch_size)
    
    if mtype == 'GP': # Gaussian process mapping
        kt, ind_list = kernel_used(mode, behav_tuple, num_induc, inner_dims, var)
        rate_model = GP_params(ind_list, kt, num_induc, inner_dims, inv_link, jitter, mean, 
                               None, learn_mean=(ll_mode != 'U'))
    elif mtype == 'ANN': # ANN mapping
        rate_model = ANN_params(in_dims, enc_layers, angle_dims, inner_dims, inv_link)
        
    likelihood = likelihood_params(ll_mode, mode, behav_tuple, num_induc, inner_dims, inv_link, tbin, jitter, 
                                   J, cutoff, neurons, mapping_net, C)
    likelihood.set_Y(rc_t, batch_size=batch_size)
    
    full = mdl.inference.VI_optimized(input_group, rate_model, likelihood)
    return full, covariates


