import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import subprocess
import os
import argparse



import pickle

import sys
sys.path.append("..") # access to library


import neuroprob as mdl
from neuroprob import utils

import models





def init_argparse():
    parser = argparse.ArgumentParser(
        usage="%(prog)s [OPTION] [FILE]...",
        description="Run diode simulations."
    )
    parser.add_argument(
        "-v", "--version", action="version",
        version = f"{parser.prog} version 1.0.0"
    )
    
    parser.add_argument('--batchsize', default=10000, type=int)
    parser.add_argument('--datatype', type=int)
    parser.add_argument('--modes', nargs='+', type=int)
    parser.add_argument('--cv', nargs='+', type=int)
    
    parser.add_argument('--ncvx', default=2, type=int)
    parser.add_argument('--gpu', default=0, type=int)
    parser.add_argument('--cov_MC', default=1, type=int)
    parser.add_argument('--ll_MC', default=10, type=int)
    
    parser.add_argument('--lr', default=1e-2, type=float)
    parser.add_argument('--lr_2', default=1e-3, type=float)
    
    args = parser.parse_args()
    return args



### synthetic data ###
class attention_bumps(mdl.parametrics.custom_wrapper):
    def __init__(self, neurons, inv_link='relu', tens_type=torch.float, active_dims=None):
        super().__init__(1, neurons, inv_link, tensor_type=tens_type, active_dims=active_dims)
    
    def set_params(self, mu, sigma, A, A_0):
        self.register_buffer('mu', torch.tensor(mu, dtype=self.tensor_type).to(self.dummy.device))
        self.register_buffer('sigma', torch.tensor(sigma, dtype=self.tensor_type).to(self.dummy.device))
        self.register_buffer('A', torch.tensor(A, dtype=self.tensor_type).to(self.dummy.device))
        self.register_buffer('A_0', torch.tensor(A_0, dtype=self.tensor_type).to(self.dummy.device))
    
    def compute_F(self, XZ):
        cov = XZ[..., self.active_dims]
        x = (cov[:, None, :, 0] - self.mu[None, :, None])/self.sigma[None, :, None]
        return self.A[None, :, None]*torch.exp(-x**2) + self.A_0[None, :, None], 0



def CMP_hdc(sample_bin, track_samples, covariates, neurons, trials=1):
    """
    CMP with separate mu and nu tuning curves
    """
    # Von Mises fields
    angle_0 = np.linspace(0, 2*np.pi, neurons+1)[:-1]
    beta = np.random.rand(neurons)*2.0 + 0.5
    rate_0 = np.random.rand(neurons)*4.0+4.0
    w = np.stack([np.log(rate_0), beta*np.cos(angle_0), beta*np.sin(angle_0)]).T # beta, phi_0 for theta modulation
    neurons = w.shape[0]

    vm_rate = mdl.rate_models.vonMises_GLM(neurons, inv_link='exp')
    vm_rate.set_params(w)

    # Dispersion tuning curve
    _angle_0 = np.random.permutation(angle_0)#angle_0 + 0.4*np.random.randn(neurons)#np.random.permutation(angle_0)
    _beta = 0.6*np.random.rand(neurons) + 0.1
    _rate_0 = np.random.rand(neurons)*0.5 + 0.5
    w = np.stack([np.log(_rate_0)+_beta, _beta*np.cos(_angle_0), _beta*np.sin(_angle_0)]).T # beta, phi_0 for theta modulation

    vm_disp = mdl.rate_models.vonMises_GLM(neurons, inv_link='identity')
    vm_disp.set_params(w)
    
    # sum for mu input
    comp_func = lambda x: ((x[0]*sample_bin)**(1/torch.exp(x[1])) - 0.5*(1/torch.exp(x[1])-1)) / sample_bin
    rate_model = mdl.parametrics.mixture_composition(1, [vm_rate, vm_disp], comp_func, inv_link='softplus')

    # CMP process output
    likelihood = mdl.likelihoods.COM_Poisson(sample_bin, neurons, 'softplus', dispersion_mapping=vm_disp)

    input_group = mdl.inference.input_group(1, [(None, None, None, 1)])
    input_group.set_XZ(covariates, track_samples, batch_size=track_samples, trials=trials)

    # NLL model
    glm = mdl.inference.VI_optimized(input_group, rate_model, likelihood)
    glm.validate_model(likelihood_set=False)
    return glm



def IP_bumps(sample_bin, track_samples, covariates, neurons, trials=1):
    """
    Poisson with spotlight attention
    """
    # Von Mises fields
    angle_0 = np.linspace(0, 2*np.pi, neurons+1)[:-1]
    beta = np.random.rand(neurons)*2.6 + 0.4
    rate_0 = np.random.rand(neurons)*4.0+3.0
    w = np.stack([np.log(rate_0), beta*np.cos(angle_0), beta*np.sin(angle_0)]).T # beta, phi_0 for theta modulation
    neurons = w.shape[0]

    vm_rate = mdl.rate_models.vonMises_GLM(neurons, inv_link='exp')
    vm_rate.set_params(w)


    att_rate = attention_bumps(neurons, active_dims=[1])
    mu = np.random.randn(neurons)
    sigma = 0.6*np.random.rand(neurons)+0.6
    A = 1.7*np.random.rand(neurons)
    A_0 = np.ones(neurons)*0.3
    att_rate.set_params(mu, sigma, A, A_0)
    
    rate_model = mdl.parametrics.product_model(2, [vm_rate, att_rate], inv_link='relu')

    input_group = mdl.inference.input_group(2, [(None, None, None, 1)]*2)
    input_group.set_XZ(covariates, track_samples, batch_size=track_samples, trials=trials)

    # Poisson process output
    likelihood = mdl.likelihoods.Poisson(sample_bin, neurons, 'relu')

    # NLL model
    glm = mdl.inference.VI_optimized(input_group, rate_model, likelihood)
    glm.validate_model(likelihood_set=False)
    return glm




### Data ###
def get_dataset(data_type):
    
    if data_type == 0:
        syn_data = np.load('./data/CMPh_HDC.npz')
        rhd_t = syn_data['rhd_t']
        ra_t = rhd_t
        
    elif data_type == 1:
        syn_data = np.load('./data/IP_HDC.npz')
        rhd_t = syn_data['rhd_t']
        ra_t = syn_data['ra_t']
    
    rc_t = syn_data['spktrain']
    tbin = syn_data['tbin']
        
    resamples = rc_t.shape[1]
    units_used = rc_t.shape[0]
    rcov = (rhd_t, ra_t, rhd_t, rhd_t, rhd_t, rhd_t)
    return rcov, units_used, tbin, resamples, rc_t



def main():
    parser = init_argparse()
    
    dev = utils.pytorch.get_device(gpu=parser.gpu)
    batch_size = parser.batchsize
    
    rcov, units_used, tbin, resamples, rc_t = get_dataset(parser.datatype)
    rhd_t = rcov[0]
    
    # GP with variable regressors model fit
    max_count = int(rc_t.max())
    print(max_count)

    nonconvex_trials = parser.ncvx
    modes_tot = [('GP', 'IP', 'hd', None, 8, 'exp', 1, [], False, 10, False, 'ew'), # 1
                 ('GP', 'hNB', 'hd', None, 8, 'exp', 1, [], False, 10, False, 'ew'), 
                 ('GP', 'U', 'hd', None, 8, 'identity', 3, [], False, 10, False, 'ew'), 
                 ('ANN', 'U', 'hd', None, 8, 'identity', 3, [], False, 10, False, 'ew'), 
                 ('GP', 'IP', 'T1', None, 8, 'exp', 1, [0], False, 10, False, 'ew'), # 5
                 ('GP', 'hNB', 'T1', None, 8, 'exp', 1, [0], False, 10, False, 'ew'), 
                 ('GP', 'U', 'T1', None, 8, 'identity', 3, [0], False, 10, False, 'ew'), 
                 ('ANN', 'U', 'T1', None, 8, 'identity', 3, [0], False, 10, False, 'ew'), 
                 ('GP', 'U', 'hdxR1', None, 16, 'identity', 3, [1], False, 10, False, 'ew')] # 9
    

    modes = [modes_tot[m] for m in parser.modes]
    cv_runs = parser.cv
    for m in modes:
        mtype, ll_mode, r_mode, spk_cpl, num_induc, inv_link, C, z_dims, delays, folds, cv_switch, basis_mode = m
        enc_layers, basis = models.hyper_params(basis_mode)
        print(m)

        shared_W = False
        if ll_mode == 'U':
            mapping_net = models.net(C, basis, max_count, units_used, shared_W)
        else:
            mapping_net = None

        for cvdata in models.get_cv_sets(m, cv_runs, parser.batchsize, rc_t, resamples, rcov):
            kcv, ftrain, fcov, vtrain, vcov, batch_size = cvdata

            lowest_loss = np.inf # nonconvex pick the best
            for kk in range(nonconvex_trials):

                retries = 0
                while True:
                    try:
                        full_model, _ = models.set_model('HDC', max_count, mtype, r_mode, ll_mode, spk_cpl, fcov, units_used, tbin, 
                                                         ftrain, num_induc, batch_size=batch_size, 
                                                         inv_link=inv_link, mapping_net=mapping_net, C=C, enc_layers=enc_layers)
                        full_model.to(dev)

                        # fit
                        sch = lambda o: optim.lr_scheduler.MultiplicativeLR(o, lambda e: 0.9)
                        opt_tuple = (optim.Adam, 100, sch)
                        opt_lr_dict = {'default': parser.lr}
                        for z_dim in z_dims:
                            opt_lr_dict['inputs.lv_std_{}'.format(z_dim)] = parser.lr_2
                            
                        full_model.set_optimizers(opt_tuple, opt_lr_dict)#, nat_grad=('rate_model.0.u_loc', 'rate_model.0.u_scale_tril'))

                        annealing = lambda x: 1.0#min(1.0, 0.002*x)

                        losses = full_model.fit(3000, loss_margin=-1e0, margin_epochs=100, kl_anneal_func=annealing, 
                                                cov_samples=parser.cov_MC, ll_samples=parser.ll_MC)
                        break
                        
                    except (RuntimeError, AssertionError):
                        print('Retrying...')
                        if retries == 3: # max retries
                            print('Stopped after max retries.')
                            raise ValueError
                        retries += 1

                if losses[-1] < lowest_loss:
                    lowest_loss = losses[-1]

                    # save model
                    name = 'valS' if shared_W else 'val'
                    if basis_mode != 'ew':
                        name += basis_mode
                    model_name = '{}{}_{}_{}_{}_C={}_{}'.format(name, parser.datatype, mtype, ll_mode, r_mode, C, kcv)
                    torch.save({'full_model': full_model.state_dict()}, './checkpoint/' + model_name)



if __name__ == "__main__":
    main()