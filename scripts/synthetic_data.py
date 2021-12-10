import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F
import torch.optim as optim

import matplotlib.pyplot as plt

import numpy as np

import sys
sys.path.append("..") # access to library



import neuroprob as mdl
from neuroprob import utils
from neuroprob import GP


dev = utils.pytorch.get_device(gpu=0)

import validation

import pickle





# Gaussian von Mises bump head direction model
sample_bin = 0.1 # 100 ms
track_samples = 10000
trials = 1

hd_t = np.empty(track_samples)

hd_t[0] = 0
rn = np.random.randn(track_samples)
for k in range(1, track_samples):
    hd_t[k] = hd_t[k-1] + 0.5*rn[k]
    
hd_t = hd_t % (2*np.pi)


# GP trajectory sample
Tl = track_samples

l = 200.*sample_bin*np.ones((1, 1))
v = np.ones(1)
kernel_tuples = [('variance', v), 
                 ('RBF', 'euclid', l)]

with torch.no_grad():
    kernel, _, _ = GP.kernels.create_kernel(kernel_tuples, 'softplus', torch.double)

    T = torch.arange(Tl)[None, None, :, None]*sample_bin
    K = kernel(T, T)[0, 0]
    K.view(-1)[::Tl+1] += 1e-6
    
L = torch.cholesky(K)
eps = torch.randn(Tl).double()
v = L @ eps
a_t = v.data.numpy()





### sample activity ###

# heteroscedastic CMP
neurons = 50

covariates = [hd_t[None, :, None].repeat(trials, axis=0)]
glm = validation.CMP_hdc(sample_bin, track_samples, covariates, neurons, trials=trials)
glm.to(dev)

XZ, rate, _ = glm.evaluate(0)
syn_train = glm.likelihood.sample(rate[0].cpu().numpy(), XZ=XZ)

trial = 0
bin_size = 1
tbin, resamples, rc_t, (rhd_t,) = utils.neural.bin_data(bin_size, sample_bin, syn_train[trial], 
                                                        track_samples, (np.unwrap(hd_t),), average_behav=True, binned=True)
rhd_t = rhd_t % (2*np.pi)


np.savez_compressed('./data/CMPh_HDC', spktrain=rc_t, rhd_t=rhd_t, tbin=tbin)
torch.save({'model': glm.state_dict()}, './data/CMPh_HDC_model')



# modulated Poisson
neurons = 50

covariates = [hd_t[None, :, None].repeat(trials, axis=0), 
              a_t[None, :, None].repeat(trials, axis=0)]
glm = validation.IP_bumps(sample_bin, track_samples, covariates, neurons, trials=trials)
glm.to(dev)


_, rate, _ = glm.evaluate(0)
syn_train = glm.likelihood.sample(rate[0].cpu().numpy())

trial = 0
bin_size = 1
tbin, resamples, rc_t, (rhd_t, ra_t) = utils.neural.bin_data(bin_size, sample_bin, syn_train[trial], 
                                                        track_samples, (np.unwrap(hd_t), a_t), average_behav=True, binned=True)
rhd_t = rhd_t % (2*np.pi)


np.savez_compressed('./data/IP_HDC', spktrain=rc_t, rhd_t=rhd_t, ra_t=ra_t, tbin=tbin)
torch.save({'model': glm.state_dict()}, './data/IP_HDC_model')