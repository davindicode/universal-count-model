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


    
### utils ###
def compute_P(full_model, covariates, show_neuron, MC=1000):
    """
    Compute predictive count distribution given X.
    """
    h = full_model.sample_F(covariates, MC, show_neuron)
    with torch.no_grad():
        logp = full_model.likelihood.get_logp(h, show_neuron).data # samples, N, time, K
    P_mc = torch.exp(logp)
    return P_mc


    
def compute_pred_P(full_model, batch, use_neuron, ts, cov_samples=10, ll_samples=1, tr=0):
    """
    Predictive posterior computed from model.
    """
    _, h, _ = full_model.evaluate(batch, obs_neuron=None, cov_samples=cov_samples, 
                                   ll_samples=ll_samples, ll_mode='MC', lv_input=None)

    with torch.no_grad():
        F_dims = full_model.likelihood._neuron_to_F(use_neuron)
        H = h[:, tr, F_dims, :]
        if ts is None:
            ts = np.arange(H.shape[-1])
        H = H[..., ts]
        logp = full_model.likelihood.get_logp(H, use_neuron) # samples, N, time, K
    P_mc = torch.exp(logp)
    return P_mc



def marginalized_P(full_model, eval_points, eval_dims, rcov, bs, use_neuron, MC=100, skip=1):
    """
    Marginalize over the behaviour p(X) for X not evaluated over.
    """
    rcov = [rc[::skip] for rc in rcov] # set dilution
    animal_T = rcov[0].shape[0]
    Ep = eval_points[0].shape[0]
    tot_len = Ep*animal_T
    
    covariates = []
    k = 0
    for d, rc in enumerate(rcov):
        if d in eval_dims:
            covariates.append(eval_points[k].repeat(animal_T))
            k += 1
        else:
            covariates.append(np.tile(rc, Ep))
    
    P_tot = np.empty((MC, len(use_neuron), Ep, full_model.likelihood.K))
    batches = int(np.ceil(animal_T / bs))
    for e in range(Ep):
        print(e)
        P_ = np.empty((MC, len(use_neuron), animal_T, full_model.likelihood.K))
        for b in range(batches):
            bcov = [c[e*animal_T:(e+1)*animal_T][b*bs:(b+1)*bs] for c in covariates]
            P_mc = compute_P(full_model, bcov, use_neuron, MC=MC).cpu()
            P_[..., b*bs:(b+1)*bs, :] = P_mc
            
        P_tot[..., e, :] = P_.mean(-2)
        
    return P_tot



def ind_to_pair(ind, N):
    """
    Convert index (k,) to a pair index (i, j) for neural population correlation
    """
    a = ind
    k = 1
    while a >= 0:
        a -= (N-k)
        k += 1
        
    n = k-1
    m = N-n + a
    return n-1, m



def get_q_Z(P, spike_binned, deq_noise=None):  
    if deq_noise is None:
        deq_noise = np.random.uniform(size=spike_binned.shape)
    else:
        deq_noise = 0

    cumP = np.cumsum(P, axis=-1) # T, K
    tt = np.arange(spike_binned.shape[0])
    quantiles = cumP[tt, spike_binned.astype(int)] - P[tt, spike_binned.astype(int)]*deq_noise
    Z = utils.stats.q_to_Z(quantiles)
    return quantiles, Z



def compute_count_stats(glm, ll_mode, tbin, spktrain, behav_list, neuron, traj_len=None, traj_spikes=None,
                        start=0, T=100000, bs=5000, mode='single'):
    """
    Compute the dispersion statistics, per neuron in a population.
    
    :param string mode: *single* mode refers to computing separate single neuron quantities, *population* mode 
                        refers to computing over a population indicated by neurons, *peer* mode involves the 
                        peer predictability i.e. conditioning on all other neurons given a subset
    """
    if mode != 'single' and mode != 'peer' and mode != 'population':
        raise ValueError()
        
    N = int(np.ceil(T/bs))
    rate_model = []
    shape_model = []
    spktrain = spktrain[:, start:start+T]
    behav_list = [b[start:start+T] for b in behav_list]
    
    count_model = (ll_mode == 'IP' or ll_mode[:2] == 'NB' or ll_mode[:3] == 'CMP' or ll_mode[:3] == 'ZIP')
    
    for k in range(N):
        covariates_ = [b[k*bs:(k+1)*bs] for b in behav_list]
        
        if glm.likelihood.filter_len > 1:
            ini_train = spktrain[None, :, :glm.likelihood.filter_len-1]
        else:
            ini_train = np.zeros((1, glm.likelihood.neurons, 1)) # used only for trial count
        
        if mode == 'single' or mode == 'population':
            ospktrain = spktrain[None, :, glm.likelihood.filter_len-1:]
        elif mode == 'peer':
            ospktrain = spktrain[None, :, glm.likelihood.filter_len-1:]
            for ne in neurons:
                ospktrain[ne, :] = 0

        rate = glm.mapping.eval_rate(covariates_, neuron, n_samp=1000)
        rate_model += [rate[0, ...]]
        
        if count_model and glm.likelihood.dispersion_mapping is not None:
            disp = glm.likelihood.dispersion_mapping.eval_rate(covariates_, neuron, n_samp=1000)
            shape_model += [disp[0, ...]]
                

    rate_model = np.concatenate(rate_model, axis=1)
    if count_model and glm.likelihood.dispersion_mapping is not None:
        shape_model = np.concatenate(shape_model, axis=1)
    
    if ll_mode == 'IP':
        shape_model = None
        f_p = lambda c, avg, shape, t: utils.stats.poiss_count_prob(c, avg, shape, t)
    elif ll_mode[:2] == 'NB':
        if count_model and glm.likelihood.dispersion_mapping is None:
            shape_model = glm.likelihood.r_inv.data.cpu().numpy()[:, None].repeat(T, axis=-1)
        f_p = lambda c, avg, shape, t: utils.stats.nb_count_prob(c, avg, shape, t)
    elif ll_mode[:3] == 'CMP':
        if count_model and glm.likelihood.dispersion_mapping is None:
            shape_model = glm.likelihood.nu.data.cpu().numpy()[:, None].repeat(T, axis=-1)
        f_p = lambda c, avg, shape, t: utils.stats.cmp_count_prob(c, avg, shape, t)
    elif ll_mode[:3] == 'ZIP':
        if count_model and glm.likelihood.dispersion_mapping is None:
            shape_model = glm.likelihood.alpha.data.cpu().numpy()[:, None].repeat(T, axis=-1)
        f_p = lambda c, avg, shape, t: utils.stats.zip_count_prob(c, avg, shape, t)
    elif ll_mode == 'U':
        f_p = lambda c, avg, shape, t: utils.stats.zip_count_prob(c, avg, shape, t)
    else:
        raise ValueError
    m_f = lambda x: x

    if shape_model is not None:
        assert traj_len == 1
    if traj_len is not None:
        traj_lens = (T // traj_len) * [traj_len]
        
    q_ = []
    for k, ne in enumerate(neuron):
        if traj_spikes is not None:
            avg_spikecnt = np.cumsum(rate_model[k]*tbin)
            nc = 1
            traj_len = 0
            for tt in range(T):
                if avg_spikecnt >= traj_spikes*nc:
                    nc += 1
                    traj_lens.append(traj_len)
                    traj_len = 0
                    continue
                traj_len += 1
                
        if shape_model is not None:
            sh = shape_model[k]
            spktr = spktrain[ne]
            rm = rate_model[k]
        else:
            sh = None
            spktr = []
            rm = []
            start = np.cumsum(traj_lens)
            for tt, traj_len in enumerate(traj_lens):
                spktr.append(spktrain[ne][start[tt]:start[tt]+traj_len].sum())
                rm.append(rate_model[k][start[tt]:start[tt]+traj_len].sum())
            spktr = np.array(spktr)
            rm = np.array(rm)
                    
        q_.append(utils.stats.count_KS_method(f_p, m_f, tbin, spktr, rm, shape=sh))

    return q_





def get_cv_sets(mode, cv_runs, batchsize, rc_t, resamples, rcov):
    mtype, ll_mode, r_mode, num_induc, inv_link, C, z_dims, delays, folds, cv_switch, basis_mode = mode
    returns = []
    
    if delays is not False:
        if delays.min() > 0:
            raise ValueError('Delay minimum must be 0 or less')
        if delays.max() < 0:
            raise ValueError('Delay maximum must be 0 or less')
            
        D_min = -delays.min()
        D_max = -delays.max()
        D = -D_max+D_min # total delay steps
        dd = delays
        
    else:
        D_min = 0
        D_max = 0
        D = 0 # total delay steps
        dd = [0]
    
    for delay in dd:
        rc_t_ = rc_t[D_min:(D_max if D_max < 0 else None)]
        _min = D_min+delay
        _max = D_max+delay
        rcov_ = [rc[_min:(_max if _max < 0 else None)] for rc in rcov]
        resamples_ = resamples - D + 1
        
        cv_sets, vstart = utils.neural.spiketrain_CV(folds, rc_t_, resamples_, rcov_)
        for kcv in cv_runs:
            if kcv >= 0:
                if cv_switch:
                    vtrain, vcov, ftrain, fcov = cv_sets[kcv]
                    batch_size = batchsize

                else:
                    ftrain, fcov, vtrain, vcov = cv_sets[kcv]

                    if len(z_dims) > 0: # has latent and CV, remove validation segment
                        segment_lengths = [vstart[kcv], resamples-vstart[kcv]-vtrain.shape[-1]]
                        batch_size = utils.neural.batch_segments(segment_lengths, batchsize)
                    else:
                        batch_size = batchsize

            else:
                ftrain, fcov = rc_t_, rcov_
                vtrain, vcov = None, None
                batch_size = batchsize

            if delays is not False:
                kcv_str = str(kcv) + 'delay' + str(delay)
            else:
                kcv_str = str(kcv)

            returns.append((kcv_str, ftrain, fcov, vtrain, vcov, batch_size))
        
    return returns


    
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
def cov_used(mode, behav_tuple, data_type):
    """
    Create the used covariates list for different models
    """
    resamples = behav_tuple[0].shape[0]
    if data_type == 'HDC':
        hd_t, w_t, s_t, x_t, y_t, time_t = behav_tuple
    elif data_type == 'CA':
        x_t, y_t, s_t, th_t, hd_t, time_t = behav_tuple
    else:
        raise ValueError('Invalid data type {}'.format(data_type))
    
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



def kernel_used(mode, behav_tuple, data_type, num_induc, outdims, var=1.):
    if data_type == 'HDC':
        hd_t, w_t, s_t, x_t, y_t, time_t = behav_tuple
        # specifics
        l_w = w_t.std()*np.ones(outdims)
        
    elif data_type == 'CA':
        x_t, y_t, s_t, th_t, hd_t, time_t = behav_tuple
    
    left_x = x_t.min()
    right_x = x_t.max()
    bottom_y = y_t.min()
    top_y = y_t.max()
    
    l = 10.*np.ones(outdims)
    l_ang = 5.*np.ones(outdims)
    l_s = 10.*np.ones(outdims)
    v = var*np.ones(outdims)
    l_time = time_t.max()/2.*np.ones(outdims)
    l_one = np.ones(outdims)
    
    factorized = 1
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
        
    elif mode == 'hd_wTs':
        factorized = 2
        ind_list = [np.linspace(0, 2*np.pi, num_induc+1)[:-1], 
                    np.random.randn(num_induc)*w_t.std(), 
                    np.random.uniform(0, s_t.std(), size=(num_induc,))]
        kt = [('variance', v), (None,), (None,), ('RBF', 'euclid', np.array([l_s]))]
        kt_ = [('variance', v), ('RBF', 'torus', np.array([l_ang])), ('RBF', 'euclid', np.array([l_w])), (None,)]
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
        
    elif mode == 'hd_w_sTpos':
        factorized = 2
        ind_list = [np.linspace(0, 2*np.pi, num_induc+1)[:-1], 
                    np.random.randn(num_induc)*w_t.std(), 
                    np.random.uniform(0, s_t.std(), size=(num_induc,)), 
                    np.random.uniform(left_x, right_x, size=(num_induc,)), 
                    np.random.uniform(bottom_y, top_y, size=(num_induc,))]
        kt = [('variance', v), (None,), (None,), (None,), ('RBF', 'euclid', np.array([l, l]))]
        kt_ = [('variance', v), ('RBF', 'torus', np.array([l_ang])), ('RBF', 'euclid', np.array([l_w, l_s])), (None,), (None,)]
        kernel_tuples = [('variance', v), 
              ('RBF', 'torus', np.array([l_ang])), 
              ('RBF', 'euclid', np.array([l_w, l_s, l, l]))]
        
    elif mode == 'hd_wTsTpos':
        factorized = 3
        ind_list = [np.linspace(0, 2*np.pi, num_induc+1)[:-1], 
                    np.random.randn(num_induc)*w_t.std(), 
                    np.random.uniform(0, s_t.std(), size=(num_induc,)), 
                    np.random.uniform(left_x, right_x, size=(num_induc,)), 
                    np.random.uniform(bottom_y, top_y, size=(num_induc,))]
        kt = [('variance', v), (None,), (None,), (None,), ('RBF', 'euclid', np.array([l, l]))]
        kt_ = [('variance', v), (None,), (None,), ('RBF', 'euclid', np.array([l_s])), (None,), (None,)]
        kt__ = [('variance', v), ('RBF', 'torus', np.array([l_ang])), ('RBF', 'euclid', np.array([l_w])), (None,), (None,), (None,)]
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
        
    if factorized == 1:
        kernel_tuples_ = (kernel_tuples,)
    elif factorized == 2:
        kernel_tuples_ = (kt, kt_, kernel_tuples)
    elif factorized == 3:
        kernel_tuples_ = (kt, kt_, kt__, kernel_tuples)
        
    return kernel_tuples_, ind_list, factorized



def get_VI_blocks(mode, behav_tuple, data_type):
    resamples = behav_tuple[0].shape[0]
    if data_type == 'HDC':
        hd_t, w_t, s_t, x_t, y_t, time_t = behav_tuple
    elif data_type == 'CA':
        x_t, y_t, s_t, th_t, hd_t, time_t = behav_tuple
        
    covariates = cov_used(mode, behav_tuple, data_type)
    
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
def GP_params(ind_list, kernel_tuples_, num_induc, neurons, inv_link, jitter, mean_func, factorized, filter_data, 
              learn_mean=True):
    """
    Create the GP object.
    """
    kernel_tuples, = kernel_tuples_
    inducing_points = np.array(ind_list).T[None, ...].repeat(neurons, axis=0)
    inpd = inducing_points.shape[-1]

    glm_rate = mdl.nonparametrics.Gaussian_process(inpd, neurons, kernel_tuples,
                                                   inv_link=inv_link, jitter=jitter, 
                                                   whiten=True, inducing_points=inducing_points, 
                                                   mean=mean_func, learn_mean=learn_mean)
        
    return glm_rate




def likelihood_params(ll_mode, mode, behav_tuple, data_type, num_induc, inner_dims, inv_link, tbin, jitter, 
                      J, cutoff, neurons, mapping_net, C):
    """
    Create the likelihood object.
    """
    if mode is not None:
        kernel_tuples_, ind_list, factorized = kernel_used(mode, behav_tuple, data_type, num_induc, inner_dims)
        if factorized > 1: # overwrite
            inv_link = 'relu'
            
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
        kt, ind_list, factorized = kernel_used(mode, behav_tuple, data_type, num_induc, inner_dims)
        gp_lvms = GP_params(ind_list, kt, num_induc, neurons, inv_link, jitter, mean_func, factorized, None, 
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



def set_model(data_type, max_count, mtype, mode, ll_mode, behav_tuple, neurons, tbin, rc_t, num_induc, 
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
    
    VI_tuples, in_dims, angle_dims = get_VI_blocks(mode, behav_tuple, data_type)
    covariates = cov_used(mode, behav_tuple, data_type)
    
    input_group = mdl.inference.input_group(in_dims, VI_tuples)
    input_group.set_XZ(covariates, resamples, batch_size=batch_size)
    
    if mtype == 'GP': # Gaussian process mapping
        kt, ind_list, factorized = kernel_used(mode, behav_tuple, data_type, num_induc, inner_dims, var)
        rate_model = GP_params(ind_list, kt, num_induc, inner_dims, inv_link, jitter, mean, 
                               factorized, None, learn_mean=(ll_mode != 'U'))
    elif mtype == 'ANN': # ANN mapping
        rate_model = ANN_params(in_dims, enc_layers, angle_dims, inner_dims, inv_link)
        
    likelihood = likelihood_params(ll_mode, mode, behav_tuple, data_type, num_induc, inner_dims, inv_link, tbin, jitter, 
                                   J, cutoff, neurons, mapping_net, C)
    likelihood.set_Y(rc_t, batch_size=batch_size)
    
    full = mdl.inference.VI_optimized(input_group, rate_model, likelihood)
    return full, covariates



### cross validation ###
def RG_pred_ll(model, r_mode, cv_set, neuron_group=None, ll_mode='GH', ll_samples=100, cov_samples=1, 
               beta=1.0, datatype='HDC', bound='ELBO'):
    """
    Compute the predictive log likelihood (ELBO).
    """
    ftrain, fcov, vtrain, vcov = cv_set
    time_steps = vtrain.shape[-1]
    
    vcov = cov_used(r_mode, vcov, datatype)
    model.inputs.set_XZ(vcov, time_steps, batch_size=time_steps)
    model.likelihood.set_Y(vtrain, batch_size=time_steps)
    model.validate_model(likelihood_set=True)
    
    return -model.objective(0, cov_samples=cov_samples, ll_mode=ll_mode, bound=bound, neuron=neuron_group, 
                            beta=beta, ll_samples=ll_samples).data.cpu().numpy()
    
    

def LVM_pred_ll(model, z_dims, mode, cv_set, f_neuron, v_neuron, eval_cov_MC=1, eval_ll_MC=100, eval_ll_mode='GH', 
                datatype='HDC', cov_MC=16, ll_MC=1, ll_mode='MC', beta=1.0, beta_z=1.0, bound='ELBO'):
    """
    Compute the predictive log likelihood (ELBO).
    """
    ftrain, fcov, vtrain, vcov = cv_set
    time_steps = vtrain.shape[-1]
    
    vcov = cov_used(mode, vcov, datatype)
    
    model.inputs.set_XZ(vcov, time_steps, batch_size=time_steps)
    model.likelihood.set_Y(vtrain, batch_size=time_steps)
    
    # fit
    sch = lambda o: optim.lr_scheduler.MultiplicativeLR(o, lambda e: 0.9)
    opt_tuple = (optim.Adam, 100, sch)
    opt_lr_dict = {'default': 0}
    for z_dim in z_dims:
        opt_lr_dict['inputs.lv_mu_{}'.format(z_dim)] = 1e-2
        opt_lr_dict['inputs.lv_std_{}'.format(z_dim)] = 1e-3

    model.set_optimizers(opt_tuple, opt_lr_dict)

    annealing = lambda x: 1.0#min(1.0, 0.002*x)
    losses = model.fit(3000, neuron=f_neuron, loss_margin=-1e0, margin_epochs=100, ll_mode=ll_mode, 
                       kl_anneal_func=annealing, cov_samples=cov_MC, ll_samples=ll_MC)
    

    return -model.objective(0, neuron=v_neuron, cov_samples=eval_cov_MC, ll_mode=eval_ll_mode, bound=bound, 
                            beta=beta, beta_z=beta_z, ll_samples=eval_ll_MC).data.cpu().numpy(), losses
