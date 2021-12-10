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
                        start=0, T=100000, bs=5000):
    """
    Compute the dispersion statistics, per neuron in a population.
    
    :param string mode: *single* mode refers to computing separate single neuron quantities, *population* mode 
                        refers to computing over a population indicated by neurons, *peer* mode involves the 
                        peer predictability i.e. conditioning on all other neurons given a subset
    """
    N = int(np.ceil(T/bs))
    rate_model = []
    shape_model = []
    spktrain = spktrain[:, start:start+T]
    behav_list = [b[start:start+T] for b in behav_list]
    
    for k in range(N):
        covariates_ = [b[k*bs:(k+1)*bs] for b in behav_list]
        
        if glm.likelihood.filter_len > 1:
            ini_train = spktrain[None, :, :glm.likelihood.filter_len-1]
        else:
            ini_train = np.zeros((1, glm.likelihood.neurons, 1)) # used only for trial count
        
        ospktrain = spktrain[None, :, glm.likelihood.filter_len-1:]

        rate = glm.mapping.eval_rate(covariates_, neuron, n_samp=1000)
        rate_model += [rate[0, ...]]
        
        if glm.likelihood.dispersion_mapping is not None: # heteroscedastic
            disp = glm.likelihood.dispersion_mapping.eval_rate(covariates_, neuron, n_samp=1000)
            shape_model += [disp[0, ...]]
                

    rate_model = np.concatenate(rate_model, axis=1)
    if glm.likelihood.dispersion_mapping is not None: # heteroscedastic
        shape_model = np.concatenate(shape_model, axis=1)
    
    if ll_mode == 'IP':
        shape_model = None
        f_p = lambda c, avg, shape, t: utils.stats.poiss_count_prob(c, avg, shape, t)
    elif ll_mode[-2:] == 'NB':
        if glm.likelihood.dispersion_mapping is None:
            shape_model = glm.likelihood.r_inv.data.cpu().numpy()[:, None].repeat(T, axis=-1)
        f_p = lambda c, avg, shape, t: utils.stats.nb_count_prob(c, avg, shape, t)
    elif ll_mode[-3:] == 'CMP':
        if glm.likelihood.dispersion_mapping is None:
            shape_model = glm.likelihood.nu.data.cpu().numpy()[:, None].repeat(T, axis=-1)
        f_p = lambda c, avg, shape, t: utils.stats.cmp_count_prob(c, avg, shape, t)
    elif ll_mode[-3:] == 'ZIP':
        if glm.likelihood.dispersion_mapping is None:
            shape_model = glm.likelihood.alpha.data.cpu().numpy()[:, None].repeat(T, axis=-1)
        f_p = lambda c, avg, shape, t: utils.stats.zip_count_prob(c, avg, shape, t)
    elif ll_mode == 'U':
        f_p = lambda c, avg, shape, t: utils.stats.zip_count_prob(c, avg, shape, t)
    else:
        raise ValueError
    m_f = lambda x: x

    # compute the activity quantiles
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


### cross validation ###
def RG_pred_ll(model, r_mode, cov_used, cv_set, neuron_group=None, ll_mode='GH', ll_samples=100, cov_samples=1, 
               beta=1.0, bound='ELBO'):
    """
    Compute the predictive log likelihood (ELBO).
    """
    ftrain, fcov, vtrain, vcov = cv_set
    time_steps = vtrain.shape[-1]
    
    vcov = cov_used(r_mode, vcov)
    model.inputs.set_XZ(vcov, time_steps, batch_size=time_steps)
    model.likelihood.set_Y(vtrain, batch_size=time_steps)
    model.validate_model(likelihood_set=True)
    
    return -model.objective(0, cov_samples=cov_samples, ll_mode=ll_mode, bound=bound, neuron=neuron_group, 
                            beta=beta, ll_samples=ll_samples).data.cpu().numpy()
    
    

def LVM_pred_ll(model, z_dims, mode, cov_used, cv_set, f_neuron, v_neuron, eval_cov_MC=1, eval_ll_MC=100, eval_ll_mode='GH', 
                cov_MC=16, ll_MC=1, ll_mode='MC', beta=1.0, beta_z=1.0, bound='ELBO', max_iters=3000):
    """
    Compute the predictive log likelihood (ELBO).
    """
    ftrain, fcov, vtrain, vcov = cv_set
    time_steps = vtrain.shape[-1]
    
    vcov = cov_used(mode, vcov)
    
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
    losses = model.fit(max_iters, neuron=f_neuron, loss_margin=-1e0, margin_epochs=100, ll_mode=ll_mode, 
                       kl_anneal_func=annealing, cov_samples=cov_MC, ll_samples=ll_MC)
    

    return -model.objective(0, neuron=v_neuron, cov_samples=eval_cov_MC, ll_mode=eval_ll_mode, bound=bound, 
                            beta=beta, beta_z=beta_z, ll_samples=eval_ll_MC).data.cpu().numpy(), losses
