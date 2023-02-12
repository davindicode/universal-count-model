import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F
import torch.optim as optim


import scipy.special as sps
import scipy.stats as scstats
import numpy as np

import pickle

import os
    
    
import sys

sys.path.append("..") # access to library
import neuroprob as nprb

sys.path.append("../scripts") # access to scripts
import models
    
    
    




use_neuron = np.arange(50)


rhd_t = rcov[0]
ra_t = rcov[1]
covariates = [rhd_t[None, :, None].repeat(trials, axis=0), 
              ra_t[None, :, None].repeat(trials, axis=0)]


# modes = [('GP', 'IP', 'hd', 8, 'exp', 1, [], False, 10, False, 'ew'), 
#          ('GP', 'U', 'hd', 8, 'identity', 3, [], False, 10, False, 'ew'),
#          ('GP', 'U', 'hdxR1', 16, 'identity', 3, [1], False, 10, False, 'ew')]
checkpoint_dir = '../scripts/checkpoint/'
config_name = 'th1_U-el-4_svgp-64_X[hd-omega-speed-x-y-time]_Z[]_40K11_0d0_10f-1'
batch_info = 500


full_model, training_loss, fit_dict, val_dict = models.load_model(
    config_name,
    checkpoint_dir,
    dataset_dict,
    batch_info,
    device,
)


def latent_variable(config_names):
    # neuron subgroup likelihood CV
    beta = 0.0
    n_group = np.arange(5)
    val_neuron = [n_group, n_group+10, n_group+20, n_group+30, n_group+40]
    ncvx = 2
    kcvs = [2, 5, 8] # validation sets chosen in 10-fold split of data

    batch_size = 5000
    cv_pll = []
    for em, name in enumerate(config_names):
        for cvdata in model_utils.get_cv_sets(mode, kcvs, batch_size, rc_t, resamples, rcov):
            _, ftrain, fcov, vtrain, vcov, cvbatch_size = cvdata
            cv_set = (ftrain, fcov, vtrain, vcov)

            if em > 1:
                for v_neuron in val_neuron:

                    prev_ll = np.inf
                    for tr in range(ncvx):
                        full_model = get_full_model(datatype, cvdata, resamples, rc_t, 100, 
                                                    mode, rcov, max_count, neurons)
                        mask = np.ones((neurons,), dtype=bool)
                        mask[v_neuron] = False
                        f_neuron = np.arange(neurons)[mask]
                        ll = model_utils.LVM_pred_ll(
                            full_model, mode[-5], mode[2], models.cov_used, cv_set, f_neuron, v_neuron, 
                            beta=beta, beta_z=0.0, max_iters=3000)[0]
                        if ll < prev_ll:
                            prev_ll = ll

                    cv_pll.append(prev_ll)

            else:
                for v_neuron in val_neuron:
                    full_model = get_full_model(datatype, cvdata, resamples, rc_t, 100, 
                                                mode, rcov, max_count, neurons)
                    cv_pll.append(model_utils.RG_pred_ll(
                        full_model, mode[2], models.cov_used, cv_set, bound='ELBO', 
                        beta=beta, neuron_group=v_neuron, ll_mode='GH', ll_samples=100))


    cv_pll = np.array(cv_pll).reshape(len(Ms), len(kcvs), len(val_neuron))




    # compute tuning curves and latent trajectories for XZ joint regression-latent model
    mode = modes[2]
    cvdata = model_utils.get_cv_sets(mode, [-1], 5000, rc_t, resamples, rcov)[0]

    full_model = get_full_model(datatype, cvdata, resamples, rc_t, 100, 
                                mode, rcov, max_count, neurons)

    # latents
    X_loc, X_std = full_model.inputs.eval_XZ()

    T = X_loc[1].shape[0]
    X_c, shift, sign, scale, _ = utils.latent.signed_scaled_shift(X_loc[1], ra_t[:T], 
                                                                  dev, 'euclid', learn_scale=True)
    X_s = scale*X_std[1]



    # tuning
    steps = 100
    covariates_z = [0.*np.ones(steps), np.linspace(X_loc[1].min(), X_loc[1].max(), steps)]
    P_mc = model_utils.compute_P(full_model, covariates_z, use_neuron, MC=1000).cpu()

    x_counts = torch.arange(max_count+1)
    avg = (x_counts[None, None, None, :]*P_mc).sum(-1)
    xcvar = ((x_counts[None, None, None, :]**2*P_mc).sum(-1)-avg**2)
    ff = xcvar/avg

    avgs = utils.signal.percentiles_from_samples(avg, percentiles=[0.05, 0.5, 0.95], 
                                                 smooth_length=5, padding_mode='replicate')
    avglower, avgmean, avgupper = [cs_.cpu().numpy() for cs_ in avgs]

    ffs = utils.signal.percentiles_from_samples(ff, percentiles=[0.05, 0.5, 0.95], 
                                                smooth_length=5, padding_mode='replicate')
    fflower, ffmean, ffupper = [cs_.cpu().numpy() for cs_ in ffs]


    covariates_z[1] = sign*scale*covariates_z[1]+shift
    grate = glm.mapping.eval_rate(covariates_z, use_neuron)[0, ...]
    gFF = np.ones_like(grate)



    latent_dict = {
        'X_c': X_c, 
        'X_s': X_s, 
        'covariates_z': covariates_z, 
        'cv_pll': cv_pll, 
        'avglower': avglower, 
        'avgmean': avgmean, 
        'avgupper': avgupper, 
        'fflower': fflower, 
        'ffmean': ffmean, 
        'ffupper': ffupper, 
        'grate': grate, 
        'gFF': gFF, 
    }
    
    return latent_dict



def variability_stats(config_names):
    # KS framework
    Qq = []
    Zz = []
    R = []
    Rp = []

    batch_size = 5000

    CV = [2, 5, 8]
    for kcv in CV:
        for en, mode in enumerate(config_names):
            cvdata = model_utils.get_cv_sets(mode, [kcv], batch_size, rc_t, resamples, rcov)[0]
            _, ftrain, fcov, vtrain, vcov, cvbatch_size = cvdata
            time_steps = ftrain.shape[-1]

            full_model = get_full_model(datatype, cvdata, resamples, rc_t, 100, 
                                        mode, rcov, max_count, neurons)

            if en > 0:
                # predictive posterior
                q_ = []
                Z_ = []
                for b in range(full_model.inputs.batches):
                    P_mc = model_utils.compute_pred_P(full_model, b, use_neuron, None, cov_samples=10, ll_samples=1, tr=0)
                    P = P_mc.mean(0).cpu().numpy()

                    for n in range(len(use_neuron)):
                        spike_binned = full_model.likelihood.spikes[b][0, n, :].numpy()
                        q, Z = model_utils.get_q_Z(P[n, ...], spike_binned, deq_noise=None)

                        if b == 0:
                            q_.append(q)
                            Z_.append(Z)
                        else:
                            q_[n] = np.concatenate((q_[n], q))
                            Z_[n] = np.concatenate((Z_[n], Z))

            elif en == 0:
                cov_used = models.cov_used(mode[2], fcov)
                q_ = model_utils.compute_count_stats(full_model, 'IP', tbin, ftrain, cov_used, use_neuron, \
                                                     traj_len=1, start=0, T=time_steps, bs=5000)
                Z_ = [utils.stats.q_to_Z(q) for q in q_]


            Pearson_s = []
            for n in range(len(use_neuron)):
                for m in range(n+1, len(use_neuron)):
                    r, r_p = scstats.pearsonr(Z_[n], Z_[m]) # Pearson r correlation test
                    Pearson_s.append((r, r_p))

            r = np.array([p[0] for p in Pearson_s])
            r_p = np.array([p[1] for p in Pearson_s])

            Qq.append(q_)
            Zz.append(Z_)
            R.append(r)
            Rp.append(r_p)



    fisher_z = []
    fisher_q = []
    for en, r in enumerate(R):
        fz = 0.5*np.log((1+r)/(1-r))*np.sqrt(time_steps-3)
        fisher_z.append(fz)
        fisher_q.append(utils.stats.Z_to_q(fz))



    q_DS_ = []
    T_DS_ = []
    T_KS_ = []
    for q in Qq:
        for qq in q:
            T_DS, T_KS, sign_DS, sign_KS, p_DS, p_KS = utils.stats.KS_statistics(qq, alpha=0.05, alpha_s=0.05)
            T_DS_.append(T_DS)
            T_KS_.append(T_KS)

            Z_DS = T_DS/np.sqrt(2/(qq.shape[0]-1))
            q_DS_.append(utils.stats.Z_to_q(Z_DS))


    q_DS_ = np.array(q_DS_).reshape(len(CV), len(Ms), -1)
    T_DS_ = np.array(T_DS_).reshape(len(CV), len(Ms), -1)
    T_KS_ = np.array(T_KS_).reshape(len(CV), len(Ms), -1)

    
    # noise correlation structure
    NN = len(use_neuron)
    R_mat_Xp = np.zeros((NN, NN))
    R_mat_X = np.zeros((NN, NN))
    R_mat_XZ = np.zeros((NN, NN))
    for a in range(len(R[0])):
        n, m = model_utils.ind_to_pair(a, NN)
        R_mat_Xp[n, m] = R[0][a]
        R_mat_X[n, m] = R[1][a]
        R_mat_XZ[n, m] = R[2][a]

    # noise correlation structure
    NN = len(use_neuron)
    R_mat_Xp = np.zeros((NN, NN))
    R_mat_X = np.zeros((NN, NN))
    R_mat_XZ = np.zeros((NN, NN))
    for a in range(len(R[0])):
        n, m = model_utils.ind_to_pair(a, NN)
        R_mat_Xp[n, m] = R[0][a]
        R_mat_X[n, m] = R[1][a]
        R_mat_XZ[n, m] = R[2][a]


    variability_dict = {
        q_DS_, 
        T_DS_, 
        T_KS_, 
        sign_DS, 
        fisher_z, 
        fisher_q, 
        Qq, 
        Zz, 
        'R': R, 
        'Rp': Rp, 
        'R_mat_Xp': R_mat_Xp, 
        'R_mat_X': R_mat_X, 
        'R_mat_XZ': R_mat_XZ, 
    }
    
    return variability_dict



def main():
    save_dir = '../output/'
    data_path = '../../data/'
    
    dev = utils.pytorch.get_device(gpu=0)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    # names
    reg_config_names = [
        'modIP1_IP-exp_svgp-8_X[hd]_Z[]_1K28_0d0_10f', 
        'modIP1_hNB-exp_svgp-8_X[hd]_Z[]_1K28_0d0_10f', 
        'modIP1_U-el-3_svgp-8_X[hd]_Z[]_1K28_0d0_10f', 
        'modIP1_U-el-3_ffnn-50-50-100_X[hd]_Z[]_1K28_0d0_10f', 
    ]
    
    lat_config_names = [
        'modIP1_IP-exp_svgp-8_X[hd]_Z[]_1K28_0d0_10f', 
        'modIP1_hNB-exp_svgp-8_X[hd]_Z[]_1K28_0d0_10f', 
        'modIP1_U-el-3_svgp-8_X[hd]_Z[]_1K28_0d0_10f', 
        'modIP1_U-el-3_ffnn-50-50-100_X[hd]_Z[]_1K28_0d0_10f', 
    ]
    
    ### load dataset ###
    data_type = 'modIP1'
    bin_size = 1
    
    dataset_dict = models.get_dataset(data_type, bin_size, data_path)
    
    ### analysis ###
    latent_dict = latent_variable()
    variability_dict = variability_stats()
    
    ### export ###
    data_run = {
        'latent': latent_dict, 
        'variability': variability_dict, 
    }

    pickle.dump(data_run, open(save_dir + 'modIP_results.p', 'wb'))
    
    
    
if __name__ == "__main__":
    main()