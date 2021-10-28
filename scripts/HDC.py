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
    parser.add_argument('--binsize', default=100, type=int)
    parser.add_argument('--session_id', default=1, type=int)
    parser.add_argument('--phase', default=1, type=int)
    parser.add_argument('--modes', nargs='+', type=int)
    parser.add_argument('--cv', nargs='+', type=int)
    
    parser.add_argument('--ncvx', default=2, type=int)
    parser.add_argument('--gpu', default=0, type=int)
    parser.add_argument('--cov_MC', default=1, type=int) # latent covariates MC samples
    parser.add_argument('--ll_MC', default=10, type=int) # likelihood MC sammples
    
    parser.add_argument('--lr', default=1e-2, type=float)
    parser.add_argument('--lr_2', default=1e-3, type=float)
    
    args = parser.parse_args()
    return args




def get_dataset(session_id, phase, bin_size):

    data = np.load('./data/{}_{}.npz'.format(session_id, phase))
    spktrain = data['spktrain']
    x_t = data['x_t']
    y_t = data['y_t']
    hd_t = data['hd_t']
    region_edge = data['region_edge']
    #arena = data['arena']

    sample_bin = 0.001

    neurons = spktrain.shape[0]
    track_samples = spktrain.shape[1]

    tbin, resamples, rc_t, (rhd_t, rx_t, ry_t) = utils.neural.bin_data(
        bin_size, sample_bin, spktrain, track_samples, 
        (np.unwrap(hd_t), x_t, y_t), average_behav=True, binned=True
    )

    rw_t = (rhd_t[1:]-rhd_t[:-1])/tbin
    rw_t = np.concatenate((rw_t, rw_t[-1:]))

    rvx_t = (rx_t[1:]-rx_t[:-1])/tbin
    rvy_t = (ry_t[1:]-ry_t[:-1])/tbin
    rs_t = np.sqrt(rvx_t**2 + rvy_t**2)
    rs_t = np.concatenate((rs_t, rs_t[-1:]))
    rtime_t = np.arange(resamples)*tbin

    units_used = rc_t.shape[0]
    rcov = (utils.signal.WrapPi(rhd_t, True), rw_t, rs_t, rx_t, ry_t, rtime_t)
    return rcov, units_used, tbin, resamples, rc_t, region_edge



def main():
    parser = init_argparse()
    
    dev = utils.pytorch.get_device(gpu=parser.gpu)
    
    session_id = ['Mouse12-120806', 'Mouse28-140313']
    phase = ['sleep', 'wake']
    
    rcov, units_used, tbin, resamples, rc_t, _ = get_dataset(session_id[parser.session_id], 
                                                             phase[parser.phase], parser.binsize)

    # GP with variable regressors model fit
    max_count = int(rc_t.max())

    nonconvex_trials = parser.ncvx
    modes_tot = [('GP', 'IP', 'hd_w_s_pos_t', 64, 'exp', 1, [], False, 10, False, 'ew'), 
                 ('GP', 'hNB', 'hd_w_s_pos_t', 64, 'exp', 1, [], False, 10, False, 'ew'), 
                 ('GP', 'U', 'hd', 8, 'identity', 3, [], False, 10, False, 'ew'),  # 3 
                 ('GP', 'U', 'hd_w_s_t', 48, 'identity', 3, [], False, 10, False, 'ew'), 
                 ('GP', 'U', 'hd_w_s_pos_t', 64, 'identity', 3, [], False, 10, False, 'ew'), # 5
                 ('GP', 'U', 'hd_w_s_pos_t_R1', 72, 'identity', 3, [6], False, 10, False, 'ew'), 
                 ('GP', 'U', 'hd_w_s_pos_t_R2', 80, 'identity', 3, [6], False, 10, False, 'ew'), 
                 ('GP', 'U', 'hd_w_s_pos_t_R3', 88, 'identity', 3, [6], False, 10, False, 'ew'), 
                 ('GP', 'U', 'hd_w_s_pos_t_R4', 96, 'identity', 3, [6], False, 10, False, 'ew'), 
                 ('GP', 'U', 'T1', 8, 'identity', 3, [0], False, 10, False, 'ew'), # 10
                 ('GP', 'IP', 'T1', 8, 'exp', 1, [0], False, 10, False, 'ew'), 
                 ('GP', 'hNB', 'T1', 8, 'exp', 1, [0], False, 10, False, 'ew')]


    modes = [modes_tot[m] for m in parser.modes]
    cv_runs = parser.cv
    for m in modes:
        mtype, ll_mode, r_mode, num_induc, inv_link, C, z_dims, delays, folds, cv_switch, basis_mode = m
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
                        full_model, _ = models.set_model('HDC', max_count, mtype, r_mode, ll_mode, fcov, units_used, tbin, 
                                                         ftrain, num_induc, batch_size=batch_size, 
                                                         inv_link=inv_link, mapping_net=mapping_net, C=C, enc_layers=enc_layers)
                        full_model.to(dev)

                        # fit
                        sch = lambda o: optim.lr_scheduler.MultiplicativeLR(o, lambda e: 0.9)
                        opt_tuple = (optim.Adam, 100, sch)
                        opt_lr_dict = {'default': parser.lr}
                        if r_mode[:2] == 'T1':
                            opt_lr_dict['mapping.kernel.kern1._lengthscale'] = parser.lr_2
                        for z_dim in z_dims:
                            opt_lr_dict['inputs.lv_std_{}'.format(z_dim)] = parser.lr_2
                            
                        full_model.set_optimizers(opt_tuple, opt_lr_dict)#, nat_grad=('rate_model.0.u_loc', 'rate_model.0.u_scale_tril'))

                        annealing = lambda x: 1.0
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
                    name = 'HDC{}'.format(parser.binsize)
                    if shared_W:
                        name += 'S'
                    if basis_mode != 'ew':
                        name += basis_mode
                        
                    model_name = '{}{}{}_{}_{}_{}_C={}_{}'.format(name, parser.session_id, parser.phase, 
                                                                  mtype, ll_mode, r_mode, C, kcv)
                    if cv_switch:
                        model_name += '_'
                    torch.save({'full_model': full_model.state_dict()}, './checkpoint/' + model_name)



if __name__ == "__main__":
    main()