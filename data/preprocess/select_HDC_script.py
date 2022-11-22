import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F
import torch.optim as optim

import matplotlib.pyplot as plt
from matplotlib import patches
import numpy as np

import pickle

import os
import sys
sys.path.append("..")

import neuroprob as mdl
import neuroprob.utils as utils



brain_regions = {'ANT': 0, 'PoS': 1, 'CA1': 2, 'mPFC': 3}

mice_sessions = {
    'Mouse12': ['120806', '120807', '120809', '120810' ], # '120808' is missing position files
    'Mouse17': ['130125', '130128', '130129', '130130', '130131', '130201', '130202', '130203', '130204'],
    'Mouse20': ['130514', '130515', '130516', '130517', '130520'],
    'Mouse24': ['131213', '131216', '131217','131218'],
    'Mouse25': ['140123', '140124', '140128', '140129', '140130', '140131', '140203', '140204', '140205', '140206'],
    'Mouse28': ['140310', '140311', '140312', '140313', '140317', '140318'],
} 

phase = 'wake'
datadir = '/scratches/ramanujan_2/dl543/HDC_PartIII'
savedir = '/scratches/ramanujan_2/dl543/HDC_PartIII'

if not os.path.exists(savedir):
    os.makedirs(savedir)
        


for mouse_id in mice_sessions.keys():
    for session_id in mice_sessions[mouse_id]:
        print(mouse_id, session_id)
        
        data_dict = pickle.load(open(datadir+'/th1_{}_{}_{}.p'.format(mouse_id, session_id, phase), 'rb'))

        # resample at 1 ms
        covariates = (
            data_dict['covariates']['x'], 
            data_dict['covariates']['y'], 
            np.unwrap(data_dict['covariates']['hd']), 
        )
        units = data_dict['neural']['units']

        bin_size = 20  # original data at 20k Hz
        tbin, resamples, rc_t, (rx_t, ry_t, rhd_t,) = utils.neural.bin_data(
            bin_size, data_dict['sample_bin'], data_dict['neural']['spike_time_inds'], data_dict['use_sample_num'], 
            covariates, average_behav=True)

        rhd_t = rhd_t % (2*np.pi)
        print('time bin: ', tbin, ' units: ', units)


        # remove invalid data
        inval_behav = data_dict['covariates']['invalid_behaviour']
        print('invalid time intervals: ', inval_behav)
        if len(inval_behav['HD']) > 0:  # assume XY invalids is subset
            if inval_behav['HD'][-1]['index'] + inval_behav['HD'][-1]['length'] == resamples:  # remove invalid region at start
                end_cut = int(np.ceil(inval_behav['HD'][-1]['length'] / bin_size))
                resamples -= end_cut
                rc_t = rc_t[:, :-end_cut]
                rx_t = rx_t[:-end_cut]
                ry_t = ry_t[:-end_cut]
                rhd_t = rhd_t[:-end_cut]
                
            if inval_behav['HD'][0]['index'] == 0:  # remove invalid region at start
                start_ind = int(np.ceil(inval_behav['HD'][0]['length'] / bin_size))
                resamples -= start_ind
                rc_t = rc_t[:, start_ind:]
                rx_t = rx_t[start_ind:]
                ry_t = ry_t[start_ind:]
                rhd_t = rhd_t[start_ind:]
        
        r_t_spike = []
        for u in range(units):
            r_t_spike.append(utils.neural.binned_to_indices(rc_t[u]))

        # binning of covariates and analysis
        bins_hd = 60
        bin_hd = np.linspace(0, 2*np.pi+1e-3, bins_hd+1)
        hd_rate, hd_prob = utils.neural.IPP_model(tbin, 0.0, (rhd_t,), (bin_hd,), r_t_spike, divide=True)
        hd_MI = utils.neural.spike_var_MI(hd_rate, hd_prob)
        filter_win = 41
        centre_win = filter_win//2
        sigma = 6
        sfilter = np.exp(-0.5*(np.arange(filter_win)-centre_win)**2/sigma**2)
        sfilter = sfilter / sfilter.sum()
        sm_tun = utils.neural.smooth_hist(hd_rate, sfilter, ['periodic'], dev='cpu')
        coherence, sparsity = utils.neural.geometric_tuning(hd_rate, sm_tun, hd_prob)
            
        # select cells based on criterion
        criterion = {'refractory': 2.0, 'spatial_info': 0.5, 'sparsity': 0.2}
        hdc_unit = np.zeros(units).astype(bool)
        for u in range(units):
            if ('spatial_info' in criterion and hd_MI[u] < criterion['spatial_info']) or \
               ('coherence' in criterion and coherence[u] < criterion['coherence']) or \
               ('refractory' in criterion and data_dict['neural']['refract_viol'][u] > criterion['refractory']) or \
               ('sparsity' in criterion and sparsity[u] < criterion['sparsity']):
                continue
            hdc_unit[u] = True

        print('num of HD cells:', (hdc_unit).sum())
        
        # put region ID for each unit
        neuron_groups = data_dict['neural']['neuron_groups']
        neuron_regions = np.empty(units)
        for key in neuron_groups.keys():
            neuron_regions[neuron_groups[key]] = brain_regions[key]

        left_x = rx_t.min()
        right_x = rx_t.max()
        bottom_y = ry_t.min()
        top_y = ry_t.max()

        arena = np.array([left_x, right_x, bottom_y, top_y])

        np.savez_compressed(savedir+'/{}_{}_{}'.format(mouse_id, session_id, phase), 
                            spktrain=rc_t, hdc_unit=hdc_unit, neuron_regions=neuron_regions, 
                            arena=arena, x_t=rx_t, y_t=ry_t, hd_t=rhd_t)