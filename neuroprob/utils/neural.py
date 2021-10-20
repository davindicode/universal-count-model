import numpy as np

from .signal import ConsecutiveArrays, TrueIslands, WrapPi

import torch
import torch.nn.functional as F



# spike train tools
def bin_data(bin_size, bin_time, spiketimes, track_samples, behaviour_data=None, average_behav=True, binned=False):
    """
    Bin the spike train into a given bin size.
    
    :param int bin_size: desired binning of original time steps into new bin
    :param float bin_time: time step of each original bin or time point
    :param np.array spiketimes: input spikes in train or index format
    :param int track_samples: number of time steps in the recording
    :param tuple behaviour_data: input behavioural time series
    :param bool average_behav: takes the middle element in bins for behavioural data if False
    :param bool binned: spiketimes is a spike train is True, otherwise it is spike time indices
    """
    tbin = bin_size*bin_time
    resamples = int(np.floor(track_samples/bin_size))
    centre = bin_size // 2
    # leave out data with full bins
    
    rcov_t = ()
    if behaviour_data is not None:
        if isinstance(average_behav, list) is False:
            average_behav = [average_behav for _ in range(len(behaviour_data))]
        
        for k, cov_t in enumerate(behaviour_data):
            if average_behav[k]:
                rcov_t += (cov_t[:resamples*bin_size].reshape(resamples, bin_size).mean(1),)
            else:
                rcov_t += (cov_t[centre:resamples*bin_size:bin_size],)
            
    if binned:
        rc_t = spiketimes[:, :resamples*bin_size].reshape(spiketimes.shape[0], resamples, bin_size).sum(-1)
    else:
        units = len(spiketimes)
        rc_t = np.zeros((units, resamples))
        for u in range(units):
            retimes = np.floor(spiketimes[u]/bin_size).astype(int)
            np.add.at(rc_t[u], retimes[retimes < resamples], 1)
        
    return tbin, resamples, rc_t, rcov_t


def binned_to_indices(spiketrain):
    """
    Converts a binned spike train into spike time indices (with duplicates)
    
    :param np.array spiketrain: the spike train to convert
    :returns: spike indices denoting spike times in units of time bins
    :rtype: np.array
    """
    spike_ind = spiketrain.nonzero()[0]
    bigger = np.where(spiketrain > 1)[0]
    add_on = (spike_ind,)
    for b in bigger:
        add_on += (b*np.ones(int(spiketrain[b])-1, dtype=int),)
    spike_ind = np.concatenate(add_on)
    return np.sort(spike_ind)


def covariates_at_spikes(spiketimes, behaviour_data):
    """
    Returns tuple of covariate arrays at spiketimes for all neurons
    """
    cov_s = tuple([] for n in behaviour_data)
    units = len(spiketimes)
    for u in range(units):
        for k, cov_t in enumerate(behaviour_data):
            cov_s[k].append(cov_t[spiketimes[u]])
            
    return cov_s


def compute_ISI_LV(sample_bin, spiketimes):
    r"""
    Compute the local variation measure and the interspike intervals.
    
    .. math::
            LV = 3 \langle \left( \frac{\Delta_{k-1} - \Delta_{k}}{\Delta_{k-1} + \Delta_{k}} \right)^2 \rangle
    
    References:
    
    [1] `A measure of local variation of inter-spike intervals',
    Shigeru Shinomoto, Keiji Miura Shinsuke Koyama (2005)
    """
    ISI = []
    LV = []
    units = len(spiketimes)
    for u in range(units):
        if len(spiketimes[u]) < 3:
            ISI.append([])
            LV.append([])
            continue
        ISI_ = (spiketimes[u][1:] - spiketimes[u][:-1])*sample_bin
        LV.append( 3 * (((ISI_[:-1] - ISI_[1:]) / (ISI_[:-1] + ISI_[1:]))**2).mean() )
        ISI.append(ISI_)
        
    return ISI, LV
        

def spiketrain_CV(folds, spiketrain, track_samples, behaviour_list=None):
    """
    Creates subsets of the the neural data (behaviour plus spike trains) and splits it 
    into folds cross-validation sets with validation data and test data as one out of the 
    folds subsets, validation data being the rest. Each cross-validation set will take 
    a different folds subset for validation data.
    
    :param int folds: number of cross-validation segments to split into
    :param np.array spiketrain: spike train of shape (neuron, timestep)
    :param int track_samples: time steps of recording
    :param list behaviour_list: list of covariate time series
    :returns: cross-validation set as tuple
    :rtype: tuple
    """
    behav = np.array(behaviour_list) # dims, timesteps
    df = track_samples // folds
    cv_set = []
    valset_start = []
    
    blocks_t = []
    blocks_c = []
    
    for f in range(folds):
        valset_start.append(df*f)
        blocks_t.append(spiketrain[:, df*f:df*(f+1)])
        blocks_c.append(behav[:, df*f:df*(f+1)])

    for f in range(folds):
        fb = np.delete(np.arange(folds), f)
        
        ftrain = np.concatenate(tuple(blocks_t[f_] for f_ in fb), axis=1)
        vtrain = blocks_t[f]
        
        if behaviour_list is not None:
            fcov_t = list(np.concatenate(tuple(blocks_c[f_] for f_ in fb), axis=1))
            vcov_t = list(blocks_c[f])
        else:
            fcov_t = None
            vcov_t = None
                    
        cv_set.append((ftrain, fcov_t, vtrain, vcov_t))

    return cv_set, valset_start



def batch_segments(segment_lengths, batch_size):
    """
    Returns list of batch size and batch links for input to the model batching argument.
    
    :param list segment_lengths: list of time step lengths of continuous segments in the data
    """
    batch_list = []
    for s in segment_lengths:
        if s == 0: # empty segment
            continue
            
        n = int(np.ceil(s / batch_size))-1
        for n_ in range(n):
            batch_list.append((batch_size, True if n_ > 0 else False))
        batch_list.append((s - n*batch_size, True if n > 0 else False))
        
    return batch_list



# spiketrain to rate processing
def spike_threshold(sample_bin, bin_thres, covariates, cov_bins, spiketimes, direct=False):
    """
    Only include spikes that correspond to bins with sufficient occupancy time. This is useful 
    when using histogram models to avoid counting undersampled bins, which suffer from huge 
    variance when computing the average firing rates in a histogram.
    
    :param float sample_bin: binning time
    :param float bin_thres: only count bins with total occupancy time above bin_thres
    :param list covariates: list of time series that describe animal behaviour
    :param tuple cov_bins: tuple of arrays describing bin boundary locations
    :param list spiketimes: list of arrays containing spike times of neurons
    """
    units = len(spiketimes)
    c_bins = ()
    tg_c = ()
    sg_c = [() for u in range(units)]
    for k, cov in enumerate(covariates):
        c_bins += (len(cov_bins[k])-1,)
        tg_c += (np.digitize(cov, cov_bins[k])-1,)
        for u in range(units):
            sg_c[u] += (np.digitize(cov[spiketimes[u]], cov_bins[k])-1,)

    # get time spent in each bin
    bin_time = np.zeros(tuple(len(bins)-1 for bins in cov_bins))
    np.add.at(bin_time, tg_c, sample_bin)

    # get activity of each bin per neuron
    activity = np.zeros((units,)+tuple(len(bins)-1 for bins in cov_bins))
    a = np.where(bin_time <= bin_thres) # delete spikes in thresholded bins

    # get flattened removal indices
    remove_glob_index = a[0]
    fac = 1
    for k, cc in enumerate(c_bins[:-1]):
        fac *= cc
        remove_glob_index += a[k+1]*fac
        
    # get flattened spike indices
    thres_spiketimes = []
    for u in range(units):
        s_glob_index = sg_c[u][0]
        fac = 1
        for k, cc in enumerate(c_bins[:-1]):
            fac *= cc
            s_glob_index += sg_c[u][k+1]*fac
            
        rem_ind = np.array([], dtype=np.int64)
        for rg in remove_glob_index:
            rem_ind = np.concatenate((rem_ind, np.where(s_glob_index == rg)[0]))
            
        t_spike = np.delete(spiketimes[u], rem_ind)
        thres_spiketimes.append(t_spike)
            
    return thres_spiketimes



def IPP_model(sample_bin, bin_thres, covariates, cov_bins, spiketimes, divide=True):
    """
    Compute the occupancy-normalized activity histogram for neural data, corresponding to maximum 
    likelihood estimation of the rate in an inhomogeneous Poisson process.
    
    :param float sample_bin: binning time
    :param float bin_thres: only count bins with total occupancy time above bin_thres
    :param list covariates: list of time series that describe animal behaviour
    :param tuple cov_bins: tuple of arrays describing bin boundary locations
    :param list spiketimes: list of arrays containing spike times of neurons
    :param bool divide: return the histogram divisions (rate and histogram probability in region) 
                        or the activity and time histograms
    """
    units = len(spiketimes)
    c_bins = ()
    tg_c = ()
    sg_c = [() for u in range(units)]
    for k, cov in enumerate(covariates):
        c_bins += (len(cov_bins[k])-1,)
        tg_c += (np.digitize(cov, cov_bins[k])-1,)
        for u in range(units):
            sg_c[u] += (np.digitize(cov[spiketimes[u]], cov_bins[k])-1,)

    # get time spent in each bin
    bin_time = np.zeros(tuple(len(bins)-1 for bins in cov_bins))
    np.add.at(bin_time, tg_c, sample_bin)

    # get activity of each bin per neuron
    activity = np.zeros((units,)+tuple(len(bins)-1 for bins in cov_bins))
    for u in range(units):
        np.add.at(activity[u], sg_c[u], 1)

    if divide:
        sess_rate = np.zeros((units,)+tuple(len(bins)-1 for bins in cov_bins))
        bin_time[bin_time <= bin_thres] = -1.0 # avoid division by zero
        for u in range(units):
            activity[u][bin_time < bin_thres] = 0.0
            sess_rate[u] = activity[u]/bin_time
        bin_time[bin_time < bin_thres] = 0.0 # take care of unvisited/uncounted bins
        prob = bin_time / bin_time.sum()

        return sess_rate, prob
    else:
        return activity, bin_time




# mutual information
def spike_var_MI(rate, prob):
    """
    Mutual information analysis for inhomogeneous Poisson process rate variable.
    
    .. math::
            I(x;\text{spike}) = \int p(x) \, \lambda(x) \, \log{\frac{\lambda(x)}{\langle \lambda \rangle}} \, \mathrm{d}x,
            
    :param np.array rate: rate variables of shape (neurons, covariate_dims...)
    :param np.array prob: occupancy values for each bin of shape (covariate_dims...)
    """
    units = rate.shape[0]
    
    MI = np.empty(units)
    logterm = rate/((rate*prob[np.newaxis, :]).sum(axis=tuple(k for k in range(1, len(rate.shape))), keepdims=True) + 1e-12)
    logterm[logterm == 0] = 1.0 # goes to zero in log terms
    for u in range(units): # MI in bits
        MI[u] = (prob * rate[u] * np.log2(logterm[u])).sum()

    return MI




def var_var_MI(sample_bin, v1_t, v2_t, v1_bin, v2_bin):
    """
    MI analysis between covariates
    """

    # Dirichlet prior on the empirical variable probabilites TODO? Bayesian point estimate

    # binning
    track_samples = len(v1_t)
    v1_bins = len(v1_bin)-1
    v2_bins = len(v2_bin)-1
    bv_1 = np.digitize(v1_t, v1_bin)-1
    bv_2 = np.digitize(v2_t, v2_bin)-1

    # get empirical probability distributions
    p_12 = np.zeros((v1_bins, v2_bins))
    np.add.at(p_12, (bv_1, bv_2), 1./track_samples)

    logterm = p_12/(p_12.sum(0, keepdims=True)*p_12.sum(1, keepdims=True) + 1e-12)
    logterm[logterm == 0] = 1.0
    return (p_12 * np.log2(logterm)).sum()




# histograms
def smooth_hist(rate_binned, sm_filter, bound, dev='cpu'):
    r"""
    Neurons is the batch dimension, parallelize the convolution for 1D, 2D or 3D
    bound indicates the padding mode ('periodic', 'repeat', 'zeros')
    sm_filter should had odd sizes for its shape
    
    :param np.array rate_binned: input histogram array of shape (units, ndim_1, ndim_2, ...)
    :param np.array sm_filter: input filter array of shape (ndim_1, ndim_2, ...)
    :param list bound: list of strings (per dimension) to indicate convolution boundary conditions
    :param string dev: device to perform convolutions on
    :returns: smoothened histograms
    :rtype: np.array
    """
    for s in sm_filter.shape:
        assert s % 2 == 1 # odd shape sizes
    units = rate_binned.shape[0]
    dim = len(rate_binned.shape)-1
    assert dim > 0
    step_sm = np.array(sm_filter.shape) // 2

    for d in range(dim):
        if d > 0:
            rate_binned = np.swapaxes(rate_binned, 1, d+1)
        if bound[d] == 'repeat':
            rate_binned = np.concatenate((np.repeat(rate_binned[:, :1, ...], step_sm[d], axis=1), \
                                          rate_binned, \
                                          np.repeat(rate_binned[:, -1:, ...], step_sm[d], axis=1)), axis=1)
        elif bound[d] == 'periodic':
            rate_binned = np.concatenate((rate_binned[:, -step_sm[d]:, ...], rate_binned, rate_binned[:, :step_sm[d], ...]), axis=1)
        elif bound[d] == 'zeros':
            zz = np.zeros_like(rate_binned[:, :step_sm[d], ...])
            rate_binned = np.concatenate((zz, rate_binned, zz), axis=1)
        else:
            raise NotImplementedError

        if d > 0:
            rate_binned = np.swapaxes(rate_binned, 1, d+1)

    if dim <= 3:
        rate_binned = torch.tensor(rate_binned, device=dev).unsqueeze(1)
        sm_filter = torch.tensor(sm_filter, device=dev).unsqueeze(0).unsqueeze(0)

    if dim == 1:
        smth_rate = F.conv1d(rate_binned, sm_filter).squeeze(1).cpu().numpy()
    elif dim == 2:
        smth_rate = F.conv2d(rate_binned, sm_filter).squeeze(1).cpu().numpy()
    elif dim == 3:
        smth_rate = F.conv3d(rate_binned, sm_filter).squeeze(1).cpu().numpy()
    else: # above 3
        smth_rate = []
        for u in range(units):
            smth_rate.append(signal.convolve(rate_binned[u], sm_filter, mode='valid'))
        smth_rate = np.array(smth_rate)

    return smth_rate



def KDE_behaviour(bins_tuple, covariates, sm_size, L, smooth_modes, dev='cpu'):
    """
    Kernel density estimation of the covariates, with Gaussian kernels.
    """
    dim = len(bins_tuple)
    assert (dim == len(covariates)) and (dim == len(sm_size)) and (dim == len(smooth_modes))
    time_samples = covariates[0].shape[0]
    c_bins = ()
    tg_c = ()
    for k, cov in enumerate(covariates):
        c_bins += (len(bins_tuple[k])-1,)
        tg_c += (np.digitize(cov, bins_tuple[k])-1,)

    # get time spent in each bin
    bin_time = np.zeros(tuple(len(bins)-1 for bins in bins_tuple))
    np.add.at(bin_time, tg_c, 1)
    bin_time /= bin_time.sum() # normalize

    sm_centre = np.array(sm_size) // 2
    sm_filter = np.ones(tuple(sm_size))
    ones_arr = np.ones_like(sm_filter)
    for d in range(dim):
        size = sm_size[d]
        centre = sm_centre[d]
        L_ = L[d]
        
        if d > 0:
            bin_time = np.swapaxes(bin_time, 0, d)
            ones_arr = np.swapaxes(ones_arr, 0, d)
            sm_filter = np.swapaxes(sm_filter, 0, d)
        
        for k in range(size):
                sm_filter[k, ...] *= np.exp(-0.5*(((k-centre)/L_)**2))*ones_arr[k, ...]
 
        if d > 0:
            bin_time = np.swapaxes(bin_time, 0, d)
            ones_arr = np.swapaxes(ones_arr, 0, d)
            sm_filter = np.swapaxes(sm_filter, 0, d)

    smth_time = smooth_hist(bin_time[None, ...], sm_filter, smooth_modes, dev=dev)
    smth_time /= smth_time.sum() # normalize
    return smth_time[0, ...], bin_time



# histogram tuning curves
def geometric_tuning(ori_rate, smth_rate, prob):
    r"""
    Compute coherence and sparsity related to the geometric properties of tuning curves.
    """
    # Pearson r correlation
    units = ori_rate.shape[0]
    coherence = np.empty(units)
    for u in range(units):
        x_1 = ori_rate[u].flatten()
        x_2 = smth_rate[u].flatten()
        coherence[u] = np.dot(x_1 - x_1.mean(), x_2 - x_2.mean()) / len(x_1) / x_1.std() / x_2.std()

    # Computes the sparsity of the tuning
    sparsity = np.empty(units)
    for u in range(units):
        sparsity[u] = 1-(prob * ori_rate[u]).sum()**2 / (prob * ori_rate[u]**2).sum()
        
    return coherence, sparsity
        


def tuning_overlap(tuning):
    r"""
    Compute the overlap of tuning curves using the inner product of normalized firing maps [1].
    
    .. math::
            [f, u] &\sim \mathcal{GP}(0, k([X, X_u], [X, X_u])),\\
            y & \sim p(y) = p(y \mid f) p(f),
    
    References:
    
    [1] `Organization of cell assemblies in the hippocampus` (supplementary),
    Kenneth D. Harris, Jozsef Csicsvari*, Hajime Hirase, George Dragoi & Gyorgy Buzsaki
    
    """
    g = len(tuning.shape)-1
    tuning_normalized = tuning.mean(axis=tuple(1+k for k in range(g)), keepdims=True)
    overlap = np.einsum('i...,j...->ij', tuning_normalized, tuning_normalized)
    return overlap
    



def spike_correlogram(spiketrain, lag_step, lag_points, segment_len, start_step=0, ref_point=0, cross=True, correlation=False, dev='cpu'):
    """
    Get the temporal correlogram of spikes in a given population. Computes 
    
    .. math::
            C_{ij}(\tau) = \langle S_i(t) S_j(t + \tau) \rangle
            
    or if correlation flag is True, it computes 
    
    .. math::
            C_{ij}(\tau) = \text{Corr}[ S_i(t), S_j(t + \tau) ]
    
    :param np.array spiketrain: array of population activity of shape (neurons, time)
    :param int lag_range: 
    :param int N_period: 
    :param 
    :param list start_points: list of integers of time stamps where to start computing the correlograms
    :param bool cross: compute the full cross-correlogram over the population, otherwise compute only auto-correlograms
    """
    units = spiketrain.shape[0]
    spikes = torch.tensor(spiketrain, device=dev).float()
    spikes_unfold = spikes[:, start_step:start_step+(lag_points-1)*lag_step+segment_len].unfold(-1, segment_len, lag_step) # n, t, f
    
    if cross:
        cg = []
        for u in range(units):
            a = spikes_unfold[u:, ref_point:ref_point+1, :]
            b = spikes_unfold[u:u+1, ...]
            if correlation:
                a_m = a.mean(-1, keepdims=True)
                b_m = b.mean(-1, keepdims=True)
                a_std = a.std(-1)
                b_std = b.std(-1)
                cg.append(((a-a_m)*(b-b_m)).mean(-1)/(a_std*b_std))
            else:
                cg.append((a*b).mean(-1)) # neurons-u, lags
                
        cg = torch.cat(cg, dim=0)
    else:
        a = spikes_unfold[:, ref_point:ref_point+1, :]
        b = spikes_unfold
        if correlation:
            a_m = a.mean(-1, keepdims=True)
            b_m = b.mean(-1, keepdims=True)
            a_std = a.std(-1)
            b_std = b.std(-1)
            cg = ((a-a_m)*(b-b_m)).mean(-1)/(a_std*b_std)
        else:
            cg = (a*b).mean(-1) # neurons, lags

    return cg.cpu().numpy()

