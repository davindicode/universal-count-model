import numpy as np

import scipy.special as sps
import scipy.stats as scstats
from scipy import signal

import torch
import torch.nn as nn



# histograms
def smooth_hist(hist_values, sm_filter, bound):
    r"""
    Neurons is the batch dimension, parallelize the convolution for 1D, 2D or 3D
    bound indicates the padding mode ('periodic', 'repeat', 'zeros')
    sm_filter should had odd sizes for its shape

    :param np.ndarray hist_values: input histogram array of shape (units, ndim_1, ndim_2, ...)
    :param np.ndarray sm_filter: input filter array of shape (ndim_1, ndim_2, ...)
    :param list bound: list of strings (per dimension) to indicate convolution boundary conditions
    :returns:
        array of smoothened histograms
    """
    for s in sm_filter.shape:
        assert s % 2 == 1  # odd shape sizes
    units = hist_values.shape[0]
    dim = len(hist_values.shape) - 1
    assert dim > 0
    step_sm = np.array(sm_filter.shape) // 2

    for d in range(dim):
        if d > 0:
            hist_values = np.swapaxes(hist_values, 1, d + 1)
        if bound[d] == "repeat":
            hist_values = np.concatenate(
                (
                    np.repeat(hist_values[:, :1, ...], step_sm[d], axis=1),
                    hist_values,
                    np.repeat(hist_values[:, -1:, ...], step_sm[d], axis=1),
                ),
                axis=1,
            )
        elif bound[d] == "periodic":
            hist_values = np.concatenate(
                (
                    hist_values[:, -step_sm[d] :, ...],
                    hist_values,
                    hist_values[:, : step_sm[d], ...],
                ),
                axis=1,
            )
        elif bound[d] == "zeros":
            zz = np.zeros_like(hist_values[:, : step_sm[d], ...])
            hist_values = np.concatenate((zz, hist_values, zz), axis=1)
        else:
            raise NotImplementedError

        if d > 0:
            hist_values = np.swapaxes(hist_values, 1, d + 1)

    hist_smth = []
    for u in range(units):
        hist_smth.append(signal.convolve(hist_values[u], sm_filter, mode="valid"))
    hist_smth = np.array(hist_smth)

    return hist_smth


def KDE_behaviour(bins_tuple, covariates, sm_size, L, smooth_modes):
    """
    Kernel density estimation of the covariates, with Gaussian kernels.
    """
    dim = len(bins_tuple)
    assert (
        (dim == len(covariates))
        and (dim == len(sm_size))
        and (dim == len(smooth_modes))
    )
    time_samples = covariates[0].shape[0]
    c_bins = ()
    tg_c = ()
    for k, cov in enumerate(covariates):
        c_bins += (len(bins_tuple[k]) - 1,)
        tg_c += (np.digitize(cov, bins_tuple[k]) - 1,)

    # get time spent in each bin
    bin_time = np.zeros(tuple(len(bins) - 1 for bins in bins_tuple))
    np.add.at(bin_time, tg_c, 1)
    bin_time /= bin_time.sum()  # normalize

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
            sm_filter[k, ...] *= (
                np.exp(-0.5 * (((k - centre) / L_) ** 2)) * ones_arr[k, ...]
            )

        if d > 0:
            bin_time = np.swapaxes(bin_time, 0, d)
            ones_arr = np.swapaxes(ones_arr, 0, d)
            sm_filter = np.swapaxes(sm_filter, 0, d)

    smth_time = smooth_hist(bin_time[None, ...], sm_filter, smooth_modes)
    smth_time /= smth_time.sum()  # normalize
    return smth_time[0, ...], bin_time


# percentiles
def percentiles_from_samples(
    samples, percentiles=[0.05, 0.5, 0.95]
):
    """
    Compute quantile intervals from samples

    :param torch.Tensor samples: input samples of shape (MC, ...)
    :param list percentiles: list of percentile values to look at
    :returns:
        list of tensors representing percentile boundaries
    """
    num_samples = samples.size(0)
    samples = samples.sort(dim=0)[0]  # sort for percentiles
    
    percentile_samples = [
        samples[int(num_samples * percentile)] for percentile in percentiles
    ]
    return percentile_samples


# count distributions
def poiss_count_prob(counts, rate, sim_time):
    """
    Evaluate count data against the Poisson process count distribution.

    :param np.ndarray n: count array (K,)
    :param np.ndarray rate: rate array
    :param scalar sim_time: time bin size
    """
    g = (rate * sim_time)[..., None]  # (..., 1)
    log_g = np.log(np.maximum(g, 1e-12))

    return np.exp(counts * log_g - g - sps.gammaln(counts + 1))


def zip_count_prob(counts, rate, alpha, sim_time):
    """
    Evaluate count data against the Poisson process count distribution:

    :param np.ndarray counts: count array (K,)
    :param np.ndarray rate: rate array
    :param np.ndarray alpha: zero inflation probability array
    :param scalar sim_time: time bin size
    """
    g = (rate * sim_time)[..., None]  # (..., 1)
    log_g = np.log(np.maximum(g, 1e-12))
    alpha = alpha[..., None]

    zero_mask = counts == 0
    p_ = (1.0 - alpha) * np.exp(counts * log_g - g - sps.gammaln(counts + 1))
    return zero_mask * (alpha + p_) + (1.0 - zero_mask) * p_


def nb_count_prob(counts, rate, r_inv, sim_time):
    """
    Negative binomial count probability. The mean is given by :math:`r \cdot \Delta t` like in
    the Poisson case.

    :param np.ndarray counts: count array (K,)
    :param np.ndarray rate: rate array
    :param np.ndarray r_inv: 1/r array
    :param scalar sim_time: time bin size
    """
    g = (rate * sim_time)[..., None]  # (..., 1)
    log_g = np.log(np.maximum(g, 1e-12))
    r_inv = r_inv[..., None]

    asymptotic_mask = r_inv < 1e-4
    r = 1.0 / (r_inv + asymptotic_mask)

    base_terms = log_g * counts - sps.gammaln(counts + 1)
    log_terms = (
        sps.loggamma(r + counts)
        - sps.loggamma(r)
        - (counts + r) * np.log(g + r)
        + r * np.log(r)
    )
    ll_r = base_terms + log_terms
    ll_r_inv = (
        base_terms
        - g
        - np.log(
            1.0 + asymptotic_mask * r_inv * (counts**2 + 1.0 - counts * (3 / 2 + g))
        )
    )

    ll = asymptotic_mask * ll_r_inv + (1 - asymptotic_mask) * ll_r
    return np.exp(ll)


def cmp_count_prob(counts, rate, nu, sim_time, J=100):
    """
    Conway-Maxwell-Poisson count distribution. The partition function is evaluated using logsumexp
    inspired methodology to avoid floating point overflows.

    :param np.ndarray counts: count array (K,)
    :param np.ndarray rate: rate array
    :param np.ndarray nu: dispersion parameter array
    :param scalar sim_time: time bin size
    """
    g = (rate * sim_time)[..., None]  # (..., 1)
    log_g = np.log(np.maximum(g, 1e-12))
    nu = nu[..., None]

    j = np.arange(J + 1)
    lnum = log_g * j
    lden = sps.gammaln(j + 1) * nu
    logsumexp_Z = sps.logsumexp(lnum - lden, axis=-1)[..., None]
    return np.exp(
        log_g * counts - logsumexp_Z - sps.gammaln(counts + 1) * nu
    )  # (..., K)


def cmp_moments(k, rate, nu, sim_time, J=100):
    """
    :param np.ndarray k: order of moment to compute
    :param np.ndarray rate: input rate of shape (neurons, timesteps)
    """
    g = rate[None, ...] * sim_time
    log_g = np.log(np.maximum(g, 1e-12))
    nu = nu[None, ...]
    k = np.array([k])[:, None, None]  # turn into array

    n = np.arange(1, J + 1)[:, None, None]
    j = np.arange(J + 1)[:, None, None]
    lnum = log_g * j
    lden = sps.gammaln(j + 1) * nu
    logsumexp_Z = sps.logsumexp(lnum - lden, axis=-1)[None, ...]
    return np.exp(
        k * np.log(n) + log_g * n - logsumexp_Z - sps.gammaln(n + 1) * nu
    ).sum(0)


# KS and dispersion statistics
def counts_to_quantiles(P_count, counts, rng):
    """
    :param np.ndarray P_count: count distribution values (ts, counts)
    :param np.ndarray counts: count values (ts,)
    """
    counts = counts.astype(int)
    deq_noise = rng.uniform(size=counts.shape)

    cumP = np.cumsum(P_count, axis=-1)  # T, K
    tt = np.arange(counts.shape[0])
    quantiles = cumP[tt, counts] - P_count[tt, counts] * deq_noise
    return quantiles


def quantile_Z_mapping(x, inverse=False, LIM=1e-15):
    """
    Transform to Z-scores from quantiles in forward mapping.
    """
    if inverse:
        Z = x
        q = scstats.norm.cdf(Z)
        return q
    else:
        q = x
        _q = 1.0 - q
        _q[_q < LIM] = LIM
        _q[_q > 1.0 - LIM] = 1.0 - LIM
        Z = scstats.norm.isf(_q)
        return Z


def KS_sampling_dist(x, samples, K=100000):
    """
    Sampling distribution for the Brownian bridge supremum (KS test sampling distribution).
    """
    k = np.arange(1, K + 1)[None, :]
    return (
        8
        * x
        * (
            (-1) ** (k - 1) * k**2 * np.exp(-2 * k**2 * x[:, None] ** 2 * samples)
        ).sum(-1)
        * samples
    )


def KS_DS_statistics(quantiles, alpha=0.05):
    """
    Kolmogorov-Smirnov and dispersion statistics using quantiles.
    """
    samples = quantiles.shape[0]
    assert samples > 1

    q_order = np.append(np.array([0]), np.sort(quantiles))
    ks_y = np.arange(samples + 1) / samples - q_order

    # dispersion scores
    T_KS = np.abs(ks_y).max()

    z = quantile_Z_mapping(quantiles)
    T_DS = np.log((z**2).mean()) + 1 / samples + 1 / 3 / samples**2
    T_DS_ = T_DS / np.sqrt(2 / (samples - 1))  # unit normal null distribution

    sign_KS = np.sqrt(-0.5 * np.log(alpha)) / np.sqrt(samples)
    sign_DS = sps.erfinv(1 - alpha / 2.0) * np.sqrt(2) * np.sqrt(2 / (samples - 1))

    p_DS = 2.0 * (1 - scstats.norm.cdf(T_DS_))
    p_KS = np.exp(-2 * samples * T_KS**2)
    return T_DS, T_KS, sign_DS, sign_KS, p_DS, p_KS
