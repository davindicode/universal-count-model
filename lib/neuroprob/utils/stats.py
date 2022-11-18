import numpy as np

import scipy.special as sps
import scipy.stats as scstats

import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.parameter import Parameter


# count statistics
def gamma_count_prob(n, rate, shape, sim_time):
    """
    Evaluate count data against the Gamma renewal process count distribution:

    .. math:: p(n) = .

    :param numpy.array alpha: Probabilities of the discrete target distribution. The
        shape of the array dimensions is (samples, index) with index size :math:`n`.
    :returns: Samples from the Gumbel-softmax indexed from :math:`0` to :math:`n-1`
    :rtype: numpy.array
    """
    g = rate * sim_time
    if g == 0:  # will get Gamma(0)*0 which is ~0/0 so manually set to 1
        return (n == 0).astype(float)

    return sps.gammainc(shape * n, g) - sps.gammainc(shape * (n + 1), g)


def poiss_count_prob(n, rate, dummy, sim_time):
    """
    Evaluate count data against the Poisson process count distribution:

    .. math:: p(n) = .

    :param numpy.array alpha: Probabilities of the discrete target distribution. The
        shape of the array dimensions is (samples, index) with index size :math:`n`.
    :returns: Samples from the Gumbel-softmax indexed from :math:`0` to :math:`n-1`
    :rtype: numpy.array
    """
    g = rate * sim_time
    if g == 0:
        return (n == 0).astype(float)

    # if g == 0:
    #    return float(n == 0)
    # if n > 10: # Stirling asymptotic series
    #    l_f = np.log(g) + 1 - np.log(n) # log e*g/n
    #    if l_f < 1e-20/n: # super small
    #        return 0.0
    #    return np.exp(n*l_f - g) / np.sqrt(2*np.pi*n)
    # else:

    return np.exp(
        n * np.log(g) - g - sps.gammaln(n + 1)
    )  # / sps.factorial(n) # scipy.special.factorial computes array! efficiently


def zip_count_prob(n, rate, alpha, sim_time):
    """
    Evaluate count data against the Poisson process count distribution:

    .. math:: p(n) = .

    :param numpy.array alpha: Probabilities of the discrete target distribution. The
        shape of the array dimensions is (samples, index) with index size :math:`n`.
    :returns: Samples from the Gumbel-softmax indexed from :math:`0` to :math:`n-1`
    :rtype: numpy.array
    """
    g = rate * sim_time
    if g == 0:
        return (n == 0).astype(float)

    zero_mask = n == 0
    p_ = (1.0 - alpha) * np.exp(
        n * np.log(g) - g - sps.gammaln(n + 1)
    )  # / sps.factorial(n))
    return zero_mask * (alpha + p_) + (1.0 - zero_mask) * p_


def nb_count_prob(n, rate, r_inv, sim_time):
    """
    Negative binomial count probability. The mean is given by :math:`r \cdot \Delta t` like in
    the Poisson case.
    """
    g = rate * sim_time
    if g == 0:
        return (n == 0).astype(float)

    asymptotic_mask = r_inv < 1e-3
    r = 1.0 / (r_inv + asymptotic_mask)

    base_terms = np.log(g) * n - sps.gammaln(n + 1)
    log_terms = (
        sps.loggamma(r + n) - sps.loggamma(r) - (n + r) * np.log(g + r) + r * np.log(r)
    )
    ll_r = base_terms + log_terms
    ll_r_inv = base_terms - g - np.log(1.0 + r_inv * (n**2 + 1.0 - n * (3 / 2 + g)))

    ll = ll_r * (1.0 - asymptotic_mask) + ll_r_inv * asymptotic_mask
    return np.exp(ll)


def cmp_count_prob(n, rate, nu, sim_time, J=100):
    """
    Conway-Maxwell-Poisson count distribution. The partition function is evaluated using logsumexp
    inspired methodology to avoid floating point overflows.
    """
    g = rate * sim_time
    if g == 0:
        return (n == 0).astype(float)

    j = np.arange(J + 1)
    lnum = np.log(g) * j
    lden = np.log(sps.factorial(j)) * nu
    dl = lnum - lden
    dl_m = dl.max()
    logsumexp_Z = np.log(np.exp(dl - dl_m).sum()) + dl_m  # numerically stable
    return np.exp(np.log(g) * n - logsumexp_Z - sps.gammaln(n + 1) * nu)


def modulated_count_dist(count_dist, mult_gain, add_gain, samples):
    """
    Generate distribution samples from the modulated count process with additive and multiplicate gains.

    .. math::
            P_{mod}(n|\lambda) = \int P_{count}(n|G_{mul}\lambda + G_{add}) P(G_{mu]}, G_{add}) \mathrm{d}G_{mul} \mathrm{d}G_{add},

        where :math:`p(y \mid f)` is the likelihood.

    References:

    [1] `Dethroning the Fano Factor: A Flexible, Model-Based Approach to Partitioning Neural Variability`,
         Adam S. Charles, Mijung Park, J. PatrickWeller, Gregory D. Horwitz, Jonathan W. Pillow (2018)

    """

    raise NotImplementedError


def cdf_count_deq(n, deq, count_prob, N_MAX=100):
    """
    Compute the dequantized cdf count.

    :param int n: integer spike count
    :param float deq: dequantization noise
    :param LambdaFunction count_prob:
    """
    N = np.arange(n + 1)
    C = count_prob(N)
    cdf = C[:n].sum() + deq * C[n]
    return cdf


def cmp_moments(k, rate, nu, sim_time, J=100):
    """
    :param np.array rate: input rate of shape (neurons, timesteps)
    """
    g = rate[None, ...] * sim_time
    nu = nu[None, ...]
    k = np.array([k])[:, None, None]  # turn into array

    n = np.arange(1, J + 1)[:, None, None]
    j = np.arange(J + 1)[:, None, None]
    lnum = np.log(g) * j
    lden = np.log(sps.factorial(j)) * nu
    dl = lnum - lden
    dl_m = dl.max(axis=0)
    logsumexp_Z = (np.log(np.exp(dl - dl_m[None, ...]).sum(0)) + dl_m)[
        None, ...
    ]  # numerically stable
    return np.exp(
        k * np.log(n) + np.log(g) * n - logsumexp_Z - sps.gammaln(n + 1) * nu
    ).sum(0)


def count_KS_method(
    count_dist, mean_func, sample_bin, spike_binned, rate, shape=None, deq_noise=None
):
    """
    Overdispersion analysis using rate-rescaled distributions and Kolmogorov-Smirnov
    statistics. Trajectory lengths are given, unless the expected count is below
    min_spikes (avoid dequantization effect)

    :param LambdaType count_dist: lambda function with input (count, rate, time) and output p
    :param LambdaType mean_func: lambda function with input (avg_rate*T) and output avg_cnt
    :param float sample_bin: length of time bin
    :param numpy.array spike_ind: indices of the spike times
    :param numpy.array rate: rate as sampled at sample_bin
    :param float alpha: significance level of the statistical test
    :returns: quantiles, q_order, ks_y, T_DS, T_KS, sign_DS, sign_KS
    :rtype: tuple
    """
    if shape is not None:
        assert rate.shape[0] == shape.shape[0]
    traj_lens = rate.shape[0]
    q_cdf = []

    if deq_noise is None:
        deq_noise = np.random.uniform(size=spike_binned.shape)

    for tt in range(traj_lens):
        f_p = lambda c: count_dist(
            c, rate[tt], shape[tt] if shape is not None else None, sample_bin
        )
        q_cdf.append(cdf_count_deq(int(spike_binned[tt]), deq_noise[tt], f_p))

    quantiles = np.array(q_cdf).flatten()
    return quantiles


# ISI statistics
def discrete_rate_rescale(sample_bin, rate, spike_ind):
    r"""
    Treats the rate as a step-wise constant continuous function, see [1]. We dequantize the
    rescaled times by adding uniform noise between discrete steps in the rescaled times.

    References:

    [1] `Rescaling, thinning or complementing? On goodness-of-fit procedures for
    point process models and Generalized Linear Models`,
    Felipe Gerhard, Wulfram Gerstner

    :param float sample_bin: time of a bin
    :param numpy.array rate: rate sampled at sample_bin
    :param numpy.array spike_ind: indices of the spike times
    :returns: Spike times as rescaled with rate rescaling in units of sample_bin
    :rtype: numpy.array
    """
    rtime = np.cumsum(rate) * sample_bin
    drtime = np.concatenate((rtime[0:1], rtime[1:] - rtime[:-1]))
    rspk = rtime[spike_ind] - np.random.rand(*spike_ind.shape) * drtime[spike_ind]
    rspk = np.sort(rspk)  # re-oder in case of duplicate spike_ind

    return rspk


def ISI_KS_method(interval_dist, sample_bin, spike_ind, rate):
    """
    Overdispersion analysis using time-rescaled ISIs and Kolmogorov-Smirnov statistics

    :param ISI_class interval_dist: the interval dist
    :param float sample_bin:
    :param numpy.array spike_ind: indices of the spike times
    :param numpy.array rate: 1D rate array, sampled at sample_bin
    :param float alpha: significance level of the statistical test (double tailed for DS)
    :returns: quantiles, q_order, ks_y, T_DS, T_KS, sign_DS, sign_KS
    :rtype: tuple
    """
    rtime = discrete_rate_rescale(sample_bin, rate, spike_ind)
    rISI = rtime[1:] - rtime[:-1]

    quantiles = interval_dist.cum_prob(rISI)
    return quantiles


def invISI_KS_method(inv_interval_dist, sample_bin, spike_ind, rate):
    """
    Overdispersion analysis using time-rescaled ISIs.

    :param ISI_class interval_dist: the interval dist
    :param float sample_bin:
    :param numpy.array spike_ind: indices of the spike times
    :param numpy.array rate: 1D rate array, sampled at sample_bin
    :param float alpha: significance level of the statistical test (double tailed for DS)
    :returns: quantiles, q_order, ks_y, T_DS, T_KS, sign_DS, sign_KS
    :rtype: tuple
    """
    rtime = discrete_rate_rescale(sample_bin, rate, spike_ind)
    rISI = rtime[1:] - rtime[:-1]

    quantiles = inv_interval_dist.cum_prob(1 / (rISI + 1e-12))  # stabilized
    return quantiles


def q_to_Z(quantiles, LIM=1e-15):
    """
    Inverse transform to Gaussian variables from uniform variables.
    """
    _q = 1.0 - quantiles
    _q[_q < LIM] = LIM
    _q[_q > 1.0 - LIM] = 1.0 - LIM
    z = scstats.norm.isf(_q)
    return z


def Z_to_q(Z, LIM=1e-15):
    """
    Inverse transform to Gaussian variables from uniform variables.
    """
    q = scstats.norm.cdf(Z)
    return q


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


def KS_statistics(quantiles, alpha=0.05, alpha_s=0.05):
    """
    Kolmogorov-Smirnov statistics using quantiles.
    """
    samples = quantiles.shape[0]
    assert samples > 1

    q_order = np.append(np.array([0]), np.sort(quantiles))
    ks_y = np.arange(samples + 1) / samples - q_order

    # dispersion scores
    T_KS = np.abs(ks_y).max()

    z = q_to_Z(quantiles)
    T_DS = np.log((z**2).mean()) + 1 / samples + 1 / 3 / samples**2
    T_DS_ = T_DS / np.sqrt(2 / (samples - 1))  # unit normal null distribution

    sign_KS = np.sqrt(-0.5 * np.log(alpha)) / np.sqrt(samples)
    sign_DS = sps.erfinv(1 - alpha / 2.0) * np.sqrt(2) * np.sqrt(2 / (samples - 1))
    # ref_sign_DS = sps.erfinv(1-alpha_s/2.)*np.sqrt(2)
    # T_DS /= ref_sign_DS
    # sign_DS /= ref_sign_DS

    p_DS = 2.0 * (1 - scstats.norm.cdf(T_DS_))
    p_KS = np.exp(-2 * samples * T_KS**2)
    return T_DS, T_KS, sign_DS, sign_KS, p_DS, p_KS


# correlations
def corr_lin_lin(x, y):
    r"""
    Linear-linear correlation coefficient.
    
    .. math::
            [f, u] &\sim \mathcal{GP}(0, k([X, X_u], [X, X_u])),\\
            y & \sim p(y) = p(y \mid f) p(f)
    
    
    :param numpy.array x: input array x of shape (samples)
    """
    return ((x - x.mean()) * (y - y.mean())).mean() / (x.std() * y.std())


def corr_lin_circ(x, theta):
    r"""
    Linear-circular correlation coefficient, using embedding approach [1].
    
    .. math::
            [f, u] &\sim \mathcal{GP}(0, k([X, X_u], [X, X_u])),\\
            y & \sim p(y) = p(y \mid f) p(f)
    
    :param numpy.array x: input array x of shape (samples,)
    :param numpy.array theta: input array theta of shape (samples,)
    
    References:
    
    [1] Mardia (1979) and Johnson and Wehrly (1977)
    
    """
    s = np.sin(theta)
    c = np.cos(theta)
    r_xs = corr_lin_lin(x, s)
    r_xc = corr_lin_lin(x, c)
    r_cs = corr_lin_lin(c, s)
    return np.sqrt((r_xs**2 + r_xc**2 - 2 * r_xs * r_xc * r_cs) / (1 - r_cs**2))


def corr_lin_circ_Kempter(x, theta, a):
    """
    Circular-linear correlation coefficient as defined by the following papers:

    :param numpy.array x: input array x of shape (samples,)
    :param numpy.array theta: input array theta of shape (samples,)
    :param float a: slope divided by 2pi

    References:

    [1] Kempter et al. (2012) Note: phi and theta are reversed
    [2] Jammalamadaka and Sengupta (2001)
    """
    theta_bar = np.arctan2(np.sum(np.sin(theta)), np.sum(np.cos(theta)))
    phi = 2 * np.pi * a * x % (2 * np.pi)
    phi_bar = np.arctan2(np.sum(np.sin(phi)), np.sum(np.cos(phi)))
    num = np.sum(np.sin(theta - theta_bar) * np.sin(phi - phi_bar))
    den = np.sqrt(
        np.sum(np.sin(theta - theta_bar) ** 2) * np.sum(np.sin(phi - phi_bar) ** 2)
    )

    return num / den


def corr_circ_circ(x, y):
    r"""
    Circular-circular correlation coefficient [1].
    
    .. math::
            [f, u] &\sim \mathcal{GP}(0, k([X, X_u], [X, X_u])),\\
            y & \sim p(y) = p(y \mid f) p(f)
    
    :param numpy.array x: input array x of shape (samples,)
    :param numpy.array y: input array y of shape (samples,)
    
    References:
    
    [1] `A Correlation Coefficient for Circular Data`, 
    N. I. Fisher and A. J. Lee
    
    """
    x_ = np.angle(np.exp(1j * x).mean())
    y_ = np.angle(np.exp(1j * y).mean())
    s_x = np.sin(x - x_)
    s_y = np.sin(y - y_)
    return (s_x * s_y).mean() / (np.sqrt((s_x**2).mean() * (s_y**2).mean()))
