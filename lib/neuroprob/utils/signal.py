import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.parameter import Parameter


### utilities
def safe_sqrt(x, eps=1e-12):
    """
    A convenient function to avoid the NaN gradient issue of :func:`torch.sqrt`
    at 0.
    """
    # Ref: https://github.com/pytorch/pytorch/issues/2421
    return (x + eps).sqrt()


def linear_regression(A, B):
    """
    linear regression with R squared
    """
    a = ((A * B).mean(-1) - A.mean(-1) * B.mean(-1)) / A.var(-1)
    b = (B.mean(-1) * (A**2).mean(-1) - (A * B).mean(-1) * A.mean(-1)) / A.var(-1)
    return a, b


def basis_function_regression(x, y, phi, lambd=0.0):
    """ """
    ph = phi(x)[None, ...]  # o, d, n
    rhs = (ph * y[:, None, :]).sum(-1)  # o, d
    d = rhs.shape[-1]
    o = rhs.shape[0]
    gram = np.matmul(ph, ph.transpose(0, 2, 1))  # o, d, d
    gram = gram.reshape(o, -1)
    gram[:, :: d + 1] += lambd
    w = np.linalg.solve(gram.reshape(-1, d, d), rhs)  # o, d
    f = (w[..., None] * ph).sum(1)  # o, n
    return w, f


def kernel_ridge_regression(x, y, k_func, lambd=0.0):
    """
    Dual of BFR
    """
    K = k_func(x, x)  # o, n, n

    return f


def lagged_input(
    input, hist_len, hist_stride=1, time_stride=1, tensor_type=torch.float
):
    """
    Introduce lagged history input from time series.

    :param torch.tensor input: input of shape (dimensions, timesteps)
    :param int hist_len:
    :param int hist_stride:
    :param int time_stride:
    :param dtype tensor_type:
    :returns: lagged input tensor of shape (dimensions, time-H+1, history)
    :rtype: torch.tensor
    """
    in_unfold = input.unfold(-1, hist_len, time_stride)[
        :, :, ::hist_stride
    ]  # (dim, time, hist)
    return in_unfold  # (dimensions, timesteps-H+1, H)


def percentiles_from_samples(
    samples, percentiles=[0.05, 0.5, 0.95], smooth_length=5, padding_mode="replicate"
):
    """
    Compute quantile intervals from samples, samples has shape (sample_dim, event_dims..., T).

    :param torch.tensor samples: input samples of shape (MC, event_dims...)
    :param list percentiles: list of percentile values to look at
    :param int smooth_length: time steps over which to smooth with uniform block
    :returns: list of tensors representing percentile boundaries
    :rtype: list
    """
    num_samples = samples.size(0)
    T = samples.size(-1)
    prev_shape = samples.shape[1:]
    if len(samples.shape) == 2:
        samples = samples[:, None, :]
    else:
        samples = samples.view(num_samples, -1, T)

    samples = samples.sort(dim=0)[0]
    percentile_samples = [
        samples[int(num_samples * percentile)] for percentile in percentiles
    ]

    with torch.no_grad():  # Smooth the samples
        Conv1D = nn.Conv1d(
            1,
            1,
            smooth_length,
            padding=smooth_length // 2,
            bias=False,
            padding_mode=padding_mode,
        ).to(samples.device)
        Conv1D.weight.fill_(1.0 / smooth_length)
        percentiles_samples = [
            Conv1D(percentile_sample[:, None, :]).view(prev_shape)
            for percentile_sample in percentile_samples
        ]

    return percentiles_samples


def eye_like(value, m, n=None):
    """
    Create an identity tensor, from Pyro [1].

    References:

    [1] `Pyro: Deep Universal Probabilistic Programming`, E. Bingham et al. (2018)

    """
    if n is None:
        n = m
    eye = torch.zeros(m, n, dtype=value.dtype, device=value.device)
    eye.view(-1)[: min(m, n) * n : n + 1] = 1
    return eye


# PCA
def PCA(x, covariance=False):
    """
    Principal component analysis.

    :param np.array x: input data of shape (dims, time)
    :param bool covariance: perform PCA on data covariance if True, otherwise on
                            data correlation
    :returns:
    """
    if covariance:
        x_std = x - x.mean(1, keepdims=True)
    else:
        x_std = (x - x.mean(1, keepdims=True)) / x.std(1, keepdims=True)

    if x_std.shape[0] < x_std.shape[1]:
        x_std = x_std
    else:
        x_std = x_std.T

    cov = np.cov(x_std)
    ev, eig = np.linalg.eig(cov)
    PCA_proj = eig.dot(x_std)
    return ev, eig, PCA_proj


# circular regression
def resultant_length(x, y):
    """
    Returns the mean resultant length R of the residual distribution
    """
    return torch.sqrt(
        torch.mean(torch.cos(x - y)) ** 2 + torch.mean(torch.sin(x - y)) ** 2
    )


def circ_lin_regression(theta, x, dev="cpu", iters=1000, lr=1e-2):
    """
    Similar to [1].

    :param np.array theta: circular input of shape (timesteps,)
    :param np.array x: linear input of shape (timesteps,)
    :param string dev:
    :param int iters: number of optimization iterations
    :param float lr: learning rate of optimization

    References:

    [1] `Quantifying circular–linear associations: Hippocampal phase precession`,
    Richard Kempter, Christian Leibold, György Buzsáki, Kamran Diba, Robert Schmidt (2012)

    """
    # lowest_loss = np.inf
    # shift = Parameter(torch.zeros(1, device=dev))
    a = Parameter(torch.zeros(1, device=dev))

    # optimizer = optim.Adam([a, shift], lr=lr)
    optimizer = optim.Adam([a], lr=lr)
    XX = torch.tensor(x, device=dev)
    HD = torch.tensor(theta, device=dev)

    losses = []
    for k in range(iters):
        optimizer.zero_grad()
        X_ = 2 * np.pi * XX * a  # + shift
        loss = -1 * resultant_length(X_, HD).mean()  # must maximise R
        # loss = (metric(X_, HD, 'torus')**2).mean()
        loss.backward()
        optimizer.step()
        losses.append(loss.cpu().item())

    a_ = a.cpu().item()
    # shift_ = shift.cpu().item()
    shift_ = np.arctan2(
        np.sum(np.sin(theta - 2 * np.pi * a_ * x)),
        np.sum(np.cos(theta - 2 * np.pi * a_ * x)),
    )

    return 2 * np.pi * x * a_ + shift_, a_, shift_, losses


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



# FFT filter
def filter_signal(signal, f_min, f_max, sample_bin):
    """
    Filter in Fourier space by multiplying with a box function for (f_min, f_max).
    """
    track_samples = signal.shape[0]
    Df = 1 / sample_bin / track_samples
    low_ind = np.floor(f_min / Df).astype(int)
    high_ind = np.ceil(f_max / Df).astype(int)

    g_fft = np.fft.rfft(signal)
    mask = np.zeros_like(g_fft)
    mask[low_ind:high_ind] = 1.0
    g_fft *= mask
    signal_ = np.fft.irfft(g_fft)

    if track_samples % 2 == 1:  # odd
        signal_ = np.concatenate((signal_, signal_[-1:]))

    return signal_


# node and anti-nodes
def find_peaks(signal, min_node=True, max_node=True):
    T = []
    for t in range(1, signal.shape[0] - 1):
        if signal[t - 1] < signal[t] and signal[t + 1] < signal[t] and max_node:
            T.append(t)
        elif signal[t - 1] > signal[t] and signal[t + 1] > signal[t] and min_node:
            T.append(t)

    return T
