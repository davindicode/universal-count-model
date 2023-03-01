import numpy as np

import torch
import torch.optim as optim
from torch.nn.parameter import Parameter

from tqdm.autonotebook import tqdm

import neuroprob as nprb
from neuroprob import utils



def ind_to_pair(ind, N):
    a = ind
    k = 1
    while a >= 0:
        a -= N - k
        k += 1

    n = k - 1
    m = N - n + a
    return n - 1, m


def metric(x, y, topology="euclid"):
    """
    Returns the geodesic displacement between x and y, (x-y).

    :param torch.tensor x: input x of any shape
    :param torch.tensor y: input y of same shape as x
    :returns: x-y tensor of geodesic distances
    :rtype: torch.tensor
    """
    if topology == "euclid":
        xy = x - y
    elif topology == "ring":
        xy = (x - y) % (2 * np.pi)
        xy[xy > np.pi] -= 2 * np.pi
    elif topology == "cosine":
        xy = 2 * (1 - torch.cos(x - y))
    else:
        raise NotImplementedError

    return xy


def resultant_length(x, y):
    """
    Returns the mean resultant length R of the residual distribution
    """
    return torch.sqrt(torch.mean(torch.cos(x-y))**2 + torch.mean(torch.sin(x-y))**2)


def signed_scaled_shift(
    x, x_ref, device="cpu", topology="ring", iters=1000, lr=1e-2, learn_scale=True
):
    """
    Shift trajectory, with scaling, reflection and translation.
    Shift trajectories to be as close as possible to each other, including
    switches in sign.

    :param np.ndarray theta: circular input array of shape (timesteps,)
    :param np.ndarray theta_ref: reference circular input array of shape (timesteps,)
    :param string device: device ('cpu', 'cuda:0' etc.) to perform computations on
    :param int iters: number of iterations of gradient descent
    :param float lr: learning rate
    :returns:
        aligned trajectory (np.ndarray), shift, sign, scale, losses (List)
    """
    XX = torch.tensor(x, device=device)
    XR = torch.tensor(x_ref, device=device)

    lowest_loss = np.inf
    for sign in [1, -1]:  # select sign automatically
        shift = Parameter(torch.zeros(1, device=device))
        p = [shift]

        if learn_scale:
            scale = Parameter(torch.ones(1, device=device))
            p += [scale]
        else:
            scale = torch.ones(1, device=device)

        optimizer = optim.Adam(p, lr=lr)
        losses = []

        iterator = tqdm(range(iters))
        for k in iterator:
            optimizer.zero_grad()
            X_ = XX * sign * scale + shift
            loss = (metric(X_, XR, topology) ** 2).mean()
            loss.backward()
            optimizer.step()

            l_ = loss.cpu().item()
            losses.append(l_)
            iterator.set_postfix(loss=l_)

        if l_ < lowest_loss:
            lowest_loss = l_
            shift_ = shift.cpu().item()
            scale_ = scale.cpu().item()
            sign_ = sign
            losses_ = losses

    aligned = x * sign_ * scale_ + shift_
    return aligned, shift_, sign_, scale_, losses_


def circ_drift_regression(x, z, t, topology, device="cpu", iters=1000, lr=1e-2, a_fac=1):
    """
    Aligned circular trajectories with a learnable linear drift in time.
    
    :param np.ndarray theta: circular input array of shape (timesteps,)
    :param np.ndarray theta_ref: reference circular input array of shape (timesteps,)
    :param string device: device ('cpu', 'cuda:0' etc.) to perform computations on
    :param int iters: number of iterations of gradient descent
    :param float lr: learning rate
    """
    t = torch.tensor(t, device=device)
    X = torch.tensor(x, device=device)
    Z = torch.tensor(z, device=device)

    lowest_loss = np.inf
    for sign in [1, -1]:  # select sign automatically
        shift = Parameter(torch.zeros(1, device=device))
        a = Parameter(torch.zeros(1, device=device))

        optimizer = optim.Adam([a, shift], lr=lr)
        losses = []

        iterator = tqdm(range(iters))
        for k in iterator:
            optimizer.zero_grad()
            Z_ = t * a_fac * a + shift + sign * Z
            loss = (metric(Z_, X, topology) ** 2).mean()
            loss.backward()
            optimizer.step()

            l_ = loss.cpu().item()
            losses.append(l_)
            iterator.set_postfix(loss=l_)

        if l_ < lowest_loss:
            lowest_loss = l_
            a_ = a.cpu().item()
            shift_ = shift.cpu().item()
            sign_ = sign
            losses_ = losses

    aligned = a_fac * a_  # aligned trajectory
    return aligned, sign_, shift_, losses_



def circ_lin_regression(theta, x, device='cpu', iters=1000, lr=1e-2):
    """
    Similar to [1].
    
    :param np.array theta: circular input of shape (timesteps,)
    :param np.array x: linear input of shape (timesteps,)
    :param string device: device ('cpu', 'cuda:0' etc.) to perform computations on
    :param int iters: number of optimization iterations
    :param float lr: learning rate of optimization
    
    References:
    
    [1] `Quantifying circular–linear associations: Hippocampal phase precession`,
    Richard Kempter, Christian Leibold, György Buzsáki, Kamran Diba, Robert Schmidt (2012)
    
    """
    a = Parameter(torch.zeros(1, device=device))

    optimizer = optim.Adam([a], lr=lr)
    XX = torch.tensor(x, device=device)
    HD = torch.tensor(theta, device=device)

    losses = []
    for k in range(iters):
        optimizer.zero_grad()
        X_ = 2*np.pi*XX*a# + shift
        loss = -1 * resultant_length(X_, HD).mean() # must maximise R
        #loss = (metric(X_, HD, 'torus')**2).mean()
        loss.backward()
        optimizer.step()
        losses.append(loss.cpu().item())

    a_ = a.cpu().item()
    #shift_ = shift.cpu().item()
    shift_ = np.arctan2(np.sum(np.sin(theta - 2*np.pi*a_*x)),
                        np.sum(np.cos(theta - 2*np.pi*a_*x)))
    
    return 2*np.pi*x*a_ + shift_, a_, shift_, losses