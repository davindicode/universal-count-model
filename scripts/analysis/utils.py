import sys

import numpy as np

import torch
import torch.optim as optim

sys.path.append("../..")
import neuroprob as nprb
from neuroprob import utils


### UCM ###
def marginalized_UCM_P_count(
    full_model, eval_points, eval_dims, rcov, bs, use_neuron, MC=100, skip=1
):
    """
    Marginalize over the behaviour p(X) for X not evaluated over.

    :param list eval_points: list of ndarrays of values that you want to compute the marginal SCD at
    :param list eval_dims: the dimensions that are not marginalized evaluated at eval_points
    :param list rcov: list of covariate time series
    :param int bs: batch size
    :param list use_neuron: list of neurons used
    :param int skip: only take every skip time points of the behaviour time series for marginalisation
    """
    rcov = [rc[::skip] for rc in rcov]  # set dilution
    animal_T = rcov[0].shape[0]
    Ep = eval_points[0].shape[0]
    tot_len = Ep * animal_T

    covariates = []
    k = 0
    for d, rc in enumerate(rcov):
        if d in eval_dims:
            covariates.append(torch.repeat_interleave(eval_points[k], animal_T))
            k += 1
        else:
            covariates.append(rc.repeat(Ep))

    km = full_model.likelihood.K + 1
    P_tot = torch.empty((MC, len(use_neuron), Ep, km), dtype=torch.float)
    batches = int(np.ceil(animal_T / bs))
    for e in range(Ep):
        print("\r" + str(e), end="", flush=True)
        P_ = torch.empty((MC, len(use_neuron), animal_T, km), dtype=torch.float)
        for b in range(batches):
            bcov = [
                c[e * animal_T : (e + 1) * animal_T][b * bs : (b + 1) * bs]
                for c in covariates
            ]
            P_mc = compute_P(full_model, bcov, use_neuron, MC=MC).cpu()
            P_[..., b * bs : (b + 1) * bs, :] = P_mc

        P_tot[..., e, :] = P_.mean(-2)

    return P_tot


### stats ###
def ind_to_pair(ind, N):
    a = ind
    k = 1
    while a >= 0:
        a -= N - k
        k += 1

    n = k - 1
    m = N - n + a
    return n - 1, m



# metrics
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


# align latent
def signed_scaled_shift(
    x, x_ref, dev="cpu", topology="ring", iters=1000, lr=1e-2, learn_scale=True
):
    """
    Shift trajectory, with scaling, reflection and translation.

    Shift trajectories to be as close as possible to each other, including
    switches in sign.

    :param np.array theta: circular input array of shape (timesteps,)
    :param np.array theta_ref: reference circular input array of shape (timesteps,)
    :param string dev:
    :param int iters:
    :param float lr:
    :returns:
    :rtype: tuple
    """
    XX = torch.tensor(x, device=dev)
    XR = torch.tensor(x_ref, device=dev)

    lowest_loss = np.inf
    for sign in [1, -1]:  # select sign automatically
        shift = Parameter(torch.zeros(1, device=dev))
        p = [shift]

        if learn_scale:
            scale = Parameter(torch.ones(1, device=dev))
            p += [scale]
        else:
            scale = torch.ones(1, device=dev)

        optimizer = optim.Adam(p, lr=lr)
        losses = []
        for k in range(iters):
            optimizer.zero_grad()
            X_ = XX * sign * scale + shift
            loss = (metric(X_, XR, topology) ** 2).mean()
            loss.backward()
            optimizer.step()
            losses.append(loss.cpu().item())

        l_ = loss.cpu().item()

        if l_ < lowest_loss:
            lowest_loss = l_
            shift_ = shift.cpu().item()
            scale_ = scale.cpu().item()
            sign_ = sign
            losses_ = losses

    return x * sign_ * scale_ + shift_, shift_, sign_, scale_, losses_


def align_CCA(X, X_tar):
    """
    :param np.array X: input variables of shape (time, dimensions)
    """
    d = X.shape[-1]
    cca = CCA(n_components=d)
    cca.fit(X, X_tar)
    X_c = cca.transform(X)
    return X_c, cca
