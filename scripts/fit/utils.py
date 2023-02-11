import sys

import numpy as np

import torch
import torch.optim as optim

sys.path.append("../..")
import neuroprob as nprb
from neuroprob import utils


### UCM ###
def marginalized_P(
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


def get_q_Z(P, spike_binned, deq_noise=None):
    if deq_noise is None:
        deq_noise = np.random.uniform(size=spike_binned.shape)
    else:
        deq_noise = 0

    cumP = np.cumsum(P, axis=-1)  # T, K
    tt = np.arange(spike_binned.shape[0])
    quantiles = (
        cumP[tt, spike_binned.astype(int)] - P[tt, spike_binned.astype(int)] * deq_noise
    )
    Z = utils.stats.q_to_Z(quantiles)
    return quantiles, Z


def compute_count_stats(
    modelfit,
    spktrain,
    behav_list,
    neuron,
    traj_len=None,
    traj_spikes=None,
    start=0,
    T=100000,
    bs=5000,
    n_samp=1000,
):
    """
    Compute the dispersion statistics for the count model

    :param string mode: *single* mode refers to computing separate single neuron quantities, *population* mode
                        refers to computing over a population indicated by neurons, *peer* mode involves the
                        peer predictability i.e. conditioning on all other neurons given a subset
    """
    mapping = modelfit.mapping
    likelihood = modelfit.likelihood
    tbin = modelfit.likelihood.tbin

    N = int(np.ceil(T / bs))
    rate_model = []
    shape_model = []
    spktrain = spktrain[:, start : start + T]
    behav_list = [b[start : start + T] for b in behav_list]

    for k in range(N):
        covariates_ = [torchb[k * bs : (k + 1) * bs] for b in behav_list]
        ospktrain = spktrain[None, ...]

        rate = posterior_rate(
            mapping, likelihood, covariates, MC, F_dims, trials=1, percentiles=[0.5]
        )  # glm.mapping.eval_rate(covariates_, neuron, n_samp=1000)
        rate_model += [rate[0, ...]]

        if likelihood.dispersion_mapping is not None:
            cov = mapping.to_XZ(covariates_, trials=1)
            disp = likelihood.sample_dispersion(cov, n_samp, neuron)
            shape_model += [disp[0, ...]]

    rate_model = np.concatenate(rate_model, axis=1)
    if count_model and glm.likelihood.dispersion_mapping is not None:
        shape_model = np.concatenate(shape_model, axis=1)

    if type(likelihood) == nprb.likelihoods.Poisson:
        shape_model = None
        f_p = lambda c, avg, shape, t: utils.stats.poiss_count_prob(c, avg, t)

    elif type(likelihood) == nprb.likelihoods.Negative_binomial:
        shape_model = (
            glm.likelihood.r_inv.data.cpu().numpy()[:, None].repeat(T, axis=-1)
        )
        f_p = lambda c, avg, shape, t: utils.stats.nb_count_prob(c, avg, shape, t)

    elif type(likelihood) == nprb.likelihoods.COM_Poisson:
        shape_model = glm.likelihood.nu.data.cpu().numpy()[:, None].repeat(T, axis=-1)
        f_p = lambda c, avg, shape, t: utils.stats.cmp_count_prob(c, avg, shape, t)

    elif type(likelihood) == nprb.likelihoods.ZI_Poisson:
        shape_model = (
            glm.likelihood.alpha.data.cpu().numpy()[:, None].repeat(T, axis=-1)
        )
        f_p = lambda c, avg, shape, t: utils.stats.zip_count_prob(c, avg, shape, t)

    elif type(likelihood) == nprb.likelihoods.hNegative_binomial:
        f_p = lambda c, avg, shape, t: utils.stats.nb_count_prob(c, avg, shape, t)

    elif type(likelihood) == nprb.likelihoods.hCOM_Poisson:
        f_p = lambda c, avg, shape, t: utils.stats.cmp_count_prob(c, avg, shape, t)

    elif type(likelihood) == nprb.likelihoods.hZI_Poisson:
        f_p = lambda c, avg, shape, t: utils.stats.zip_count_prob(c, avg, shape, t)

    else:
        raise ValueError

    m_f = lambda x: x

    if shape_model is not None:
        assert traj_len == 1
    if traj_len is not None:
        traj_lens = (T // traj_len) * [traj_len]

    q_ = []
    for k, ne in enumerate(neuron):
        if traj_spikes is not None:
            avg_spikecnt = np.cumsum(rate_model[k] * tbin)
            nc = 1
            traj_len = 0
            for tt in range(T):
                if avg_spikecnt >= traj_spikes * nc:
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
                spktr.append(spktrain[ne][start[tt] : start[tt] + traj_len].sum())
                rm.append(rate_model[k][start[tt] : start[tt] + traj_len].sum())
            spktr = np.array(spktr)
            rm = np.array(rm)

        q_.append(utils.stats.count_KS_method(f_p, m_f, tbin, spktr, rm, shape=sh))

    return q_


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
