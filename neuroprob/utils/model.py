import numpy as np

import torch
import torch.optim as optim

from tqdm.autonotebook import tqdm

from ..likelihoods import Universal

from ..likelihoods.base import mc_gen


def marginal_posterior_samples(mapping, inv_link, covariates, MC, F_dims, trials=1):
    """
    Sample f(F) from diagonalized variational posterior.

    :returns: F of shape (MCxtrials, outdims, time)
    """
    cov = mapping.to_XZ(covariates, trials)

    if mapping.MC_only:
        samples = mapping.sample_F(cov)[:, F_dims, :]

    else:
        F_mu, F_var = mapping.compute_F(cov)
        samples = mc_gen(F_mu, F_var, MC, F_dims)

    samples = inv_link(
        samples.view(-1, trials, *samples.shape[1:]) if trials > 1 else samples
    )
    return samples


def sample_tuning_curves(mapping, likelihood, covariates, MC, F_dims, trials=1):
    """
    Sample joint posterior samples for tuning curve draws
    """
    cov = mapping.to_XZ(covariates, trials)
    eps = torch.randn(
        (MC * trials, *cov.shape[1:-1]),
        dtype=mapping.tensor_type,
        device=mapping.dummy.device,
    )
    
    samples = mapping.sample_F(cov, eps)
    if trials > 1:
        samples = samples.view(-1, trials, *samples.shape[1:])

    return samples


### UCM ###
def compute_UCM_P_count(
    mapping, likelihood, covariates, show_neuron, MC=1000, trials=1
):
    """
    Compute predictive count distribution given X.
    """
    assert type(likelihood) == Universal

    F_dims = likelihood._neuron_to_F(show_neuron)
    h = marginal_posterior_samples(
        mapping, lambda x: x, covariates, MC, F_dims, trials=trials
    )
    logp = likelihood.get_logp(h, show_neuron)  # samples, N, time, K

    P_mc = torch.exp(logp)
    return P_mc


def marginalize_UCM_P_count(
    mapping, likelihood, eval_points, eval_dims, rcov, bs, use_neuron, MC=100, skip=1
):
    """
    Marginalize over the behaviour p(X) for X not evaluated over.

    :param List eval_points: list of ndarrays of values that you want to compute the marginal SCD at
    :param List eval_dims: the dimensions that are not marginalized evaluated at eval_points
    :param List rcov: list of covariate time series
    :param int bs: batch size
    :param List use_neuron: list of neurons used
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
    P_tot = torch.empty((MC, len(use_neuron), Ep, km), dtype=mapping.tensor_type)
    
    batches = int(np.ceil(animal_T / bs))
    iterator = tqdm(range(Ep))
    for e in iterator:
        print("\r" + str(e), end="", flush=True)
        P_ = torch.empty((MC, len(use_neuron), animal_T, km), dtype=mapping.tensor_type)
        for b in range(batches):
            bcov = [
                c[e * animal_T : (e + 1) * animal_T][b * bs : (b + 1) * bs]
                for c in covariates
            ]
            P_mc = compute_UCM_P_count(full_model, bcov, use_neuron, MC=MC).cpu()
            P_[..., b * bs : (b + 1) * bs, :] = P_mc

        P_tot[..., e, :] = P_.mean(-2)

    return P_tot