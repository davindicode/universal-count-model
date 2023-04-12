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
def compute_UCM_P_count(mapping, likelihood, covariates, out_inds, MC=1000, trials=1):
    """
    :param nn.Module mapping: the input maping module
    :param nn.Module likelihood: Universal likelihood module
    :param List covariates: list of tensors with covariates

    Compute predictive count distribution given X.
    """
    assert type(likelihood) == Universal

    F_dims = likelihood._out_inds_to_F(out_inds)
    h = marginal_posterior_samples(
        mapping, lambda x: x, covariates, MC, F_dims, trials=trials
    )
    logp = likelihood.get_logp(h, out_inds)  # samples, N, time, K

    P_mc = torch.exp(logp)
    return P_mc


def marginalize_UCM_P_count(
    mapping,
    likelihood,
    eval_covariates,
    eval_dims,
    sample_covariates_all,
    batch_size,
    out_inds,
    MC=100,
    sample_skip=1,
):
    """
    Marginalize over the behaviour p(X) for X not evaluated over specified by evaluation dimensions.
    This is equivalent to regressing a model leaving out marginalized input dimensions if the input 
    occupancy distribution p(X) factorizes between the evaluation and marginalization input subsets.

    :param List eval_covariates: list of tensors that specify where want to compute the marginal SCD
    :param List eval_dims: the dimensions that are not marginalized
    :param List sample_covariates_all: list of covariate time series (all input dimensions)
    :param int batch_size: batch size for evaluating marginalization
    :param List out_inds: list of output dimensions used
    :param int MC: number of Monte Carlo samples for evaluating posterior SCDs
    :param int skip: only take every skip time points of the behaviour time series for marginalisation
    """
    scovs = [
        rc[::sample_skip] for rc in sample_covariates_all
    ]  # dilution of sample trajectory

    sample_points = scovs[0].shape[0]
    eval_points = eval_covariates[0].shape[0]
    tot_len = eval_points * sample_points

    # repeat points based on evaluation points to get extended input timeseries
    covariates = []
    k = 0
    for d, rc in enumerate(scovs):
        if d in eval_dims:
            covariates.append(
                torch.repeat_interleave(eval_covariates[k], sample_points)
            )
            k += 1
        else:
            covariates.append(rc.repeat(eval_points))

    # allocate marginalized SCD over all evaluation points
    km = likelihood.K + 1
    P_marg = torch.empty(
        (MC, len(out_inds), eval_points, km),
        dtype=mapping.tensor_type,
        device=likelihood.tbin.device,
    )

    # compute SCD over extended input timeseries with averaging over samples
    batches = int(np.ceil(sample_points / batch_size))
    iterator = tqdm(range(eval_points))
    for e in iterator:  # iterate per evaluation point
        P = torch.empty(
            (MC, len(out_inds), sample_points, km),
            dtype=mapping.tensor_type,
            device=likelihood.tbin.device,
        )

        bcovs = [c[e * sample_points : (e + 1) * sample_points] for c in covariates]
        for b in range(batches):  # average over sample_covariates_all trajectory
            batched_covs = [
                bc[b * batch_size : (b + 1) * batch_size] for bc in bcovs
            ]  # evaluate in batches
            P_mc = compute_UCM_P_count(
                mapping, likelihood, batched_covs, out_inds, MC=MC
            )
            P[..., b * batch_size : (b + 1) * batch_size, :] = P_mc

        P_marg[..., e, :] = P.mean(-2)

    return P_marg
