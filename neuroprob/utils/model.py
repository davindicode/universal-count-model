import numpy as np

import torch
import torch.optim as optim

from ..likelihoods.base import mc_gen
from ..likelihoods import Universal



def marginal_posterior_samples(
    mapping, inv_link, covariates, MC, F_dims, trials=1
):
    """
    Sample f(F) from diagonalized variational posterior.

    :returns: F of shape (MCxtrials, outdims, time)
    """
    cov = mapping.to_XZ(covariates, trials)
    if mapping.MC_only:
        samples = mapping.sample_F(cov)[:, F_dims, :]
        
    else:
        F_mu, F_var = mapping.compute_F(cov)
        with torch.no_grad():
            samples = mc_gen(F_mu[:, F_dims, :], F_var[:, F_dims, :], MC, list(range(len(F_dims))))
        
    samples = inv_link(samples.view(-1, trials, *samples.shape[1:]) if trials > 1 else samples)
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


def sample_Y(mapping, likelihood, covariates, trials, MC=1):
    """
    Use the posterior mean rates. Sampling gives np.ndarray
    """
    cov = mapping.to_XZ(covariates, trials)

    with torch.no_grad():

        F_mu, F_var = mapping.compute_F(cov)
        rate = likelihood.sample_rate(
            F_mu, F_var, trials, MC
        )  # MC, trials, neuron, time

        rate = rate.mean(0).cpu().numpy()
        syn_train = likelihood.sample(rate, XZ=cov)

    return syn_train


def compute_UCM_P_count(mapping, likelihood, covariates, show_neuron, MC=1000, trials=1):
    """
    Compute predictive count distribution given X.
    """
    assert type(likelihood) == Universal
    
    F_dims = likelihood._neuron_to_F(show_neuron)
    with torch.no_grad():
        h = marginal_posterior_samples(
            mapping, lambda x: x, covariates, MC, F_dims, trials=trials
        )
        logp = likelihood.get_logp(h, show_neuron)  # samples, N, time, K

    P_mc = torch.exp(logp)
    return P_mc