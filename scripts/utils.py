import sys

import numpy as np

import torch
import torch.optim as optim

sys.path.append("../lib/")
import neuroprob as nprb
from neuroprob import utils


### model ###
def sample_F(mapping, likelihood, covariates, MC, F_dims, trials=1, eps=None):
    """
    Sample F from diagonalized variational posterior.

    :returns: F of shape (MCxtrials, outdims, time)
    """
    cov = mapping.to_XZ(covariates, trials)
    if mapping.MC_only or eps is not None:
        samples = mapping.sample_F(cov, eps=eps)[
            :, F_dims, :
        ]  # TODO: cov_samples vs ll_samples?
        h = samples.view(-1, trials, *samples.shape[1:])
    else:
        F_mu, F_var = mapping.compute_F(cov)
        h = likelihood.mc_gen(F_mu, F_var, MC, F_dims)

    return h


def posterior_rate(
    mapping, likelihood, covariates, MC, F_dims, trials=1, percentiles=[0.05, 0.5, 0.95]
):
    """
    Sample F from diagonalized variational posterior.

    :returns: F of shape (MCxtrials, outdims, time)
    """
    cov = mapping.to_XZ(covariates, trials)
    if mapping.MC_only:
        F = mapping.sample_F(cov)[:, F_dims, :]  # TODO: cov_samples vs ll_samples?
        samples = likelihood.f(F.view(-1, trials, *samples.shape[1:]))
    else:
        F_mu, F_var = mapping.compute_F(cov)
        samples = likelihood.sample_rate(
            F_mu[:, F_dims, :], F_var[:, F_dims, :], trials, MC
        )

    return utils.signal.percentiles_from_samples(samples, percentiles)


def sample_tuning_curves(mapping, likelihood, covariates, MC, F_dims, trials=1):
    """ """
    cov = mapping.to_XZ(covariates, trials)
    eps = torch.randn(
        (MC * trials, *cov.shape[1:-1]),
        dtype=mapping.tensor_type,
        device=mapping.dummy.device,
    )
    # mapping.jitter = 1e-4
    samples = mapping.sample_F(cov, eps)
    T = samples.view(-1, trials, *samples.shape[1:])

    return T


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


### UCM ###
def compute_P(full_model, covariates, show_neuron, MC=1000, trials=1):
    """
    Compute predictive count distribution given X.
    """
    F_dims = full_model.likelihood._neuron_to_F(show_neuron)
    h = sample_F(
        full_model.mapping, full_model.likelihood, covariates, MC, F_dims, trials=trials
    )
    logp = full_model.likelihood.get_logp(h, show_neuron).data  # samples, N, time, K

    P_mc = torch.exp(logp)
    return P_mc


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


### cross validation ###
def RG_pred_ll(
    model,
    validation_set,
    neuron_group=None,
    ll_mode="GH",
    ll_samples=100,
    cov_samples=1,
    beta=1.0,
    IW=False,
):
    """
    Compute the predictive log likelihood (ELBO).
    """
    vcov, vtrain, vbatch_info = validation_set
    time_steps = vtrain.shape[-1]
    print("Data segment timesteps: {}".format(time_steps))

    model.input_group.set_XZ(vcov, time_steps, batch_info=vbatch_info)
    model.likelihood.set_Y(vtrain, batch_info=vbatch_info)
    model.validate_model(likelihood_set=True)

    # batching
    pll = []
    for b in range(model.input_group.batches):
        pll.append(
            -model.objective(
                b,
                cov_samples=cov_samples,
                ll_mode=ll_mode,
                neuron=neuron_group,
                beta=beta,
                ll_samples=ll_samples,
                importance_weighted=IW,
            ).item()
        )

    return np.array(pll).mean()


def LVM_pred_ll(
    model,
    validation_set,
    f_neuron,
    v_neuron,
    eval_cov_MC=1,
    eval_ll_MC=100,
    eval_ll_mode="GH",
    annealing=lambda x: 1.0,  # min(1.0, 0.002*x)
    cov_MC=16,
    ll_MC=1,
    ll_mode="MC",
    beta=1.0,
    IW=False,
    max_iters=3000,
):
    """
    Compute the predictive log likelihood (ELBO).
    """
    vcov, vtrain, vbatch_info = validation_set
    time_steps = vtrain.shape[-1]
    print("Data segment timesteps: {}".format(time_steps))

    model.input_group.set_XZ(vcov, time_steps, batch_info=vbatch_info)
    model.likelihood.set_Y(vtrain, batch_info=vbatch_info)
    model.validate_model(likelihood_set=True)

    # fit
    sch = lambda o: optim.lr_scheduler.MultiplicativeLR(o, lambda e: 0.9)
    opt_tuple = (optim.Adam, 100, sch)
    opt_lr_dict = {"default": 0}
    for z_dim in model.input_group.latent_dims:
        opt_lr_dict["input_group.input_{}.variational.mu".format(z_dim)] = 1e-2
        opt_lr_dict["input_group.input_{}.variational.finv_std".format(z_dim)] = 1e-3

    model.set_optimizers(
        opt_tuple, opt_lr_dict
    )  # , nat_grad=('rate_model.0.u_loc', 'rate_model.0.u_scale_tril'))

    losses = model.fit(
        max_iters,
        neuron=f_neuron,
        loss_margin=-1e0,
        margin_epochs=100,
        ll_mode=ll_mode,
        kl_anneal_func=annealing,
        cov_samples=cov_MC,
        ll_samples=ll_MC,
    )

    pll = []
    for b in range(model.input_group.batches):
        pll.append(
            -model.objective(
                b,
                neuron=v_neuron,
                cov_samples=eval_cov_MC,
                ll_mode=eval_ll_mode,
                beta=beta,
                ll_samples=eval_ll_MC,
                importance_weighted=IW,
            ).item()
        )

    return np.array(pll).mean(), losses



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
    elif topology == "torus":
        xy = (x - y) % (2 * np.pi)
        xy[xy > np.pi] -= 2 * np.pi
    elif topology == "circ":
        xy = 2 * (1 - torch.cos(x - y))
    else:
        raise NotImplementedError
    # xy[xy < 0] = -xy[xy < 0] # abs
    return xy



# align latent
def signed_scaled_shift(
    x, x_ref, dev="cpu", topology="torus", iters=1000, lr=1e-2, learn_scale=True
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