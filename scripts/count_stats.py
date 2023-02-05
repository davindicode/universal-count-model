import random

import numpy as np
import scipy

import torch


### tuning ###
def T_funcs(P):
    """
    Outputs (mean, var, FF) in last dimension
    """
    x_counts = torch.arange(P.shape[-1])
    x_count_ = x_counts[None, None, None, :]

    mu_ = (x_count_ * P).sum(-1)  # mc, N, T
    var_ = (x_count_**2 * P).sum(-1) - mu_**2  # mc, N, T
    FF_ = var_ / (mu_ + 1e-12)  # mc, N, T

    return torch.stack((mu_, var_, FF_), dim=-1)


def TI(T):
    """
    tuning indices
    """
    return torch.abs(
        (T.max(dim=-1)[0] - T.min(dim=-1)[0]) / (T.max(dim=-1)[0] + T.min(dim=-1)[0])
    )


def R2(T_marg, T_full):
    """
    explained variance, R squared
    total variance decomposition of posterior mean values
    """
    return 1 - ((T_full - T_marg) ** 2).mean(-1) / T_full.var(-1)


def RV(T_marg, T_full):
    """
    relative variance
    """
    return T_marg.var(-1) / T_full.var(-1)


def marginalized_T(
    full_model,
    T_funcs,
    T_num,
    eval_points,
    eval_dims,
    rcov,
    bs,
    use_neuron,
    MC=100,
    skip=1,
):
    """
    Marginalize over the behaviour p(X) for X not evaluated over.
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

    T_tot = torch.empty((MC, len(use_neuron), Ep, T_num), dtype=torch.float)
    batches = int(np.ceil(animal_T / bs))
    for e in range(Ep):
        print(e)
        T_ = torch.empty((MC, len(use_neuron), animal_T, T_num), dtype=torch.float)
        for b in range(batches):
            bcov = [
                c[e * animal_T : (e + 1) * animal_T][b * bs : (b + 1) * bs]
                for c in covariates
            ]
            P_mc = compute_P(full_model, bcov, use_neuron, MC=MC).cpu()
            T_[..., b * bs : (b + 1) * bs, :] = T_funcs(P_mc)

        T_tot[..., e, :] = T_.mean(-2)

    return T_tot


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
        f_p = lambda c, avg, shape, t: utils.stats.poiss_count_prob(c, avg, shape, t)

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


### spike conversions ###
def spikeinds_from_count(counts, bin_elements):
    """
    convert to spike timings

    returns: list of list of array of spike time indices
    """
    spikes = np.zeros(counts.shape + (bin_elements,))

    spikeinds = []
    trs, neurons, T = counts.shape
    for tr in range(trs):
        spikeinds_ = []
        for n in range(neurons):
            spikeinds__ = []
            cur_steps = 0
            for t in range(T):
                c = int(counts[tr, n, t])
                # spikeinds__.append(np.sort(random.sample(range(bin_elements), c)) + cur_steps) without replacement
                spikeinds__.append(
                    np.sort(random.choices(range(bin_elements), k=c)) + cur_steps
                )  # with replacement
                cur_steps += bin_elements

            spikeinds_.append(np.concatenate(spikeinds__))
        spikeinds.append(spikeinds_)

    return spikeinds


def rebin_spikeinds(spikeinds, bin_sizes, dt, behav_tuple, average_behav=False):
    """
    rebin data
    """
    datasets = []
    for bin_size in bin_sizes:
        tbin, resamples, rc_t, rbehav_tuple = utils.neural.bin_data(
            bin_size,
            dt,
            spikeinds,
            track_samples * C,
            behav_tuple,
            average_behav=average_behav,
            binned=False,
        )

        datasets.append((tbin, resamples, rc_t, rbehav_tuple))
    return datasets
    # res_ind = [] # spike times
    # for r in res:
    #    res_ind.append(utils.neural.binned_to_indices(r))


### variability ###
def variability_stats(modelfit, gp_dev="cpu", MC=100, bs=10000, jitter=1e-5):
    """
    full input data
    """
    units_used = modelfit.likelihood.neurons

    P_full = []
    with torch.no_grad():
        for b in range(modelfit.input_group.batches):
            XZ = modelfit.input_group.sample_XZ(b, 1, None, False)[0]
            P_mc = utils.compute_P(modelfit, XZ, list(range(units_used)), MC=MC).cpu()
            P_full.append(P_mc)

    P_full = torch.cat(P_full, dim=-2)

    ### variability ###
    T_full = lib.helper.T_funcs(P_full)
    mu_full, var_full, FF_full = T_full[..., 0], T_full[..., 1], T_full[..., 2]

    # posterior mean stats
    A = mu_full.mean(0)
    B = FF_full.mean(0)
    C = var_full.mean(0)
    resamples = A.shape[-1]

    # linear variance rate
    a, b = utils.signal.linear_regression(A, C)

    B_fit = a[:, None] + b[:, None] / (A + 1e-12)
    C_fit = a[:, None] * A + b[:, None]
    # R^2 of fit
    R2_ff = 1 - (B - B_fit).var(-1) / B.var(-1)
    R2_var = 1 - (C - C_fit).var(-1) / C.var(-1)

    linvar_tuple = (a, b, R2_ff, R2_var)

    # constant FF
    mff = B.mean(-1)  # R^2 by definition 0
    C_fit = mff[:, None] * A
    R2_var = 1 - (C - C_fit).var(-1) / C.var(-1)
    constFF_tuple = (mff, R2_var)

    # linear fits (doubly stochastic models, refractory models)
    a, b = utils.signal.linear_regression(A, B)

    B_fit = a[:, None] * A + b[:, None]
    C_fit = a[:, None] * A**2 + b[:, None] * A
    # R^2 of fit
    R2_ff = 1 - (B - B_fit).var(-1) / B.var(-1)
    R2_var = 1 - (C - C_fit).var(-1) / C.var(-1)

    linff_tuple = (a, b, R2_ff, R2_var)

    # nonparametric fits
    y_dims = units_used
    covariates = []
    for ne in range(y_dims):
        covariates += [np.linspace(0, A[ne, :].max() * 1.01, 100)]
    covariates = torch.tensor(covariates)

    np_tuple = (covariates,)
    for yd in [B, C]:
        v = 1.0 * torch.ones(y_dims)
        l = 1.0 * torch.ones(1, y_dims)

        constraints = []
        krn_1 = nprb.kernels.kernel.Constant(variance=v, tensor_type=torch.float)
        krn_2 = nprb.kernels.kernel.SquaredExponential(
            input_dims=1,
            lengthscale=l,
            topology="torus",
            f="softplus",
            track_dims=[0],
            tensor_type=torch.float,
        )

        kernel = nprb.kernels.kernel.Product(krn_1, krn_2)

        num_induc = 8
        Xu = []
        for ne in range(y_dims):
            Xu += [np.linspace(0, A[ne, :].max(), num_induc)]
        Xu = torch.tensor(Xu)[..., None]
        inducing_points = nprb.kernels.kernel.inducing_points(y_dims, Xu, constraints)

        input_data = [A[None, :, :, None]]  # tr, N, T, D

        # mapping
        in_dims = Xu.shape[-1]

        gp = nprb.mappings.GP.SVGP(
            in_dims,
            y_dims,
            kernel,
            inducing_points=inducing_points,
            whiten=True,
            jitter=jitter,
            mean=torch.zeros(y_dims),
            learn_mean=True,
        )

        ### inputs and likelihood ###
        input_group = nprb.inference.input_group()
        input_group.set_XZ(input_data, resamples, batch_info=bs)

        likelihood = nprb.likelihoods.Gaussian(
            y_dims, "exp", log_var=torch.zeros(y_dims)
        )
        likelihood.set_Y(yd, batch_info=bs)

        gpr = nprb.inference.VI_optimized(input_group, gp, likelihood)
        gpr.to(gp_dev)

        # fitting
        sch = lambda o: optim.lr_scheduler.MultiplicativeLR(o, lambda e: 0.9)
        opt_tuple = (optim.Adam, 100, sch)
        opt_lr_dict = {"default": 5 * 1e-3}

        gpr.set_optimizers(opt_tuple, opt_lr_dict)

        annealing = lambda x: 1.0
        losses = gpr.fit(
            3000,
            loss_margin=0.0,
            margin_epochs=100,
            kl_anneal_func=annealing,
            cov_samples=1,
            ll_samples=10,
            ll_mode="MC",
        )

        plt.figure()
        plt.plot(losses)
        plt.xlabel("epoch")
        plt.ylabel("NLL per time sample")
        plt.show()

        with torch.no_grad():
            lw, mn, up = lib.helper.posterior_rate(
                gp,
                likelihood,
                covariates[None, ..., None],
                10000,
                F_dims=list(range(units_used)),
                percentiles=[0.05, 0.5, 0.95],
            )
            lw = lw[0, ...].cpu()
            mn = mn[0, ...].cpu()
            up = up[0, ...].cpu()

        _fit_ = []
        bs = 100
        batches = int(np.ceil(A.shape[-1] / bs))
        with torch.no_grad():
            for b in range(batches):
                _fit_.append(
                    lib.helper.posterior_rate(
                        gp,
                        likelihood,
                        [A[None, :, b * bs : (b + 1) * bs, None]],
                        10000,
                        F_dims=list(range(units_used)),
                        percentiles=[0.5],
                    )[0][0, ...].cpu()
                )

        _fit = torch.cat(_fit_, dim=-1)
        R2 = 1 - (yd - _fit).var(-1) / yd.var(-1)

        np_tuple += (lw, mn, up, R2)

    ### collect ###
    variability_stats = (
        T_full.permute(0, 1, -1, -2),
        linvar_tuple,
        constFF_tuple,
        linff_tuple,
        np_tuple,
    )
    return variability_stats


def marginal_stats(
    modelfit,
    rcov_used,
    T_full,
    dimx_list,
    ang_dims=[],
    MC=100,
    skip=10,
    batchsize=10000,
    grid_size_pos=(50, 40),
    grid_size_1d=101,
):
    """
    marginalized data
    2D tuning curves have image convention T(y, x) (row-column)
    """
    units_used = modelfit.likelihood.neurons

    marginal_stats = []
    for dimx in dimx_list:
        if len(dimx) == 1:
            if dimx in ang_dims:
                cov_list = [torch.linspace(0, 2 * np.pi, grid_size_1d)]
            else:
                cov_list = [
                    torch.linspace(
                        rcov_used[dimx[0]].min(), rcov_used[dimx[0]].max(), grid_size_1d
                    )
                ]

        elif len(dimx) == 2:
            A, B = grid_size_pos
            cov_list = [
                torch.linspace(rcov_used[dimx[0]].min(), rcov_used[dimx[0]].max(), A)[
                    None, :
                ]
                .repeat(B, 1)
                .flatten(),
                torch.linspace(rcov_used[dimx[1]].min(), rcov_used[dimx[1]].max(), B)[
                    :, None
                ]
                .repeat(1, A)
                .flatten(),
            ]

        with torch.no_grad():
            # [torch.linspace(-2*np.pi*(10/100.), 2*np.pi*(110./100.), 121) # approximate periodic boundaries
            P_marg = lib.helper.marginalized_P(
                modelfit,
                cov_list,
                dimx,
                rcov_used,
                batchsize,
                list(range(units_used)),
                MC=MC,
                skip=skip,
            )  # mc, N, T, count

            T_marg = lib.helper.marginalized_T(
                modelfit,
                lib.helper.T_funcs,
                3,
                cov_list,
                dimx,
                rcov_used,
                batchsize,
                list(range(units_used)),
                MC=MC,
                skip=skip,
            ).permute(
                0, 1, -1, -2
            )  # mc, N, moment_dim, T

        T_marg_d = lib.helper.T_funcs(P_marg).permute(
            0, 1, -1, -2
        )  # mc, N, moment_dim, T

        ### tuning curves ###
        # mu_marg, var_marg, FF_marg = T_marg[..., 0], T_marg[..., 1], T_marg[..., 2]
        # mu_marg_d, var_marg_d, FF_marg_d = T_marg_d[..., 0], T_marg_d[..., 1], T_marg_d[..., 2]

        ### measures ###
        a, b, c, d = T_marg.shape  # mc, N, moment_dim, T

        if len(dimx) == 1:
            Ns = cov_list[0].shape[0]

            T_marg_x = utils.signal.cubic_interpolation(
                cov_list[0],
                T_marg.mean(0).reshape(-1, Ns),
                rcov_used[dimx[0]].float(),
                integrate=False,
            ).view(b, c, -1)

            T_marg_d_x = utils.signal.cubic_interpolation(
                cov_list[0],
                T_marg_d.mean(0).reshape(-1, Ns),
                rcov_used[dimx[0]].float(),
                integrate=False,
            ).view(b, c, -1)

        elif len(dimx) == 2:  # note it is represented as flat vector N_x*N_y
            Ns_x, Ns_y = grid_size_pos
            cl_x = cov_list[0][:Ns_x].numpy()
            cl_y = cov_list[1][::Ns_x].numpy()
            # xx, yy = np.meshgrid(cl_x, cl_y)

            x = rcov_used[dimx[0]].numpy()
            y = rcov_used[dimx[1]].numpy()
            # inp = np.stack((x, y))

            z_T = T_marg.numpy().reshape(-1, Ns_y, Ns_x)
            z_T_d = T_marg_d.numpy().reshape(-1, Ns_y, Ns_x)

            # steps = x.shape[0]
            # bs = 1000
            # batches = int(np.ceil(steps/bs))

            T_marg_x_ = []
            T_marg_d_x_ = []
            for ne in range(z_T.shape[0]):
                # RegularGridInterpolator((cl_y, cl_x), z_T[ne]) # linear
                f = scipy.interpolate.RectBivariateSpline(cl_y, cl_x, z_T[ne])
                T_marg_x_.append(
                    torch.from_numpy(f(x, y, grid=False)).float().flatten()
                )

                f = scipy.interpolate.RectBivariateSpline(cl_y, cl_x, z_T_d[ne])
                T_marg_d_x_.append(
                    torch.from_numpy(f(x, y, grid=False)).float().flatten()
                )

                """
                T_marg_x = utils.signal.bilinear_interpolation(
                    cov_list[0], cov_list[1], T_marg.permute(0, 1, -1, -2).reshape(-1, Ns_x, Ns_y), 
                    rcov_used[dimx[0]].float(), rcov_used[dimx[1]].float(), integrate=False
                ).view(a, b, d, -1).permute(0, 1, -1, -2)

                T_marg_d_x = utils.signal.bilinear_interpolation(
                    cov_list[0], cov_list[1], T_marg_d.permute(0, 1, -1, -2).reshape(-1, Ns_x, Ns_y), 
                    rcov_used[dimx[0]].float(), integrate=False
                ).view(a, b, d, -1).permute(0, 1, -1, -2)
                """
            T_marg_x = torch.stack(T_marg_x_, dim=0).view(a, b, c, -1)
            T_marg_d_x = torch.stack(T_marg_d_x_, dim=0).view(a, b, c, -1)

        # mu_full, var_full, FF_full = T_full[..., 0], T_full[..., 1], T_full[..., 2]
        # mu_marg_x, var_marg_x, FF_marg_x = T_marg_x[..., 0], T_marg_x[..., 1], T_marg_x[..., 2]
        # mu_marg_d_x, var_marg_d_x, FF_marg_d_x = T_marg_d_x[..., 0], T_marg_d_x[..., 1], T_marg_d_x[..., 2]
        T_full_pm = T_full

        TI_d = lib.helper.TI(T_marg_d)
        R2_d = lib.helper.R2(T_marg_d_x, T_full_pm)
        RV_d = lib.helper.RV(T_marg_d_x, T_full_pm)

        TI = lib.helper.TI(T_marg)
        R2 = lib.helper.R2(T_marg_x, T_full_pm)
        RV = lib.helper.RV(T_marg_x, T_full_pm)

        marginal_stats.append(
            (cov_list, T_marg, T_marg_d, TI_d, R2_d, RV_d, TI, R2, RV)
        )

    return marginal_stats
