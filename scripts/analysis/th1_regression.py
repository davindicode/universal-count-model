import argparse

import os

import pickle

import sys

import numpy as np
import scipy.stats as scstats
import torch

sys.path.append("../..")  # access to library
import neuroprob as nprb

sys.path.append("..")  # access to scripts
import models

import utils


def regression(
    checkpoint_dir, reg_config_names, subset_config_names, dataset_dict, rng, batch_info, device
):
    tbin = dataset_dict["tbin"]
    max_count = dataset_dict["max_count"]
    neurons = dataset_dict["neurons"]
    pick_neurons = list(range(neurons))

    x_counts = torch.arange(max_count + 1)
    HD_offset = -1.0  # global shift makes plots look better

    ### cross validation ###
    kcvs = [1, 2, 3, 5, 6, 8]  # validation segments from splitting data into 10

    RG_liks_cv_ll = []
    for name in reg_config_names:  # likelihood
        for kcv in kcvs:
            config_name = name + str(kcv)

            full_model, _, _, val_dict = models.load_model(
                config_name,
                checkpoint_dir,
                dataset_dict,
                batch_info,
                device,
            )

            RG_liks_cv_ll.append(
                models.RG_Ell(
                    full_model,
                    val_dict,
                    neuron_group=None,
                    ll_mode="GH",
                    ll_samples=100,
                    cov_samples=1,
                    beta=0.0,
                )
            )

    RG_liks_cv_ll = np.array(RG_liks_cv_ll).reshape(len(reg_config_names), len(kcvs))

    RG_subsets_cv_ll = []
    for name in subset_config_names:  # input space
        for kcv in kcvs:
            config_name = name + str(kcv)

            full_model, _, _, val_dict = models.load_model(
                config_name,
                checkpoint_dir,
                dataset_dict,
                batch_info,
                device,
            )

            RG_subsets_cv_ll.append(
                models.RG_Ell(
                    full_model,
                    val_dict,
                    neuron_group=None,
                    ll_mode="GH",
                    ll_samples=100,
                    cov_samples=1,
                    beta=0.0,
                )
            )

    RG_subsets_cv_ll = np.array(RG_subsets_cv_ll).reshape(
        len(subset_config_names), len(kcvs)
    )

    ### KS test ###
    KS_kcvs = [2, 5, 8]  # validation segments from splitting data into 10

    N = len(pick_neurons)
    Qq, Zz, R, Rp = [], [], [], []
    for en, name in enumerate(reg_config_names):
        for kcv in KS_kcvs:
            config_name = name + str(kcv)

            full_model, _, fit_dict, _ = models.load_model(
                config_name,
                checkpoint_dir,
                dataset_dict,
                batch_info,
                device,
            )
            ts = fit_dict["spiketrain"].shape[-1]

            P = []
            for batch in range(full_model.input_group.batches):
                covariates, _ = full_model.input_group.sample_XZ(batch, samples=1)

                if type(full_model.likelihood) == nprb.likelihoods.Poisson:
                    rate = nprb.utils.model.marginal_posterior_samples(
                        full_model.mapping,
                        full_model.likelihood.f,
                        covariates,
                        1000,
                        pick_neurons,
                    )

                    rate = rate.mean(0).cpu().numpy()  # posterior mean

                    P.append(
                        nprb.utils.stats.poiss_count_prob(
                            np.arange(max_count + 1), rate, tbin
                        )
                    )

                elif type(full_model.likelihood) == nprb.likelihoods.hNegative_binomial:
                    rate = nprb.utils.model.marginal_posterior_samples(
                        full_model.mapping,
                        full_model.likelihood.f,
                        covariates,
                        1000,
                        pick_neurons,
                    )
                    r_inv = nprb.utils.model.marginal_posterior_samples(
                        full_model.likelihood.dispersion_mapping,
                        full_model.likelihood.dispersion_mapping_f,
                        covariates,
                        1000,
                        pick_neurons,
                    )

                    rate = rate.mean(0).cpu().numpy()  # posterior mean
                    r_inv = r_inv.mean(0).cpu().numpy()

                    P.append(
                        nprb.utils.stats.nb_count_prob(
                            np.arange(max_count + 1), rate, r_inv, tbin
                        )
                    )

                else:  # UCM
                    P_mc = nprb.utils.model.compute_UCM_P_count(
                        full_model.mapping,
                        full_model.likelihood,
                        covariates,
                        pick_neurons,
                        MC=100,
                    )
                    P.append(P_mc.mean(0).cpu().numpy())  # take mean over MC samples

            P = np.concatenate(
                P, axis=1
            )  # count probabilities of shape (neurons, timesteps, count)

            q_ = []
            Z_ = []
            for n in range(len(pick_neurons)):
                spike_binned = full_model.likelihood.all_spikes[
                    0, pick_neurons[n], :
                ].numpy()
                q = nprb.utils.stats.counts_to_quantiles(P[n, ...], spike_binned, rng)

                q_.append(q)
                Z_.append(nprb.utils.stats.quantile_Z_mapping(q))

            Qq.append(q_)
            Zz.append(Z_)

            Pearson_s = []
            for n in range(N):
                for m in range(n + 1, N):
                    r, r_p = scstats.pearsonr(
                        Z_[n], Z_[m]
                    )  # Pearson r correlation test
                    Pearson_s.append((r, r_p))

            r = np.array([p[0] for p in Pearson_s])
            r_p = np.array([p[1] for p in Pearson_s])

            R.append(r)
            Rp.append(r_p)

    q_DS, T_DS, T_KS = [], [], []
    for q in Qq:
        for qq in q:
            (
                T_DS_,
                T_KS_,
                sign_DS,
                sign_KS,
                p_DS,
                p_KS,
            ) = nprb.utils.stats.KS_DS_statistics(qq, alpha=0.05, alpha_s=0.05)
            T_DS.append(T_DS_)
            T_KS.append(T_KS_)

            Z_DS = T_DS / np.sqrt(2 / (qq.shape[0] - 1))
            q_DS.append(utils.stats.Z_to_q(Z_DS))

    Qq = np.array(Qq).reshape(len(reg_config_names), len(KS_kcvs), -1)
    Zz = np.array(Zz).reshape(len(reg_config_names), len(KS_kcvs), -1)
    R = np.array(R).reshape(len(reg_config_names), len(KS_kcvs), -1)
    Rp = np.array(Rp).reshape(len(reg_config_names), len(KS_kcvs), -1)

    q_DS = np.array(q_DS).reshape(len(reg_config_names), len(KS_kcvs), -1)
    T_DS = np.array(T_DS).reshape(len(reg_config_names), len(KS_kcvs), -1)
    T_KS = np.array(T_KS).reshape(len(reg_config_names), len(KS_kcvs), -1)

    regression_dict = {
        "Ell_likelihood": PLL_rg_ll,
        "Ell_subsets": PLL_rg_cov,
        "quantiles": Qq,
        "Z_scores": Zz,
        "correlations": R,
        "significance_corrs": Rp,
        "q_DS": q_DS,
        "T_DS": T_DS,
        "T_KS": T_KS,
        "significance_KS": sign_KS,
        "significance_DS": sign_DS,
    }

    return regression_dict


def binning_variability(checkpoint_dir, config_names, binnings, dataset_dict, batch_info, device):
    neurons = dataset_dict["neurons"]
    pick_neurons = list(range(neurons))

    ### statistics over the behaviour ###
    avg_binnings, var_binnings, FF_binnings = [], [], []
    for name in config_names:
        config_name = name + "-1"

        full_model, _, fit_dict, _ = models.load_model(
            config_name,
            checkpoint_dir,
            dataset_dict,
            batch_info,
            device,
        )
        ts = fit_dict["spiketrain"].shape[-1]

        avg_model, var_model, ff_model = [], [], []
        for batch in range(full_model.input_group.batches):
            covariates, _ = full_model.input_group.sample_XZ(batch, samples=1)

            P_mc = (
                nprb.utils.model.compute_UCM_P_count(
                    full_model.mapping,
                    full_model.likelihood,
                    covariates,
                    pick_neurons,
                    MC=100,
                )
                .mean(0)
                .cpu()
                .numpy()
            )  # count probabilities of shape (neurons, timesteps, count)

            max_count = P_mc.shape[-1] - 1
            x_counts = np.arange(max_count + 1)

            avg = (x_counts[None, None, None, :] * P_mc).sum(-1)
            var = (x_counts[None, None, None, :] ** 2 * P_mc).sum(-1) - avg**2
            ff = var / (avg + 1e-12)
            avg_model.append(avg)
            var_model.append(var)
            ff_model.append(ff)

        avg_binnings.append(np.concatenate(avg_model, axis=-1).mean(0))
        var_binnings.append(np.concatenate(var_model, axis=-1).mean(0))
        FF_binnings.append(np.concatenate(ff_model, axis=-1).mean(0))

    # compute the Pearson correlation between Fano factors and mean firing rates
    bin_sel = 1  # 40 ms

    Pearson_avg_FF = []
    ratio_avg_FF = []
    for avg, ff in zip(avg_binnings[b], FF_binnings[b]):
        r, r_p = scstats.pearsonr(ff, avg)  # Pearson r correlation test
        Pearson_avg_FF.append((r, r_p))
        ratio_avg_FF.append(ff.std() / avg.std())

    binning_dict = {
        "bin_sizes": binnings,
        "avg_binnings": avg_binnings,
        "var_binnings": var_binnings,
        "FF_binnings": FF_binnings,
        "Pearson_avg_FF": Pearson_avg_FF,
        "ratio_avg_FF": ratio_avg_FF,
    }

    return binning_dict


def tunings(checkpoint_dir, model_name, dataset_dict, device):
    tbin = dataset_dict["tbin"]
    max_count = dataset_dict["max_count"]
    neurons = dataset_dict["neurons"]
    pick_neurons = list(range(neurons))

    x_t = dataset_dict["covariates"]["x_t"]
    y_t = dataset_dict["covariates"]["y_t"]
    left_x, right_x = x_t.min(), x_t.max()
    bottom_y, top_y = y_t.min(), y_t.max()

    ### load model ###
    full_model, training_loss, fit_dict, val_dict = models.load_model(
        config_name,
        checkpoint_dir,
        dataset_dict,
        batch_info,
        device,
    )

    TT = tbin * resamples

    # marginalized tuning curves
    MC = 100
    skip = 10
    batch_size = 10000

    ### hd ###
    steps = 100
    P_tot = utils.marginalized_UCM_P_count(
        full_model,
        [np.linspace(0, 2 * np.pi, steps)],
        [0],
        rcov,
        batch_size,
        pick_neurons,
        MC=MC,
        skip=skip,
    )
    avg = (x_counts[None, None, None, :] * P_tot).sum(-1)
    var = (x_counts[None, None, None, :] ** 2 * P_tot).sum(-1) - avg**2
    ff = var / avg

    avgs = nprb.utils.stats.percentiles_from_samples(
        avg, percentiles=[0.05, 0.5, 0.95], smooth_length=5, padding_mode="circular"
    )
    marg_avg_hd_percentiles = [cs_.cpu().numpy() for cs_ in avgs]

    ffs = nprb.utils.stats.percentiles_from_samples(
        ff, percentiles=[0.05, 0.5, 0.95], smooth_length=5, padding_mode="circular"
    )
    marg_FF_hd_percentiles = [cs_.cpu().numpy() for cs_ in ffs]

    # total variance decomposition
    hd_mean_EV = avg.var(0).mean(-1)
    hd_mean_VE = avg.mean(0).var(-1)
    hd_ff_EV = avg.var(0).mean(-1)
    hd_ff_VE = avg.mean(0).var(-1)

    # TI
    hd_avg_tf = (mhd_mean.max(dim=-1)[0] - mhd_mean.min(dim=-1)[0]) / (
        mhd_mean.max(dim=-1)[0] + mhd_mean.min(dim=-1)[0]
    )
    hd_FF_tf = (mhd_ffmean.max(dim=-1)[0] - mhd_ffmean.min(dim=-1)[0]) / (
        mhd_ffmean.max(dim=-1)[0] + mhd_ffmean.min(dim=-1)[0]
    )

    ### omega ###
    steps = 100
    w_edge = (-rcov[1].min() + rcov[1].max()) / 2.0
    covariates_w = np.linspace(-w_edge, w_edge, steps)
    P_tot = model_utils.marginalized_P(
        full_model, [covariates_w], [1], rcov, batch_size, pick_neurons, MC=MC, skip=skip
    )
    avg = (x_counts[None, None, None, :] * P_tot).sum(-1)
    var = (x_counts[None, None, None, :] ** 2 * P_tot).sum(-1) - avg**2
    ff = var / avg

    mw_mean = avg.mean(0)
    mw_ff = ff.mean(0)
    omega_avg_tf = (mw_mean.max(dim=-1)[0] - mw_mean.min(dim=-1)[0]) / (
        mw_mean.max(dim=-1)[0] + mw_mean.min(dim=-1)[0]
    )
    omega_FF_tf = (mw_ff.max(dim=-1)[0] - mw_ff.min(dim=-1)[0]) / (
        mw_ff.max(dim=-1)[0] + mw_ff.min(dim=-1)[0]
    )

    ### speed ###
    steps = 100
    P_tot = utils.marginalized_UCM_P_count(
        full_model,
        [np.linspace(0, 30.0, steps)],
        [2],
        rcov,
        batch_size,
        pick_neurons,
        MC=MC,
        skip=skip,
    )
    avg = (x_counts[None, None, None, :] * P_tot).sum(-1)
    var = (x_counts[None, None, None, :] ** 2 * P_tot).sum(-1) - avg**2
    ff = var / avg

    ms_mean = avg.mean(0)
    ms_ff = ff.mean(0)
    speed_avg_tf = (ms_mean.max(dim=-1)[0] - ms_mean.min(dim=-1)[0]) / (
        ms_ff.max(dim=-1)[0] + ms_ff.min(dim=-1)[0]
    )
    speed_FF_tf = (ms_ff.max(dim=-1)[0] - ms_ff.min(dim=-1)[0]) / (
        ms_ff.max(dim=-1)[0] + ms_ff.min(dim=-1)[0]
    )

    ### time ###
    steps = 100
    P_tot = model_utils.marginalized_P(
        full_model,
        [np.linspace(0, TT, steps)],
        [5],
        rcov,
        batch_size,
        pick_neurons,
        MC=MC,
        skip=skip,
    )
    avg = (x_counts[None, None, None, :] * P_tot).sum(-1)
    var = (x_counts[None, None, None, :] ** 2 * P_tot).sum(-1) - avg**2
    ff = var / avg

    mt_mean = avg.mean(0)
    mt_ff = ff.mean(0)
    time_avg_tf = (mt_mean.max(dim=-1)[0] - mt_mean.min(dim=-1)[0]) / (
        mt_ff.max(dim=-1)[0] + mt_ff.min(dim=-1)[0]
    )
    time_FF_tf = (mt_ff.max(dim=-1)[0] - mt_ff.min(dim=-1)[0]) / (
        mt_ff.max(dim=-1)[0] + mt_ff.min(dim=-1)[0]
    )

    ### position ###
    grid_size_pos = (12, 10)
    grid_shape_pos = [[left_x, right_x], [bottom_y, top_y]]

    steps = np.product(grid_size_pos)
    A, B = grid_size_pos

    cov_list = [
        np.linspace(left_x, right_x, A)[:, None].repeat(B, axis=1).flatten(),
        np.linspace(bottom_y, top_y, B)[None, :].repeat(A, axis=0).flatten(),
    ]

    P_tot = model_utils.marginalized_P(
        full_model, cov_list, [3, 4], rcov, 10000, pick_neurons, MC=MC, skip=skip
    )
    avg = (x_counts[None, None, None, :] * P_tot).sum(-1)
    var = (x_counts[None, None, None, :] ** 2 * P_tot).sum(-1) - avg**2
    ff = var / avg

    mpos_mean = avg.mean(0)
    mpos_ff = ff.mean(0)
    pos_avg_tf = (mpos_mean.max(dim=-1)[0] - mpos_mean.min(dim=-1)[0]) / (
        mpos_mean.max(dim=-1)[0] + mpos_mean.min(dim=-1)[0]
    )
    pos_FF_tf = (mpos_ff.max(dim=-1)[0] - mpos_ff.min(dim=-1)[0]) / (
        mpos_ff.max(dim=-1)[0] + mpos_ff.min(dim=-1)[0]
    )
    marg_pos_mean = mpos_mean.reshape(-1, A, B)
    marg_pos_FF = mpos_ff.reshape(-1, A, B)

    marginal_tunings = {
        "hd_avg_tf": hd_avg_tf,
        "hd_FF_tf": hd_FF_tf,
        "omega_avg_tf": omega_avg_tf,
        "omega_FF_tf": omega_FF_tf,
        "speed_avg_tf": speed_avg_tf,
        "speed_FF_tf": speed_FF_tf,
        "pos_avg_tf": pos_avg_tf,
        "pos_FF_tf": pos_FF_tf,
        "time_avg_tf": time_avg_tf,
        "time_FF_tf": time_FF_tf,
        "marg_hd_avg": mhd_mean,
        "marg_hd_FF": mhd_ff,
        "marg_omega_avg": mw_mean,
        "marg_omega_FF": mw_ff,
        "marg_speed_avg": ms_mean,
        "marg_speed_FF": ms_ff,
        "marg_pos_avg": mpos_mean,
        "marg_pos_FF": mpos_ff,
        "marg_time_avg": mt_mean,
        "marg_time_FF": mt_ff,
    }

    # special joint tuning curves

    ### hd omega ###
    grid_size_hdw = (51, 41)
    grid_shape_hdw = [[0, 2 * np.pi], [-10.0, 10.0]]
    grid_hd_omega = {"size": grid_size_hdw, "shape": grid_shape_hdw}

    steps = np.product(grid_size_hdw)
    A, B = grid_size_hdw
    covariates = [
        np.linspace(0, 2 * np.pi, A)[:, None].repeat(B, axis=1).flatten(),
        np.linspace(-10.0, 10.0, B)[None, :].repeat(A, axis=0).flatten(),
        0.0 * np.ones(steps),
        (left_x + right_x) / 2.0 * np.ones(steps),
        (bottom_y + top_y) / 2.0 * np.ones(steps),
        0.0 * np.ones(steps),
    ]

    P_mean = (
        model_utils.compute_P(full_model, covariates, pick_neurons, MC=MC).mean(0).cpu()
    )
    field_hd_omega = (
        (x_counts[None, None, :] * P_mean).sum(-1).reshape(-1, A, B).numpy()
    )

    # compute preferred HD
    grid = (101, 21)
    grid_shape = [[0, 2 * np.pi], [-10.0, 10.0]]

    steps = np.product(grid)
    A, B = grid

    w_arr = np.linspace(-10.0, 10.0, B)
    covariates = [
        np.linspace(0, 2 * np.pi, A)[:, None].repeat(B, axis=1).flatten(),
        w_arr[None, :].repeat(A, axis=0).flatten(),
        0.0 * np.ones(steps),
        (left_x + right_x) / 2.0 * np.ones(steps),
        (bottom_y + top_y) / 2.0 * np.ones(steps),
        0.0 * np.ones(steps),
    ]

    P_mean = (
        model_utils.compute_P(full_model, covariates, pick_neurons, MC=MC).mean(0).cpu()
    )
    field = (x_counts[None, None, :] * P_mean).sum(-1).reshape(-1, A, B).numpy()

    Z = np.cos(covariates[0]) + np.sin(covariates[0]) * 1j  # CoM angle
    Z = Z[None, :].reshape(-1, A, B)
    pref_hd_omega = np.angle((Z * field).mean(1)) % (2 * np.pi)  # neurons, w

    # ATI
    ATI = []
    res_var = []
    for k in range(neurons):
        _, a, shift, losses = nprb.utils.stats.circ_lin_regression(
            pref_hd_omega[k, :], w_arr / (2 * np.pi), dev="cpu", iters=1000, lr=1e-2
        )
        ATI.append(-a)
        res_var.append(losses[-1])
    ATI = np.array(ATI)
    res_var = np.array(res_var)

    ### hd time ###
    grid_size_hdt = (51, 41)
    grid_shape_hdt = [[0, 2 * np.pi], [0.0, TT]]
    grid_hd_time = {"size": grid_size_hdt, "shape": grid_shape_hdt}

    steps = np.product(grid_size_hdt)
    A, B = grid_size_hdt
    covariates = [
        np.linspace(0, 2 * np.pi, A)[:, None].repeat(B, axis=1).flatten(),
        0.0 * np.ones(steps),
        0.0 * np.ones(steps),
        (left_x + right_x) / 2.0 * np.ones(steps),
        (bottom_y + top_y) / 2.0 * np.ones(steps),
        np.linspace(0.0, TT, B)[None, :].repeat(A, axis=0).flatten(),
    ]

    P_mean = (
        nprb.utils.model.compute_UCM_P_count(
            full_model.mapping, full_model.likelihood, covariates, pick_neurons, MC=MC_
        )
        .mean(0)
        .cpu()
    )
    field_hd_time = (x_counts[None, None, :] * P_mean).sum(-1).reshape(-1, A, B).numpy()

    # drift and similarity matrix
    grid = (201, 16)
    grid_shape = [[0, 2 * np.pi], [0.0, TT]]

    steps = np.product(grid)
    A, B = grid

    t_arr = np.linspace(0.0, TT, B)
    dt_arr = t_arr[1] - t_arr[0]
    covariates = [
        np.linspace(0, 2 * np.pi, A)[:, None].repeat(B, axis=1).flatten(),
        0.0 * np.ones(steps),
        0.0 * np.ones(steps),
        (left_x + right_x) / 2.0 * np.ones(steps),
        (bottom_y + top_y) / 2.0 * np.ones(steps),
        t_arr[None, :].repeat(A, axis=0).flatten(),
    ]

    P_mean = (
        model_utils.compute_P(full_model, covariates, pick_neurons, MC=MC_).mean(0).cpu()
    )
    field = (x_counts[None, None, :] * P_mean).sum(-1).reshape(-1, A, B).numpy()

    Z = np.cos(covariates[0]) + np.sin(covariates[0]) * 1j  # CoM angle
    Z = Z[None, :].reshape(-1, A, B)
    E_exp = (Z * field).sum(-2) / field.sum(-2)
    pref_hd_time = np.angle(E_exp) % (2 * np.pi)  # neurons, t

    tun_width = 1.0 - np.abs(E_exp)
    amp_t = field.mean(-2)  # mean amplitude
    ampm_t = field.max(-2)

    sim_mat = []
    act = (field - field.mean(-2, keepdims=True)) / field.std(-2, keepdims=True)
    en = np.argsort(pref_hd_time, axis=0)
    for t in range(B):
        a = act[en[:, t], :, t]
        sim_mat = (a[:, None, :] * a[None, ...]).mean(-1)

    drift = []
    res_var_drift = []
    for k in range(len(pick_neurons)):
        _, a, shift, losses = nprb.utils.stats.circ_lin_regression(
            pref_hd_time[k, :],
            t_arr / (2 * np.pi) / 1e2,
            dev="cpu",
            iters=1000,
            lr=1e-2,
        )
        drift.append(a / 1e2)
        res_var_drift.append(losses[-1])
    drift = np.array(drift)
    res_var_drift = np.array(res_var_drift)

    joint_tunings = {
        "grid_hd_omega": grid_hd_omega,
        "field_hd_omega": field_hd_omega,
        "pref_hd_omega": pref_hd_omega,
        "grid_hd_time": grid_hd_time,
        "field_hd_time": field_hd_time,
        "pref_hd_time": pref_hd_time,
        "ATI": ATI,
        "res_var": res_var,
        "drift": drift,
        "res_var_drift": res_var_drift,
        "tun_width": tun_width,
        "amp_t": amp_t,
        "ampm_t": ampm_t,
        "sim_mat": sim_mat,
    }

    # conditional tuning curves
    MC = 300
    MC_ = 100

    ### head direction tuning ###
    steps = 100
    eval_hd = np.linspace(0, 2 * np.pi, steps)

    covariates = [
        np.linspace(0, 2 * np.pi, steps) - HD_offset,
        0.0 * np.ones(steps),
        0.0 * np.ones(steps),
        (left_x + right_x) / 2.0 * np.ones(steps),
        (bottom_y + top_y) / 2.0 * np.ones(steps),
        0.0 * np.ones(steps),
    ]

    P_mc = utils.compute_UCM_P_count(full_model, covariates, pick_neuron, MC=MC).cpu()

    avg = (x_counts[None, None, None, :] * P_mc).sum(-1)
    xcvar = (x_counts[None, None, None, :] ** 2 * P_mc).sum(-1) - avg**2
    ff = xcvar / avg

    avgs = nprb.utils.stats.percentiles_from_samples(
        avg, percentiles=[0.05, 0.5, 0.95], smooth_length=5, padding_mode="circular"
    )
    avg_hd_percentiles = [cs_.cpu().numpy() for cs_ in avgs]

    ffs = nprb.utils.stats.percentiles_from_samples(
        ff, percentiles=[0.05, 0.5, 0.95], smooth_length=5, padding_mode="circular"
    )
    FF_hd_percentiles = [cs_.cpu().numpy() for cs_ in ffs]

    ### omega tuning ###
    avg_omega_percentiles = []
    FF_omega_percentiles = []

    steps = 100
    w_edge = (-rcov[1].min() + rcov[1].max()) / 2.0
    eval_omega = np.linspace(-w_edge, w_edge, steps)
    for en, n in enumerate(pick_neurons):
        covariates = [
            pref_hd_omega[en, len(w_arr) // 2] * np.ones(steps),
            eval_omega,
            0.0 * np.ones(steps),
            (left_x + right_x) / 2.0 * np.ones(steps),
            (bottom_y + top_y) / 2.0 * np.ones(steps),
            0.0 * np.ones(steps),
        ]

        P_mc = utils.compute_UCM_P_count(full_model, covariates, [n], MC=MC)[
            :, 0, ...
        ].cpu()

        avg = (x_counts[None, None, :] * P_mc).sum(-1)
        xcvar = (x_counts[None, None, :] ** 2 * P_mc).sum(-1) - avg**2
        ff = xcvar / avg

        avgs = nprb.utils.stats.percentiles_from_samples(
            avg,
            percentiles=[0.05, 0.5, 0.95],
            smooth_length=5,
            padding_mode="replicate",
        )
        avg_percentiles = [cs_.cpu().numpy() for cs_ in avgs]

        ffs = nprb.utils.stats.percentiles_from_samples(
            ff, percentiles=[0.05, 0.5, 0.95], smooth_length=5, padding_mode="replicate"
        )
        FF_percentiles = [cs_.cpu().numpy() for cs_ in ffs]

        avg_omega_percentiles.append(avg_percentiles)
        FF_omega_percentiles.append(FF_percentiles)

    ### speed ###
    avg_speed_percentiles = []
    FF_speed_percentiles = []

    steps = 100
    eval_speed = np.linspace(0, 30.0, steps)
    for en, n in enumerate(pick_neurons):
        covariates = [
            pref_hd_omega[en, len(w_arr) // 2] * np.ones(steps),
            0.0 * np.ones(steps),
            eval_speed,
            (left_x + right_x) / 2.0 * np.ones(steps),
            (bottom_y + top_y) / 2.0 * np.ones(steps),
            0.0 * np.ones(steps),
        ]

        P_mc = utils.compute_UCM_P_count(
            full_model.mapping, full_model.likelihood, covariates, [n], MC=MC
        )[:, 0, ...].cpu()

        avg = (x_counts[None, None, :] * P_mc).sum(-1)
        xcvar = (x_counts[None, None, :] ** 2 * P_mc).sum(-1) - avg**2
        ff = xcvar / avg

        avgs = nprb.utils.stats.percentiles_from_samples(
            avg,
            percentiles=[0.05, 0.5, 0.95],
            smooth_length=5,
            padding_mode="replicate",
        )
        avg_percentiles = [cs_.cpu().numpy() for cs_ in avgs]

        ffs = nprb.utils.stats.percentiles_from_samples(
            ff, percentiles=[0.05, 0.5, 0.95], smooth_length=5, padding_mode="replicate"
        )
        FF_percentiles = [cs_.cpu().numpy() for cs_ in ffs]

        avg_speed_percentiles.append(avg_percentiles)
        FF_speed_percentiles.append(FF_percentiles)

    ### time ###
    avg_time_percentiles = []
    FF_time_percentiles = []

    steps = 100
    eval_time = np.linspace(0, TT, steps)
    for en, n in enumerate(pick_neurons):
        covariates = [
            pref_hd_omega[en, len(w_arr) // 2] * np.ones(steps),
            0.0 * np.ones(steps),
            0.0 * np.ones(steps),
            (left_x + right_x) / 2.0 * np.ones(steps),
            (bottom_y + top_y) / 2.0 * np.ones(steps),
            eval_time,
        ]

        P_mc = utils.compute_UCM_P_count(full_model, covariates, [n], MC=MC)[
            :, 0, ...
        ].cpu()

        avg = (x_counts[None, None, :] * P_mc).sum(-1)
        xcvar = (x_counts[None, None, :] ** 2 * P_mc).sum(-1) - avg**2
        ff = xcvar / avg

        avgs = nprb.utils.stats.percentiles_from_samples(
            avg,
            percentiles=[0.05, 0.5, 0.95],
            smooth_length=5,
            padding_mode="replicate",
        )
        avg_percentiles = [cs_.cpu().numpy() for cs_ in avgs]

        ffs = nprb.utils.stats.percentiles_from_samples(
            ff, percentiles=[0.05, 0.5, 0.95], smooth_length=5, padding_mode="replicate"
        )
        FF_percentiles = [cs_.cpu().numpy() for cs_ in ffs]

        avg_time_percentiles.append(avg_percentiles)
        FF_time_percentiles.append(FF_percentiles)

    ### pos ###
    grid_shape_pos = [[left_x, right_x], [bottom_y, top_y]]
    H = grid_shape_pos[1][1] - grid_shape_pos[1][0]
    W = grid_shape_pos[0][1] - grid_shape_pos[0][0]
    grid_size_pos = (int(41 * W / H), 41)
    grid_pos = {"size": grid_size_pos, "shape": grid_shape_pos}

    steps = np.product(grid_size_pos)
    A, B = grid_size_pos

    avg_pos = []
    FF_pos = []
    for en, n in enumerate(pick_neurons):
        covariates = [
            pref_hd_omega[en, len(w_arr) // 2] * np.ones(steps),
            0.0 * np.ones(steps),
            0.0 * np.ones(steps),
            np.linspace(left_x, right_x, A)[:, None].repeat(B, axis=1).flatten(),
            np.linspace(bottom_y, top_y, B)[None, :].repeat(A, axis=0).flatten(),
            t * np.ones(steps),
        ]

        P_mc = utils.compute_UCM_P_count(full_model, covariates, [n], MC=MC_)[
            :, 0, ...
        ].cpu()
        avg = (x_counts[None, None, :] * P_mc).sum(-1).reshape(-1, A, B).numpy()
        var = (x_counts[None, None, :] ** 2 * P_mc).sum(-1).reshape(-1, A, B).numpy()
        xcvar = var - avg**2

        avg_pos.append(avg.mean(0))
        FF_pos.append((xcvar / (avg + 1e-12)).mean(0))

    avg_pos = np.stack(avg_pos)
    FF_pos = np.stack(FF_pos)

    conditional_tunings = {
        "eval_hd": eval_hd,
        "avg_hd_percentiles": avg_hd_percentiles,
        "FF_hd_percentiles": FF_hd_percentiles,
        "eval_omega": eval_omega,
        "avg_omega_percentiles": avg_omega_percentiles,
        "FF_omega_percentiles": FF_omega_percentiles,
        "eval_speed": eval_speed,
        "avg_speed_percentiles": avg_speed_percentiles,
        "FF_speed_percentiles": FF_speed_percentiles,
        "eval_time": eval_time,
        "avg_time_percentiles": avg_time_percentiles,
        "FF_time_percentiles": FF_time_percentiles,
        "grid_pos": grid_pos,
        "avg_pos": avg_pos,
        "FF_pos": FF_pos,
    }

    # total dictionary
    tunings_dict = {
        "marginal_tunings": marginal_tunings,
        "conditional_tunings": conditional_tunings,
        "joint_tunings": joint_tunings,
    }

    return tunings_dict


def main():
    ### parser ###
    parser = argparse.ArgumentParser(
        usage="%(prog)s [OPTION] [FILE]...",
        description="Analysis of th1 regression models.",
    )
    parser.add_argument(
        "-v", "--version", action="version", version=f"{parser.prog} version 1.0.0"
    )

    parser.add_argument("--seed", default=123, type=int)
    parser.add_argument("--savedir", default="../output/", type=str)
    parser.add_argument("--datadir", default="../../data/", type=str)
    parser.add_argument("--checkpointdir", default="../checkpoint/", type=str)

    parser.add_argument("--batch_size", default=5000, type=int)
    
    parser.add_argument("--gpu", default=0, type=int)
    parser.add_argument("--cpu", dest="cpu", action="store_true")
    parser.set_defaults(cpu=False)

    args = parser.parse_args()

    ### setup ###
    save_dir = args.savedir
    data_path = args.datadir
    checkpoint_dir = args.checkpointdir

    if args.cpu:
        device = "cpu"
    else:
        device = nprb.inference.get_device(gpu=args.gpu)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    rng = np.random.default_rng(args.seed)
    batch_info = args.batch_size

    ### names ###
    subset_config_names = [
        "th1_U-el-3_svgp-64_X[hd]_Z[]_40K11_0d0_10f",
        "th1_U-el-3_svgp-64_X[hd-omega-speed-time]_Z[]_40K11_0d0_10f",
        "th1_U-el-3_svgp-64_X[hd-omega-speed-x-y-time]_Z[]_40K11_0d0_10f",
    ]

    reg_config_names = [
        "th1_IP-exp_svgp-64_X[hd-omega-speed-x-y-time]_Z[]_40K11_0d0_10f",
        "th1_hNB-exp_svgp-64_X[hd-omega-speed-x-y-time]_Z[]_40K11_0d0_10f",
        "th1_U-el-3_svgp-64_X[hd-omega-speed-x-y-time]_Z[]_40K11_0d0_10f",
    ]

    binnings = [20, 40, 100, 200, 500]
    binning_config_names = [
        "th1_U-el-3_svgp-64_X[hd-omega-speed-x-y-time]_Z[]_20K6_0d0_10f",
        "th1_U-el-3_svgp-64_X[hd-omega-speed-x-y-time]_Z[]_40K11_0d0_10f",
        "th1_U-el-3_svgp-64_X[hd-omega-speed-x-y-time]_Z[]_100K25_0d0_10f",
        "th1_U-el-3_svgp-64_X[hd-omega-speed-x-y-time]_Z[]_200K48_0d0_10f",
        "th1_U-el-3_svgp-64_X[hd-omega-speed-x-y-time]_Z[]_500K25_0d0_10f",
    ]

    tuning_model_name = (
        "th1_U-el-3_svgp-64_X[hd-omega-speed-x-y-time]_Z[]_40K11_0d0_10f-1"
    )

    ### load dataset ###
    data_type = "th1"
    bin_size = 40

    dataset_dict = models.get_dataset(data_type, bin_size, data_path)
    neuron_regions = dataset_dict["metainfo"]["neuron_regions"]

    ### analysis ###
    regression_dict = regression(
        checkpoint_dir, reg_config_names, subset_config_names, dataset_dict, rng, batch_info, device
    )
    binning_dict = binning_variability(
        checkpoint_dir, binning_config_names, binnings, dataset_dict, batch_info, device
    )
    tunings_dict = tunings(checkpoint_dir, tuning_model_name, dataset_dict, batch_info, device)

    ### export ###
    data_run = {
        "neuron_regions": neuron_regions,
        "regression": regression_dict,
        "binning_dict": binning_dict,
        "tunings": tunings_dict,
    }

    pickle.dump(data_run, open(save_dir + "th1_RG_results.p", "wb"))


if __name__ == "__main__":
    main()
