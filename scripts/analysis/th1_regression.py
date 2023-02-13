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


def variability_stats(config_names):
    left_x = rcov[3].min()
    right_x = rcov[3].max()
    bottom_y = rcov[4].min()
    top_y = rcov[4].max()

    pick_neuron = list(range(neurons))

    ### statistics over the behaviour ###
    avg_models = []
    var_models = []
    ff_models = []

    for bn in binnings:

        rcov, neurons, tbin, resamples, rc_t, region_edge = HDC.get_dataset(
            session_id, phase, bn, "../scripts/data"
        )
        max_count = int(rc_t.max())
        x_counts = torch.arange(max_count + 1)

        mode = modes[2]
        cvdata = model_utils.get_cv_sets(mode, [2], 5000, rc_t, resamples, rcov)[0]
        full_model = get_full_model(
            session_id,
            phase,
            cvdata,
            resamples,
            bn,
            mode,
            rcov,
            max_count,
            neurons,
            gpu=gpu_dev,
        )

        avg_model = []
        var_model = []
        ff_model = []

        for b in range(full_model.inputs.batches):
            P_mc = utils.compute_UCM_P_countred_P(
                full_model, b, pick_neuron, None, cov_samples=10, ll_samples=1, tr=0
            ).cpu()

            avg = (x_counts[None, None, None, :] * P_mc).sum(-1)
            var = (x_counts[None, None, None, :] ** 2 * P_mc).sum(-1) - avg**2
            ff = var / (avg + 1e-12)
            avg_model.append(avg)
            var_model.append(var)
            ff_model.append(ff)

        avg_models.append(torch.cat(avg_model, dim=-1).mean(0).numpy())
        var_models.append(torch.cat(var_model, dim=-1).mean(0).numpy())
        ff_models.append(torch.cat(ff_model, dim=-1).mean(0).numpy())

    # compute the Pearson correlation between Fano factors and mean firing rates
    b = 1
    Pearson_ff = []
    ratio = []
    for avg, ff in zip(avg_models[b], ff_models[b]):
        r, r_p = scstats.pearsonr(ff, avg)  # Pearson r correlation test
        Pearson_ff.append((r, r_p))
        ratio.append(ff.std() / avg.std())

    # KS framework for regression models
    CV = [2, 5, 8]

    ### KS test over binnings ###
    Qq_bn = []
    Zz_bn = []
    R_bn = []
    Rp_bn = []
    mode = modes[2]

    N = len(pick_neuron)
    for kcv in CV:
        for en, bn in enumerate(binnings):
            cvdata = model_utils.get_cv_sets(mode, [kcv], 3000, rc_t, resamples, rcov)[
                0
            ]
            _, ftrain, fcov, vtrain, vcov, cvbatch_size = cvdata
            cv_set = (ftrain, fcov, vtrain, vcov)
            time_steps = ftrain.shape[-1]

            full_model = get_full_model(
                session_id,
                phase,
                cvdata,
                resamples,
                bn,
                mode,
                rcov,
                max_count,
                neurons,
                gpu=gpu_dev,
            )

            if en == 0:
                q_ = []
                Z_ = []
                for b in range(full_model.inputs.batches):  # predictive posterior
                    P_mc = nprb.utils.model.compute_UCM_P_count(
                        full_model,
                        b,
                        pick_neuron,
                        None,
                        cov_samples=10,
                        ll_samples=1,
                        tr=0,
                    )
                    P = P_mc.mean(0).cpu().numpy()

                    for n in range(N):
                        spike_binned = full_model.likelihood.spikes[b][
                            0, pick_neuron[n], :
                        ].numpy()
                        q, Z = model_utils.get_q_Z(
                            P[n, ...], spike_binned, deq_noise=None
                        )
                        q_.append(q)
                        Z_.append(Z)

                q = []
                Z = []
                for n in range(N):
                    q.append(np.concatenate(q_[n::N]))
                    Z.append(np.concatenate(Z_[n::N]))

            elif en > 0:
                cov_used = models.cov_used(mode[2], fcov)
                q = model_utils.compute_count_stats(
                    full_model,
                    mode[1],
                    tbin,
                    ftrain,
                    cov_used,
                    pick_neuron,
                    traj_len=1,
                    start=0,
                    T=time_steps,
                    bs=5000,
                )
                Z = [utils.stats.q_to_Z(q_) for q_ in q]

            Pearson_s = []
            for n in range(len(pick_neuron)):
                for m in range(n + 1, len(pick_neuron)):
                    r, r_p = scstats.pearsonr(Z[n], Z[m])  # Pearson r correlation test
                    Pearson_s.append((r, r_p))

            r = np.array([p[0] for p in Pearson_s])
            r_p = np.array([p[1] for p in Pearson_s])

            Qq_bn.append(q)
            Zz_bn.append(Z)
            R_bn.append(r)
            Rp_bn.append(r_p)

    q_DS_bn = []
    T_DS_bn = []
    T_KS_bn = []
    for q in Qq_bn:
        for qq in q:
            (
                T_DS,
                T_KS,
                sign_DS,
                sign_KS,
                p_DS,
                p_KS,
            ) = nprb.utils.stats.KS_DS_statistics(qq, alpha=0.05, alpha_s=0.05)
            T_DS_ll.append(T_DS)
            T_KS_ll.append(T_KS)

            Z_DS = T_DS / np.sqrt(2 / (qq.shape[0] - 1))
            q_DS_ll.append(utils.stats.Z_to_q(Z_DS))

    Qq_bn = np.array(Qq_bn).reshape(len(CV), len(binnings), -1)
    Zz_bn = np.array(Zz_bn).reshape(len(CV), len(binnings), -1)
    R_bn = np.array(R_bn).reshape(len(CV), len(binnings), -1)
    Rp_bn = np.array(Rp_bn).reshape(len(CV), len(binnings), -1)

    q_DS_bn = np.array(q_DS_bn).reshape(len(CV), len(binnings), -1)
    T_DS_bn = np.array(T_DS_bn).reshape(len(CV), len(binnings), -1)
    T_KS_bn = np.array(T_KS_bn).reshape(len(CV), len(binnings), -1)

    variability_dict = {
        "avg_models": avg_models,
        "var_models": var_models,
        "ff_models": ff_models,
        "Pearson_ff": Pearson_ff,
        "ratio": ratio,
    }

    return variability_dict


def regression():

    max_count = int(rc_t.max())
    x_counts = torch.arange(max_count + 1)

    HD_offset = (
        -1.0
    )  # global shift of head direction coordinates, makes plots better as the preferred head directions are not at axis lines

    # cross validation
    PLL_rg_ll = []
    PLL_rg_cov = []
    kcvs = [1, 2, 3, 5, 6, 8]  # validation segments from splitting data into 10

    beta = 0.0
    batchsize = 5000

    PLL_rg_ll = []
    Ms = modes[2:5]
    for mode in Ms:  # likelihood

        for cvdata in model_utils.get_cv_sets(
            mode, kcvs, batchsize, rc_t, resamples, rcov
        ):
            _, ftrain, fcov, vtrain, vcov, cvbatch_size = cvdata
            cv_set = (ftrain, fcov, vtrain, vcov)

            full_model = get_full_model(
                session_id,
                phase,
                cvdata,
                resamples,
                bn,
                mode,
                rcov,
                max_count,
                neurons,
                gpu=gpu_dev,
            )
            PLL_rg_ll.append(
                nprb.utils.model.RG_pred_ll(
                    full_model,
                    mode[2],
                    models.cov_used,
                    cv_set,
                    bound="ELBO",
                    beta=beta,
                    neuron_group=None,
                    ll_mode="GH",
                    ll_samples=100,
                )
            )

    PLL_rg_ll = np.array(PLL_rg_ll).reshape(len(Ms), len(kcvs))

    CV = [2, 5, 8]  # validation segments from splitting data into 10

    ### KS test ###
    Qq_ll = []
    Zz_ll = []
    R_ll = []
    Rp_ll = []

    batch_size = 3000
    N = len(pick_neuron)
    for kcv in CV:
        for en, mode in enumerate(Ms):
            cvdata = model_utils.get_cv_sets(
                mode, [kcv], batch_size, rc_t, resamples, rcov
            )[0]
            _, ftrain, fcov, vtrain, vcov, cvbatch_size = cvdata
            cv_set = (ftrain, fcov, vtrain, vcov)
            time_steps = ftrain.shape[-1]

            full_model = get_full_model(
                session_id,
                phase,
                cvdata,
                resamples,
                bn,
                mode,
                rcov,
                max_count,
                neurons,
                gpu=gpu_dev,
            )

            if en == 0:
                q_ = []
                Z_ = []
                for b in range(full_model.inputs.batches):  # predictive posterior
                    P_mc = nprb.utils.model.compute_UCM_P_count(
                        full_model,
                        b,
                        pick_neuron,
                        None,
                        cov_samples=10,
                        ll_samples=1,
                        tr=0,
                    )
                    P = P_mc.mean(0).cpu().numpy()

                    for n in range(N):
                        spike_binned = full_model.likelihood.spikes[b][
                            0, pick_neuron[n], :
                        ].numpy()
                        q, Z = model_utils.get_q_Z(
                            P[n, ...], spike_binned, deq_noise=None
                        )
                        q_.append(q)
                        Z_.append(Z)

                q = []
                Z = []
                for n in range(N):
                    q.append(np.concatenate(q_[n::N]))
                    Z.append(np.concatenate(Z_[n::N]))

            elif en > 0:
                cov_used = models.cov_used(mode[2], fcov)
                q = utils.compute_count_stats(
                    full_model,
                    mode[1],
                    tbin,
                    ftrain,
                    cov_used,
                    pick_neuron,
                    traj_len=1,
                    start=0,
                    T=time_steps,
                    bs=5000,
                )
                Z = [utils.stats.q_to_Z(q_) for q_ in q]

            Pearson_s = []
            for n in range(len(pick_neuron)):
                for m in range(n + 1, len(pick_neuron)):
                    r, r_p = scstats.pearsonr(Z[n], Z[m])  # Pearson r correlation test
                    Pearson_s.append((r, r_p))

            r = np.array([p[0] for p in Pearson_s])
            r_p = np.array([p[1] for p in Pearson_s])

            Qq_ll.append(q)
            Zz_ll.append(Z)
            R_ll.append(r)
            Rp_ll.append(r_p)

    q_DS_ll = []
    T_DS_ll = []
    T_KS_ll = []
    for q in Qq_ll:
        for qq in q:
            (
                T_DS,
                T_KS,
                sign_DS,
                sign_KS,
                p_DS,
                p_KS,
            ) = nprb.utils.stats.KS_DS_statistics(qq, alpha=0.05, alpha_s=0.05)
            T_DS_ll.append(T_DS)
            T_KS_ll.append(T_KS)

            Z_DS = T_DS / np.sqrt(2 / (qq.shape[0] - 1))
            q_DS_ll.append(utils.stats.Z_to_q(Z_DS))

    Qq_ll = np.array(Qq_ll).reshape(len(CV), len(Ms), -1)
    Zz_ll = np.array(Zz_ll).reshape(len(CV), len(Ms), -1)
    R_ll = np.array(R_ll).reshape(len(CV), len(Ms), -1)
    Rp_ll = np.array(Rp_ll).reshape(len(CV), len(Ms), -1)

    q_DS_ll = np.array(q_DS_ll).reshape(len(CV), len(Ms), -1)
    T_DS_ll = np.array(T_DS_ll).reshape(len(CV), len(Ms), -1)
    T_KS_ll = np.array(T_KS_ll).reshape(len(CV), len(Ms), -1)

    PLL_rg_cov = []
    kcvs = [1, 2, 3, 5, 6, 8]  # validation segments from splitting data into 10

    Ms = modes[:3]
    for mode in Ms:  # input space

        for cvdata in model_utils.get_cv_sets(
            mode, kcvs, batchsize, rc_t, resamples, rcov
        ):
            _, ftrain, fcov, vtrain, vcov, cvbatch_size = cvdata
            cv_set = (ftrain, fcov, vtrain, vcov)

            full_model = get_full_model(
                session_id,
                phase,
                cvdata,
                resamples,
                bn,
                mode,
                rcov,
                max_count,
                neurons,
                gpu=gpu_dev,
            )
            PLL_rg_cov.append(
                nprb.utils.model.RG_pred_ll(
                    full_model,
                    mode[2],
                    models.cov_used,
                    cv_set,
                    bound="ELBO",
                    beta=beta,
                    neuron_group=None,
                    ll_mode="GH",
                    ll_samples=100,
                )
            )

    PLL_rg_cov = np.array(PLL_rg_cov).reshape(len(Ms), len(kcvs))

    regression_dict = {
        "Ell_likelihood": PLL_rg_ll,
        "Ell_subsets": PLL_rg_cov,
        "quantiles": Qq_ll,
        "Z_scores": Zz_ll,
        "correlations": R_ll,
        "corr_significance": Rp_ll,
        "q_DS": q_DS_ll,
        "T_DS": T_DS_ll,
        "T_KS": T_KS_ll,
        "KS_significance": sign_KS,
        "DS_significance": sign_DS,
    }

    return regression_dict


def tunings(model_name):
    # load universal regression model
    checkpoint_dir = "../scripts/checkpoint/"
    config_name = "th1_U-el-3_svgp-64_X[hd-omega-speed-x-y-time]_Z[]_40K11_0d0_10f-1"
    batch_info = 500

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
        pick_neuron,
        MC=MC,
        skip=skip,
    )
    avg = (x_counts[None, None, None, :] * P_tot).sum(-1)
    var = (x_counts[None, None, None, :] ** 2 * P_tot).sum(-1) - avg**2
    ff = var / avg

    avgs = utils.signal.percentiles_from_samples(
        avg, percentiles=[0.05, 0.5, 0.95], smooth_length=5, padding_mode="circular"
    )
    marg_avg_hd_percentiles = [cs_.cpu().numpy() for cs_ in avgs]

    ffs = utils.signal.percentiles_from_samples(
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
        full_model, [covariates_w], [1], rcov, batch_size, pick_neuron, MC=MC, skip=skip
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
    P_tot = model_utils.marginalized_P(
        full_model,
        [np.linspace(0, 30.0, steps)],
        [2],
        rcov,
        batch_size,
        pick_neuron,
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
        pick_neuron,
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
        full_model, cov_list, [3, 4], rcov, 10000, pick_neuron, MC=MC, skip=skip
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
        hd_avg_tf,
        hd_FF_tf,
        omega_avg_tf,
        omega_FF_tf,
        speed_avg_tf,
        speed_FF_tf,
        pos_avg_tf,
        pos_FF_tf,
        time_avg_tf,
        time_FF_tf,
        mhd_mean,
        mhd_ff,
        mw_mean,
        mw_ff,
        ms_mean,
        ms_ff,
        mpos_mean,
        mpos_ff,
        mt_mean,
        mt_ff,
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

    avgs = utils.signal.percentiles_from_samples(
        avg, percentiles=[0.05, 0.5, 0.95], smooth_length=5, padding_mode="circular"
    )
    avg_hd_percentiles = [cs_.cpu().numpy() for cs_ in avgs]

    ffs = utils.signal.percentiles_from_samples(
        ff, percentiles=[0.05, 0.5, 0.95], smooth_length=5, padding_mode="circular"
    )
    FF_hd_percentiles = [cs_.cpu().numpy() for cs_ in ffs]

    ### omega tuning ###
    avg_omega_percentiles = []
    FF_omega_percentiles = []

    steps = 100
    w_edge = (-rcov[1].min() + rcov[1].max()) / 2.0
    eval_omega = np.linspace(-w_edge, w_edge, steps)
    for en, n in enumerate(pick_neuron):
        covariates = [
            pref_hdw[en, len(w_arr) // 2] * np.ones(steps),
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

        avgs = utils.signal.percentiles_from_samples(
            avg,
            percentiles=[0.05, 0.5, 0.95],
            smooth_length=5,
            padding_mode="replicate",
        )
        avg_percentiles = [cs_.cpu().numpy() for cs_ in avgs]

        ffs = utils.signal.percentiles_from_samples(
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
    for en, n in enumerate(pick_neuron):
        covariates = [
            pref_hdw[en, len(w_arr) // 2] * np.ones(steps),
            0.0 * np.ones(steps),
            eval_speed,
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

        avgs = utils.signal.percentiles_from_samples(
            avg,
            percentiles=[0.05, 0.5, 0.95],
            smooth_length=5,
            padding_mode="replicate",
        )
        avg_percentiles = [cs_.cpu().numpy() for cs_ in avgs]

        ffs = utils.signal.percentiles_from_samples(
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
    for en, n in enumerate(pick_neuron):
        covariates = [
            pref_hdw[en, len(w_arr) // 2] * np.ones(steps),
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

        avgs = utils.signal.percentiles_from_samples(
            avg,
            percentiles=[0.05, 0.5, 0.95],
            smooth_length=5,
            padding_mode="replicate",
        )
        avg_percentiles = [cs_.cpu().numpy() for cs_ in avgs]

        ffs = utils.signal.percentiles_from_samples(
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

    steps = np.product(grid_size_pos)
    A, B = grid_size_pos

    avg_pos = []
    FF_pos = []
    for en, n in enumerate(pick_neuron):
        covariates = [
            pref_hdw[en, len(w_arr) // 2] * np.ones(steps),
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
        eval_hd,
        avg_hd_percentiles,
        FF_hd_percentiles,
        eval_omega,
        avg_omega_percentiles,
        FF_omega_percentiles,
        eval_speed,
        avg_speed_percentiles,
        FF_speed_percentiles,
        eval_time,
        avg_time_percentiles,
        FF_time_percentiles,
        grid_size_pos,
        grid_shape_pos,
        avg_pos,
        FF_pos,
    }
    
    
    
    # special joint tuning curves
    joint_tunings = {
        grid_size_hdw,
        grid_shape_hdw,
        field_hdw,
        grid_size_hdt,
        grid_shape_hdt,
        field_hdt,
        pref_hdw,
        ATI,
        res_var,
        pref_hdt,
        drift,
        res_var_drift,
        tun_width,
        amp_t,
        ampm_t,
        sim_mat,
    }
    

    tunings_dict = {
        marginal_tunings, 
        conditional_tunings, 
        joint_tunings, 
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

    parser.add_argument("--savedir", default="../output/", type=str)
    parser.add_argument("--datadir", default="../../data/", type=str)
    parser.add_argument("--checkpointdir", default="../checkpoint/", type=str)

    parser.add_argument("--gpu", default=0, type=int)
    parser.add_argument("--cpu", dest="cpu", action="store_true")
    parser.set_defaults(cpu=False)

    args = parser.parse_args()

    ### setup ###
    save_dir = args.savedir
    data_path = args.datadir
    checkpoint_dir = args.checkpointdir

    if args.cpu:
        dev = "cpu"
    else:
        dev = nprb.inference.get_device(gpu=args.gpu)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    ### names ###
    modes = [
        ("GP", "U", "hd", 8, "identity", 3, [], False, 10, False, "ew"),
        ("GP", "U", "hd_w_s_t", 48, "identity", 3, [], False, 10, False, "ew"),
        ("GP", "U", "hd_w_s_pos_t", 64, "identity", 3, [], False, 10, False, "ew"),
        ("GP", "IP", "hd_w_s_pos_t", 64, "exp", 1, [], False, 10, False, "ew"),
        ("GP", "hNB", "hd_w_s_pos_t", 64, "exp", 1, [], False, 10, False, "ew"),
        ("GP", "U", "hd_w_s_pos_t", 64, "identity", 3, [], False, 10, False, "qd"),
    ]

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

    ### load dataset ###
    data_path = "../data/"
    data_type = "th1"
    bin_size = 40

    dataset_dict = models.get_dataset(data_type, bin_size, data_path)

    ### analysis ###
    regression_dict = regression()
    variability_dict = variability_stats()
    tunings_dict = tunings()

    ### export ###
    data_run = {
        "regression": regression_dict,
        "variability": variability_dict,
        "tunings": tunings_dict,
    }

    pickle.dump(data_run, open("./saves/th1_RG.p", "wb"))


if __name__ == "__main__":
    main()
