import argparse

import os
import pickle

import sys

import numpy as np
import torch

sys.path.append("../..")  # access to library
import neuroprob as nprb

sys.path.append("..")  # access to scripts
import models

import utils


def regression(
    checkpoint_dir, config_names, data_path, data_type, dataset_dict, device
):
    tbin = dataset_dict["tbin"]
    max_count = dataset_dict["max_count"]
    neurons = dataset_dict["neurons"]
    pick_neurons = list(range(neurons))

    ### cross-validation of regression models ###
    kcvs = [2, 5, 8]  # validation sets chosen in 10-fold split of data
    batch_info = 5000  # batch size for cross-validation

    RG_cv_ll = []
    for name in config_names:
        for kcv in kcvs:
            config_name = name + str(kcv)

            full_model, training_loss, fit_dict, val_dict = models.load_model(
                config_name,
                checkpoint_dir,
                dataset_dict,
                batch_info,
                device,
            )

            RG_cv_ll.append(
                models.RG_pred_ll(
                    full_model,
                    val_dict,
                    neuron_group=None,
                    ll_mode="GH",
                    ll_samples=100,
                    cov_samples=1,
                    beta=0.0,
                )
            )

    RG_cv_ll = np.array(RG_cv_ll).reshape(len(config_names), len(kcvs))

    ### tuning curves of ground truth model ###
    hCMP = np.load(data_path + "hCMP1.npz")
    hd = hCMP["covariates"][:, 0]

    gt_lamb = np.maximum(hCMP["gt_lamb"], 1e-12)  # avoid divergence at 0
    gt_nu = hCMP["gt_nu"]

    gt_mean = nprb.utils.stats.cmp_moments(1, gt_lamb, gt_nu, tbin, J=10000)
    gt_var = (
        nprb.utils.stats.cmp_moments(2, gt_lamb, gt_nu, tbin, J=10000) - gt_mean**2
    )
    gt_FF = gt_var / (gt_mean + 1e-12)

    ### compute UCM SCDs ###
    x_counts = torch.arange(max_count + 1)

    U_gp_config_name = config_names[2] + "-1"
    batch_info = 5000

    full_model, training_loss, fit_dict, val_dict = models.load_model(
        U_gp_config_name,
        checkpoint_dir,
        dataset_dict,
        batch_info,
        device,
    )

    covariates = [torch.from_numpy(hd)]

    P_mc = nprb.utils.model.compute_UCM_P_count(
        full_model.mapping, full_model.likelihood, covariates, pick_neurons, MC=1000
    )
    P_rg = P_mc.mean(0).cpu().numpy()

    # count distributions
    ref_prob = []
    hd = [20, 50, 80]
    for hd_ in hd:
        for n in range(len(pick_neurons)):
            ref_prob.append(
                [
                    nprb.utils.stats.cmp_count_prob(
                        xc, gt_lamb[n, hd_], gt_nu[n, hd_], tbin
                    )
                    for xc in x_counts.numpy()
                ]
            )
    ref_prob = np.array(ref_prob).reshape(len(hd), len(pick_neurons), -1)

    cs = nprb.utils.stats.percentiles_from_samples(
        P_mc[..., hd, :], percentiles=[0.05, 0.5, 0.95], smooth_length=1
    )
    cnt_percentiles = [cs_.cpu().numpy() for cs_ in cs]

    # tuning curves
    avg = (x_counts[None, None, None, :] * P_mc.cpu()).sum(-1)
    xcvar = (x_counts[None, None, None, :] ** 2 * P_mc.cpu()).sum(-1) - avg**2
    ff = xcvar / avg

    avgs = nprb.utils.stats.percentiles_from_samples(
        avg, percentiles=[0.05, 0.5, 0.95], smooth_length=5, padding_mode="circular"
    )
    avg_percentiles = [cs_.cpu().numpy() for cs_ in avgs]

    ffs = nprb.utils.stats.percentiles_from_samples(
        ff, percentiles=[0.05, 0.5, 0.95], smooth_length=5, padding_mode="circular"
    )
    FF_percentiles = [cs_.cpu().numpy() for cs_ in ffs]

    regression_dict = {
        "RG_cv_ll": RG_cv_ll,
        "hd": hd,
        "P_rg": P_rg,
        "gt_mean": gt_mean,
        "gt_FF": gt_FF,
        "ref_prob": ref_prob,
        "cnt_percentiles": cnt_percentiles,
        "avg_percentiles": avg_percentiles,
        "FF_percentiles": FF_percentiles,
    }

    return regression_dict



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



def variability_stats(checkpoint_dir, config_names, dataset_dict, device):
    tbin = dataset_dict["tbin"]
    neurons = dataset_dict["neurons"]
    pick_neurons = list(range(neurons))

    ### KS framework ###
    Qq_rg = []
    Zz_rg = []

    kcvs = [2, 5, 8]  # validation sets chosen in 10-fold split of data
    batch_info = 5000  # batch size for cross-validation

    RG_cv_ll = []
    for en, name in enumerate(config_names):
        for kcv in kcvs:
            config_name = name + str(kcv)

            full_model, training_loss, fit_dict, val_dict = models.load_model(
                config_name,
                checkpoint_dir,
                dataset_dict,
                batch_info,
                device,
            )

            if en > 1:
                # predictive posterior
                P_mc = nprb.utils.model.compute_UCM_P_count(
                    full_model.mapping,
                    full_model.likelihood,
                    pick_neurons,
                    None,
                    cov_samples=10,
                    ll_samples=1,
                    tr=0,
                )
                P = P_mc.mean(0).cpu().numpy()

                q_ = []
                Z_ = []
                for n in range(len(pick_neurons)):
                    spike_binned = full_model.likelihood.spikes[0][
                        0, pick_neurons[n], :
                    ].numpy()
                    q, Z = utils.get_q_Z(P[n, ...], spike_binned, deq_noise=None)
                    q_.append(q)
                    Z_.append(Z)

            elif en < 2:
                fit_train = fit_dict["spiketrain"]
                fit_covs = fit_dict["covariates"]
                time_steps = fit_train.shape[-1]

                q_ = utils.compute_count_stats(
                    full_model,
                    mode[1],
                    tbin,
                    fit_train,
                    fit_covs,
                    pick_neurons,
                    traj_len=1,
                    start=0,
                    T=time_steps,
                    bs=5000,
                )
                Z_ = [nprb.utils.stats.q_to_Z(q) for q in q_]

            Qq_rg.append(q_)
            Zz_rg.append(Z_)

    q_DS_rg = []
    T_DS_rg = []
    T_KS_rg = []
    for q in Qq_rg:
        for qq in q:
            (
                T_DS,
                T_KS,
                sign_DS,
                sign_KS,
                p_DS,
                p_KS,
            ) = nprb.utils.stats.KS_DS_statistics(qq, alpha=0.05, alpha_s=0.05)
            T_DS_rg.append(T_DS)
            T_KS_rg.append(T_KS)

            Z_DS = T_DS * np.sqrt((qq.shape[0] - 1) / 2)
            q_DS_rg.append(nprb.utils.stats.Z_to_q(Z_DS))

    q_DS_rg = np.array(q_DS_rg).reshape(len(CV), len(M), -1)
    T_DS_rg = np.array(T_DS_rg).reshape(len(CV), len(M), -1)
    T_KS_rg = np.array(T_KS_rg).reshape(len(CV), len(M), -1)

    dispersion_dict = {
        "q_DS": q_DS_rg,
        "T_DS": T_DS_rg,
        "T_KS": T_KS_rg,
        "significance_DS": sign_DS,
        "significance_KS": sign_KS,
        "quantiles": Qq_rg,
        "Z_scores": Zz_rg,
    }

    return dispersion_dict


def latent_variable(checkpoint_dir, config_names, dataset_dict, device):

    ### aligning trajectory and computing RMS for different models ###
    cvK = 90
    CV = [15, 30, 45, 60, 75]

    batch_info = 5000

    RMS_cv = []
    for name in config_names:
        config_name = name + "-1"

        full_model, training_loss, fit_dict, val_dict = models.load_model(
            config_name,
            checkpoint_dir,
            dataset_dict,
            batch_info,
            device,
        )

        X_loc, X_std = full_model.inputs.eval_XZ()
        cvT = X_loc[0].shape[0]
        tar_t = rhd_t[:cvT]
        lat = X_loc[0]

        for rn in CV:
            eval_range = np.arange(cvT // cvK) + rn * cvT // cvK

            _, shift, sign, _, _ = utils.signed_scaled_shift(
                lat[eval_range],
                tar_t[eval_range],
                topology="ring",
                dev=dev,
                learn_scale=False,
            )

            mask = np.ones((cvT,), dtype=bool)
            mask[eval_range] = False

            lat_t = torch.tensor((sign * lat + shift) % (2 * np.pi))
            D = (
                utils.latent.metric(torch.tensor(tar_t)[mask], lat_t[mask], topology)
                ** 2
            )
            RMS_cv.append(np.sqrt(D.mean().item()))

    RMS_cv = np.array(RMS_cv).reshape(len(lat_config_names), len(CV))

    # neuron subgroup likelihood CV for latent models
    seeds = [123, 1234, 12345]

    n_group = np.arange(5)
    kcvs = [2, 5, 8]  # validation sets chosen in 10-fold split of data
    val_neuron = [n_group, n_group + 10, n_group + 20, n_group + 30, n_group + 40]

    batch_info = 5000

    LVM_cv_ll = []
    for name in lat_config_names:
        for kcv in kcvs:
            config_name = name + str(kcv)

            full_model, training_loss, fit_dict, val_dict = models.load_model(
                config_name,
                checkpoint_dir,
                dataset_dict,
                batch_info,
                device,
            )

            for v_neuron in val_neuron:

                prev_ll = np.inf
                for seed in seeds:  # pick best fit with different seeds
                    full_model = get_full_model(
                        datatype,
                        cvdata,
                        resamples,
                        rc_t,
                        100,
                        mode,
                        rcov,
                        max_count,
                        neurons,
                    )

                    mask = np.ones((neurons,), dtype=bool)
                    mask[v_neuron] = False
                    f_neuron = np.arange(neurons)[mask]

                    ll = models.LVM_pred_ll(
                        full_model,
                        mode[-5],
                        mode[2],
                        models.cov_used,
                        cv_set,
                        f_neuron,
                        v_neuron,
                        beta=0.0,
                        beta_z=0.0,
                        max_iters=3000,
                    )[0]

                    if ll < prev_ll:
                        prev_ll = ll

                LVM_cv_ll.append(prev_ll)

    LVM_cv_ll = np.array(LVM_cv_ll).reshape(
        len(lat_config_names), len(kcvs), len(val_neuron)
    )

    ### compute tuning curves and latent trajectory of latent UCM ###
    lat_t_ = []
    lat_std_ = []
    P_ = []

    comp_grate = []
    comp_gdisp = []
    comp_gFF = []
    comp_gvar = []

    comp_avg = []
    comp_ff = []
    comp_var = []

    for name in config_names[-2:]:
        config_name = name + "-1"

        full_model, training_loss, fit_dict, val_dict = models.load_model(
            config_name,
            checkpoint_dir,
            dataset_dict,
            batch_info,
            device,
        )

        # predict latents
        X_loc, X_std = full_model.inputs.eval_XZ()
        cvT = X_loc[0].shape[0]

        lat_t, shift, sign, _, _ = utils.latent.signed_scaled_shift(
            X_loc[0], rhd_t[:cvT], dev, learn_scale=False
        )
        lat_t_.append(utils.signal.WrapPi(lat_t, True))
        lat_std_.append(X_std[0])

        # P
        steps = 100
        covariates_aligned = [
            (sign * (np.linspace(0, 2 * np.pi, steps) - shift)) % (2 * np.pi)
        ]
        P_mc = model_utils.compute_P(
            full_model, covariates_aligned, use_neuron, MC=1000
        ).cpu()

        x_counts = torch.arange(max_count + 1)
        avg = (x_counts[None, None, None, :] * P_mc).sum(-1)
        xcvar = (x_counts[None, None, None, :] ** 2 * P_mc).sum(-1) - avg**2
        ff = xcvar / (avg + 1e-12)

        avgs = utils.signal.percentiles_from_samples(
            avg, percentiles=[0.05, 0.5, 0.95], smooth_length=5, padding_mode="circular"
        )
        comp_avg.append([cs_.cpu().numpy() for cs_ in avgs])

        ffs = utils.signal.percentiles_from_samples(
            ff, percentiles=[0.05, 0.5, 0.95], smooth_length=5, padding_mode="circular"
        )
        comp_ff.append([cs_.cpu().numpy() for cs_ in ffs])

    latent_dict = {
        "covariates_aligned": covariates_aligned,
        "lat_t_": lat_t_,
        "lat_std_": lat_std_,
        "comp_avg": comp_avg,
        "comp_ff": comp_ff,
        "LVM_cv_ll": LVM_cv_ll,
        "RMS_cv": RMS_cv,
    }

    return latent_dict


def main():
    ### parser ###
    parser = argparse.ArgumentParser(
        usage="%(prog)s [OPTION] [FILE]...",
        description="Analysis of heteroscedastic CMP dataset.",
    )
    parser.add_argument(
        "-v", "--version", action="version", version=f"{parser.prog} version 1.0.0"
    )

    parser.add_argument("--dataseed", default=1, type=int)
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
    data_type = "hCMP{}".format(args.dataseed)

    reg_config_names = [
        data_type + "_IP-exp_svgp-8_X[hd]_Z[]_1K18_0d0_10f",
        data_type + "_hNB-exp_svgp-8_X[hd]_Z[]_1K18_0d0_10f",
        data_type + "_U-el-3_svgp-8_X[hd]_Z[]_1K18_0d0_10f",
        data_type + "_U-el-3_ffnn-50-50-100_X[hd]_Z[]_1K18_0d0_10f",
    ]

    lat_config_names = [
        data_type + "_IP-exp_svgp-8_X[]_Z[T1]_1K18_0d0_10f",
        data_type + "_hNB-exp_svgp-8_X[]_Z[T1]_1K18_0d0_10f",
        data_type + "_U-el-3_svgp-8_X[]_Z[T1]_1K18_0d0_10f",
        data_type + "_U-el-3_ffnn-50-50-100_X[]_Z[T1]_1K18_0d0_10f",
    ]

    ### load dataset ###
    bin_size = 1
    dataset_dict = models.get_dataset(data_type, bin_size, data_path)

    ### analysis ###
    regression_dict = regression(
        checkpoint_dir, reg_config_names, data_path, data_type, dataset_dict, dev
    )
    variability_dict = variability_stats(
        checkpoint_dir, reg_config_names, dataset_dict, dev
    )
    latent_dict = latent_variable(checkpoint_dir, lat_config_names, dataset_dict, dev)

    ### export ###
    data_run = {
        "regression": regression_dict,
        "variability": variability_dict,
        "latent": latent_dict,
    }

    pickle.dump(data_run, open(save_dir + "hCMP_results.p", "wb"))


if __name__ == "__main__":
    main()
