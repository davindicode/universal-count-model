import argparse

import os
import pickle

import sys

import numpy as np
import torch

import neuroprob as nprb

sys.path.append("..")  # access to scripts
import models

import utils


def regression(checkpoint_dir, config_names, dataset_dict, gt_hCMP, batch_info, device):
    tbin = dataset_dict["tbin"]
    max_count = dataset_dict["max_count"]
    neurons = dataset_dict["neurons"]
    pick_neurons = list(range(neurons))

    ### cross-validation of regression models ###
    kcvs = [2, 5, 8]  # validation sets chosen in 10-fold split of data

    RG_cv_ll = []
    for name in config_names:
        for kcv in kcvs:
            config_name = name + str(kcv)

            full_model, _, _, val_dict = models.load_model(
                config_name,
                checkpoint_dir,
                dataset_dict,
                batch_info,
                device,
            )

            RG_cv_ll.append(
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

    RG_cv_ll = np.array(RG_cv_ll).reshape(len(config_names), len(kcvs))

    ### tuning curves of ground truth model ###
    hd = gt_hCMP["gt_covariates"][:, 0]

    gt_lamb = gt_hCMP["gt_lamb"]
    gt_nu = gt_hCMP["gt_nu"]

    gt_mean = nprb.utils.stats.cmp_moments(1, gt_lamb, gt_nu, tbin, J=10000)
    gt_var = (
        nprb.utils.stats.cmp_moments(2, gt_lamb, gt_nu, tbin, J=10000) - gt_mean**2
    )
    gt_FF = gt_var / (gt_mean + 1e-12)

    ### compute UCM SCDs ###
    x_counts = torch.arange(max_count + 1)

    U_gp_config_name = config_names[2] + "-1"

    full_model, _, _, _ = models.load_model(
        U_gp_config_name,
        checkpoint_dir,
        dataset_dict,
        batch_info,
        device,
    )

    covariates = [torch.from_numpy(hd)]

    with torch.no_grad():
        P_mc = nprb.utils.model.compute_UCM_P_count(
            full_model.mapping, full_model.likelihood, covariates, pick_neurons, MC=1000
        )
    P_rg = P_mc.mean(0).cpu().numpy()

    # count distributions
    ref_prob = []
    eval_hd_inds = [20, 50, 80]
    for hd_ind in eval_hd_inds:
        for n in range(len(pick_neurons)):
            ref_prob.append(
                [
                    nprb.utils.stats.cmp_count_prob(
                        xc, gt_lamb[n, hd_ind], gt_nu[n, hd_ind], tbin
                    )
                    for xc in x_counts.numpy()
                ]
            )
    ref_prob = np.array(ref_prob).reshape(len(eval_hd_inds), len(pick_neurons), -1)

    cs = nprb.utils.stats.percentiles_from_samples(
        P_mc[..., eval_hd_inds, :], percentiles=[0.05, 0.5, 0.95], smooth_length=1
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
        "covariates_hd": hd,
        "eval_hd_inds": eval_hd_inds,
        "UCM_P_count": P_rg,
        "gt_mean": gt_mean,
        "gt_FF": gt_FF,
        "gt_P_count": ref_prob,
        "cnt_percentiles": cnt_percentiles,
        "avg_percentiles": avg_percentiles,
        "FF_percentiles": FF_percentiles,
    }

    return regression_dict


def variability_stats(checkpoint_dir, config_names, dataset_dict, rng, batch_info, device):
    tbin = dataset_dict["tbin"]
    max_count = dataset_dict["max_count"]
    neurons = dataset_dict["neurons"]
    pick_neurons = list(range(neurons))

    ### KS framework ###
    Qq, Zz = [], []

    kcvs = [2, 5, 8]  # validation sets chosen in 10-fold split of data

    RG_cv_ll = []
    for name in config_names:
        for kcv in kcvs:
            config_name = name + str(kcv)

            full_model, _, _, _ = models.load_model(
                config_name,
                checkpoint_dir,
                dataset_dict,
                batch_info,
                device,
            )

            P = []
            with torch.no_grad():
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
            ) = nprb.utils.stats.KS_DS_statistics(qq, alpha=0.05)
            T_DS.append(T_DS_)
            T_KS.append(T_KS_)

            Z_DS = T_DS_ * np.sqrt((qq.shape[0] - 1) / 2)
            q_DS.append(nprb.utils.stats.quantile_Z_mapping(Z_DS, inverse=True))

    q_DS = np.array(q_DS).reshape(len(config_names), len(kcvs), -1)
    T_DS = np.array(T_DS).reshape(len(config_names), len(kcvs), -1)
    T_KS = np.array(T_KS).reshape(len(config_names), len(kcvs), -1)

    variability_dict = {
        "q_DS": q_DS,
        "T_DS": T_DS,
        "T_KS": T_KS,
        "significance_DS": sign_DS,
        "significance_KS": sign_KS,
        "quantiles": Qq,
        "Z_scores": Zz,
    }

    return variability_dict


def latent_variable(checkpoint_dir, config_names, dataset_dict, seed, batch_info, device):
    tbin = dataset_dict["tbin"]
    ts = dataset_dict["timesamples"]
    gt_hd = dataset_dict["covariates"]["hd"]
    max_count = dataset_dict["max_count"]
    neurons = dataset_dict["neurons"]
    pick_neurons = list(range(neurons))

    ### neuron subgroup likelihood CV for latent models ###
    seeds = [seed, seed + 1]

    n_group = np.arange(5)
    kcvs = [2, 5, 8]  # validation sets chosen in 10-fold split of data
    val_neuron = [n_group, n_group + 10, n_group + 20, n_group + 30, n_group + 40]

    LVM_cv_ll = []
    for name in config_names:
        for kcv in kcvs:
            config_name = name + str(kcv)

            full_model, _, _, val_dict = models.load_model(
                config_name,
                checkpoint_dir,
                dataset_dict,
                batch_info,
                device,
            )

            for v_neuron in val_neuron:

                prev_ll = np.inf
                for sd in seeds:  # pick best fit with different random seeds
                    torch.manual_seed(sd)

                    mask = np.ones((neurons,), dtype=bool)
                    mask[v_neuron] = False
                    f_neuron = np.arange(neurons)[mask]

                    ll, _ = models.LVM_Ell(
                        full_model,
                        val_dict,
                        f_neuron,
                        v_neuron,
                        beta=0.0,
                        max_iters=3,
                    )

                    if ll < prev_ll:
                        prev_ll = ll

                LVM_cv_ll.append(prev_ll)

    LVM_cv_ll = np.array(LVM_cv_ll).reshape(
        len(config_names), len(kcvs), len(val_neuron)
    )

    ### aligning trajectory and computing RMS for different models ###
    splits = 90  # split trajectory into 90 parts
    kcvs = [15, 30, 45, 60, 75]

    RMS_cv = []
    for name in config_names:
        config_name = name + "-1"

        full_model, _, _, _ = models.load_model(
            config_name,
            checkpoint_dir,
            dataset_dict,
            batch_info,
            device,
        )

        X_loc, _ = full_model.input_group.input_0.variational.eval_moments(0, ts)
        latents = X_loc.data.cpu().numpy()[:, 0]

        for kcv in kcvs:
            fit_range = np.arange(ts // splits) + kcv * ts // splits

            _, shift, sign, _, _ = utils.signed_scaled_shift(
                latents[fit_range],
                gt_hd[fit_range],
                topology="ring",
                dev=device,
                learn_scale=False,
                iters=1000,
            )

            mask = np.ones((ts,), dtype=bool)
            mask[fit_range] = False

            lat_t = torch.tensor((sign * latents + shift) % (2 * np.pi))
            D = (
                utils.metric(torch.tensor(gt_hd)[mask], lat_t[mask], topology="ring")
                ** 2
            )
            RMS_cv.append(np.sqrt(D.mean().item()))

    RMS_cv = np.array(RMS_cv).reshape(len(config_names), len(kcvs))

    ### compute tuning curves and latent trajectory of UCMs ###
    x_counts = torch.arange(max_count + 1)

    latent_mu = []
    latent_std = []

    comp_avg = []
    comp_FF = []

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
        X_loc, X_std = full_model.input_group.input_0.variational.eval_moments(0, ts)
        X_loc = X_loc.data.cpu().numpy()[:, 0]
        X_std = X_std.data.cpu().numpy()[:, 0]

        lat_t, shift, sign, _, _ = utils.signed_scaled_shift(
            X_loc, gt_hd, device, learn_scale=False, iters=1000
        )
        latent_mu.append(lat_t % (2 * np.pi))
        latent_std.append(X_std)

        # P
        steps = 100
        covariates_aligned = (sign * (np.linspace(0, 2 * np.pi, steps) - shift)) % (
            2 * np.pi
        )

        cov_list = [torch.from_numpy(covariates_aligned)]
        with torch.no_grad():
            P_mc = nprb.utils.model.compute_UCM_P_count(
                full_model.mapping, full_model.likelihood, cov_list, pick_neurons, MC=1000
            ).cpu()

        avg = (x_counts[None, None, None, :] * P_mc).sum(-1)
        xcvar = (x_counts[None, None, None, :] ** 2 * P_mc).sum(-1) - avg**2
        ff = xcvar / (avg + 1e-12)

        avgs = nprb.utils.stats.percentiles_from_samples(
            avg, percentiles=[0.05, 0.5, 0.95], smooth_length=5, padding_mode="circular"
        )
        comp_avg.append([cs_.cpu().numpy() for cs_ in avgs])

        ffs = nprb.utils.stats.percentiles_from_samples(
            ff, percentiles=[0.05, 0.5, 0.95], smooth_length=5, padding_mode="circular"
        )
        comp_FF.append([cs_.cpu().numpy() for cs_ in ffs])

    latent_dict = {
        "LVM_cv_ll": LVM_cv_ll,
        "RMS_cv": RMS_cv,
        "gt_hd": gt_hd, 
        "latent_mu": latent_mu,
        "latent_std": latent_std,
        "covariates_aligned": covariates_aligned,
        "avg_percentiles": comp_avg,
        "FF_percentiles": comp_FF,
    }

    return latent_dict


def main():
    ### parser ###
    parser = argparse.ArgumentParser(
        usage="%(prog)s [options]",
        description="Analysis of heteroscedastic CMP dataset.",
    )

    parser.add_argument("--seed", default=123, type=int)
    parser.add_argument("--dataseed", default=1, type=int)
    parser.add_argument("--savedir", default="../output/", type=str)
    parser.add_argument("--datadir", default="../../data/", type=str)
    parser.add_argument("--checkpointdir", default="../checkpoint/", type=str)

    parser.add_argument("--batch_size", default=5000, type=int)
    
    parser.add_argument("--gpu", default=0, type=int)
    parser.add_argument("--cpu", dest="cpu", action="store_true")
    parser.set_defaults(cpu=False)

    args = parser.parse_args()
    batch_info = args.batch_size
    
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
    gt_hCMP = np.load(data_path + data_type + ".npz")

    ### analysis ###
    regression_dict = regression(
        checkpoint_dir, reg_config_names, dataset_dict, gt_hCMP, batch_info, device
    )
    variability_dict = variability_stats(
        checkpoint_dir, reg_config_names, dataset_dict, rng, batch_info, device
    )
    latent_dict = latent_variable(
        checkpoint_dir, lat_config_names, dataset_dict, args.seed, batch_info, device
    )

    ### export ###
    data_run = {
        "regression": regression_dict,
        "variability": variability_dict,
        "latent": latent_dict,
    }

    pickle.dump(data_run, open(save_dir + "hCMP_results.p", "wb"))


if __name__ == "__main__":
    main()
