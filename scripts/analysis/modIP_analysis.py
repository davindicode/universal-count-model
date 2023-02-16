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


def latent_observed(checkpoint_dir, config_names, dataset_dict, seed, device):
    tbin = dataset_dict["tbin"]
    ts = dataset_dict["timesamples"]
    gt_a = dataset_dict["covariates"]["a"]
    max_count = dataset_dict["max_count"]
    neurons = dataset_dict["neurons"]
    pick_neurons = list(range(neurons))

    ### neuron subgroup likelihood CV ###
    seeds = [seed, seed + 1]

    n_group = np.arange(5)
    val_neuron = [n_group, n_group + 10, n_group + 20, n_group + 30, n_group + 40]

    kcvs = [2, 5, 8]  # validation sets chosen in 10-fold split of data
    batch_info = 5000

    cv_Ell = []
    for en, name in enumerate(config_names):
        for kcv in kcvs:
            config_name = name + str(kcv)

            full_model, _, _, val_dict = models.load_model(
                config_name,
                checkpoint_dir,
                dataset_dict,
                batch_info,
                device,
            )

            if en > 1:
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
                            max_iters=3000,
                        )

                        if ll < prev_ll:
                            prev_ll = ll

                    cv_Ell.append(prev_ll)

            else:  # no latent space
                for v_neuron in val_neuron:
                    cv_Ell.append(
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

    cv_Ell = np.array(cv_Ell).reshape(len(config_names), len(kcvs), len(val_neuron))

    ### compute tuning curves and latent trajectories for XZ joint regression-latent model ###
    config_name = config_names[-1] + "-1"
    batch_info = 1000

    full_model, training_loss, fit_dict, val_dict = models.load_model(
        config_name,
        checkpoint_dir,
        dataset_dict,
        batch_info,
        device,
    )

    # latents
    X_loc, X_std = full_model.input_group.input_1.variational.eval_moments(0, ts)
    X_loc = X_loc.data.cpu().numpy()[:, 0]
    X_std = X_std.data.cpu().numpy()[:, 0]

    X_c, shift, sign, scale, _ = utils.signed_scaled_shift(
        X_loc, gt_a, device, "euclid", learn_scale=True
    )
    X_s = scale * X_std

    # tuning
    steps = 100
    covariates_a = np.linspace(X_loc.min(), X_loc.max(), steps)
    covs_list = [0.0 * torch.ones(steps), torch.from_numpy(covariates_a)]
    P_mc = nprb.utils.model.compute_UCM_P_count(
        full_model.mapping, full_model.likelihood, covs_list, pick_neurons, MC=1000
    ).cpu()

    x_counts = torch.arange(max_count + 1)
    avg = (x_counts[None, None, None, :] * P_mc).sum(-1)
    xcvar = (x_counts[None, None, None, :] ** 2 * P_mc).sum(-1) - avg**2
    ff = xcvar / avg

    avgs = nprb.utils.stats.percentiles_from_samples(
        avg, percentiles=[0.05, 0.5, 0.95], smooth_length=5, padding_mode="replicate"
    )
    avg_percentiles = [cs_.cpu().numpy() for cs_ in avgs]

    ffs = nprb.utils.stats.percentiles_from_samples(
        ff, percentiles=[0.05, 0.5, 0.95], smooth_length=5, padding_mode="replicate"
    )
    FF_percentiles = [cs_.cpu().numpy() for cs_ in ffs]

    covariates_a = sign * scale * covariates_a + shift

    # true tuning
    modIP = np.load(data_path + data_type + ".npz")
    hd = modIP["covariates"][:, 0]

    gt_mean = modIP["gt_rate"] * tbin
    gt_FF = np.ones_like(gt_mean)

    latent_observed_dict = {
        "cv_Ell": cv_Ell,
        "X_c": X_c,
        "X_s": X_s,
        "covariates_a": covariates_a,
        "avg_percentiles": avg_percentiles,
        "FF_percentiles": FF_percentiles,
        "gt_a": gt_a,
        "gt_mean": gt_mean,
        "gt_FF": gt_FF,
    }

    return latent_observed_dict


def variability_stats(checkpoint_dir, config_names, dataset_dict, rng, device):
    tbin = dataset_dict["tbin"]
    max_count = dataset_dict["max_count"]
    neurons = dataset_dict["neurons"]
    pick_neurons = list(range(neurons))
    NN = len(pick_neurons)

    # KS framework
    kcvs = [2, 5, 8]  # validation sets chosen in 10-fold split of data
    batch_info = 5000

    Qq, Zz, R, Rp, Fisher_z, Fisher_q = [], [], [], [], [], []
    for name in config_names:
        for kcv in kcvs:
            config_name = name + str(kcv)

            full_model, _, fit_dict, _ = models.load_model(
                config_name,
                checkpoint_dir,
                dataset_dict,
                batch_info,
                device,
            )

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

                else:  # UCM
                    P_mc = nprb.utils.model.compute_UCM_P_count(
                        full_model.mapping,
                        full_model.likelihood,
                        covariates,
                        pick_neurons,
                        MC=30,
                    )
                    P.append(P_mc.mean(0).cpu().numpy())  # take mean over MC samples

            P = np.concatenate(
                P, axis=1
            )  # count probabilities of shape (neurons, timesteps, count)

            q_ = []
            Z_ = []
            for n in range(NN):
                spike_binned = full_model.likelihood.all_spikes[
                    0, pick_neurons[n], :
                ].numpy()
                q = nprb.utils.stats.counts_to_quantiles(P[n, ...], spike_binned, rng)

                q_.append(q)
                Z_.append(nprb.utils.stats.quantile_Z_mapping(q))

            Qq.append(q_)
            Zz.append(Z_)

            Pearson_s = []
            for n in range(NN):
                for m in range(n + 1, NN):
                    r, r_p = scstats.pearsonr(
                        Z_[n], Z_[m]
                    )  # Pearson r correlation test
                    Pearson_s.append((r, r_p))

            r = np.array([p[0] for p in Pearson_s])
            r_p = np.array([p[1] for p in Pearson_s])

            R.append(r)
            Rp.append(r_p)

            ts = fit_dict["spiketrain"].shape[-1]
            fz = 0.5 * np.log((1 + r) / (1 - r)) * np.sqrt(ts - 3)
            Fisher_z.append(fz)
            Fisher_q.append(nprb.utils.stats.quantile_Z_mapping(fz, inverse=True))

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

            Z_DS = T_DS_ / np.sqrt(2 / (qq.shape[0] - 1))
            q_DS.append(nprb.utils.stats.quantile_Z_mapping(Z_DS, inverse=True))

    q_DS = np.array(q_DS).reshape(len(config_names), len(kcvs), -1)
    T_DS = np.array(T_DS).reshape(len(config_names), len(kcvs), -1)
    T_KS = np.array(T_KS).reshape(len(config_names), len(kcvs), -1)

    # noise correlation structure
    R_mat_Xp = np.zeros((NN, NN))
    R_mat_X = np.zeros((NN, NN))
    R_mat_XZ = np.zeros((NN, NN))
    for a in range(len(R[0])):
        n, m = utils.ind_to_pair(a, NN)
        R_mat_Xp[n, m] = R[0][a]
        R_mat_X[n, m] = R[1][a]
        R_mat_XZ[n, m] = R[2][a]

    variability_dict = {
        "q_DS": q_DS,
        "T_DS": T_DS,
        "T_KS": T_KS,
        "DS_significance": sign_DS,
        "Fisher_Z": Fisher_z,
        "Fisher_q": Fisher_q,
        "quantiles": Qq,
        "Z_scores": Zz,
        "R": R,
        "Rp": Rp,
        "R_Poisson_X": R_mat_Xp,
        "R_Universal_X": R_mat_X,
        "R_Universal_XZ": R_mat_XZ,
    }

    return variability_dict


def main():
    ### parser ###
    parser = argparse.ArgumentParser(
        usage="%(prog)s [OPTION] [FILE]...",
        description="Analysis of modulated Poisson dataset.",
    )
    parser.add_argument(
        "-v", "--version", action="version", version=f"{parser.prog} version 1.0.0"
    )

    parser.add_argument("--seed", default=123, type=int)
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
        device = "cpu"
    else:
        device = nprb.inference.get_device(gpu=args.gpu)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    rng = np.random.default_rng(args.seed)

    ### names ###
    data_type = "modIP{}".format(args.dataseed)

    nc_config_names = [
        data_type + "_IP-exp_svgp-8_X[hd]_Z[]_1K28_0d0_10f",
        data_type + "_U-el-3_svgp-8_X[hd]_Z[]_1K28_0d0_10f",
        data_type + "_U-el-3_svgp-16_X[hd]_Z[R1]_1K28_0d0_10f",
    ]

    ### load dataset ###
    bin_size = 1
    dataset_dict = models.get_dataset(data_type, bin_size, data_path)

    ### analysis ###
    latent_observed_dict = latent_observed(
        checkpoint_dir, nc_config_names, dataset_dict, args.seed, device
    )
    variability_dict = variability_stats(
        checkpoint_dir, nc_config_names, dataset_dict, rng, device
    )

    ### export ###
    data_run = {
        "latent_observed": latent_observed_dict,
        "variability": variability_dict,
    }

    pickle.dump(data_run, open(save_dir + "modIP_results.p", "wb"))


if __name__ == "__main__":
    main()
