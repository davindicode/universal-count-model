import argparse

import os

import pickle

import sys

import numpy as np
import scipy.stats as scstats
import torch

sys.path.append("..")  # access to library
import neuroprob as nprb

sys.path.append("..")  # access to scripts
import models


def latent_observed(
    checkpoint_dir, config_names, data_path, data_type, dataset_dict, device
):
    tbin = dataset_dict["tbin"]
    neurons = dataset_dict["neurons"]
    pick_neurons = list(range(neurons))

    rhd_t = rcov[0]
    ra_t = rcov[1]
    covariates = [
        rhd_t[None, :, None].repeat(trials, axis=0),
        ra_t[None, :, None].repeat(trials, axis=0),
    ]

    checkpoint_dir = "../scripts/checkpoint/"
    config_name = "th1_U-el-4_svgp-64_X[hd-omega-speed-x-y-time]_Z[]_40K11_0d0_10f-1"
    batch_info = 500

    full_model, training_loss, fit_dict, val_dict = models.load_model(
        config_name,
        checkpoint_dir,
        dataset_dict,
        batch_info,
        device,
    )

    # neuron subgroup likelihood CV
    seeds = [123, 1234, 12345]

    n_group = np.arange(5)
    val_neuron = [n_group, n_group + 10, n_group + 20, n_group + 30, n_group + 40]

    kcvs = [2, 5, 8]  # validation sets chosen in 10-fold split of data

    batch_size = 5000
    cv_pll = []
    for em, name in enumerate(lat_config_names):
        for kcv in kcvs:
            config_name = name + str(kcv)

            if em > 1:
                for v_neuron in val_neuron:

                    prev_ll = np.inf
                    for seed in range(seeds):
                        (
                            full_model,
                            training_loss,
                            fit_dict,
                            val_dict,
                        ) = models.load_model(
                            config_name,
                            checkpoint_dir,
                            dataset_dict,
                            batch_info,
                            device,
                        )

                        mask = np.ones((neurons,), dtype=bool)
                        mask[v_neuron] = False
                        f_neuron = np.arange(neurons)[mask]

                        ll = model_utils.LVM_pred_ll(
                            full_model,
                            mode[-5],
                            mode[2],
                            models.cov_used,
                            cv_set,
                            f_neuron,
                            v_neuron,
                            beta=beta,
                            beta_z=0.0,
                            max_iters=3000,
                        )[0]
                        if ll < prev_ll:
                            prev_ll = ll

                    cv_pll.append(prev_ll)

            else:
                for v_neuron in val_neuron:
                    full_model, training_loss, fit_dict, val_dict = models.load_model(
                        config_name,
                        checkpoint_dir,
                        dataset_dict,
                        batch_info,
                        device,
                    )

                    cv_pll.append(
                        model_utils.RG_pred_ll(
                            full_model,
                            mode[2],
                            models.cov_used,
                            cv_set,
                            bound="ELBO",
                            beta=beta,
                            neuron_group=v_neuron,
                            ll_mode="GH",
                            ll_samples=100,
                        )
                    )

    cv_pll = np.array(cv_pll).reshape(len(Ms), len(kcvs), len(val_neuron))

    # compute tuning curves and latent trajectories for XZ joint regression-latent model
    checkpoint_dir = "../scripts/checkpoint/"
    config_name = "modIP_U-el-3_svgp-16_X[hd]_Z[T1]_1K28_0d0_10f-1"
    batch_info = 500

    full_model, training_loss, fit_dict, val_dict = models.load_model(
        config_name,
        checkpoint_dir,
        dataset_dict,
        batch_info,
        device,
    )

    # latents
    X_loc, X_std = full_model.inputs.eval_XZ()

    T = X_loc[1].shape[0]
    X_c, shift, sign, scale, _ = utils.latent.signed_scaled_shift(
        X_loc[1], ra_t[:T], dev, "euclid", learn_scale=True
    )
    X_s = scale * X_std[1]

    # tuning
    steps = 100
    covariates_z = [
        0.0 * np.ones(steps),
        np.linspace(X_loc[1].min(), X_loc[1].max(), steps),
    ]
    P_mc = model_utils.compute_P(full_model, covariates_z, pick_neurons, MC=1000).cpu()

    x_counts = torch.arange(max_count + 1)
    avg = (x_counts[None, None, None, :] * P_mc).sum(-1)
    xcvar = (x_counts[None, None, None, :] ** 2 * P_mc).sum(-1) - avg**2
    ff = xcvar / avg

    avgs = utils.signal.percentiles_from_samples(
        avg, percentiles=[0.05, 0.5, 0.95], smooth_length=5, padding_mode="replicate"
    )
    avg_percentiles = [cs_.cpu().numpy() for cs_ in avgs]

    ffs = utils.signal.percentiles_from_samples(
        ff, percentiles=[0.05, 0.5, 0.95], smooth_length=5, padding_mode="replicate"
    )
    ff_percentiles = [cs_.cpu().numpy() for cs_ in ffs]

    covariates_z[1] = sign * scale * covariates_z[1] + shift

    # true tuning
    modIP = np.load(data_path + data_type + ".npz")
    hd = modIP["covariates"][:, 0]

    gt_rate = modIP["gt_rate"]
    gt_FF = np.ones_like(gt_rate)

    latent_dict = {
        "cv_pll": cv_pll,
        "X_c": X_c,
        "X_s": X_s,
        "covariates_z": covariates_z,
        "avg_percentiles": avg_percentiles,
        "ff_percentiles": ff_percentiles,
        "gt_rate": gt_rate,
        "gt_FF": gt_FF,
    }

    return latent_dict


def variability_stats(checkpoint_dir, config_names, dataset_dict, rng, device):
    tbin = dataset_dict["tbin"]
    neurons = dataset_dict["neurons"]
    pick_neurons = list(range(neurons))
    
    # KS framework
    Qq = []
    Zz = []
    R = []
    Rp = []

    kcvs = [2, 5, 8]  # validation sets chosen in 10-fold split of data
    batch_info = 5000

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

            P = []
            for b in range(full_model.inputs.batches):
                XZ, _, _, _ = self.inputs.sample_XZ(batch, samples=1)
                
                if type(full_model.likelihood) == nprb.likelihoods.Poisson:
                    rate = nprb.utils.model.posterior_rate()

                    P = utils.stats.poiss_count_prob(c, rate, tbin)

                else:  # UCM
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
                    
            P = np.concatenate(P, axis=1)  # count probabilities of shape (neurons, timesteps, count)
                
            q_ = []
            Z_ = []
            for n in range(len(pick_neurons)):
                spike_binned = full_model.likelihood.spikes[0][
                    0, pick_neurons[n], :
                ].numpy()
                q = nprb.utils.stats.counts_to_quantiles(P[n, ...], spike_binned, rng)

                q_.append(q)
                Z_.append(nprb.utils.stats.quantile_Z_mapping(q))

            Qq.append(q_)
            Zz.append(Z_)


            Pearson_s = []
            for n in range(len(pick_neurons)):
                for m in range(n + 1, len(pick_neurons)):
                    r, r_p = scstats.pearsonr(
                        Z_[n], Z_[m]
                    )  # Pearson r correlation test
                    Pearson_s.append((r, r_p))

            r = np.array([p[0] for p in Pearson_s])
            r_p = np.array([p[1] for p in Pearson_s])

            Qq.append(q_)
            Zz.append(Z_)
            R.append(r)
            Rp.append(r_p)

    fisher_z, fisher_q = [], []
    for en, r in enumerate(R):
        fz = 0.5 * np.log((1 + r) / (1 - r)) * np.sqrt(time_steps - 3)
        fisher_z.append(fz)
        fisher_q.append(utils.stats.quantile_Z_mapping(fz, inverse=True))

    q_DS, T_DS, T_KS = [], [], []
    for q in Qq:
        for qq in q:
            T_DS, T_KS, sign_DS, sign_KS, p_DS, p_KS = utils.stats.KS_DS_statistics(
                qq, alpha=0.05, alpha_s=0.05
            )
            T_DS_.append(T_DS)
            T_KS_.append(T_KS)

            Z_DS = T_DS / np.sqrt(2 / (qq.shape[0] - 1))
            q_DS_.append(utils.stats.quantile_Z_mapping(Z_DS, inverse=True))

    q_DS = np.array(q_DS).reshape(len(CV), len(Ms), -1)
    T_DS = np.array(T_DS).reshape(len(CV), len(Ms), -1)
    T_KS = np.array(T_KS).reshape(len(CV), len(Ms), -1)

    # noise correlation structure
    NN = len(pick_neurons)
    
    R_mat_Xp = np.zeros((NN, NN))
    R_mat_X = np.zeros((NN, NN))
    R_mat_XZ = np.zeros((NN, NN))
    for a in range(len(R[0])):
        n, m = model_utils.ind_to_pair(a, NN)
        R_mat_Xp[n, m] = R[0][a]
        R_mat_X[n, m] = R[1][a]
        R_mat_XZ[n, m] = R[2][a]
        

    variability_dict = {
        "q_DS": q_DS,
        "T_DS": T_DS,
        "T_KS": T_KS,
        "DS_significance": sign_DS,
        "Fisher_Z": fisher_z,
        "Fisher_q": fisher_q,
        "quantiles": Qq,
        "Z_scores": Zz,
        "R": R,
        "Rp": Rp,
        "R_mat_Xp": R_mat_Xp,
        "R_mat_X": R_mat_X,
        "R_mat_XZ": R_mat_XZ,
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
        checkpoint_dir, config_names, data_path, data_type, dataset_dict, dev
    )
    variability_dict = variability_stats()

    ### export ###
    data_run = {
        "latent": latent_dict,
        "variability": variability_dict,
    }

    pickle.dump(data_run, open(save_dir + "modIP_results.p", "wb"))


if __name__ == "__main__":
    main()
