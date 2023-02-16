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


def latent_variable(checkpoint_dir, config_names, dataset_dict, seed, batch_info, device):
    tbin = dataset_dict["tbin"]
    ts = dataset_dict["timesamples"]
    max_count = dataset_dict["max_count"]
    gt_hd = dataset_dict["covariates"]["hd"]
    neurons = dataset_dict["neurons"]
    pick_neurons = list(range(neurons))

    ### likelihood CV over subgroups of neurons as well as validation runs ###
    seeds = [seed, seed + 1]

    n_group = np.arange(5)
    val_neuron = [
        n_group,
        n_group + 5,
        n_group + 10,
        n_group + 15,
        n_group + 20,
        n_group + 25,
        np.arange(3) + 30,
    ]
    kcvs = [1, 2, 3, 5, 6, 8]  # validation segments from splitting data into 10

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

    ### trajectory regression to align to data and compute drifts ###
    splits = 3  # split trajectory into 3 parts
    kcvs = [0, 1, 2]

    RMS_cv = []
    drifts_lv = []
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

            drift, sign, shift, _ = utils.circ_drift_regression(
                gt_hd[fit_range],
                latents[fit_range],
                fit_range * tbin,
                topology="ring",
                dev=device,
                a_fac=1e-5,
            )

            mask = np.ones((ts,), dtype=bool)
            mask[fit_range] = False

            latent_aligned = torch.tensor(
                (np.arange(ts) * tbin * drift + shift + sign * latents) % (2 * np.pi)
            )
            D = (
                utils.metric(
                    torch.tensor(gt_hd)[mask], latent_aligned[mask], topology="ring"
                )
                ** 2
            )
            RMS_cv.append(np.sqrt(D.mean().item()))
            drifts_lv.append(drift)

    RMS_cv = np.array(RMS_cv).reshape(len(config_names), len(kcvs))
    drifts_lv = np.array(drifts_lv).reshape(len(config_names), len(kcvs))

    # compute delays in latent trajectory w.r.t. data, see which one fits best in RMS
    D = 5
    delays = np.arange(-D, D + 1)

    config_name = config_names[2] + "-1"  # UCM

    full_model, _, _, _ = models.load_model(
        config_name,
        checkpoint_dir,
        dataset_dict,
        batch_info,
        device,
    )

    X_loc, X_std = full_model.input_group.input_0.variational.eval_moments(0, ts)
    X_loc = X_loc.data.cpu().numpy()[:, 0]
    X_std = X_std.data.cpu().numpy()[:, 0]

    delay_RMS = []
    for delay in delays:
        cvT = X_loc[0].shape[0] - len(delays) + 1
        tar_t = rhd_t[D + delay : cvT + D + delay]
        lat = X_loc[0][D : cvT + D]

        for rn in CV:
            fit_range = np.arange(cvT // cvK) + rn * cvT // cvK

            drift, sign, shift, _ = utils.circ_drift_regression(
                tar_t[fit_range],
                lat[fit_range],
                fit_range * tbin,
                topology,
                dev=dev,
                a_fac=1e-5,
            )

            mask = np.ones((cvT,), dtype=bool)
            mask[fit_range] = False

            lat_ = torch.tensor(
                (np.arange(cvT) * tbin * drift + shift + sign * lat) % (2 * np.pi)
            )
            Dd = (
                utils.latent.metric(torch.tensor(tar_t)[mask], lat_[mask], topology)
                ** 2
            )
            delay_RMS.append(Dd.mean().item())

    delay_RMS = np.array(delay_RMS).reshape(len(delays), len(CV))

    # get the latent inferred trajectory of UCM
    drift, sign, shift, _ = circ_drift_regression(
        gt_hd[fit_range],
        X_loc[fit_range],
        fit_range * tbin,
        topology,
        dev=dev,
        a_fac=1e-5,
    )

    latent_mu = (np.arange(rhd_t.shape[0]) * tbin * drift + shift + sign * lat) % (
        2 * np.pi
    )

    latent_dict = {
        "LVM_cv_ll": LVM_cv_ll,
        "RMS_cv": RMS_cv,
        "drifts_lv": drifts_lv,
        "delay_RMS": delay_RMS,
        "latent_mu": latent_mu,
        "latent_std": X_std,
    }

    return latent_dict


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

    ### names ###
    lat_config_names = [
        "th1_IP-exp_svgp-8_X[]_Z[T1]_100K25_0d0_10f",
        "th1_hNB-exp_svgp-8_X[]_Z[T1]_100K25_0d0_10f",
        "th1_U-el-3_svgp-8_X[]_Z[T1]_100K25_0d0_10f",
    ]

    ### load dataset ###
    data_type = "th1"
    bin_size = 100

    dataset_dict = models.get_dataset(data_type, bin_size, data_path)

    ### analysis ###
    latent_dict = latent_variable(
        checkpoint_dir, lat_config_names, dataset_dict, args.seed, batch_info, device
    )

    ### export ###
    data_run = {
        "latent": latent_dict,
    }

    pickle.dump(data_run, open(save_dir + "th1_LVM_results.p", "wb"))


if __name__ == "__main__":
    main()
