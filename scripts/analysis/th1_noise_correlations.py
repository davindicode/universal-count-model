import argparse
import os
import pickle

import sys

import numpy as np
import scipy.stats as scstats
import torch

import neuroprob as nprb

sys.path.append("..")  # access to scripts
import models

import utils


def variability_stats(checkpoint_dir, config_names, dataset_dict, batch_info, device):
    tbin = dataset_dict["tbin"]
    max_count = dataset_dict["max_count"]
    neurons = dataset_dict["neurons"]
    pick_neurons = list(range(neurons))
    
    ### statistics over the behaviour ###
    MC = 100
    x_counts = np.arange(max_count + 1)
    kcv = 2
    
    avg_binnings, var_binnings, FF_binnings = [], [], []
    for name in config_names:
        config_name = name + str(kcv)

        full_model, _, _, _ = models.load_model(
            config_name,
            checkpoint_dir,
            dataset_dict,
            batch_info,
            device,
        )

        avg_model, var_model, ff_model = [], [], []
        with torch.no_grad():
            for batch in range(full_model.input_group.batches):
                covariates, _ = full_model.input_group.sample_XZ(batch, samples=1)

                P_mc = (
                    nprb.utils.model.compute_UCM_P_count(
                        full_model.mapping,
                        full_model.likelihood,
                        covariates,
                        pick_neurons,
                        MC=MC,
                    )
                    .mean(0)
                    .cpu()
                    .numpy()
                )  # count probabilities of shape (neurons, timesteps, count)

                avg = (x_counts[None, None, None, :] * P_mc).sum(-1)
                var = (x_counts[None, None, None, :] ** 2 * P_mc).sum(-1) - avg**2
                ff = var / (avg + 1e-12)
                
                avg_model.append(avg)
                var_model.append(var)
                ff_model.append(ff)

        avg_binnings.append(np.concatenate(avg_model, axis=-1).mean(0))
        var_binnings.append(np.concatenate(var_model, axis=-1).mean(0))
        FF_binnings.append(np.concatenate(ff_model, axis=-1).mean(0))

    Pearson_avg_FF, ratio_avg_FF = [], []
    for d in range(len(avg_binnings)):  # iterate over models
        Pearson = []
        ratio = []
        for avg, ff in zip(avg_binnings[d], FF_binnings[d]):
            r, r_p = scstats.pearsonr(ff, avg)  # Pearson r correlation test
            Pearson.append((r, r_p))
            ratio.append(ff.std() / avg.std())

        Pearson_avg_FF.append(Pearson)
        ratio_avg_FF.append(ratio)

    variability_dict = {
        "avg_binnings": avg_binnings,
        "var_binnings": var_binnings,
        "FF_binnings": FF_binnings,
        "Pearson_avg_FF": Pearson_avg_FF,
        "ratio_avg_FF": ratio_avg_FF,
    }
    return variability_dict


def noise_correlations(checkpoint_dir, config_names, dataset_dict, seed, batch_info, device):
    tbin = dataset_dict["tbin"]
    max_count = dataset_dict["max_count"]
    neurons = dataset_dict["neurons"]
    pick_neurons = list(range(neurons))

    x_counts = torch.arange(max_count + 1)

    ### ELBO for models of different dimensions ###
    kcvs = [2, 5, 8]  # cross validation folds

    ELBO = []
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

            batches = full_model.likelihood.batches

            elbo = []
            for b in range(batches):
                elbo.append(
                    full_model.objective(
                        b,
                        neuron=None,
                        beta=1.0,
                        cov_samples=1, 
                        ll_samples=100,
                        ll_mode="GH",
                    )
                    .data.cpu()
                    .numpy()
                )
            ELBO.append(np.array(elbo).mean())

    ELBO = np.array(ELBO).reshape(len(config_names), len(kcvs))

    ### cross validation for dimensionality ###
    seeds = [seed, seed + 1]
    n_group = np.arange(5)
    val_neuron = [
        n_group,
        n_group + 5,
#         n_group + 10,
#         n_group + 15,
#         n_group + 20,
#         n_group + 25,
#         np.arange(3) + 30,
    ]

    kcvs = [1, 2]#, 3, 5, 6, 8]  # validation segments from splitting data into 10

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
                            max_iters=3,
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
                            neuron_group=v_neuron, 
                            ll_mode="GH",
                            ll_samples=100,
                            cov_samples=1,
                            beta=0.0,
                        )
                    )

    cv_Ell = np.array(cv_Ell).reshape(len(config_names), len(kcvs), len(val_neuron))

    noisecorr_dict = {
        "ELBO": ELBO,
        "cv_Ell": cv_Ell,
    }

    return noisecorr_dict


def best_model(checkpoint_dir, model_name, dataset_dict, batch_info, device):
    tbin = dataset_dict["tbin"]
    max_count = dataset_dict["max_count"]
    neurons = dataset_dict["neurons"]
    pick_neurons = list(range(neurons))
    
    x_counts = torch.arange(max_count + 1)

    covariates = dataset_dict["covariates"]
    TT = tbin * dataset_dict["timesamples"]
    
    
    ### load model ###
    config_name = model_name

    full_model, training_loss, fit_dict, _ = models.load_model(
        config_name,
        checkpoint_dir,
        dataset_dict,
        batch_info,
        device,
    )

    # latents
    X_loc, X_std = full_model.input_group.input_6.variational.eval_moments(0, ts)
    X_loc = X_loc.data.cpu().numpy()[:, 0]
    X_std = X_std.data.cpu().numpy()[:, 0]

    X_c = X_loc
    X_s = X_std
    z_tau = tbin / (1 - torch.sigmoid(full_model.input_group.input_6.prior.mu).data.cpu().numpy())

    t_lengths = (
        full_model.mapping.kernel.kern1.lengthscale[:, 0, 0, -3].data.cpu().numpy()
    )

    ### covariates ###
    
    # compute timescales for input dimensions from ACG
    delays = 5000
    Tsteps = rcov[0].shape[0]
    L = Tsteps - delays + 1
    acg_rc = {}

    for name, cov in covariates.items():
        if name == 'hd':  # angular
            acg = np.empty(delays)
            for d in range(delays):
                A = rc[d : d + L]
                B = rc[:L]
                acg[d] = utils.stats.corr_circ_circ(A, B)
        else:
            acg = np.empty(delays)
            for d in range(delays):
                A = rc[d : d + L]
                B = rc[:L]
                acg[d] = ((A - A.mean()) * (B - B.mean())).mean() / A.std() / B.std()

        acg_rc[name] = acg

    for en, rc in enumerate(X_c.T):
        acg = np.empty(delays)
        for d in range(delays):
            A = rc[d : d + L]
            B = rc[:L]
            acg[d] = ((A - A.mean()) * (B - B.mean())).mean() / A.std() / B.std()
        
        acg_rc['z_{}'.format(en+1)] = acg

    timescales = {}
    for name, acg in acg_rc.items():
        timescales[name] = np.where(acg < np.exp(-1))[0][0] * tbin


    covariates_dict = {
        "X_mu": X_c,
        "X_std": X_s,
        "z_tau": z_tau, 
        "t_lengths": t_lengths,
        "timescales": timescales,
        "acg_rc": acg_rc,
    }

    ### head direction tuning ###
    MC = 100

    steps = 100
    covariates = [
        np.linspace(0, 2 * np.pi, steps),
        0.0 * np.ones(steps),
        0.0 * np.ones(steps),
        (left_x + right_x) / 2.0 * np.ones(steps),
        (bottom_y + top_y) / 2.0 * np.ones(steps),
        0.0 * np.ones(steps),
        0.0 * np.ones(steps),
        0.0 * np.ones(steps),
    ]

    with torch.no_grad():
        P_mc = nprb.utils.model.compute_UCM_P_count(
            full_model.mapping, 
            full_model.likelihood, 
            [torch.from_numpy(c) for c in fit_dict['covariates']], 
            pick_neurons, 
            MC=MC, 
        ).cpu()

    avg = (x_counts[None, None, None, :] * P_mc).sum(-1).mean(0).numpy()
    pref_hd = covariates[0][np.argmax(avg, axis=1)]

    # marginalized tuning curves
    rcovz = list(rcov) + [X_c[:, 0], X_c[:, 1]]
    MC = 100
    skip = 10

    ### TI to latent space ###
    step = 100
    with torch.no_grad():
        P_tot = nprb.utils.model.marginalize_UCM_P_count(
            full_model.mapping,
            full_model.likelihood,
            [torch.linspace(-0.2, 0.2, step)],
            [6],
            [torch.from_numpy(c) for c in fit_dict['covariates']],
            batch_size,
            pick_neurons,
            MC=MC,
            sample_skip=skip,
        ).cpu()
    avg = (x_counts[None, None, None, :] * P_tot).sum(-1)
    var = (x_counts[None, None, None, :] ** 2 * P_tot).sum(-1) - avg**2
    ff = var / avg

    marg_z1_avg = avg.mean(0)
    marg_z1_FF = ff.mean(0)
    
    z1_avg_tf = (mz1_mean.max(dim=-1)[0] - mz1_mean.min(dim=-1)[0]) / (
        mz1_mean.max(dim=-1)[0] + mz1_mean.min(dim=-1)[0]
    )
    z1_FF_tf = (mz1_ff.max(dim=-1)[0] - mz1_ff.min(dim=-1)[0]) / (
        mz1_ff.max(dim=-1)[0] + mz1_ff.min(dim=-1)[0]
    )

    step = 100
    with torch.no_grad():
        P_tot = nprb.utils.model.marginalized_UCM_P_count(
            full_model.mapping,
            full_model.likelihood,
            [torch.linspace(-0.2, 0.2, step)],
            [7],
            [torch.from_numpy(c) for c in fit_dict['covariates']],
            batch_size,
            pick_neurons,
            MC=MC,
            sample_skip=skip,
        ).cpu()
    avg = (x_counts[None, None, None, :] * P_tot).sum(-1)
    var = (x_counts[None, None, None, :] ** 2 * P_tot).sum(-1) - avg**2
    ff = var / avg

    marg_z2_avg = avg.mean(0)
    marg_z2_FF = ff.mean(0)
    
    z2_avg_tf = (mz2_mean.max(dim=-1)[0] - mz2_mean.min(dim=-1)[0]) / (
        mz2_mean.max(dim=-1)[0] + mz2_mean.min(dim=-1)[0]
    )
    z2_FF_tf = (mz2_ff.max(dim=-1)[0] - mz2_ff.min(dim=-1)[0]) / (
        mz2_ff.max(dim=-1)[0] + mz2_ff.min(dim=-1)[0]
    )

    marginal_tunings = {
        "marg_z1_avg": marg_z1_avg,
        "marg_z1_FF": marg_z1_FF,
        "z1_avg_tf": z1_avg_tf,
        "z1_FF_tf": z1_FF_tf,
        "marg_z2_avg": marg_z2_avg,
        "marg_z2_FF": marg_z2_FF,
        "z2_avg_tf": z2_avg_tf,
        "z2_FF_tf": z2_FF_tf,
    }

    # compute 2D latent model properties of conditional tuning curves
    grid_size_zz = (41, 41)
    grid_shape_zz = [[-0.2, 0.2], [-0.2, 0.2]]
    grid_zz = {"size": grid_size_zz, "shape": grid_shape_zz}

    steps = np.product(grid_size_zz)
    A, B = grid_size_zz

    avg_zz, FF_zz = [], []
    t = 0
    for en, n in enumerate(pick_neurons):
        covariates = [
            pref_hd[n] * np.ones(steps),
            0.0 * np.ones(steps),
            0.0 * np.ones(steps),
            (left_x + right_x) / 2.0 * np.ones(steps),
            (bottom_y + top_y) / 2.0 * np.ones(steps),
            t * np.ones(steps),
            np.linspace(-0.2, 0.2, A)[:, None].repeat(B, axis=1).flatten(),
            np.linspace(-0.2, 0.2, B)[None, :].repeat(A, axis=0).flatten(),
        ]

        with torch.no_grad():
            P_mean = (
                nprb.utils.model.compute_UCM_P_count(
                    full_model.mapping, 
                    full_model.likelihood, 
                    covariates, 
                    [n], 
                    MC=MC, 
                ).mean(0).cpu()
            )
        avg = (x_counts[None, :] * P_mean[0, ...]).sum(-1).reshape(A, B).numpy()
        var = (x_counts[None, :] ** 2 * P_mean[0, ...]).sum(-1).reshape(A, B).numpy()
        xcvar = var - avg**2

        avg_zz.append(avg)
        FF_zz.append(xcvar / avg)

    avg_zz = np.stack(avg_zz)
    FF_zz = np.stack(FF_zz)

    # KS framework for latent models, including Fisher Z scores
    kcvs = [2, 5, 8]

    ### KS test ###
    N = len(pick_neurons)
    Qq, Zz, R, Rp = [], [], [], []

    for en, name in enumerate(config_names):
        for kcv in kcvs:

            config_name = name

            full_model, training_loss, fit_dict, _ = models.load_model(
                config_name,
                checkpoint_dir,
                dataset_dict,
                batch_info,
                device,
            )

            q_ = []
            Z_ = []
            for b in range(full_model.inputs.batches):  # predictive posterior
                with torch.no_grad():
                    P = nprb.utils.model.compute_UCM_P_count(
                        full_model.mapping, 
                        full_model.likelihood, 
                        covariates, 
                        pick_neurons, 
                        MC=MC, 
                    ).mean(0).cpu().numpy()

                for n in range(N):
                    spike_binned = full_model.likelihood.all_spikes[
                        0, pick_neurons[n], :
                    ].numpy()

                    q, Z = model_utils.get_q_Z(P[n, ...], spike_binned, deq_noise=None)
                    q_.append(q)
                    Z_.append(Z)

            q = []
            Z = []
            for n in range(N):
                q.append(np.concatenate(q_[n::N]))
                Z.append(np.concatenate(Z_[n::N]))

            Pearson_s = []
            for n in range(len(pick_neurons)):
                for m in range(n + 1, len(pick_neurons)):
                    r, r_p = scstats.pearsonr(Z[n], Z[m])  # Pearson r correlation test
                    Pearson_s.append((r, r_p))

            r = np.array([p[0] for p in Pearson_s])
            r_p = np.array([p[1] for p in Pearson_s])

            Qq.append(q)
            Zz.append(Z)
            R.append(r)
            Rp.append(r_p)

    Fisher_Z, Fisher_q = [], []
    for en, r in enumerate(R):
        fz = 0.5 * np.log((1 + r) / (1 - r)) * np.sqrt(time_steps - 3)
        Fisher_z.append(fz)
        Fisher_q.append(utils.stats.Z_to_q(fz))

    q_DS, T_DS, T_KS = [], [], []
    for q in Qq:
        for qq in q:
            T_DS_, T_KS_, sign_DS, sign_KS, p_DS, p_KS = utils.stats.KS_DS_statistics(
                qq, alpha=0.05, 
            )
            T_DS.append(T_DS_)
            T_KS.append(T_KS_)

            Z_DS = T_DS / np.sqrt(2 / (qq.shape[0] - 1))
            q_DS.append(utils.stats.Z_to_q(Z_DS))

    Fisher_Z = np.array(Fisher_Z).reshape(len(modes), len(kcvs), -1)
    Fisher_q = np.array(Fisher_q).reshape(len(modes), len(kcvs), -1)

    Qq = np.array(Qq).reshape(len(modes), len(kcvs), N, -1)
    Zz = np.array(Zz).reshape(len(modes), len(kcvs), -1)
    R = np.array(R).reshape(len(modes), len(kcvs), N, -1)
    Rp = np.array(Rp).reshape(len(modes), len(kcvs), N, -1)

    q_DS = np.array(q_DS).reshape(len(modes), len(kcvs), N, -1)
    T_DS = np.array(T_DS).reshape(len(modes), len(kcvs), N, -1)
    T_KS = np.array(T_KS).reshape(len(modes), len(kcvs), N, -1)

    # KS test on population statistics
    T_KS_fishq = []
    p_KS_fishq = []
    for q in fisher_q:
        for qq in q:
            _, T_KS, _, _, _, p_KS = utils.stats.KS_DS_statistics(
                qq, alpha=0.05, 
            )
            T_KS_fishq.append(T_KS)
            p_KS_fishq.append(p_KS)

    T_KS_fishq = np.array(T_KS_fishq).reshape(len(modes), len(kcvs))
    p_KS_fishq = np.array(p_KS_fishq).reshape(len(modes), len(kcvs))

    T_KS_ks = []
    p_KS_ks = []
    for q in Qq:
        for qq in q:
            for qqq in qq:
                _, T_KS, _, _, _, p_KS = utils.stats.KS_DS_statistics(
                    qqq, alpha=0.05, 
                )
                T_KS_ks.append(T_KS)
                p_KS_ks.append(p_KS)

    T_KS_ks = np.array(T_KS_ks).reshape(len(modes), len(kcvs), N)
    p_KS_ks = np.array(p_KS_ks).reshape(len(modes), len(kcvs), N)

    population_KS = {
        "T_KS_fishq": T_KS_fishq,
        "significance_KS_fishq": significance_KS_fishq,
        "T_KS_ks": T_KS_ks,
        "significance_KS_ks": significance_KS_ks,
    }

    # delayed noise or spatiotemporal correlations
    delays = np.arange(5)
    R_mat_spt = np.empty((len(modes), len(delays), N, N))
    R_mat_sptp = np.empty((len(modes), len(delays), N, N))

    kcv_ind = 1
    for d, Z_ in enumerate(Zz[kcv_ind]):
        steps = len(Z_[0]) - len(delays)

        for en, t in enumerate(delays):
            Pearson_s = []
            for n in range(N):
                for m in range(N):
                    r, r_p = scstats.pearsonr(
                        Z_[n][t : t + steps], Z_[m][: -len(delays)]
                    )  # Pearson r correlation test
                    R_mat_spt[d, en, n, m] = r
                    R_mat_sptp[d, en, n, m] = r_p

    bestmodel_dict = {
        "covariates": covariates_dict,
        "marginal_tunings": marginal_tunings,
        "pref_hd": pref_hd,
        "grid_zz": grid_zz,
        "avg_zz": avg_zz,
        "FF_zz": FF_zz,
        "q_DS": q_DS,
        "T_DS": T_DS,
        "T_KS": T_KS,
        "quantiles": Qq,
        "Z_scores": Zz,
        "R": R,
        "Rp": Rp,
        "Fisher_Z": Fisher_Z,
        "Fisher_q": Fisher_q,
        "population_KS": population_KS,
        "R_delays": R_mat_spt,
        "Rp_delays": R_mat_sptp,
    }
    return bestmodel_dict


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

    parser.add_argument("--batch_size", default=500, type=int)
    
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
        
    batch_info = args.batch_size

    ### names ###
    nc_config_names = [
        "th1_U-el-3_svgp-64_X[hd-omega-speed-x-y-time]_Z[]_40K11_0d0_10f",
        "th1_U-el-3_svgp-72_X[hd-omega-speed-x-y-time]_Z[R1]_40K11_0d0_10f",
        "th1_U-el-3_svgp-80_X[hd-omega-speed-x-y-time]_Z[R2]_40K11_0d0_10f",
        "th1_U-el-3_svgp-88_X[hd-omega-speed-x-y-time]_Z[R3]_40K11_0d0_10f",
        "th1_U-el-3_svgp-96_X[hd-omega-speed-x-y-time]_Z[R4]_40K11_0d0_10f",
    ]

    best_name = "th1_U-el-3_svgp-80_X[hd-omega-speed-x-y-time]_Z[R2]_40K11_0d0_10f-1"

    ### load dataset ###
    data_type = "th1"
    bin_size = 40

    dataset_dict = models.get_dataset(data_type, bin_size, data_path)

    ### analysis ###
    variability_dict = variability_stats(
        checkpoint_dir, nc_config_names, dataset_dict, batch_info, device
    )
    noisecorr_dict = noise_correlations(
        checkpoint_dir, nc_config_names, dataset_dict, args.seed, batch_info, device
    )
    bestmodel_dict = best_model(
        checkpoint_dir, best_name, dataset_dict, batch_info, device
    )

    ### export ###
    data_run = {
        "variability": variability_dict,
        "noise_correlations": noisecorr_dict,
        "best_model": bestmodel_dict,
    }

    pickle.dump(data_run, open(save_dir + "th1_NC_results.p", "wb"))


if __name__ == "__main__":
    main()
