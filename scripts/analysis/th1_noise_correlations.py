import argparse
import os
import pickle

import sys

import numpy as np
import scipy.stats as scstats
import torch

sys.path.append("..")  # access to library
import neuroprob as nprb

sys.path.append("../scripts")  # access to scripts
import models


def variability_stats(checkpoint_dir, config_names, dataset_dict, batch_info, device):
    tbin = dataset_dict["tbin"]
    max_count = dataset_dict["max_count"]
    neurons = dataset_dict["neurons"]
    pick_neurons = list(range(neurons))

    ### statistics over the behaviour ###
    avg_models, var_models, FF_models = [], [], []
    
    kcv = 2
    bn = 40

    for name in config_names:
        config_name = name + str(kcv)

        rcov, neurons, tbin, resamples, rc_t, region_edge = HDC.get_dataset(
            session_id, phase, bn, "../scripts/data"
        )
        max_count = int(rc_t.max())
        x_counts = torch.arange(max_count + 1)

        cvdata = model_utils.get_cv_sets(
            mode, [kcv], batch_size, rc_t, resamples, rcov
        )[0]
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
            P_mc = model_utils.compute_pred_P(
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
        FF_models.append(torch.cat(ff_model, dim=-1).mean(0).numpy())

    b = 1
    Pearson_avg_FF = []
    ratio_avg_FF = []

    for d in range(len(avg_models)):
        Pearson = []
        ratio = []
        for avg, ff in zip(avg_models[d], FF_models[d]):
            r, r_p = scstats.pearsonr(ff, avg)  # Pearson r correlation test
            Pearson.append((r, r_p))
            ratio.append(ff.std() / avg.std())

        Pearson_avg_FF.append(Pearson)
        ratio_avg_FF.append(ratio)

    variability_dict = {
        "avg_models": avg_models,
        "var_models": var_models,
        "FF_models": FF_models,
        "Pearson_avg_FF": Pearson_avg_FF,
        "ratio_avg_FF": ratio_avg_FF,
    }
    return variability_dict


def noise_correlations(checkpoint_dir, config_names, dataset_dict, batch_info, device):
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

            elbo_ = []
            for b in range(batches):
                elbo_.append(
                    full_model.objective(
                        b,
                        cov_samples=1,
                        ll_mode="GH",
                        bound="ELBO",
                        neuron=None,
                        beta=1.0,
                        ll_samples=100,
                    )
                    .data.cpu()
                    .numpy()
                )
            ELBO.append(np.array(elbo_).mean())

    ELBO = np.array(ELBO).reshape(len(config_names), len(kcvs))

    ### cross validation for dimensionality ###
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
    Ms = modes[:5]

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

    noisecorr_dict = {
        "ELBO": ELBO,
        "cv_Ell": cv_Ell,
    }

    return noisecorr_dict


def best_model(checkpoint_dir, model_name, dataset_dict, batch_info, device):
    # load model
    config_name = model_name

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

    X_c = X_loc[6]
    X_s = X_std[6]
    z_tau = tbin / (1 - torch.sigmoid(full_model.inputs.p_mu_6).data.cpu().numpy())

    t_lengths = (
        full_model.mapping.kernel.kern1.lengthscale[:, 0, 0, -3].data.cpu().numpy()
    )

    # covariates
    # compute timescales for input dimensions from ACG
    delays = 5000
    Tsteps = rcov[0].shape[0]
    L = Tsteps - delays + 1
    acg_rc = []

    for rc in rcov[:1]:  # angular
        acg = np.empty(delays)
        for d in range(delays):
            A = rc[d : d + L]
            B = rc[:L]
            acg[d] = utils.stats.corr_circ_circ(A, B)
        acg_rc.append(acg)

    for rc in rcov[1:-1]:
        acg = np.empty(delays)
        for d in range(delays):
            A = rc[d : d + L]
            B = rc[:L]
            acg[d] = ((A - A.mean()) * (B - B.mean())).mean() / A.std() / B.std()
        acg_rc.append(acg)

    acg_z = []
    for rc in X_c.T:
        acg = np.empty(delays)
        for d in range(delays):
            A = rc[d : d + L]
            B = rc[:L]
            acg[d] = ((A - A.mean()) * (B - B.mean())).mean() / A.std() / B.std()
        acg_z.append(acg)

    timescales = []

    for d in range(len(rcov) - 1):
        timescales.append(np.where(acg_rc[d] < np.exp(-1))[0][0] * tbin)

    for d in range(X_c.shape[-1]):
        timescales.append(np.where(acg_z[d] < np.exp(-1))[0][0] * tbin)

    covariates_dict = {
        "X_mu": X_c,
        "X_std": X_s,
        "z_tau": z_tau,
        "timescales": timescales,
        "acg_rc": acg_rc,
        "acg_z": acg_z,
        "t_lengths": t_lengths,
    }

    # load regression model with most input dimensions
    mode = modes[4]
    cvdata = model_utils.get_cv_sets(mode, [-1], 5000, rc_t, resamples, rcov)[0]
    full_model = get_full_model(
        session_id,
        phase,
        cvdata,
        resamples,
        40,
        mode,
        rcov,
        max_count,
        neurons,
        gpu=gpu_dev,
    )

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

    P_mc = model_utils.compute_P(full_model, covariates, pick_neuron, MC=MC).cpu()

    avg = (x_counts[None, None, None, :] * P_mc).sum(-1).mean(0).numpy()
    pref_hd = covariates[0][np.argmax(avg, axis=1)]

    # marginalized tuning curves
    rcovz = list(rcov) + [X_c[:, 0], X_c[:, 1]]
    MC = 10
    skip = 10

    ### TI to latent space ###
    step = 100
    P_tot = model_utils.marginalized_P(
        full_model,
        [np.linspace(-0.2, 0.2, step)],
        [6],
        rcovz,
        10000,
        pick_neuron,
        MC=MC,
        skip=skip,
    )
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
    P_tot = utils.marginalized_UCM_P_count(
        full_model,
        [np.linspace(-0.2, 0.2, step)],
        [7],
        rcovz,
        10000,
        pick_neuron,
        MC=MC,
        skip=skip,
    )
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
    for en, n in enumerate(pick_neuron):
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

        P_mean = (
            nprb.utils.model.compute_UCM_P_count(full_model, covariates, [n], MC=100)
            .mean(0)
            .cpu()
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
    bn = 40

    ### KS test ###
    Qq, Zz, R, Rp = [], [], [], []

    N = len(pick_neuron)
    for en, mode in enumerate(modes):
        for kcv in kcvs:

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

            q_ = []
            Z_ = []
            for b in range(full_model.inputs.batches):  # predictive posterior
                P_mc = model_utils.compute_pred_P(
                    full_model, b, pick_neuron, None, cov_samples=10, ll_samples=1, tr=0
                )
                P = P_mc.mean(0).cpu().numpy()

                for n in range(N):
                    spike_binned = full_model.likelihood.spikes[b][
                        0, pick_neuron[n], :
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
            for n in range(len(pick_neuron)):
                for m in range(n + 1, len(pick_neuron)):
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
                qq, alpha=0.05, alpha_s=0.05
            )
            T_DS.append(T_DS_)
            T_KS.append(T_KS_)

            Z_DS = T_DS / np.sqrt(2 / (qq.shape[0] - 1))
            q_DS.append(utils.stats.Z_to_q(Z_DS))

    Fisher_Z = np.array(Fisher_Z).reshape(len(CV), len(Ms), -1)
    Fisher_q = np.array(Fisher_q).reshape(len(CV), len(Ms), -1)

    Qq = np.array(Qq).reshape(len(CV), len(Ms), len(pick_neuron), -1)
    Zz = np.array(Zz).reshape(len(CV), len(Ms), len(pick_neuron), -1)
    R = np.array(R).reshape(len(CV), len(Ms), len(pick_neuron), -1)
    Rp = np.array(Rp).reshape(len(CV), len(Ms), len(pick_neuron), -1)

    q_DS = np.array(q_DS).reshape(len(CV), len(Ms), len(pick_neuron), -1)
    T_DS = np.array(T_DS).reshape(len(CV), len(Ms), len(pick_neuron), -1)
    T_KS = np.array(T_KS).reshape(len(CV), len(Ms), len(pick_neuron), -1)

    # KS test on population statistics
    T_KS_fishq = []
    p_KS_fishq = []
    for q in fisher_q:
        for qq in q:
            _, T_KS, _, _, _, p_KS = utils.stats.KS_DS_statistics(
                qq, alpha=0.05, alpha_s=0.05
            )
            T_KS_fishq.append(T_KS)
            p_KS_fishq.append(p_KS)

    T_KS_fishq = np.array(T_KS_fishq).reshape(len(CV), len(Ms))
    p_KS_fishq = np.array(p_KS_fishq).reshape(len(CV), len(Ms))

    T_KS_ks = []
    p_KS_ks = []
    for q in Qq:
        for qq in q:
            for qqq in qq:
                _, T_KS, _, _, _, p_KS = utils.stats.KS_DS_statistics(
                    qqq, alpha=0.05, alpha_s=0.05
                )
                T_KS_ks.append(T_KS)
                p_KS_ks.append(p_KS)

    T_KS_ks = np.array(T_KS_ks).reshape(len(CV), len(Ms), len(pick_neuron))
    p_KS_ks = np.array(p_KS_ks).reshape(len(CV), len(Ms), len(pick_neuron))

    population_KS = {
        "T_KS_fishq": T_KS_fishq,
        "significance_KS_fishq": significance_KS_fishq,
        "T_KS_ks": T_KS_ks,
        "significance_KS_ks": significance_KS_ks,
    }

    # delayed noise or spatiotemporal correlations
    NN = len(pick_neuron)
    delays = np.arange(5)
    R_mat_spt = np.empty((len(Ms), len(delays), NN, NN))
    R_mat_sptp = np.empty((len(Ms), len(delays), NN, NN))

    kcv_ind = 1
    for d, Z_ in enumerate(Zz[kcv_ind]):
        steps = len(Z_[0]) - len(delays)

        for en, t in enumerate(delays):
            Pearson_s = []
            for n in range(NN):
                for m in range(NN):
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
    nc_config_names = [
        "th1_U-el-3_svgp-64_X[hd-omega-speed-x-y-time]_Z[]_40K11_0d0_10f",
        "th1_U-el-3_svgp-72_X[hd-omega-speed-x-y-time]_Z[R1]_40K11_0d0_10f",
        "th1_U-el-3_svgp-80_X[hd-omega-speed-x-y-time]_Z[R2]_40K11_0d0_10f",
        "th1_U-el-3_svgp-88_X[hd-omega-speed-x-y-time]_Z[R3]_40K11_0d0_10f",
        "th1_U-el-3_svgp-96_X[hd-omega-speed-x-y-time]_Z[R4]_40K11_0d0_10f",
    ]

    best_name = [
        "th1_U-el-3_svgp-80_X[hd-omega-speed-x-y-time]_Z[R2]_40K11_0d0_10f-1",
    ]

    ### load dataset ###
    data_type = "th1"
    bin_size = 40

    dataset_dict = models.get_dataset(data_type, bin_size, data_path)

    ### analysis ###
    variability_dict = variability_stats(
        checkpoint_dir, nc_config_names, dataset_dict, device
    )
    noisecorr_dict = noise_correlations(
        checkpoint_dir, nc_config_names, dataset_dict, device
    )
    bestmodel_dict = best_model(checkpoint_dir, best_name, dataset_dict, device)

    ### export ###
    data_run = {
        "variability": variability_dict,
        "noise_correlations": noisecorr_dict,
        "best_model": bestmodel_dict,
    }

    pickle.dump(data_run, open(save_dir + "th1_NC_results.p", "wb"))


if __name__ == "__main__":
    main()
