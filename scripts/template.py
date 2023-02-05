import argparse
import os

import pickle
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.parameter import Parameter

sys.path.append("../lib/")

import neuroprob as nprb
from neuroprob import kernels, utils



### GP ###
def create_kernel(kernel_tuples, kern_f, tensor_type):
    """
    Helper function for creating kernel triplet tuple
    """
    track_dims = 0
    kernelobj = 0

    for k, k_tuple in enumerate(kernel_tuples):

        if k_tuple[0] is not None:

            if k_tuple[0] == "variance":
                krn = kernels.Constant(variance=k_tuple[1], tensor_type=tensor_type)

            else:
                kernel_type = k_tuple[0]
                topology = k_tuple[1]
                lengthscales = k_tuple[2]

                act = []
                for _ in lengthscales:
                    act += [track_dims]
                    track_dims += 1

                if kernel_type == "SE":
                    krn = kernels.SquaredExponential(
                        input_dims=len(lengthscales),
                        lengthscale=lengthscales,
                        track_dims=act,
                        topology=topology,
                        f=kern_f,
                        tensor_type=tensor_type,
                    )

                elif kernel_type == "DSE":
                    if topology != "euclid":
                        raise ValueError("Topology must be euclid")
                    lengthscale_beta = k_tuple[3]
                    beta = k_tuple[4]
                    krn = kernels.DSE(
                        input_dims=len(lengthscales),
                        lengthscale=lengthscales,
                        lengthscale_beta=lengthscale_beta,
                        beta=beta,
                        track_dims=act,
                        f=kern_f,
                        tensor_type=tensor_type,
                    )

                elif kernel_type == "OU":
                    krn = kernels.Exponential(
                        input_dims=len(lengthscales),
                        lengthscale=lengthscales,
                        track_dims=act,
                        topology=topology,
                        f=kern_f,
                        tensor_type=tensor_type,
                    )

                elif kernel_type == "RQ":
                    scale_mixture = k_tuple[3]
                    krn = kernels.RationalQuadratic(
                        input_dims=len(lengthscales),
                        lengthscale=lengthscales,
                        scale_mixture=scale_mixture,
                        track_dims=act,
                        topology=topology,
                        f=kern_f,
                        tensor_type=tensor_type,
                    )

                elif kernel_type == "Matern32":
                    krn = kernels.Matern32(
                        input_dims=len(lengthscales),
                        lengthscale=lengthscales,
                        track_dims=act,
                        topology=topology,
                        f=kern_f,
                        tensor_type=tensor_type,
                    )

                elif kernel_type == "Matern52":
                    krn = kernels.Matern52(
                        input_dims=len(lengthscales),
                        lengthscale=lengthscales,
                        track_dims=act,
                        topology=topology,
                        f=kern_f,
                        tensor_type=tensor_type,
                    )

                elif kernel_type == "linear":
                    if topology != "euclid":
                        raise ValueError("Topology must be euclid")
                    krn = kernels.Linear(
                        input_dims=len(lengthscales), track_dims=act, f=kern_f
                    )

                elif kernel_type == "polynomial":
                    if topology != "euclid":
                        raise ValueError("Topology must be euclid")
                    degree = k_tuple[3]
                    krn = kernels.Polynomial(
                        input_dims=len(lengthscales),
                        bias=lengthscales,
                        degree=degree,
                        track_dims=act,
                        f=kern_f,
                        tensor_type=tensor_type,
                    )

                else:
                    raise NotImplementedError("Kernel type is not supported.")

            kernelobj = kernels.Product(kernelobj, krn) if kernelobj != 0 else krn

        else:
            track_dims += 1

    return kernelobj


def latent_kernel(z_mode, num_induc, out_dims):
    """ """
    z_mode_comps = z_mode.split("-")

    ind_list = []
    kernel_tuples = []

    l_one = np.ones(out_dims)

    for zc in z_mode_comps:
        if zc[:1] == "R":
            dz = int(zc[1:])
            for h in range(dz):
                ind_list += [np.random.randn(num_induc)]
            ls = np.array([l_one] * dz)
            kernel_tuples += [("SE", "euclid", torch.tensor(ls))]

        elif zc[:1] == "T":
            dz = int(zc[1:])
            for h in range(dz):
                ind_list += [np.linspace(0, 2 * np.pi, num_induc + 1)[:-1]]
            ls = np.array([10.0 * l_one] * dz)
            kernel_tuples += [("SE", "euclid", torch.tensor(ls))]

        elif zc != "":
            raise ValueError("Invalid latent covariate type")

    return kernel_tuples, ind_list


### model components ###
def latent_objects(z_mode, d_x, timesamples, tensor_type):
    """
    Create latent state prior and variational distribution
    """
    z_mode_comps = z_mode.split("-")

    tot_d_z, latents = 0, []
    for zc in z_mode_comps:
        d_z = 0

        if zc[:1] == "R":
            d_z = int(zc[1:])

            if d_z == 1:
                p = nprb.inputs.priors.AR1(
                    torch.tensor(0.0), torch.tensor(4.0), 1, tensor_type=tensor_type
                )
            else:
                p = nprb.inputs.priors.AR1(
                    torch.tensor([0.0] * d_z),
                    torch.tensor([4.0] * d_z),
                    d_z,
                    tensor_type=tensor_type,
                )

            v = nprb.inputs.variational.IndNormal(
                torch.rand(timesamples, d_z) * 0.1,
                torch.ones((timesamples, d_z)) * 0.01,
                "euclid",
                d_z,
                tensor_type=tensor_type,
            )

            latents += [nprb.inputs.prior_variational_pair(d_z, p, v)]

        elif zc[:1] == "T":
            d_z = int(zc[1:])

            if d_z == 1:
                p = nprb.inputs.priors.dAR1(
                    torch.tensor(0.0),
                    torch.tensor(4.0),
                    "ring",
                    1,
                    tensor_type=tensor_type,
                )
            else:
                p = nprb.inputs.priors.dAR1(
                    torch.tensor([0.0] * d_z),
                    torch.tensor([4.0] * d_z),
                    "ring",
                    d_z,
                    tensor_type=tensor_type,
                )

            v = nprb.inputs.variational.IndNormal(
                torch.rand(timesamples, 1) * 2 * np.pi,
                torch.ones((timesamples, 1)) * 0.1,  # 0.01
                "ring",
                d_z,
                tensor_type=tensor_type,
            )

            latents += [nprb.inputs.prior_variational_pair(_z, p, v)]

        elif zc != "":
            raise ValueError("Invalid latent covariate type")

        tot_d_z += d_z

    return latents, tot_d_z


def inputs_used(model_dict, covariates, batch_info):
    """
    Create the used covariates list.
    """
    x_mode, z_mode, tensor_type = (
        model_dict["x_mode"],
        model_dict["z_mode"],
        model_dict["tensor_type"],
    )
    x_mode_comps = x_mode.split("-")

    input_data = []
    for xc in x_mode_comps:
        if xc == "":
            continue
        input_data.append(torch.from_numpy(covariates[xc]))

    d_x = len(input_data)

    timesamples = list(covariates.values())[0].shape[0]
    latents, d_z = latent_objects(z_mode, d_x, timesamples, tensor_type)
    input_data += latents
    return input_data, d_x, d_z


def get_likelihood(model_dict, cov, enc_used):
    """
    Create the likelihood object.
    """
    ll_mode, tensor_type = model_dict["ll_mode"], model_dict["tensor_type"]
    ll_mode_comps = ll_mode.split("-")
    C = int(ll_mode_comps[2]) if ll_mode_comps[0] == "U" else 1

    max_count, neurons, tbin = (
        model_dict["max_count"],
        model_dict["neurons"],
        model_dict["tbin"],
    )
    inner_dims = model_dict["map_outdims"]

    if ll_mode[0] == "h":
        hgp = enc_used(model_dict, cov, learn_mean=True)

    if ll_mode_comps[0] == "U":
        inv_link = "identity"
    elif ll_mode == "IBP":
        inv_link = lambda x: torch.sigmoid(x) / tbin
    elif ll_mode_comps[-1] == "exp":
        inv_link = "exp"
    elif ll_mode_comps[-1] == "spl":
        inv_link = "softplus"
    else:
        raise ValueError("Likelihood inverse link function not defined")

    if ll_mode_comps[0] == "IBP":
        likelihood = nprb.likelihoods.Bernoulli(
            tbin, inner_dims, tensor_type=tensor_type
        )

    elif ll_mode_comps[0] == "IP":
        likelihood = nprb.likelihoods.Poisson(
            tbin, inner_dims, inv_link, tensor_type=tensor_type
        )

    elif ll_mode_comps[0] == "ZIP":
        alpha = 0.1 * torch.ones(inner_dims)
        likelihood = nprb.likelihoods.ZI_Poisson(
            tbin, inner_dims, inv_link, alpha, tensor_type=tensor_type
        )

    elif ll_mode_comps[0] == "hZIP":
        likelihood = nprb.likelihoods.hZI_Poisson(
            tbin, inner_dims, inv_link, hgp, tensor_type=tensor_type
        )

    elif ll_mode_comps[0] == "NB":
        r_inv = 10.0 * torch.ones(inner_dims)
        likelihood = nprb.likelihoods.Negative_binomial(
            tbin, inner_dims, inv_link, r_inv, tensor_type=tensor_type
        )

    elif ll_mode_comps[0] == "hNB":
        likelihood = nprb.likelihoods.hNegative_binomial(
            tbin, inner_dims, inv_link, hgp, tensor_type=tensor_type
        )

    elif ll_mode_comps[0] == "CMP":
        J = int(ll_mode_comps[1])
        log_nu = torch.zeros(inner_dims)
        likelihood = nprb.likelihoods.COM_Poisson(
            tbin, inner_dims, inv_link, log_nu, J=J, tensor_type=tensor_type
        )

    elif ll_mode_comps[0] == "hCMP":
        J = int(ll_mode[4:])
        likelihood = nprb.likelihoods.hCOM_Poisson(
            tbin, inner_dims, inv_link, hgp, J=J, tensor_type=tensor_type
        )

    elif ll_mode_comps[0] == "U":
        basis_mode = ll_mode_comps[1]
        likelihood = nprb.likelihoods.Universal(
            neurons, C, basis_mode, inv_link, max_count, tensor_type=tensor_type
        )

    else:
        raise NotImplementedError

    return likelihood


def gen_name(model_dict, delay, fold):
    delaystr = "".join(str(d) for d in model_dict["delays"])

    name = model_dict[
        "model_name"
    ] + "_{}_{}_X[{}]_Z[{}]_{}K{}_{}d{}_{}f{}".format(
        model_dict["ll_mode"],
        model_dict["map_mode"],
        model_dict["x_mode"],
        model_dict["z_mode"],
        model_dict["bin_size"],
        model_dict["max_count"],
        delaystr,
        delay,
        model_dict["folds"],
        fold,
    )
    return name


### script ###
def standard_parser(usage, description):
    """
    Parser arguments belonging to training loop
    """
    parser = argparse.ArgumentParser(usage=usage, description=description)
    parser.add_argument(
        "-v", "--version", action="version", version=f"{parser.prog} version 1.0.0"
    )
    parser.add_argument(
        "--checkpoint_dir", default="./checkpoint/", action="store", type=str
    )

    parser.add_argument("--tensor_type", default="float", action="store", type=str)

    parser.add_argument("--batch_size", default=10000, type=int)
    parser.add_argument("--cv", nargs="+", type=int)
    parser.add_argument("--cv_folds", default=5, type=int)
    parser.add_argument("--bin_size", type=int)
    parser.add_argument("--single_spikes", dest="single_spikes", action="store_true")
    parser.set_defaults(single_spikes=False)

    parser.add_argument("--seeds", default=[123], nargs="+", type=int)
    parser.add_argument("--gpu", default=0, type=int)
    parser.add_argument("--cov_MC", default=1, type=int)
    parser.add_argument("--ll_MC", default=10, type=int)
    parser.add_argument("--integral_mode", default="MC", action="store", type=str)

    parser.add_argument("--jitter", default=1e-5, type=float)
    parser.add_argument("--max_epochs", default=3000, type=int)
    parser.add_argument("--lr", default=1e-2, type=float)
    parser.add_argument("--lr_2", default=1e-3, type=float)

    parser.add_argument("--scheduler_factor", default=0.9, type=float)
    parser.add_argument("--scheduler_interval", default=100, type=int)
    parser.add_argument("--loss_margin", default=-1e0, type=float)
    parser.add_argument("--margin_epochs", default=100, type=int)

    parser.add_argument("--likelihood", action="store", type=str)
    parser.add_argument("--mapping", default="", action="store", type=str)
    parser.add_argument("--x_mode", default="", action="store", type=str)
    parser.add_argument("--z_mode", default="", action="store", type=str)
    parser.add_argument("--delays", nargs="+", default=[0], type=int)
    return parser


def preprocess_data(
    dataset_dict, folds, delays, cv_runs, batchsize, has_latent=False
):
    """
    Returns delay shifted cross-validated data for training
    rcov list of arrays of shape (neurons, time, 1)
    rc_t array of shape (trials, neurons, time) or (neurons, time)

    Data comes in as stream of data, trials are appended consecutively
    """
    rc_t, resamples, rcov = (
        dataset_dict["spiketrains"],
        dataset_dict["timesamples"],
        dataset_dict["covariates"],
    )
    trial_sizes = dataset_dict["trial_sizes"]
    returns = []

    # need trial as tensor dimension for these
    if trial_sizes is not None:
        if delays != [0]:
            raise ValueError("Delays not supported in appended trials")

    if delays != [0]:
        if min(delays) > 0:
            raise ValueError("Delay minimum must be 0 or less")
        if max(delays) < 0:
            raise ValueError("Delay maximum must be 0 or less")

        D_min = -min(delays)
        D_max = -max(delays)
        dd = delays

    else:
        D_min = 0
        D_max = 0
        dd = [0]

    # history of spike train filter
    #rcov = {n: rc[hist_len:] for n, rc in rcov.items()}
    #resamples -= hist_len

    D = -D_max + D_min  # total delay steps - 1
    for delay in dd:

        # delays
        rc_t_ = rc_t[..., D_min : (D_max if D_max < 0 else None)]
        _min = D_min + delay
        _max = D_max + delay

        rcov_ = {n: rc[_min : (_max if _max < 0 else None)] for n, rc in rcov.items()}
        resamples_ = resamples - D

        # get cv datasets
        if trial_sizes is not None and has_latent:  # trials and has latent
            cv_sets, cv_inds = utils.neural.spiketrials_CV(
                folds, rc_t_, resamples_, rcov_, trial_sizes
            )
        else:
            cv_sets, vstart = utils.neural.spiketrain_CV(
                folds, rc_t_, resamples_, rcov_, spk_hist_len=0
            )

        for kcv in cv_runs:
            if kcv >= 0:  # CV data
                ftrain, fcov, vtrain, vcov = cv_sets[kcv]

                if has_latent:
                    if (
                        trial_sizes is None
                    ):  # continual, has latent and CV, removed validation segment is temporal disconnect
                        segment_lengths = [
                            vstart[kcv],
                            resamples_ - vstart[kcv] - vtrain.shape[-1],
                        ]
                        trial_ids = [0] * len(segment_lengths)
                        fbatch_info = utils.neural.batch_segments(
                            segment_lengths, trial_ids, batchsize
                        )
                        vbatch_info = batchsize

                    else:
                        ftr_inds, vtr_inds = cv_inds[kcv]
                        fbatch_info = utils.neural.batch_segments(
                            [trial_sizes[ind] for ind in ftr_inds], ftr_inds, batchsize
                        )
                        vbatch_info = utils.neural.batch_segments(
                            [trial_sizes[ind] for ind in vtr_inds], vtr_inds, batchsize
                        )

                else:
                    fbatch_info = batchsize
                    vbatch_info = batchsize

            else:  # full data
                ftrain, fcov = rc_t_, rcov_
                if trial_sizes is not None and has_latent:
                    trial_ids = list(range(len(trial_sizes)))
                    fbatch_info = utils.neural.batch_segments(
                        trial_sizes, trial_ids, batchsize
                    )
                else:
                    fbatch_info = batchsize

                vtrain, vcov = None, None
                vbatch_info = None

            preprocess_dict = {
                "fold": kcv,
                "delay": delay,
                "spiketrain_fit": ftrain,
                "covariates_fit": fcov,
                "batching_info_fit": fbatch_info,
                "spiketrain_val": vtrain,
                "covariates_val": vcov,
                "batching_info_val": vbatch_info,
            }
            returns.append(preprocess_dict)

    return returns


def setup_model(data_tuple, model_dict, enc_used):
    """ "
    Assemble the encoding model
    """
    spktrain, cov, batch_info = data_tuple
    neurons, timesamples = spktrain.shape[0], spktrain.shape[-1]

    ll_mode, map_mode, x_mode, z_mode, tensor_type = (
        model_dict["ll_mode"],
        model_dict["map_mode"],
        model_dict["x_mode"],
        model_dict["z_mode"],
        model_dict["tensor_type"],
    )

    # seed everything
    seed = model_dict["seed"]
    np.random.seed(seed)
    torch.manual_seed(seed)

    # inputs
    input_data, d_x, d_z = inputs_used(model_dict, cov, batch_info)
    model_dict["map_xdims"], model_dict["map_zdims"] = d_x, d_z

    input_group = nprb.inputs.input_group(tensor_type)
    input_group.set_XZ(input_data, timesamples, batch_info=batch_info)

    # encoder mapping
    ll_mode_comps = ll_mode.split("-")
    C = int(ll_mode_comps[2]) if ll_mode_comps[0] == "U" else 1
    model_dict["map_outdims"] = (
        neurons * C
    )  # number of output dimensions of the input_mapping

    learn_mean = (ll_mode_comps[0] != "U")
    mapping = enc_used(model_dict, cov, learn_mean)

    # likelihood
    likelihood = get_likelihood(model_dict, cov, enc_used)
    likelihood.set_Y(torch.from_numpy(spktrain), batch_info=batch_info)

    full = nprb.inference.VI_optimized(input_group, mapping, likelihood)
    return full


def extract_model_dict(config, dataset_dict):
    if config.tensor_type == "float":
        tensor_type = torch.float
    elif config.tensor_type == "double":
        tensor_type = torch.double
    else:
        raise ValueError("Invalid tensor type in arguments")

    folds = config.cv_folds
    delays = config.delays

    # mode
    ll_mode = config.likelihood
    map_mode = config.mapping
    x_mode = config.x_mode
    z_mode = config.z_mode

    model_dict = {
        "ll_mode": ll_mode,
        "map_mode": map_mode,
        "x_mode": x_mode,
        "z_mode": z_mode,
        "folds": folds,
        "delays": delays,
        "neurons": dataset_dict["neurons"],
        "max_count": dataset_dict["max_count"],
        "bin_size": dataset_dict["bin_size"],
        "tbin": dataset_dict["tbin"],
        "model_name": dataset_dict["name"],
        "tensor_type": tensor_type,
        "jitter": config.jitter,
    }
    return model_dict


def train_model(dev, parser_args, dataset_dict, enc_used):
    """
    General training loop

    def inputs_used(model_dict, covariates, batch_info):
        Get inputs for model

    def enc_used(model_dict, covariates, inner_dims):
        Function for generating encoding model
    """
    checkpoint_dir = parser_args.checkpoint_dir
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    seeds = parser_args.seeds

    folds = parser_args.cv_folds
    delays = parser_args.delays
    cv_runs = parser_args.cv
    batchsize = parser_args.batch_size
    z_mode = parser_args.z_mode

    model_dict = extract_model_dict(parser_args, dataset_dict)

    # training
    has_latent = False if z_mode == "" else True
    preprocessed = preprocess_data(
        dataset_dict, folds, delays, cv_runs, batchsize, has_latent
    )
    for cvdata in preprocessed:
        fitdata = (
            cvdata["spiketrain_fit"],
            cvdata["covariates_fit"],
            cvdata["batching_info_fit"],
        )
        model_name = gen_name(model_dict, cvdata["delay"], cvdata["fold"])
        print(model_name)

        # fitting
        for seed in seeds:
            model_dict["seed"] = seed
            print("seed: {}".format(seed))

            try:  # attempt to fit model
                full_model = setup_model(fitdata, model_dict, enc_used)
                full_model.to(dev)

                sch = lambda o: optim.lr_scheduler.MultiplicativeLR(
                    o, lambda e: parser_args.scheduler_factor
                )
                opt_tuple = (optim.Adam, parser_args.scheduler_interval, sch)
                opt_lr_dict = {"default": parser_args.lr}
                if z_mode == "T1":
                    opt_lr_dict["mapping.kernel.kern1._lengthscale"] = parser_args.lr_2
                for z_dim in full_model.input_group.latent_dims:
                    opt_lr_dict[
                        "input_group.input_{}.variational.finv_std".format(z_dim)
                    ] = parser_args.lr_2

                full_model.set_optimizers(opt_tuple, opt_lr_dict)

                annealing = lambda x: 1.0
                losses = full_model.fit(
                    parser_args.max_epochs,
                    loss_margin=parser_args.loss_margin,
                    margin_epochs=parser_args.margin_epochs,
                    kl_anneal_func=annealing,
                    cov_samples=parser_args.cov_MC,
                    ll_samples=parser_args.ll_MC,
                    ll_mode=parser_args.integral_mode,
                )

                # save and progress
                if os.path.exists(
                    checkpoint_dir + model_name + "_result.p"
                ):  # check previous best losses
                    with open(checkpoint_dir + model_name + "_result.p", "rb") as f:
                        results = pickle.load(f)
                        lowest_loss = results["training_loss"][-1]
                else:
                    lowest_loss = np.inf  # nonconvex optimization, pick the best

                if losses[-1] < lowest_loss:
                    if not os.path.exists(checkpoint_dir):
                        os.makedirs(checkpoint_dir)
                    
                    # save model
                    torch.save(
                        {
                            "full_model": full_model.state_dict(), 
                            "training_loss": losses,
                            "seed": seed,
                            "delay": cvdata["delay"],
                            "cv_run": cvdata["fold"],
                            "config": parser_args,
                        }, 
                        checkpoint_dir + model_name + ".pt",
                    )

            except (ValueError, RuntimeError) as e:
                print(e)


def load_model(
    config_name,
    checkpoint_dir,
    dataset_dict,
    enc_used,
    batch_info,
    device,
):
    """
    Load the model with cross-validated data structure
    """
    with open(checkpoint_dir + config_name + ".p", "rb") as f:
        training_results = pickle.load(f)

    delay, cv_run = training_results["delay"], training_results["cv_run"]
    config = training_results["config"]
    model_dict = extract_model_dict(config, dataset_dict)
    model_dict["seed"] = training_results["seed"]

    has_latent = False if model_dict["z_mode"] == "" else True
    cvdata = preprocess_data(
        dataset_dict,
        model_dict["folds"],
        [delay],
        [cv_run],
        batch_info,
        has_latent,
    )[0]

    fit_data = (
        cvdata["spiketrain_fit"],
        cvdata["covariates_fit"],
        cvdata["batching_info_fit"],
    )
    val_data = (
        cvdata["spiketrain_val"],
        cvdata["covariates_val"],
        cvdata["batching_info_val"],
    )
    fit_set = (
        inputs_used(model_dict, fit_data[1], batch_info)[0],
        torch.from_numpy(fit_data[0]),
        fit_data[2],
    )
    validation_set = (
        inputs_used(model_dict, val_data[1], batch_info)[0]
        if val_data[1] is not None
        else None,
        torch.from_numpy(val_data[0]) if val_data[0] is not None else None,
        val_data[2],
    )

    ### model ###
    full_model = setup_model(fit_data, model_dict, enc_used)
    full_model.to(device)

    ### load ###
    model_name = gen_name(model_dict, delay, cv_run)
    checkpoint = torch.load(
        checkpoint_dir + model_name + ".pt", map_location=device
    )
    full_model.load_state_dict(checkpoint["full_model"])

    return full_model, training_results, fit_set, validation_set
