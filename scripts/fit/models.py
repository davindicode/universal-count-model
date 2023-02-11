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

sys.path.append("../..")

import neuroprob as nprb
from neuroprob import kernels, utils
from neuroprob.mappings.base import _input_mapping


### data ###
def get_dataset(data_type, bin_size, path):

    if data_type == "th1" or data_type == "th1leftover":
        data = np.load(path + "Mouse28_140313_wake.npz")

        if data_type == "th1":
            sel_unit = data["hdc_unit"]
        else:
            sel_unit = ~data["hdc_unit"]

        neuron_regions = data["neuron_regions"][sel_unit]  # 1 is ANT, 0 is PoS
        spktrain = data["spktrain"][sel_unit, :]

        x_t = data["x_t"]
        y_t = data["y_t"]
        hd_t = data["hd_t"]

        sample_bin = 0.001
        track_samples = spktrain.shape[1]

        tbin, resamples, rc_t, (rhd_t, rx_t, ry_t) = utils.neural.bin_data(
            bin_size,
            sample_bin,
            spktrain,
            track_samples,
            (np.unwrap(hd_t), x_t, y_t),
            average_behav=True,
            binned=True,
        )

        # recompute velocities
        rw_t = (rhd_t[1:] - rhd_t[:-1]) / tbin
        rw_t = np.concatenate((rw_t, rw_t[-1:]))

        rvx_t = (rx_t[1:] - rx_t[:-1]) / tbin
        rvy_t = (ry_t[1:] - ry_t[:-1]) / tbin
        rs_t = np.sqrt(rvx_t**2 + rvy_t**2)
        rs_t = np.concatenate((rs_t, rs_t[-1:]))
        rtime_t = np.arange(resamples) * tbin

        rcov = {
            "hd": rhd_t % (2 * np.pi),
            "omega": rw_t,
            "speed": rs_t,
            "x": rx_t,
            "y": ry_t,
            "time": rtime_t,
        }

        metainfo = {
            "neuron_regions": neuron_regions,
        }

    else:  # synthetic
        assert bin_size == 1
        syn_data = np.load(path + data_type + ".npz")
        metainfo = {}

        if data_type[:4] == "hCMP":
            rcov = {
                "hd": syn_data["hd_t"],
            }

        elif data_type[:5] == "modIP":
            rcov = {"hd": syn_data["hd_t"], "a": syn_data["a_t"]}

        rc_t = syn_data["spktrain"]
        tbin = syn_data["tbin"].item()

        units_used, resamples = rc_t.shape

    name = data_type

    # export
    units_used = rc_t.shape[0]
    max_count = int(rc_t.max())

    dataset_dict = {
        "name": name,
        "covariates": rcov,
        "spiketrains": rc_t,
        "neurons": units_used,
        "metainfo": metainfo,
        "tbin": tbin,
        "timesamples": resamples,
        "max_count": max_count,
        "bin_size": bin_size,
        "trial_sizes": None,
    }
    return dataset_dict


### model ###
class Siren(nn.Module):
    """
    Sinusoidal activation function class (SIREN)
    """

    def __init__(self):
        super().__init__()

    def forward(self, input):
        return torch.sin(input)


class FFNN(_input_mapping):
    """
    Artificial neural network rate model.
    """

    def __init__(
        self,
        input_dim,
        angle_dims,
        out_dims,
        layers,
        nonlin=Siren,
        tensor_type=torch.float,
        active_dims=None,
        bias=True,
    ):
        """
        :param nn.Module mu_ANN: ANN parameterizing the mean function mapping
        :param nn.Module sigma_ANN: ANN paramterizing the standard deviation mapping if stochastic
        """
        super().__init__(input_dim, out_dims, tensor_type, active_dims)
        euclid_dims = input_dim - angle_dims
        self.angle_dims = angle_dims

        in_dims = 2 * angle_dims + euclid_dims
        net = nn.ModuleList([])
        if len(layers) == 0:
            net.append(nn.Linear(in_dims, out_dims, bias=bias))
        else:
            net.append(nn.Linear(in_dims, layers[0], bias=bias))
            net.append(nonlin())
            for k in range(len(layers) - 1):
                net.append(nn.Linear(layers[k], layers[k + 1], bias=bias))
                net.append(nonlin())
                # net.append(nn.BatchNorm1d())
            net.append(nn.Linear(layers[-1:][0], out_dims, bias=bias))

        self.add_module("net", nn.Sequential(*net))

    def compute_F(self, XZ):
        """
        The input to the ANN will be of shape (samples*timesteps, dims).

        :param torch.Tensor cov: covariates with shape (samples, timesteps, dims)
        :returns: inner product with shape (samples, neurons, timesteps)
        :rtype: torch.tensor
        """
        XZ = self._XZ(XZ)
        incov = XZ.view(-1, XZ.shape[-1])

        embed = torch.cat(
            (
                torch.cos(incov[:, : self.angle_dims]),
                torch.sin(incov[:, : self.angle_dims]),
                incov[:, self.angle_dims :],
            ),
            dim=-1,
        )
        mu = self.net(embed).view(*XZ.shape[:2], -1).permute(0, 2, 1)
        return mu, 0

    def sample_F(self, XZ):
        self.compute_F(XZ)[0]


def enc_used(model_dict, covariates, learn_mean):
    """
    Construct the neural encoding mapping module
    """
    ll_mode, map_mode, x_mode, z_mode = (
        model_dict["ll_mode"],
        model_dict["map_mode"],
        model_dict["x_mode"],
        model_dict["z_mode"],
    )
    jitter, tensor_type = model_dict["jitter"], model_dict["tensor_type"]
    neurons, in_dims = (
        model_dict["neurons"],
        model_dict["map_xdims"] + model_dict["map_zdims"],
    )

    out_dims = model_dict["map_outdims"]
    mean = torch.zeros((out_dims)) if learn_mean else 0  # not learnable vs learnable

    map_mode_comps = map_mode.split("-")
    x_mode_comps = x_mode.split("-")

    def get_inducing_locs_and_ls(comp):
        if comp == "hd":
            locs = np.linspace(0, 2 * np.pi, num_induc + 1)[:-1]
            ls = 5.0 * np.ones(out_dims)
        elif comp == "omega":
            scale = covariates["omega"].std()
            locs = scale * np.random.randn(num_induc)
            ls = scale * np.ones(out_dims)
        elif comp == "speed":
            scale = covariates["speed"].std()
            locs = np.random.uniform(0, scale, size=(num_induc,))
            ls = 10.0 * np.ones(out_dims)
        elif comp == "x":
            left_x = covariates["x"].min()
            right_x = covariates["x"].max()
            locs = np.random.uniform(left_x, right_x, size=(num_induc,))
            ls = (right_x - left_x) / 10.0 * np.ones(out_dims)
        elif comp == "y":
            bottom_y = covariates["y"].min()
            top_y = covariates["y"].max()
            locs = np.random.uniform(bottom_y, top_y, size=(num_induc,))
            ls = (top_y - bottom_y) / 10.0 * np.ones(out_dims)
        elif comp == "time":
            scale = covariates["time"].max()
            locs = np.linspace(0, scale, num_induc)
            ls = scale / 2.0 * np.ones(out_dims)
        else:
            raise ValueError("Invalid covariate type")

        return locs, ls, (comp == "hd")

    if map_mode_comps[0] == "svgp":
        num_induc = int(map_mode_comps[1])

        var = 1.0  # initial kernel variance
        v = var * torch.ones(out_dims)

        ind_list = []
        kernel_tuples = [("variance", v)]
        ang_ls, euclid_ls = [], []

        # x
        for xc in x_mode_comps:
            if xc == "":
                continue

            locs, ls, angular = get_inducing_locs_and_ls(xc)

            ind_list += [locs]
            if angular:
                ang_ls += [ls]
            else:
                euclid_ls += [ls]

        if len(ang_ls) > 0:
            ang_ls = np.array(ang_ls)
            kernel_tuples += [("SE", "ring", torch.tensor(ang_ls))]
        if len(euclid_ls) > 0:
            euclid_ls = np.array(euclid_ls)
            kernel_tuples += [("SE", "euclid", torch.tensor(euclid_ls))]

        # z
        latent_k, latent_u = latent_kernel(z_mode, num_induc, out_dims)
        kernel_tuples += latent_k
        ind_list += latent_u

        # objects
        kernelobj = create_kernel(kernel_tuples, "exp", tensor_type)

        Xu = torch.tensor(np.array(ind_list)).T[None, ...].repeat(out_dims, 1, 1)
        inpd = Xu.shape[-1]
        inducing_points = nprb.mappings.inducing_points(
            out_dims,
            Xu,
            jitter=jitter,
            tensor_type=tensor_type,
        )

        mapping = nprb.mappings.SVGP(
            in_dims,
            out_dims,
            kernelobj,
            inducing_points=inducing_points,
            jitter=jitter,
            whiten=True,
            mean=mean,
            learn_mean=learn_mean,
            tensor_type=tensor_type,
        )

    elif map_mode_comps[0] == "ffnn":  # feedforward neural network mapping
        angle_dims = 0

        for xc in x_mode_comps:

            if xc == "hd" or xc == "omega":
                angle_dims += 1

        for zc in x_mode_comps:
            if zc[:1] == "T":
                dz = int(zc[1:])
                angle_dims += dz

        enc_layers = [int(ls) for ls in map_mode_comps[1:]]
        mapping = FFNN(
            in_dims, angle_dims, out_dims, enc_layers, tensor_type=torch.float
        )

    else:
        raise ValueError

    return mapping


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
    """
    Create the kernel tuples and inducing point lists for latent spaces
    """
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
                p = nprb.inputs.priors.tAR1(
                    torch.tensor(0.0),
                    torch.tensor(4.0),
                    "ring",
                    1,
                    tensor_type=tensor_type,
                )
            else:
                p = nprb.inputs.priors.tAR1(
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

            latents += [nprb.inputs.prior_variational_pair(d_z, p, v)]

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
        r_inv = 1.0 * torch.ones(inner_dims)
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
        J = int(ll_mode_comps[1])
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

    name = model_dict["model_name"] + "_{}_{}_X[{}]_Z[{}]_{}K{}_{}d{}_{}f{}".format(
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
        "--checkpoint_dir", action="store", type=str
    )

    parser.add_argument("--tensor_type", default="float", action="store", type=str)
    parser.add_argument("--cpu", dest="cpu", action="store_true")
    parser.set_defaults(cpu=False)

    parser.add_argument("--batch_size", default=10000, type=int)
    parser.add_argument("--cv", nargs="+", type=int)  # if -1, fit to all data
    parser.add_argument("--cv_folds", default=5, type=int)
    parser.add_argument("--bin_size", type=int)

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
    parser.add_argument(
        "--scheduler_interval", default=100, type=int
    )  # if 0, no scheduler applied
    parser.add_argument(
        "--loss_margin", default=-1e0, type=float
    )  # minimum loss increase
    parser.add_argument(
        "--margin_epochs", default=100, type=int
    )  # epochs over which this must happen

    parser.add_argument("--likelihood", action="store", type=str)
    parser.add_argument("--mapping", default="", action="store", type=str)
    parser.add_argument("--x_mode", default="", action="store", type=str)
    parser.add_argument("--z_mode", default="", action="store", type=str)
    parser.add_argument("--delays", nargs="+", default=[0], type=int)
    return parser


def preprocess_data(dataset_dict, folds, delays, cv_runs, batchsize, has_latent=False):
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


def setup_model(data_dict, model_dict, enc_used):
    """ "
    Assemble the encoding model
    """
    spktrain = data_dict["spiketrain"]
    cov = data_dict["covariates"]
    batch_info = data_dict["batch_info"]

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

    learn_mean = ll_mode_comps[0] != "U"
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


def train_model(dev, parser_args, dataset_dict):
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
        fit_data = {
            "spiketrain": cvdata["spiketrain_fit"],
            "covariates": cvdata["covariates_fit"],
            "batch_info": cvdata["batching_info_fit"],
        }
        model_name = gen_name(model_dict, cvdata["delay"], cvdata["fold"])
        print(model_name)

        # fitting
        for seed in seeds:
            model_dict["seed"] = seed
            print("seed: {}".format(seed))

            try:  # attempt to fit model
                full_model = setup_model(fit_data, model_dict, enc_used)
                full_model.to(dev)

                sch = lambda o: optim.lr_scheduler.MultiplicativeLR(
                    o, lambda e: parser_args.scheduler_factor
                )
                opt_lr_dict = {"default": parser_args.lr}

                # set learning rates for special cases
                if z_mode == "T1":
                    opt_lr_dict["mapping.kernel.kern1._lengthscale"] = parser_args.lr_2
                for z_dim in full_model.input_group.latent_dims:
                    opt_lr_dict[
                        "input_group.input_{}.variational.finv_std".format(z_dim)
                    ] = parser_args.lr_2

                full_model.set_optimizers(
                    optim.Adam, sch, parser_args.scheduler_interval, opt_lr_dict
                )

                losses = full_model.fit(
                    parser_args.max_epochs,
                    loss_margin=parser_args.loss_margin,
                    margin_epochs=parser_args.margin_epochs,
                    kl_anneal_func=lambda x: 1.0,  # no KL annealing
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
    batch_info,
    device,
):
    """
    Load the model with cross-validated data structure
    """
    checkpoint = torch.load(checkpoint_dir + config_name + ".pt", map_location=device)

    delay, cv_run = checkpoint["delay"], checkpoint["cv_run"]
    config = checkpoint["config"]
    model_dict = extract_model_dict(config, dataset_dict)
    model_dict["seed"] = checkpoint["seed"]

    has_latent = False if model_dict["z_mode"] == "" else True
    cvdata = preprocess_data(
        dataset_dict,
        model_dict["folds"],
        [delay],
        [cv_run],
        batch_info,
        has_latent,
    )[0]

    fit_data = {
        "spiketrain": cvdata["spiketrain_fit"],
        "covariates": cvdata["covariates_fit"],
        "batch_info": cvdata["batching_info_fit"],
    }

    ### model ###
    full_model = setup_model(fit_data, model_dict, enc_used)
    full_model.to(device)

    ### load ###
    model_name = gen_name(model_dict, delay, cv_run)
    full_model.load_state_dict(checkpoint["full_model"])

    # return
    fit_dict = {
        "covariates": inputs_used(model_dict, cvdata["covariates_fit"], batch_info)[0],
        "spiketrain": torch.from_numpy(cvdata["spiketrain_fit"]),
        "batch_info": cvdata["batching_info_fit"],
    }
    val_dict = {
        "covariates": inputs_used(model_dict, cvdata["covariates_val"], batch_info)[0]
        if cvdata["covariates_val"] is not None
        else None,
        "spiketrain": torch.from_numpy(cvdata["spiketrain_val"])
        if cvdata["spiketrain_val"] is not None
        else None,
        "batch_info": cvdata["batching_info_val"],
    }
    training_loss = checkpoint["training_loss"]

    return full_model, training_loss, fit_dict, val_dict


### cross validation ###
def RG_pred_ll(
    model,
    val_dict,
    neuron_group=None,
    ll_mode="GH",
    ll_samples=100,
    cov_samples=1,
    beta=1.0,
):
    """
    Compute the variational log likelihood (beta = 0.) or ELBO (beta = 1.)
    """
    vtrain = val_dict["spiketrain"]
    vcov = val_dict["covariates"]
    vbatch_info = val_dict["batch_info"]

    time_steps = vtrain.shape[-1]
    # print("Data segment timesteps: {}".format(time_steps))

    model.input_group.set_XZ(vcov, time_steps, batch_info=vbatch_info)
    model.likelihood.set_Y(vtrain, batch_info=vbatch_info)
    model.validate_model(likelihood_set=True)

    # batching
    pll = []
    for b in range(model.input_group.batches):
        pll.append(
            -model.objective(
                b,
                cov_samples=cov_samples,
                ll_mode=ll_mode,
                neuron=neuron_group,
                beta=beta,
                ll_samples=ll_samples,
            ).item()
        )

    return np.array(pll).mean()  # mean over each subsampled estimate


def LVM_pred_ll(
    model,
    val_dict,
    fit_neurons,
    val_neurons,
    eval_cov_MC=1,
    eval_ll_MC=100,
    eval_ll_mode="GH",
    annealing=lambda x: 1.0,  # min(1.0, 0.002*x)
    cov_MC=16,
    ll_MC=1,
    ll_mode="MC",
    beta=1.0,
    max_iters=3000,
):
    """
    Compute the variational log likelihood (beta = 0.) or ELBO (beta = 1.)
    """
    vtrain = val_dict["spiketrain"]
    vcov = val_dict["covariates"]
    vbatch_info = val_dict["batch_info"]

    time_steps = vtrain.shape[-1]
    # print("Data segment timesteps: {}".format(time_steps))

    model.input_group.set_XZ(vcov, time_steps, batch_info=vbatch_info)
    model.likelihood.set_Y(vtrain, batch_info=vbatch_info)
    model.validate_model(likelihood_set=True)

    # fit
    sch = lambda o: optim.lr_scheduler.MultiplicativeLR(o, lambda e: 0.9)
    opt_tuple = (optim.Adam, 100, sch)
    opt_lr_dict = {"default": 0}
    for z_dim in model.input_group.latent_dims:
        opt_lr_dict["input_group.input_{}.variational.mu".format(z_dim)] = 1e-2
        opt_lr_dict["input_group.input_{}.variational.finv_std".format(z_dim)] = 1e-3

    model.set_optimizers(
        opt_tuple, opt_lr_dict
    )  # , nat_grad=('rate_model.0.u_loc', 'rate_model.0.u_scale_tril'))

    losses = model.fit(
        max_iters,
        neuron=fit_neurons,
        loss_margin=-1e0,
        margin_epochs=100,
        ll_mode=ll_mode,
        kl_anneal_func=annealing,
        cov_samples=cov_MC,
        ll_samples=ll_MC,
    )

    pll = []
    for b in range(model.input_group.batches):
        pll.append(
            -model.objective(
                b,
                neuron=val_neurons,
                cov_samples=eval_cov_MC,
                ll_mode=eval_ll_mode,
                beta=beta,
                ll_samples=eval_ll_MC,
            ).item()
        )

    return np.array(pll).mean(), losses  # mean over each subsampled estimate


### main ###
def main():
    parser = standard_parser("%(prog)s [OPTION] [FILE]...", "Fit model to data.")
    parser.add_argument("--data_path", action="store", type=str)
    parser.add_argument("--data_type", action="store", type=str)

    args = parser.parse_args()

    if args.cpu:
        dev = "cpu"
    else:
        dev = nprb.inference.get_device(gpu=args.gpu)

    dataset_dict = get_dataset(args.data_type, args.bin_size, args.data_path)

    train_model(dev, args, dataset_dict)


if __name__ == "__main__":
    main()
