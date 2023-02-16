import torch

from ..base import _data_object
from . import base


def _expand_cov(cov):
    if len(cov.shape) == 1:  # expand arrays from (timesteps,)
        cov = cov[None, None, :, None]
    elif len(cov.shape) == 2:  # expand arrays (timesteps, dims)
        cov = cov[None, None, ...]
    elif len(cov.shape) == 3:
        cov = cov[None, ...]  # expand arrays (out, timesteps, dims)

    if len(cov.shape) != 4:  # trials, out, timesteps, dims
        raise ValueError(
            "Shape of input covariates at most trials x out x timesteps x dims"
        )

    return cov


### inputs ###
class prior_variational_pair(base._VI_object):
    """
    Module pairing priors and corresponding variational posteriors
    """

    def __init__(self, dims, prior, variational):
        """
        Delta when std = 0
        """
        super().__init__(dims, prior.tensor_type)
        if prior.tensor_type != variational.tensor_type:
            raise ValueError("Tensor types of prior and variational do not match")
        self.add_module("prior", prior)
        self.add_module("variational", variational)
        self.tsteps = self.variational.tsteps

    def validate(self, tsteps, trials, batch_info):
        self.prior.validate(tsteps, trials)
        self.variational.validate(tsteps, trials)

        if self.prior.topo != self.variational.topo:
            raise ValueError(
                "Topologies in prior and variational distributions do not match"
            )

    def sample(self, b, batch_info, samples):
        """ """
        batch_edge, batch_link, batch_initial = batch_info

        offs = self.prior.AR_p if batch_link[b] else 0
        t_lower = batch_edge[b] - offs
        t_upper = batch_edge[b + 1]

        v_samp, nlog_q = self.variational.sample(t_lower, t_upper, offs, samples)
        log_p = self.prior.log_p(v_samp, batch_initial[b])
        v_samp_ = v_samp[:, offs:]

        if len(v_samp_.shape) == 2:  # one-dimensional latent variables expanded
            v_samp_ = v_samp_[:, None, :, None]
        else:
            v_samp_ = v_samp_[:, None, ...]

        nqlog_pq = -(log_p + nlog_q).mean()  # mean over trials and MC samples

        fac = self.tsteps / v_samp_.shape[-2]  # subsampling in batched mode
        return v_samp_, fac * nqlog_pq


class probabilistic_mapping(base._VI_object):
    """
    A stochastic mapping layer
    """

    def __init__(self, input_group, mapping, joint_samples=False):
        """
        The input_group and mapping form an input-output pair for the stochastic map.
        This component can be used to build the input_group to another mapping, allowing
        one to build hierarchical models as deep Gaussian processes.
        """
        super().__init__(mapping.out_dims, mapping.tensor_type)
        if mapping.tensor_type != input_group.tensor_type:
            raise ValueError("Mapping and input group tensor types do not match")
        self.add_module("mapping", mapping)
        self.add_module("input_group", input_group)
        self.joint_samples = joint_samples

    def constrain(self):
        self.mapping.constrain()
        self.input_group.constrain()

    def validate(self, tsteps, trials, batch_info):
        if tsteps != self.input_group.tsteps:
            raise ValueError(
                "Time steps of mapping input does not match expected time steps"
            )
        if trials != self.input_group.trials:
            raise ValueError(
                "Trial count of mapping input does not match expected trial count"
            )
        if batch_info != self.input_group.batch_info:
            raise ValueError("Nested input batching structures do not match")

    def sample(self, b, batch_info, samples):
        """
        Sample from the posterior of the mapping
        """
        t_, KL_prior = self.input_group.sample_XZ(
            b,
            1,
        )  # samples, timesteps, dims
        KL_prior = KL_prior + self.mapping.KL_prior()

        if self.joint_samples:
            f = self.mapping.sample_F(t_)  # batch, outdims, time
        else:
            f_mu, f_var = self.mapping.compute_F(t_)
            f = f_mu + torch.sqrt(f_var) * torch.randn_like(f_mu)

        f = f.permute(0, 2, 1)[:, None, ...]  # batch, neurons, time, d

        return f, KL_prior


# main group
class input_group(_data_object):
    """
    with priors and variational distributions attached for VI inference.
    To set up the input mapping, one has to call *preprocess()* to initialize the input :math:`X` and latent
    :math:`Z` variables.

    Allows priors and SVI for latent variables, and regressors when using observed variables.

    Input variables :math:`X` and :math:`Z` with reparameterizable variational distributions.


    Prepare covariate data for optimization, batches the data and initializes the variational parameters and
    the prior distributions. Note that the data is batched and remains on the host, moved to the target device
    only during fitting, per batch.

    Regression supports parallel trial structure common in neural recordings. This is achieved with covariates
    of shape (trials, timesteps, dims) and spikes of shape (trials, neurons, timesteps). Latent variable models
    with trial structure is indicated through the batch_size argument, with a single integer indicating a
    continuous recording setting whereas a list of integers indicates separate trials of correspondings lengths,
    which are fed in as concatenated together along the time dimension. Note this supports unequal trial lengths,
    but does not parallelize the inference unlike in the regression case.

    :param list covariates: a list of the input covariates, which are np.arrays when indicating observed
                            variables, and are multi-dimensional np.arrays when indicating latent variables
                            i.e. initializing the variational distributions, in case of self.maps > 1 the
                            last element is a tuple for HMM variables, containing (hmm_T, p_0) or (hmm_state)
    :param int timesteps: total number of time points of the covariates
    :param np.ndarray spikes: input spikes of shape (neurons, timesteps)
    :param int/tuple batch_size: the size of batches in time for optimization. If it is a tuple, this indicates
                                 trial structure in the data, treating each batch as a separate trial (relevant
                                 for AR priors to be disconnected). If trials are short, this leads to slower
                                 performance on GPUs typically due to the small batch size and high number of
                                 batches
    """

    def __init__(self, tensor_type=torch.float):
        super().__init__()
        self.register_buffer("dummy", torch.empty(0))  # keeping track of device
        self.tensor_type = tensor_type
        self.XZ = None  # label as data not set

    def set_XZ(self, input_data, timesteps, batch_info, trials=1):
        """
        Preprocesses input data for training. Batches the input data and takes care of setting
        priors on latent dimensions, as well as initializing the SVI framework. Modules and Parameters
        are moved to the current device.

        The priors are specified by setting the variables `p_mu_{}` and `p_std_{}`. In the case of a GP
        prior ('mapping'), the mapping module is placed in the variable `p_mapping_{}` with its input
        group in the variable `p_inputs_{}`. The differential mapping prior ('RW_mapping') requires
        specification of both the GP module and prior distributions.

        The latent variables are initialized by specifying the moments of the variational distribution
        through the variables `lv_mu_{}` and `lv_std_{}`, representing the mean and the scale. The mean
        can be higher dimensional, the scale value is one-dimensional.

        The random walk prior ('RW') in the torus has a learnable offset and standard deviation for the
        transition distribution. The euclidean version is the linear dynamical system, with p_mu being
        the decay factor, p_std the stationary standard deviation. p_mu < 1 to remain bounded.

        Batching AR processes or GLM with history takes into account the overlap of batches when continuous
        i.e. batches are linked temporally.

        :param list input_data: input array of observed regressors of shape (timesteps,) or (timesteps,
                                    dims) when the event shape of this block is bigger than 1, or
                                    input array of latent initialization as a list of [prior_tuple, var_tupe]
                                    with (p_mu, p_std, learn_mu, learn_std, GP_module) as prior_tuple and
                                    var_tuple (mean, std)
        :param int/list batch_size: batch size to use, if list this indicates batches separated temporally
        """
        self.setup_batching(batch_info, timesteps, trials)

        ### read the input ###
        self.latent_dims = []  # which dimensions are latent
        cov_split = []
        self.regressor_mode = (
            True  # if covariates is all regressor values, pack into array later
        )
        self.dims = 0
        for k, in_ in enumerate(input_data):

            if isinstance(in_, torch.Tensor):  # observed
                cov = _expand_cov(in_.type(self.tensor_type))
                self.dims += cov.shape[-1]

                if cov.shape[0] != self.trials:
                    raise ValueError(
                        "Trial count does not match trial count in covariates"
                    )
                if cov.shape[-2] != self.tsteps:
                    raise ValueError(
                        "Expected time steps do not match given covariates"
                    )

                cov_split.append(cov)

            else:  # latent
                self.regressor_mode = False
                in_.validate(timesteps, trials, self.batch_info)
                if in_.tensor_type != self.tensor_type:
                    raise ValueError(
                        "VI input object tensor type does not match input group"
                    )

                self.add_module("input_{}".format(k), in_.to(self.dummy.device))
                self.dims += in_.dims
                cov_split.append(getattr(self, "input_{}".format(k)))
                self.latent_dims.append(k)

        ### assign input to storage ###
        if self.regressor_mode:  # turn into compact tensor arrays
            self.XZ = torch.cat(cov_split, dim=-1)
        else:  # XZ is a list of tensors and/or nn.Modules
            self.XZ = cov_split

    def constrain(self):
        """ """
        for k, in_ in enumerate(self.XZ):

            if isinstance(in_, torch.Tensor):  # regressor variable
                continue

            in_.constrain()

    def _XZ(self, XZ, samples):
        """
        Expand XZ to standard shape with MC copies
        """
        trials, out, bs, inpd = XZ.shape

        if trials != self.trials:
            raise ValueError(
                "Trial number in input does not match expected trial number"
            )

        if (
            trials > 1
        ):  # cannot rely on broadcasting to get MCxtrials in first dimension
            _XZ = (
                XZ[None, ...]
                .repeat(samples, 1, 1, 1, 1)
                .view(-1, out, bs, inpd)
                .to(self.dummy.device)
            )
        else:
            _XZ = XZ.expand(samples, out, bs, inpd).to(self.dummy.device)

        return _XZ

    def sample_XZ(self, b, samples):
        """
        Draw samples from the covariate distribution, provides an implementation of SVI.
        In [1] we amortise the variational parameters with a recognition network. Note the input
        to this network is the final output, i.e. all latents in each layer of a deep GP are mapped
        from the final output layer.
        In the sparse GP version we have [2].

        History in GLMs is incorporated through the offset before the batch_edge values indicating the
        batching timepoints. AR(1) structure is incorporated by appending the first timestep of the next
        batch into the current one, which is then used for computing the Markovian prior.

        Note that in LVM + GLM, the first filter_len-1 number of LVs is not inferred.

        Note when input is trialled, we can only have MC samples flag equal to trials (one per trial).

        :param int b: the batch ID to sample from
        :param int samples: the number of samples to take
        :param torch.Tensor net_input: direct input for VAE amortization of shape (dims, time)
        :returns: covariates sample of shape (samples, timesteps, dims), log_prior
        :rtype: tuple

        References:

        [1] `Variational auto-encoded deep Gaussian Processes`,
        Zhenwen Dai, Andreas Damianou, Javier Gonzalez & Neil Lawrence

        [2] `Scalable Gaussian Process Variational Autoencoders',
        Metod Jazbec, Matthew Ashman, Vincent Fortuin, Michael Pearce, Stephan Mandt, Gunnar R¨atsch

        """
        batch_edge, batch_link, _ = self.batch_info

        if self.regressor_mode:  # regressor mode, no lv, trial blocks are repeated
            XZ = self._XZ(self.XZ[..., batch_edge[b] : batch_edge[b + 1], :], samples)
            # dummy = torch.zeros(samples*self.trials, dtype=self.tensor_type, device=self.dummy.device)
            return XZ, 0

        ### LVM ###
        KL_prior = 0
        cov = []

        for k, in_ in enumerate(self.XZ):

            if isinstance(in_, torch.Tensor):  # regressor variable
                cov.append(
                    self._XZ(in_[..., batch_edge[b] : batch_edge[b + 1], :], samples)
                )
                continue

            ### continuous latent variables ###
            v_samp, kl_p = in_.sample(b, self.batch_info, samples)

            KL_prior = KL_prior + kl_p
            cov.append(v_samp)

        # XZ, kl_stim = self._time_slice(torch.cat(cov, dim=-1))
        return torch.cat(cov, dim=-1), KL_prior
