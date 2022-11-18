import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.parameter import Parameter

from tqdm.autonotebook import tqdm

from . import base, distributions as dist
from .likelihoods import point_process as point_process


def get_device(gpu=0):
    """
    Enable PyTorch with CUDA if available.

    :param int gpu: device number for CUDA
    :returns: device name for CUDA
    :rtype: string
    """
    print("PyTorch version: %s" % torch.__version__)
    dev = (
        torch.device("cuda:{}".format(gpu))
        if torch.cuda.is_available()
        else torch.device("cpu")
    )
    print("Using device: %s" % dev)
    return dev


### inputs ###
class prior_variational_pair(base._VI_object):
    """ """

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

    def sample(self, b, batch_info, samples, net_input, importance_weighted):
        """ """
        batch_edge, batch_link, batch_initial = batch_info

        offs = self.prior.AR_p if batch_link[b] else 0
        t_lower = batch_edge[b] - offs
        t_upper = batch_edge[b + 1]

        v_samp, nlog_q = self.variational.sample(
            t_lower, t_upper, offs, samples, net_input
        )
        log_p = self.prior.log_p(v_samp, batch_initial[b])
        v_samp_ = v_samp[:, offs:]

        if len(v_samp_.shape) == 2:  # one-dimensional latent variables expanded
            v_samp_ = v_samp_[:, None, :, None]
        else:
            v_samp_ = v_samp_[:, None, ...]

        if importance_weighted:
            qlog_pq = (log_p + nlog_q).view(
                cov_samples, self.input_group.trials
            ) - np.log(cov_samples)
            nqlog_pq = -torch.logsumexp(qlog_pq, dim=-2).mean()  # mean over trials
        else:
            nqlog_pq = -(log_p + nlog_q).mean()  # mean over trials and MC samples

        fac = self.tsteps / v_samp_.shape[-2]  # subsampling in batched mode
        return v_samp_, fac * nqlog_pq


class probabilistic_mapping(base._VI_object):
    """
    Takes in an _input_mapping object
    """

    def __init__(self, input_group, mapping):
        """ """
        super().__init__(mapping.out_dims, mapping.tensor_type)
        if mapping.tensor_type != input_group.tensor_type:
            raise ValueError("Mapping and input group tensor types do not match")
        self.add_module("mapping", mapping)
        self.add_module("input_group", input_group)

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

    def sample(self, b, batch_info, samples, net_input, importance_weighted):
        """ """
        t_, KL_prior = self.input_group.sample_XZ(
            b, 1, average_nlq, net_input
        )  # samples, timesteps, dims
        KL_prior = KL_prior + self.mapping.KL_prior()
        f = self.mapping.sample_F(t_)  # batch, outdims, time
        f = f.permute(0, 2, 1)[:, None, ...]  # batch, neurons, time, d

        return f, KL_prior


class filtered_input(base._VI_object):
    """
    Stimulus filtering as in GLMs
    """

    def __init__(self, input_series, stimulus_filter, tensor_type=torch.float):
        self.register_buffer("input_series", input_series.type(tensor_type))

        self.add_module("filter", stimulus_filter)
        self.history_len = (
            self.filter.history_len
        )  # history excludes instantaneous part

    def sample(self, b, batch_info, samples, net_input, importance_weighted):
        """ """
        _XZ = self.stimulus_filter(XZ.permute(0, 2, 1))[0].permute(
            0, 2, 1
        )  # ignore filter variance
        KL_prior = self.stimulus_filter.KL_prior()

        return _XZ, KL_prior


# discrete variables
class discrete_latent(base._VI_object):
    """
    Discrete variables are added as extra dimensions to the left hand side, and we perform enumeration
    We add an input dimension with the integer categories
    When enumeration is performed, this dimension is coupled to the left hand side extra dimensions
    When performing SVI, we sample from the posterior...
    """

    def __init__(self, p_0, K, learn_p_0=False, tensor_type=torch.float):
        if learn_p_0:
            self.register_buffer("learn_p_0", learn_p_0)
        else:
            self.register_parameter("learn_p_0", Parameter(learn_p_0))

        # self.filter_len = filter_len
        # if self.stimulus_filter is not None and self.stimulus_filter.history_len != filter_len:
        #    raise ValueError('Stimulus filter length and input filtering length do not match')

    def _time_slice(self, XZ):
        """ """
        if self.stimulus_filter is not None:
            _XZ = self.stimulus_filter(XZ.permute(0, 2, 1))[0].permute(
                0, 2, 1
            )  # ignore filter variance
            KL_prior = self.stimulus_filter.KL_prior()
        else:
            _XZ = XZ[
                :, self.filter_len - 1 :, :
            ]  # covariates has initial history part excluded
            KL_prior = 0

        return _XZ, KL_prior


class HMM_latent(base._VI_object):
    """ """

    def __init__(self, stimulus_filter):
        self.add_module("stimulus_filter", stimulus_filter)
        self.filter_len = filter_len

    def _time_slice(self, XZ):
        """ """
        if self.stimulus_filter is not None:
            _XZ = self.stimulus_filter(XZ.permute(0, 2, 1))[0].permute(
                0, 2, 1
            )  # ignore filter variance
            KL_prior = self.stimulus_filter.KL_prior()
        else:
            _XZ = XZ[
                :, self.filter_len - 1 :, :
            ]  # covariates has initial history part excluded
            KL_prior = 0

        return _XZ, KL_prior


# main group
class input_group(base._data_object):
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
                cov = base._expand_cov(in_.type(self.tensor_type))
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
        return

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

    def sample_XZ(self, b, samples, net_input=None, importance_weighted=False):
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
            v_samp, kl_p = in_.sample(
                b, self.batch_info, samples, net_input, importance_weighted
            )

            KL_prior = KL_prior + kl_p
            cov.append(v_samp)

        # XZ, kl_stim = self._time_slice(torch.cat(cov, dim=-1))
        return torch.cat(cov, dim=-1), KL_prior


### variational inference class ###
class VI_optimized(nn.Module):
    """
    Inference model with log likelihood based objectives using variational inference.
    Depending on the variational distribution, this covers marginal likelihood models as well
    as point estimate ML/MAP optimization.

    Adds in spike couplings and stimulus filters [1].
    Conditional renewal process inference model (both u(t) and h(t)) [2].
    Includes Hidden Markov Model rate maps [3].

    General structure for usage:
    - Initialize the model (specify parameters, kernels etc.), involves the functions
      *__init__()* and *set_params()* if parameters need to be set
    - Preprocess the data (allocates latent variables etc.), involves the functions
      *set_data()*
    - Fit the model with the data input provided, involves the function *fit()*
    - Visualize the fit, involves the functions *eval_rate()* and so on
    - Sample from the model or predictive posterior using *sample()*

    For example, we can do dimensional reduction on Iris dataset as follows:
        >>> # With y as the 2D Iris data of shape 150x4 and we want to reduce its dimension
        >>> # to a tensor X of shape 150x2, we will use GPLVM.
        .. doctest::
           :hide:
            >>> # Simulating iris data.
            >>> y = torch.stack([dist.Normal(4.8, 0.1).sample((150,)),
            ...                  dist.Normal(3.2, 0.3).sample((150,)),
            ...                  dist.Normal(1.5, 0.4).sample((150,)),
            ...                  dist.Exponential(0.5).sample((150,))])
        >>> # First, define the initial values for X parameter:
        >>> X_init = torch.zeros(150, 2)
        >>> # Then, define a Gaussian Process model with input X_init and output y:
        >>> kernel = gp.kernels.RBF(input_dim=2, lengthscale=torch.ones(2))
        >>> Xu = torch.zeros(20, 2)  # initial inducing inputs of sparse model
        >>> gpmodule = gp.models.SparseGPRegression(X_init, y, kernel, Xu)
        >>> # Finally, wrap gpmodule by GPLVM, optimize, and get the "learned" mean of X:
        >>> gplvm = gp.models.GPLVM(gpmodule)
        >>> gp.util.train(gplvm)  # doctest: +SKIP
        >>> X = gplvm.X

    References:

    [1] `Capturing the Dynamical Repertoire of Single Neurons with Generalized Linear Models`,
        Alison I. Weber, Jonathan W. Pillow (2017)

    [2] `Time-rescaling methods for the estimation and assessment of non-Poisson neural encoding
         models`,
         Jonathan W. Pillow (2009)

    [3] `Hidden Markov models for the stimulus-response relationships of multistate neural systems`,
        Escola, S., Fontanini, A., Katz, D. and Paninski, L. (2011)

    """

    def __init__(self, input_group, mapping, likelihood):
        """
        Constructor for the VI_optimized class.

        :param _input_mapping mapping: list of nn.Module rate model which maps input to rate distribution
        :param _likelihood likelihood: likelihood model connecting data with rates
        :param int h_len: the history length of the spike coupling and stimulus history term
        :param int u_basis: number of stimulus history basis functions
        :param int h_basis: number of spike coupling basis functions
        :param bool spike_filter: indicator whether we want spike coupling
        :param _input_mapping dispersion_mapping: nn.Module model for dispersion parameter
        """
        super().__init__()
        self.set_model(input_group, mapping, likelihood)

    ### model structure ###
    def set_model(self, input_group, mapping, likelihood):
        """
        Model is p(Y|F)p(F|X,Z)p(Z).
        Setting data corresponds to setting observed Y and X, setting p(Z) and initializing q(Z).
        Heteroscedastic likelihoods allow p(Y|F,X,Z).
        """
        self.add_module("input_group", input_group)  # p(x)p(Z) and q(Z)
        self.add_module("mapping", mapping)  # p(F|X)
        self.add_module("likelihood", likelihood)  # p(Y|F)

        if (
            max(self.mapping.active_dims) >= self.input_group.dims
        ):  # avoid illegal access
            raise ValueError(
                "Mapping active dimensions are (partly) outside input space"
            )
        if self.mapping.out_dims != self.likelihood.F_dims:
            raise ValueError(
                "Mapping output dimensions do not match likelihood input dimensions"
            )
        if self.mapping.tensor_type != self.likelihood.tensor_type:
            raise ValueError("Mapping and likelihood tensor types do not match")
        if self.mapping.tensor_type != self.input_group.tensor_type:
            raise ValueError("Mapping and input group tensor types do not match")

    def validate_model(self, likelihood_set=True):
        """
        Checks model validity and consistency, after setting data.

        When setting up a model for sampling, call this method with likelihood_set depending on
        whether spike trains have been loaded (needed for sampling conditional processes).
        """
        if self.input_group.XZ is None:
            raise ValueError("Inputs object has not been set with data using .set_XZ()")

        if likelihood_set:
            if self.likelihood.all_spikes is None:
                raise ValueError(
                    "Likelihood object has not been set with data using .set_Y()"
                )
            if self.likelihood.tsteps != self.input_group.tsteps:
                raise ValueError("Input and output time dimensions do not match")
            if self.likelihood.batch_info != self.input_group.batch_info:
                raise ValueError("Input and output batch counts do not match")
            if self.likelihood.trials != self.input_group.trials:
                raise ValueError("Input and output trial numbers do not match")

    def constrain(self):
        """
        Constrain the optimization step, in particular constrain rate model parameters, stimulus history and spike
        history/coupling filters, the HMM parameters to the simplex.
        """
        self.input_group.constrain()
        self.mapping.constrain()
        self.likelihood.constrain()

    ### log likelihood computation ###
    def _nll(self, mean, variance, ll_samples, ll_mode, b, neuron, XZ, sum_time=False):
        _nll, ws = self.likelihood.objective(
            mean, variance, XZ, b, neuron, samples=ll_samples, mode=ll_mode
        )  # (sample, time)
        _nll = _nll.view(
            -1, self.input_group.trials, _nll.shape[-1]
        )  # (MC/GH sample, trials, time)
        if (
            sum_time
        ):  # need separate NLL values at time steps for enumeration (e.g. HMM)
            _nll = _nll.sum(-1)

        if len(ws.shape) > 0:
            if sum_time:
                ws = ws.view(-1, self.input_group.trials)
            else:
                ws = ws.view(-1, self.input_group.trials, 1)

        ws *= self.input_group.trials
        return _nll, ws  # (sample, trial, time) tensors

    def objective(
        self,
        b,
        neuron=None,
        beta=1.0,
        cov_samples=1,
        ll_samples=10,
        importance_weighted=False,
        ll_mode="MC",
        lv_input=None,
    ):
        """
        Compute the rate and then spike train log likelihood.
        Mapping prior is for SVGP the KL term in the variational ELBO.
        When rate output is deterministic (inner_var=0) the ll_samples parameter is irrelevant in the
        'MC' in that case, where no sampling will occur. Note that MC is used for the expectation over
        the latent variable distribution as well, hence ll_mode 'MC' leads to double MC estimation.

        :param int b: batch index used for picking the batch to fit
        :param np.ndarray neuron: indices of neurons for which to evaluate the likelihood term (observed neurons)
        :param float beta: prior annealing or KL annealing via beta
        :param int cov_samples: number of samples to draw from covariates distribution
        :param int ll_samples: number of samples to evaluate likelihood per covariate sample
        :param string bound: type of variational objective ('ELBO', 'PLL' [1] or 'IWAE' with K=mc [2])
        :param string ll_mode: likelihood evaluation mode (Monte Carlo `MC` or Gauss-Hermite quadratures `GH`)
        :param bool enuerate_z: use enumerate for discrete latent (marginalizing out all states)
        :param torch.Tensor lv_input: input tensor for the latent variable computation (e.g. in the case
                                      of amortized inference we want self.likelihood.all_spikes)
        :returns: variational approximation to negative marginal log likelihood
        :rtype: torch.tensor

        References:

        [1] `Parametric Gaussian Process Regressors`, Martin Jankowiak, Geoff Pleiss, Jacob R. Gardner (2019)

        [2] `Importance Weighted Autoencoders`, Yuri Burda, Roger Grosse, Ruslan Salakhutdinov (2015)

        [3] `Deep Gaussian Processes with Importance-Weighted Variational Inference`,
            Hugh Salimbeni, Vincent Dutordoir, James Hensman, Marc Peter Deisenroth (2019)
        """
        neuron = self.likelihood._validate_neuron(neuron)

        ### sample input ###
        XZ, KL_prior_in = self.input_group.sample_XZ(
            b, cov_samples, lv_input, importance_weighted
        )
        if len(XZ.shape) > 4:  # normally (samples, outdims, timesteps, dims)
            enumeration = True
        else:
            enumeration = False

        ### compute mapping ###
        if self.mapping.MC_only:  # SVI only
            ll_mode = "direct"  # carry samples directly
            F = self.mapping.sample_F(
                XZ, ll_samples
            )  # directly carries likelihood samples
        else:  # VI type mapping i.e. computing first and second moments
            F, F_var = self.mapping.compute_F(
                XZ
            )  # mean and covariance of input mapping (samples, neurons, timesteps)

        KL_prior_m = self.mapping.KL_prior(
            importance_weighted
        )  # prior may need quantities computed in compute_F, e.g. Luu, KTT
        if ll_mode == "direct":
            F_var = None

        ### evaluating likelihood expectation and prior terms ###
        batch_size = F.shape[-1]
        fac = (
            self.likelihood.tsteps / batch_size
        )  # n_data/batch_size, as we subsample temporally (batching)
        KL_prior_l = self.likelihood.KL_prior(importance_weighted)

        _nll, ws = self._nll(
            F, F_var, ll_samples, ll_mode, b, neuron, XZ, sum_time=True
        )

        ### bounds ###
        if (
            importance_weighted
        ):  # mean is taken over the sample dimension i.e. log E_{q(f)q(z)}[...]
            nll_term = -torch.logsumexp(
                -_nll + torch.log(ws), dim=-2
            )  # sum over MC/GH samples
        else:
            nll_term = (_nll * ws).sum(-2)  # sum over MC/GH samples

        nll_term = nll_term.mean()  # mean over trials

        ### compute objective ###
        KL_prior = (
            KL_prior_in + KL_prior_m + KL_prior_l
        )  # model priors integrated out (KL terms)
        return nll_term * fac + beta * KL_prior  # negative (approximate) evidence

    ### optimization ###
    def set_optimizers(
        self, optim_tuple, opt_lr_dict, special_grads=[], extra_update_args=None
    ):
        """
        Set the optimizers and the optimization hyperparameters.

        :param tuple optim_tuple: tuple containing PyTorch (optimizer, schedule_steps, lr_scheduler)
        :param dict opt_lr_dict: is a dictionary with parameter name and lr, needs to supply 'default': lr
        :param list special_grads: list of tuples ([parameter_names,...], lr, optim_step_function)
        :param tuple extra_update_args: tuple of update function and args
        """
        opt, self.sch_st, sch = optim_tuple
        assert "default" in opt_lr_dict  # default learning rate

        if opt == optim.LBFGS:  # special case
            d = []
            for param in self.parameters():
                # special gradient treatment
                for sg in special_grads:
                    if name in sg[0]:  # filter parameters
                        continue

                d.append(param)

            # for param, lr in extra_params:
            #    d.append(param)

            history_size = opt_lr_dict["history"] if "history" in opt_lr_dict else 10
            max_iter = opt_lr_dict["max_iter"] if "max_iter" in opt_lr_dict else 4

            self.optim_base = opt(
                d,
                lr=opt_lr_dict["default"],
                history_size=history_size,
                max_iter=max_iter,
            )
        else:
            for key in opt_lr_dict:
                if key == "default":
                    continue
                if not key in dict(self.named_parameters()):
                    print("The parameter {} is not found in the model.".format(key))

            d = []
            for name, param in self.named_parameters():
                # special gradient treatment
                for sg in special_grads:
                    if name in sg[0]:  # filter parameters
                        continue

                opt_dict = {}
                opt_dict["params"] = param

                for key in opt_lr_dict:
                    if key == name:
                        opt_dict["lr"] = opt_lr_dict[key]
                        break

                d.append(opt_dict)

            # for param, lr in extra_params:
            #    opt_dict = {}
            #    opt_dict['params'] = param
            #    opt_dict['lr'] = lr
            #    d.append(opt_dict)

            self.optim_base = opt(d, lr=opt_lr_dict["default"])
        self.sch = sch(self.optim_base)

        # special gradient treatment
        pd = dict(self.named_parameters())

        self.special_grads = []
        for sg in special_grads:
            p_list = []
            for key in sg[0]:
                p_list.append(pd[key])

            lr = sg[1]
            optim_step_function = sg[2]
            container = (optim.SGD(p_list, lr=lr), p_list, optim_step_function)
            self.special_grads.append(container)

        self.extra_update_args = extra_update_args

    def fit(
        self,
        max_epochs,
        margin_epochs=10,
        loss_margin=0.0,
        neuron=None,
        kl_anneal_func=None,
        retain_graph=False,
        cov_samples=1,
        ll_samples=1,
        importance_weighted=False,
        ll_mode="MC",
        lv_input=None,
        callback=None,
    ):
        """
        Can fit to all neurons present in the data (default), or fits only a subset of neurons.
        margin_epochs sets the early stop when the loss is still above lowest loss in given iterations.

        :param int max_epochs: maximum number of iterations over the dataset
        :param int margin_epochs: number of iterations above margin loss reduction to tolerate
        :param float loss_margin: tolerated loss increase during training
        :param list neuron: neuron indices to fit to
        :param string grad_type: gradient type used in optimization (`vanilla` or `natural` [1])
        :returns: time series of loss during training
        :rtype: list
        """
        self.validate_model()
        batches = self.likelihood.batches

        if kl_anneal_func is None:
            kl_anneal_func = lambda x: 1.0

        tracked_loss = []
        minloss = np.inf
        cnt = 0
        iterator = tqdm(range(max_epochs))

        # optimizers
        optimizer = [self.optim_base]
        for sg in self.special_grads:
            optimizer += [sg[0]]

        # iterate over dataset
        for epoch in iterator:

            sloss = 0
            for b in range(batches):

                def closure():
                    if torch.is_grad_enabled():
                        for o in optimizer:
                            o.zero_grad()

                    anneal_t = epoch / max_epochs
                    loss = self.objective(
                        b,
                        neuron,
                        beta=kl_anneal_func(anneal_t),
                        importance_weighted=importance_weighted,
                        cov_samples=cov_samples,
                        ll_samples=ll_samples,
                        ll_mode=ll_mode,
                        lv_input=lv_input,
                    )

                    if loss.requires_grad:
                        # with profiler.profile(record_shapes=True, use_cuda=True) as profi:
                        #    with profiler.record_function("backwards"):
                        loss.backward(retain_graph=retain_graph)

                        # special gradient steps
                        for sg in self.special_grads:
                            sg[2](sg[1], loss)

                        # extra updates
                        if self.extra_update_args is not None:
                            self.extra_update_args[0](*self.extra_update_args[1])

                    return loss

                if isinstance(optimizer[0], optim.LBFGS):
                    optimizer[0].step(closure)
                    with torch.no_grad():
                        loss = closure()
                else:
                    loss = closure()
                    optimizer[0].step()

                for o in optimizer[1:]:  # additional optimizers
                    o.step()

                self.constrain()
                if callback is not None:  # additional operations per gradient step
                    callback(self)

                if torch.isnan(loss.data).any() or torch.isinf(loss.data).any():
                    raise ValueError("Loss diverged")
                sloss += loss.item()

            if self.sch_st is not None and epoch % self.sch_st == self.sch_st - 1:
                self.sch.step()

            sloss /= batches  # average over batches, each batch loss is subsampled estimator of full loss
            iterator.set_postfix(loss=sloss)
            tracked_loss.append(sloss)

            if sloss <= minloss + loss_margin:
                cnt = 0
            else:
                cnt += 1

            if sloss < minloss:
                minloss = sloss

            if cnt > margin_epochs:
                print("\nStopped at epoch {}.".format(epoch + 1))
                break

        return tracked_loss
