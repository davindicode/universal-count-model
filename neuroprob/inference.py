import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.parameter import Parameter

from tqdm.autonotebook import tqdm

from . import base, distributions as dist


def get_device(gpu=0):
    """
    Enable PyTorch with CUDA if available.

    :param int gpu: device number for CUDA
    :returns:
        device name string for CUDA gpu
    """
    print("PyTorch version: %s" % torch.__version__)
    dev = (
        torch.device("cuda:{}".format(gpu))
        if torch.cuda.is_available()
        else torch.device("cpu")
    )
    print("Using device: %s" % dev)
    return dev


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
        Model is p(Y|F) p(F|X,Z) p(Z).
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
            if self.likelihood.Y is None:
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
    def _nll(
        self, mean, variance, ll_samples, ll_mode, b, out_inds, XZ, sum_time=False
    ):
        _nll, ws = self.likelihood.objective(
            mean, variance, XZ, b, out_inds, samples=ll_samples, mode=ll_mode
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
        out_inds=None,
        beta=1.0,
        cov_samples=1,
        ll_samples=10,
        ll_mode="MC",
    ):
        """
        Compute the rate and then spike train log likelihood.
        Mapping prior is for SVGP the KL term in the variational ELBO.
        When rate output is deterministic (inner_var=0) the ll_samples parameter is irrelevant in the
        'MC' in that case, where no sampling will occur. Note that MC is used for the expectation over
        the latent variable distribution as well, hence ll_mode 'MC' leads to double MC estimation.

        :param int b: batch index used for picking the batch to fit
        :param np.ndarray out_inds: indices of output dimensions for which to evaluate the likelihood term
        :param float beta: scalar multiplier of prior/KL term
        :param int cov_samples: number of samples to draw from covariates distribution
        :param int ll_samples: number of samples to evaluate likelihood per covariate sample
        :param string ll_mode: likelihood evaluation mode (Monte Carlo `MC` or Gauss-Hermite quadratures `GH`)
        :returns:
            variational approximation to negative marginal log likelihood
        """
        out_inds = self.likelihood._validate_out_inds(out_inds)

        ### sample input ###
        XZ, KL_prior_in = self.input_group.sample_XZ(b, cov_samples)
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
            )  # mean and covariance of input mapping (samples, out_dims, timesteps)

        KL_prior_m = (
            self.mapping.KL_prior()
        )  # prior may need quantities computed in compute_F, e.g. Luu, KTT
        if ll_mode == "direct":
            F_var = None

        ### evaluating likelihood expectation and prior terms ###
        batch_size = F.shape[-1]
        fac = (
            self.likelihood.tsteps / batch_size
        )  # n_data/batch_size, as we subsample temporally (batching)
        KL_prior_l = self.likelihood.KL_prior()

        _nll, ws = self._nll(
            F, F_var, ll_samples, ll_mode, b, out_inds, XZ, sum_time=True
        )

        ### bounds ###
        nll_term = (_nll * ws).sum(-2)  # sum over MC/GH samples

        nll_term = nll_term.mean()  # mean over trials

        ### compute objective ###
        KL_prior = (
            KL_prior_in + KL_prior_m + KL_prior_l
        )  # model priors integrated out (KL terms)
        return nll_term * fac + beta * KL_prior  # negative (approximate) evidence

    ### optimization ###
    def set_optimizers(
        self,
        optimizer,
        scheduler,
        scheduler_interval,
        opt_lr_dict,
        special_grads=[],
        extra_update_args=None,
    ):
        """
        Set the optimizers and the optimization hyperparameters.

        :param torch.Optimizer optimizer: PyTorch optimizer
        :param _LRScheduler scheduler: Learning rate scheduler
        :param dict opt_lr_dict: is a dictionary with parameter name and lr, needs to supply 'default': lr
        :param list special_grads: list of tuples ([parameter_names,...], lr, optim_step_function)
        :param tuple extra_update_args: tuple of update function and args
        """
        assert "default" in opt_lr_dict  # default learning rate

        if optimizer == optim.LBFGS:  # special case
            d = []
            for param in self.parameters():
                # special gradient treatment
                for sg in special_grads:
                    if name in sg[0]:  # filter parameters
                        continue

                d.append(param)

            history_size = opt_lr_dict["history"] if "history" in opt_lr_dict else 10
            max_iter = opt_lr_dict["max_iter"] if "max_iter" in opt_lr_dict else 4

            self.optim_base = optimizer(
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

            self.optim_base = optimizer(d, lr=opt_lr_dict["default"])

        # scheduler if any
        self.sch = scheduler(self.optim_base) if scheduler is not None else None
        self.sch_st = scheduler_interval

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
        out_inds=None,
        kl_anneal_func=None,
        retain_graph=False,
        cov_samples=1,
        ll_samples=1,
        ll_mode="MC",
        callback=None,
    ):
        """
        Can fit to all output dimensions present in the data (default), or fits only a subset of output dimensions.
        margin_epochs sets the early stop when the loss is still above lowest loss in given iterations.

        :param int max_epochs: maximum number of iterations over the dataset
        :param int margin_epochs: number of iterations above margin loss reduction to tolerate
        :param float loss_margin: tolerated loss increase during training
        :param List out_inds: out_inds indices to fit to
        :param str grad_type: gradient type used in optimization (`vanilla` or `natural` [1])
        :returns:
            time series list of loss during training
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
                        out_inds,
                        beta=kl_anneal_func(anneal_t),
                        cov_samples=cov_samples,
                        ll_samples=ll_samples,
                        ll_mode=ll_mode,
                    )

                    if loss.requires_grad:
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

            if self.sch_st > 0 and epoch % self.sch_st == self.sch_st - 1:
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
