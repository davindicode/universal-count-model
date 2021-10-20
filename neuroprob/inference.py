import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.optim as optim

from . import base
from . import point_process
from . import distributions as dist
from .utils import pytorch

import numpy as np

from tqdm.autonotebook import tqdm



class input_group(base._input):
    """
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
    :param np.array spikes: input spikes of shape (neurons, timesteps)
    :param int/tuple batch_size: the size of batches in time for optimization. If it is a tuple, this indicates 
                                 trial structure in the data, treating each batch as a separate trial (relevant
                                 for AR priors to be disconnected). If trials are short, this leads to slower 
                                 performance on GPUs typically due to the small batch size and high number of 
                                 batches   
    """
    def __init__(self, dims, VI_tuples, tensor_type=torch.float, latent_f='softplus', stimulus_filter=None):
        super().__init__(dims, VI_tuples, tensor_type, latent_f, stimulus_filter)



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
    def __init__(self, inputs, mapping, likelihood):
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
        self.set_model(inputs, mapping, likelihood)
            
            
    ### model structure ###
    def set_model(self, inputs, mapping, likelihood):
        """
        Model is p(Y|F)p(F|X,Z)p(Z).
        Setting data corresponds to setting observed Y and X, setting p(Z) and initializing q(Z).
        Heteroscedastic likelihoods allow p(Y|F,X,Z).
        """
        self.add_module('inputs', inputs) # p(x)p(Z) and q(Z)
        self.add_module('mapping', mapping) # p(F|X)
        self.add_module('likelihood', likelihood) # p(Y|F)
        
        if max(self.mapping.active_dims) >= self.inputs.dims: # avoid illegal access
            raise ValueError('Mapping active dimensions are (partly) outside input space')
        if self.mapping.out_dims != self.likelihood.F_dims:
            raise ValueError('Mapping output dimensions do not match likelihood input dimensions')
        if self.mapping.inv_link != self.likelihood.inv_link:
            raise ValueError('Mapping and likelihood link functions do not match')
        if self.mapping.tensor_type != self.likelihood.tensor_type:
            raise ValueError('Mapping and likelihood tensor types do not match')
        #self.F_dims = self.likelihood.F_dims
        
        
    def validate_model(self, likelihood_set=True):
        """
        Checks model validity and consistency, after setting data.
        
        When setting up a model for sampling, call this method with likelihood_set depending on 
        whether spike trains have been loaded (needed for sampling conditional processes).
        """
        if self.inputs.XZ is None:
            raise ValueError('Inputs object has not been set with data using .set_XZ()')
        if likelihood_set:
            if self.likelihood.spikes is None:
                raise ValueError('Likelihood object has not been set with data using .set_Y()')
            if self.likelihood.tsteps != self.inputs.tsteps:
                raise ValueError('Input and output time dimensions do not match')
            if self.likelihood.filter_len != self.inputs.filter_len:
                raise ValueError('Input and likelihood filtering lengths do not match')
            if self.likelihood.batches != self.inputs.batches:
                raise ValueError('Input and output batching do not match')
            if self.likelihood.trials != self.inputs.trials:
                raise ValueError('Input and output trial numbers do not match')
            
            
    def constrain(self):
        """
        Constrain the optimization step, in particular constrain rate model parameters, stimulus history and spike 
        history/coupling filters, the HMM parameters to the simplex.
        """
        self.inputs.constrain()
        self.mapping.constrain()
        self.likelihood.constrain()

            
    ### log likelihood computation ###
    def _nll(self, mean, variance, maps, ll_samples, ll_mode, b, neuron, XZ, sum_time=False):
        _nll, ws = self.likelihood.objective(mean, variance, XZ, b, neuron, 
                                             samples=ll_samples, mode=ll_mode) # (sample, time)
        _nll = _nll.view(-1, self.inputs.trials, _nll.shape[-1]) # (MC/GH sample, trials, time)
        if sum_time: # need separate NLL values at time steps for enumeration (e.g. HMM)
            _nll = _nll.sum(-1)

        if len(ws.shape) > 0:
            if sum_time:
                ws = ws.view(-1, self.inputs.trials)
            else:
                ws = ws.view(-1, self.inputs.trials, 1)
                
        ws *= self.inputs.trials
        return _nll, ws # (sample, trial, time) tensors
        
        
    def objective(self, b, neuron=None, beta=1.0, beta_z=1.0, cov_samples=1, ll_samples=10, bound='ELBO', ll_mode='MC', 
                  lv_input=None):
        """
        Compute the rate and then spike train log likelihood.
        Mapping prior is for SVGP the KL term in the variational ELBO.
        When rate output is deterministic (inner_var=0) the ll_samples parameter is irrelevant in the 
        'MC' in that case, where no sampling will occur. Note that MC is used for the expectation over 
        the latent variable distribution as well, hence ll_mode 'MC' leads to double MC estimation.
        
        :param int b: batch index used for picking the batch to fit
        :param np.array neuron: indices of neurons for which to evaluate the likelihood term (observed neurons)
        :param float beta: prior annealing or KL annealing via beta
        :param int cov_samples: number of samples to draw from covariates distribution
        :param int ll_samples: number of samples to evaluate likelihood per covariate sample
        :param string bound: type of variational objective ('ELBO', 'PLL' [1] or 'IWAE' [2])
        :param string ll_mode: likelihood evaluation mode (Monte Carlo `MC` or Gauss-Hermite quadratures `GH`)
        :param bool enuerate_z: use enumerate for discrete latent (marginalizing out all states)
        :param torch.tensor lv_input: input tensor for the latent variable computation (e.g. in the case 
                                      of amortized inference we want self.likelihood.all_spikes)
        :returns: variational approximation to negative marginal log likelihood
        :rtype: torch.tensor
        
        References:
        
        [1] `Parametric Gaussian Process Regressors`, Martin Jankowiak, Geoff Pleiss, Jacob R. Gardner (2019)
        
        [2] `Importance Weighted Autoencoders`, Yuri Burda, Roger Grosse, Ruslan Salakhutdinov (2015)
            
        """
        if (bound != 'ELBO') and (bound != 'PLL') and (bound != 'IWAE'):
            raise ValueError('Bound parameter not known')
        entropy = False if bound == 'IWAE' else True
        neuron = self.likelihood._validate_neuron(neuron)
            
        ### sample input ###
        XZ, lvm_lprior, lvm_nlq, KL_prior_in = self.inputs.sample_XZ(b, cov_samples, lv_input, entropy) # samples, timesteps, dims
        
        ### mean and covariance of input mapping ###
        F_mu, F_var = self.mapping.compute_F(XZ) # samples, neurons, timesteps
        KL_prior_m = self.mapping.KL_prior() # prior may need quantities computed in compute_F, e.g. Luu
        
        ### evaluating likelihood expectation and prior terms ###
        maps = len(F_mu) if isinstance(F_mu, list) else 1
        batch_size = F_mu[0].shape[-1] if isinstance(F_mu, list) else F_mu.shape[-1]
        fac = self.likelihood.tsteps / batch_size # n_data/batch_size, as we subsample temporally (batching)
        KL_prior_l = self.likelihood.KL_prior()

        if bound == 'IWAE':
            if maps != 1:
                raise ValueError('Enumeration not supported for IWAE')
                
            _nll, ws = self._nll(F_mu, F_var, maps, ll_samples, ll_mode, b, neuron, XZ, sum_time=True) # sum over time or dummy
            ll_samples = _nll.shape[0] // cov_samples # could be less than requested if exact nll
            
            tot_samp = ll_samples*cov_samples*self.inputs.trials
            lvm_lprior = lvm_lprior.view(cov_samples, self.inputs.trials).repeat(ll_samples, 1) / tot_samp
            lvm_nlq = lvm_nlq.view(cov_samples, self.inputs.trials).repeat(ll_samples, 1) / tot_samp
            
            log_pq = (lvm_lprior + lvm_nlq) * beta_z
            model_nlog_pq = -torch.logsumexp((-_nll + log_pq + \
                                              torch.log(ws)), dim=0).mean() * fac # mean over trials

        else: # lvm_nlq is scalar as entropy is True
            if maps > 1:
                nll = []
                for k in range(maps): # enumeration
                    mean = F_mu[k] if maps > 1 else F_mu
                    variance = F_var[k] if maps > 1 else F_var

                    _nll, ws = self._nll(mean, variance, maps, ll_samples, ll_mode, b, neuron, XZ)
                    if bound == 'PLL': # mean is taken over the sample dimension i.e. log E_{q(f)q(z)}[...]
                        nll.append(-torch.logsumexp(-_nll+torch.log(ws), dim=0))
                    else:
                        nll.append((_nll*ws).sum(0)) # sum over MC/GH samples
                nll_term = self.mapping.objective(torch.stack(nll))
                
            else:
                _nll, ws = self._nll(F_mu, F_var, maps, ll_samples, ll_mode, b, neuron, XZ, sum_time=True)
                if bound == 'PLL': # mean is taken over the sample dimension i.e. log E_{q(f)q(z)}[...]
                    nll_term = -torch.logsumexp(-_nll+torch.log(ws), dim=0) # sum over MC/GH samples
                else:
                    nll_term = (_nll*ws).sum(0) # sum over MC/GH samples
                nll_term = nll_term.mean() # mean over trials
            
            log_pq = (lvm_lprior.mean(0) + lvm_nlq) * beta_z
            model_nlog_pq = (nll_term - log_pq) * fac
            
        KL_prior = KL_prior_in + KL_prior_m + KL_prior_l # model priors integrated out (KL terms)
        return model_nlog_pq - beta*KL_prior # negative (approximate) evidence
    
    
    ### evaluating the inference model ###
    def evaluate(self, batch, obs_neuron=None, cov_samples=1, ll_samples=1, ll_mode='MC', lv_input=None):
        """
        Sample from the predictive posterior.
        
        Sample from the neural encoding model. Note when sampling the latent state (neuron=[], >1 HMM states), 
        the likelihood module will be set with *preprocess()* and affects model fitting after calling this.
        
        .. note:: The model is evaluated with its variational posterior mean values, which is identical to the 
            ML/MAP parameters when non-probabilistic model fitting is performed.
        
        Filtered and HMM inference require likelihood to be set with data Y, then validated with *validate_model()*
        
        Sampling from a conditional Poisson process, with choice of neurons to condition on.
        covariates has timesteps T. The ini_train will be taken as the spike train corresponding to the stimulus history 
        segment at the start, which is relevant with temporal filtering. Without temporal filtering, ini_train will be 
        used to extract the trial size of the sampled spike train.
        
        :param list covariates: input covariates list of time series of shape (T,), initialize hmm tuple (hmm_t, logp_0), 
                                None uses learned values stored. Discrete state is inferred with Viterbi, np.array gives 
                                observed HMM state. This is read from the last element of covariates list when self.maps>1
        :param np.array ini_spktrain: initial spike train segment of shape (trials, neurons, h_len-1)
        :param np.array neuron: neuron indices that we want to sample from, conditioned on rest
        :param np.array obs_spktrain: observed spike train of shape (trials, neurons, T)
        :param bool stochastic: take into account variational variance in the inferred rate and dispersion
        
        :returns: [samples, trial, neuron, timestep] of spiketrain generated, rate, hmm_state
        :rtype: tuple of np.array
        """
        with torch.no_grad():
            XZ, _, _, _ = self.inputs.sample_XZ(batch, cov_samples, lv_input, False)
            timesteps = XZ.shape[1]
            obs_neuron = self.likelihood._validate_neuron(obs_neuron)
            F_mu, F_var = self.mapping.compute_F(XZ)

            maps = len(F_mu) if isinstance(F_mu, list) else 1
            if maps > 1:
                if self.mapping.state_obs:
                    discrete = self.hmm_state[0].to(self.mapping.dummy.device)
                else:
                    nll = torch.stack(self._nll(F_mu, F_var, maps, 'ELBO', ll_samples, 
                                                ll_mode, batch, obs_neuron, XZ, 1.)) # (state, time)
                    discrete = self.mapping.sample_state(timesteps, self.logp_0, trials=self.inputs.trials, 
                                                         cond_nll=nll, viterbi=True)
                    
                F_mu = self.slice_F(torch.stack(F_mu), discrete)
                F_var = self.slice_F(torch.stack(F_var), discrete)
                
            else: # only one state
                discrete = torch.zeros((self.inputs.trials, timesteps), dtype=torch.long).to(self.mapping.dummy.device)
        
        rate = self.likelihood.sample_rate(F_mu, F_var, self.inputs.trials, ll_samples) # MC, trials, neuron, time   
        return XZ, rate, discrete
    
    
    
    def sample_F(self, covariates, MC, neuron):
        """
        """
        cov = self.mapping.to_XZ(covariates)
        F_dims = self.likelihood._neuron_to_F(neuron)
        
        with torch.no_grad():
            F_mu, F_var = self.mapping.compute_F(cov)
            
            maps = len(F_mu) if isinstance(F_mu, list) else 1
            if maps > 1:
                h = []                
                for m in range(maps):
                    h.append(self.likelihood.mc_gen(F_mu[m], F_var[m], MC, F_dims))
            else:
                h = self.likelihood.mc_gen(F_mu, F_var, MC, F_dims)
            
        return h


    
    ### optimization ###
    def set_optimizers(self, optim_tuple, opt_lr_dict, extra_params=(), nat_grad=(), nat_lr=1e0, 
                       newton_grad=(), newton_lr=1e-1):
        """
        Set the optimizers and the optimization hyperparameters.
        
        :param tuple optim_tuple: tuple containing PyTorch (optimizer, schedule_steps, lr_scheduler)
        :param dict opt_lr_dict: is a dictionary with parameter name and lr, needs to supply 'default': lr
        :param tuple extra_params: tuple of tuples containing (params, lr)
        :param tuple nat_grad: tuple of parameter names for natural gradient [1]
        
        References:
        
        [1] `Natural Gradients in Practice: Non-Conjugate Variational Inference in Gaussian Process Models`,
            Hugh Salimbeni, Stefanos Eleftheriadis, James Hensman (2018)
            
        """
        opt, self.sch_st, sch = optim_tuple
        assert 'default' in opt_lr_dict # default learning rate
        
        if opt == optim.LBFGS: # special case
            d = []
            for param in self.parameters():
                if name in nat_grad or name in newton_grad: # filter parameters
                    continue

                d.append(param)

            for param, lr in extra_params:
                d.append(param)
                
            history_size = opt_lr_dict['history'] if 'history' in opt_lr_dict else 10
            max_iter = opt_lr_dict['max_iter'] if 'max_iter' in opt_lr_dict else 4
                
            self.optim_base = opt(d, lr=opt_lr_dict['default'], 
                                  history_size=history_size, max_iter=max_iter)
        else:
            for key in opt_lr_dict:
                if key == 'default':
                    continue
                if not key in dict(self.named_parameters()):
                    print('The parameter {} is not found in the model.'.format(key))
                    
            d = []
            for name, param in self.named_parameters():

                if name in nat_grad or name in newton_grad: # filter parameters
                    continue

                opt_dict = {}
                opt_dict['params'] = param

                for key in opt_lr_dict:
                    if key == name:
                        opt_dict['lr'] = opt_lr_dict[key]
                        break

                d.append(opt_dict)

            for param, lr in extra_params:
                opt_dict = {}
                opt_dict['params'] = param
                opt_dict['lr'] = lr
                d.append(opt_dict)
            
            self.optim_base = opt(d, lr=opt_lr_dict['default'])
        self.sch = sch(self.optim_base)
        
        # natural gradients
        self.nat_grad = []
        pd = dict(self.named_parameters())
        assert len(nat_grad) % 2 == 0 # m, S pairs
        for key in nat_grad:
            self.nat_grad.append(pd[key])
            
        if len(self.nat_grad) > 0:
            self.optim_nat = optim.SGD(self.nat_grad, lr=nat_lr)
            
        # Newton gradients
        self.newton_grad = []
        pd = dict(self.named_parameters())
        for key in newton_grad:
            self.newton_grad.append(pd[key])
            
        if len(self.newton_grad) > 0:
            self.optim_newton = optim.SGD(self.newton_grad, lr=newton_lr)
        
        
    def fit(self, max_epochs, margin_epochs=10, loss_margin=0.0, neuron=None, kl_anneal_func=None,
            z_anneal_func=None, retain_graph=False, cov_samples=1, ll_samples=1, bound='ELBO', ll_mode='MC', 
            lv_input=None, callback=None):
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
        if z_anneal_func is None:
            z_anneal_func = lambda x: 1.0
            
        tracked_loss = []
        minloss = np.inf
        cnt = 0
        iterator = tqdm(range(max_epochs))
        
        optimizer = [self.optim_base]
        if len(self.nat_grad) > 0:
            optimizer += [self.optim_nat]
        if len(self.newton_grad) > 0:
            optimizer += [self.optim_newton]

        for epoch in iterator:
            
            sloss = 0
            for b in range(batches):
                
                def closure():
                    if torch.is_grad_enabled():
                        for o in optimizer:
                            o.zero_grad()
                            
                    anneal_t = epoch/max_epochs
                    loss = self.objective(b, neuron, beta=kl_anneal_func(anneal_t), beta_z=z_anneal_func(anneal_t), 
                                          cov_samples=cov_samples, ll_samples=ll_samples, bound=bound, 
                                          ll_mode=ll_mode, lv_input=lv_input)
                    
                    if loss.requires_grad:
                        #with profiler.profile(record_shapes=True, use_cuda=True) as profi:
                        #    with profiler.record_function("backwards"):
                        loss.backward(retain_graph=retain_graph)
                        
                        if len(self.nat_grad) > 0: # compute natural gradient
                            pytorch.compute_nat_grads(self.nat_grad)
                        if len(self.newton_grad) > 0: # compute Newton gradient
                            pytorch.compute_newton_grads(self.newton_grad, loss)
                    return loss
                
                if isinstance(optimizer[0], optim.LBFGS):
                    optimizer[0].step(closure)
                    with torch.no_grad():
                        loss = closure()
                else:
                    loss = closure()
                    optimizer[0].step()
                    
                for o in optimizer[1:]: # additional optimizers
                    o.step()

                self.constrain()
                if callback is not None: # additional operations per gradient step
                    callback(self)
                    
                if torch.isnan(loss.data).any() or torch.isinf(loss.data).any():
                    raise ValueError('Loss diverged')
                sloss += loss.item()
                
            if self.sch_st is not None and epoch % self.sch_st == self.sch_st-1:
                self.sch.step()

            sloss /= batches # average over batches, each batch loss is subsampled estimator of full loss
            iterator.set_postfix(loss=sloss)
            tracked_loss.append(sloss)
            
            if sloss <= minloss + loss_margin:
                cnt = 0
            else:
                cnt += 1
                
            if sloss < minloss:
                minloss = sloss

            if cnt > margin_epochs:
                print("\nStopped at epoch {}.".format(epoch+1))
                break
        
        return tracked_loss
    
    

### causal decoding ###
def VI_filtering(model, ini_X, spikes, VI_steps, fitting_options, past_spikes=None):
    """
    Decode causally taking into account covariate prior. In the most general case, treat model as LVM 
    but fix the latent variables at all timesteps except the last and fit model with fixed tuning. Note 
    filtering has to be done recursively, simultaneous inference of LVM is smoothing.

    Past covariate values before the decoding window (relevant in e.g. GP priors) are inserted via ``ini_X``, 
    identical to the procedure of initialization with *preprocess()*. Observed covariates will be variational 
    distributions with zero standard deviation.

    All model parameters are kept constant, only the decoded variables are optimized. Optimizers need to 
    be set up before with *set_optimizers()*.

    Reduces to Kalman filtering when Gaussian and Euclidean.

    :param list ini_X: 
    :param np.array spikes: 
    :param np.array past_spikes: spikes before the decoding window (relevant for GLMs with filter_len>1)
    """
    assert self.maps == 1 # no HMM supported

    rc_t = np.concatenate((past_spikes, spikes), dim=-1) if past_spikes is not None else spikes
    decode_len = spikes.shape[-1]
    past_len = past_spikes.shape[-1] if past_spikes is not None else 0
    resamples = rc_t.shape[-1]

    # set all parameters to not learnable
    label_for_constrain = []
    for name, param in self.named_parameters():
        if name[:16] == 'inputs.lv_':
            label_for_constrain.append(param)
            continue
        param.requires_grad = False

    def func(obj):
        decode_past = obj.saved_covariates[0].shape[0]
        k = 0
        for name, param in obj.named_parameters():
            if name[:16] == 'inputs.lv_':
                param.data[:decode_past] = obj.saved_covariates[k].shape[0]
                k += 1

    for k in range(past_len+1, past_len+decode_len):
        bs = [k+1, resamples-k-1] if resamples-k-1 > 0 else k+1
        model.inputs.set_XZ(ini_X, resamples, rc_t, bs) # non-continous batches
        model.likelihood.set_Y(rc_t)
        
        model.batches = 1 # only optimize first batch
        model.saved_covariates = []
        for p in label_for_constrain:
            model.saved_covariates.append(p.data[:k-1])

        # infer only last LV in batch 0
        model.fit(VI_steps, *fitting_options, callback=func) #loss_margin=-1e2, margin_epochs=100, anneal_func=annealing, 
             #cov_samples=32, ll_samples=1, bound='ELBO', ll_mode='MC'

    # set all parameters to learnable
    for param in self.parameters():
        param.requires_grad = True

    del model.saved_covariates

    return



def particle_filtering(self):
    """
    Suitable for AR(1) priors.
    """
    def __init__(self, covariates, bounds, samples=1000):
        """
        Rejection sampling to get the posterior samples
        Dynamic spatial maps supported
        """
        #self.register_buffer('samples', torch.empty((samples, covariates)))
        #self.register_buffer('weights', torch.empty((samples)))
        self.sam_cnt = samples
        self.eff_cnt = samples
        self.samples = np.empty((samples, covariates))
        self.weights = np.empty((samples))
        self.bounds = bounds

        
    def check_bounds(self):
        for k in range(self.bounds.shape[0]):
            np.clip(self.samples[:, k], self.bounds[k, 0], self.bounds[k, 1], out=self.samples[:, k])   

            
    def initialize(self, mu, std):
        """
        Initial samples positions, uniform weights
        """
        self.samples = std*np.random.randn(*self.samples.shape) + mu
        self.check_bounds()
        self.weights.fill(1/self.sam_cnt)

        
    def predict(self, sigma):
        """
        Prior distribution propagation
        """
        gauss = np.random.randn(*self.samples.shape)
        self.samples += sigma*gauss
        self.check_bounds()

        
    def update(self, eval_rate, activity, tbin):
        """
        Use IPP likelihood for cognitive maps to assign weights
        TODO use NLL directly
        """
        units = len(activity)
        fact = factorial(activity)
        w = 1.
        for u in range(units):
            rT = tbin*eval_rate[u]((self.samples[:, 0], self.samples[:, 1], np.zeros_like(self.samples[:, 0])))
            if activity[u] > 0:
                w *= rT**activity[u] / fact[u] * np.exp(-rT)

        self.weights *= w
        self.weights += 1e-12
        self.weights /= self.weights.sum()

        self.eff_cnt = np.sum(np.square(self.weights))

        
    def resample(self):
        """
        Stratified resampling algorithm
        """
        positions = (np.random.rand(self.sam_cnt) + np.arange(self.sam_cnt)) / self.sam_cnt
        indexes = np.zeros(self.sam_cnt, 'i')
        cumulative_sum = np.cumsum(self.weights)
        i, j = 0, 0
        while i < self.sam_cnt:
            if positions[i] < cumulative_sum[j]:
                indexes[i] = j
                i += 1
            else:
                j += 1
        self.samples = self.samples[indexes, :]
        self.weights.fill(1/self.sam_cnt)

        
    def estimate(self):
        """
        Get moments of particle distribution
        """
        rew = self.samples*self.weights[:, None]
        mu = rew.sum(0)
        cov = np.cov(rew.T)
        return mu, cov
    
    
    
# Markov chain Monte Carlo
def MCMC(prob_model):
    """
    Samples exact posterior samples from the probabilistic model, which has been fit with VI before.
    
    :param VI_optimized prob_model: the probabilistic model to sample from
    """
    return