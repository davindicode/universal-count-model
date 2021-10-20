import torch
from torch.nn.parameter import Parameter


import numbers
import numpy as np

from scipy.special import factorial

from . import point_process
from . import distributions as dist

from . import base



class count_model(base._likelihood):
    """
    Count likelihood base class.
    """
    def __init__(self, tbin, neurons, dispersion_mapping, inv_link, tensor_type):
        super().__init__(tbin, neurons, neurons, inv_link, tensor_type)
        self.strict_likelihood = True
        if dispersion_mapping is not None:
            self.add_module('dispersion_mapping', dispersion_mapping)
        else:
            self.dispersion_mapping = None
        
        
    def set_params(self, strict_likelihood=None):
        """
        :param float tbin: time bin duration in some time unit (sets time unit in model)
        :param bool strict_likelihood: flag for whether to compute the count probability (involves 
                                       constants to be loaded into memory)
        :param float jitter: value for stabilization of matrix inverses
        """
        if strict_likelihood is not None:
            self.strict_likelihood = strict_likelihood
        
        
    def set_Y(self, spikes, batch_size, filter_len=1):
        """
        Get all the activity into batches useable format for quick log-likelihood evaluation
        Tensor shapes: self.spikes (neuron_dim, batch_dim)
        
        tfact is the log of time_bin times the spike count
        lfact is the log (spike count)!
        """
        super().set_Y(spikes, batch_size, filter_len=filter_len)
        
        self.lfact = []
        self.tfact = []
        self.totspik = []
        for b in range(self.batches):
            spikes = self.spikes[b][..., self.filter_len-1:]
            self.totspik.append(spikes.sum(-1))
            self.tfact.append(spikes*torch.log(self.tbin.cpu()))
            self.lfact.append(torch.lgamma(spikes+1.))
            
            
    def KL_prior(self):
        """
        """
        if self.dispersion_mapping is not None:
            return self.dispersion_mapping.KL_prior()
        else:
            return 0
            
            
    def sample_helper(self, h, b, neuron, samples):
        """
        NLL helper function for sample evaluation. Note that spikes is batched including history 
        when the model uses history couplings, hence we sample the spike batches without the 
        history segments from this function.
        """
        rates = self.f(h) # watch out for underflow or overflow here
        spikes = self.spikes[b][:, neuron, self.filter_len-1:].to(self.tbin.device)
        if self.trials != 1 and samples > 1 and self.trials < h.shape[0]: # cannot rely on broadcasting
            spikes = spikes.repeat(samples, 1, 1) # MCxtrials
        
        if self.inv_link == 'exp': # spike count times log rate
            l_rates = (spikes*h)
        else:
            l_rates = (spikes*torch.log(rates+1e-12))
            
        return rates, l_rates, spikes
    
    
    def eval_dispersion_mapping(self, XZ, samples, neuron):
        """
        Posterior predictive mean of the dispersion model.
        """
        disp, disp_var = self.dispersion_mapping.compute_F(XZ)
        dh = self.mc_gen(disp, disp_var, samples, neuron)
        return self.dispersion_mapping.f(dh).mean(0) # watch out for underflow or overflow here
    
    
    def objective(self, F_mu, F_var, XZ, b, neuron, samples=10, mode='MC'):
        """
        Computes the terms for variational expectation :math:`\mathbb{E}_{q(f)q(z)}[]`, which 
        can be used to compute different likelihood objectives.
        The returned tensor will have sample dimension as MC over :math:`q(z)`, depending 
        on the evaluation mode will be MC or GH or exact over the likelihood samples. This 
        is all combined in to the same dimension to be summed over. The weights :math:`w_s` 
        are the quadrature weights or equal weights for MC, with appropriate normalization.
        
        :param int samples: number of MC samples or GH points (exact will ignore and give 1)
        
        :returns: negative likelihood term of shape (samples, timesteps), sample weights (samples, 1
        :rtype: tuple of torch.tensors
        """
        if mode == 'MC':
            h = self.mc_gen(F_mu, F_var, samples, neuron) # h has only observed neurons
            rates, l_rates, spikes = self.sample_helper(h, b, neuron, samples)
            ws = torch.tensor(1./rates.shape[0])
        elif mode == 'GH':
            h, ws = self.gh_gen(F_mu, F_var, samples, neuron)
            rates, l_rates, spikes = self.sample_helper(h, b, neuron, samples)
            ws = ws[:, None]
        else:
            raise NotImplementedError
            
        if self.dispersion_mapping is None:
            disper_param = None
        else: # MC sampling
            disper_param = self.eval_dispersion_mapping(XZ, samples, neuron)
        
        return self.nll(b, rates, l_rates, spikes, neuron, disper_param), ws



class renewal_model(base._likelihood):
    """
    Renewal model base class
    """
    def __init__(self, tbin, neurons, inv_link, tensor_type, allow_duplicate, dequantize):
        super().__init__(tbin, neurons, neurons, inv_link, tensor_type)
        self.allow_duplicate = allow_duplicate
        self.dequant = dequantize
        
        
    def train_to_ind(self, train):
        if self.allow_duplicate:
            duplicate = False
            spike_ind = train.nonzero().flatten()
            bigger = torch.where(train > 1)[0]
            add_on = (spike_ind,)
            for b in bigger:
                add_on += (b*torch.ones(int(train[b])-1, device=train.device, dtype=int),)

            if len(add_on) > 1:
                duplicate = True
            spike_ind = torch.cat(add_on)
            return torch.sort(spike_ind)[0], duplicate
        else:
            return torch.nonzero(train).flatten(), False
        
        
    def ind_to_train(self, ind, timesteps):
        train = torch.zeros((timesteps))
        train[ind] += 1
        return train
        
        
    def rate_rescale(self, neuron, spike_ind, rates, duplicate, minimum=1e-8):
        """
        Rate rescaling with option to dequantize, which will be random per sample.
        
        :param torch.tensor rates: input rates of shape (trials, neurons, timesteps)
        :returns: list of rescaled ISIs, list index over neurons, elements of shape (trials, ISIs)
        :rtype: list
        """
        rtime = torch.cumsum(rates, dim=-1)*self.tbin
        samples = rtime.shape[0]
        rISI = []
        for tr in range(self.trials):
            isis = []
            for en, n in enumerate(neuron):
                if len(spike_ind[tr][n]) > 1:
                    if self.dequant:
                        deqn = torch.rand(
                            samples, 
                            *spike_ind[tr][n].shape, 
                            device=rates.device
                        )*rates[tr::self.trials, en, spike_ind[tr][n]]*self.tbin # assume spike at 0
                        
                        tau = rtime[tr::self.trials, en, spike_ind[tr][n]] - deqn
                        if duplicate[n]: # re-oder in case of duplicate spike_ind
                            tau = torch.sort(tau, dim=-1)[0]
                    else:
                        tau = rtime[tr::self.trials, en, spike_ind[tr][n]]
                        
                    a = tau[:, 1:]-tau[:, :-1]
                    a[a < minimum] = minimum # don't allow near zero ISI
                    isis.append(a) # samples, order
                else:
                    isis.append([])
            rISI.append(isis)

        return rISI
        
        
    def set_Y(self, spikes, batch_size, filter_len=1):
        """
        Get all the activity into batches useable format for quick log-likelihood evaluation
        Tensor shapes: self.act [neuron_dim, batch_dim]
        """
        if self.allow_duplicate is False and spikes.max() > 1: # only binary trains
            raise ValueError('Only binary spike trains are accepted in set_Y() here')
        super().set_Y(spikes, batch_size, filter_len=filter_len)
        
        self.spiketimes = []
        self.intervals = torch.empty((self.batches, self.trials, self.neurons))
        self.duplicate = np.empty((self.batches, self.trials, self.neurons), dtype=bool)
        for b, spk in enumerate(self.spikes):
            spiketimes = []
            for tr in range(self.trials):
                cont = []
                for k in range(self.neurons):
                    s, self.duplicate[b, tr, k] = self.train_to_ind(spk[tr, k])
                    cont.append(s)
                    self.intervals[b, tr, k] = len(s)-1
                spiketimes.append(cont)
            self.spiketimes.append(spiketimes) # batch list of trial list of spike times list over neurons
        
        
    def sample_helper(self, h, b, neuron, scale, samples):
        """
        MC estimator for NLL function.
        
        :param torch.tensor scale: additional scaling of the rate rescaling to preserve the ISI mean
        
        :returns: tuple of rates, spikes*log(rates*scale), rescaled ISIs
        :rtype: tuple
        """
        scale = scale.expand(1, self.F_dims)[:, neuron, None] # rescale to get mean 1 in renewal distribution
        rates = self.f(h)*scale
        spikes = self.spikes[b][:, neuron, self.filter_len-1:].to(self.tbin.device)
        if self.trials != 1 and samples > 1 and self.trials < h.shape[0]: # cannot rely on broadcasting
            spikes = spikes.repeat(samples, 1, 1) # trial blocks are preserved, concatenated in first dim
            
        if self.inv_link == 'exp': # bit masking seems faster than integer indexing using spiketimes
            l_rates = (spikes*(h+torch.log(scale))).sum(-1)
        else:
            l_rates = (spikes*torch.log(rates+1e-12)).sum(-1) # rates include scaling
        
        spiketimes = [[s.to(self.tbin.device) for s in ss] for ss in self.spiketimes[b]]
        rISI = self.rate_rescale(neuron, spiketimes, rates, self.duplicate[b])
        return rates, l_rates, rISI
    
    
    def objective(self, F_mu, F_var, XZ, b, neuron, scale, samples=10, mode='MC'):
        """
        :param torch.tensor F_mu: model output F mean values of shape (samplesxtrials, neurons, time)
        
        :returns: negative likelihood term of shape (samples, timesteps), sample weights (samples, 1
        :rtype: tuple of torch.tensors
        """
        if mode == 'MC':
            h = self.mc_gen(F_mu, F_var, samples, neuron)
            rates, l_rates, rISI = self.sample_helper(h, b, neuron, scale, samples)
            ws = torch.tensor(1./rates.shape[0])
        else:
            raise NotImplementedError
            
        return self.nll(l_rates, rISI, neuron), ws
    
    
    def sample(self, rate, neuron=None, XZ=None):
        """
        Sample spike trains from the modulated renewal process.
        
        :param numpy.array rate: input rate of shape (trials, neuron, timestep)
        :returns: spike train of shape (trials, neuron, timesteps)
        :rtype: np.array
        """
        neuron = self._validate_neuron(neuron)
        spiketimes = point_process.gen_IRP(self.ISI_dist(neuron), rate[:, neuron, :], self.tbin.item())
            
        tr_t_spike = []
        for sp in spiketimes:
            tr_t_spike.append(self.ind_to_train(torch.tensor(sp), rate.shape[-1]).numpy())

        return np.array(tr_t_spike).reshape(rate.shape[0], -1, rate.shape[-1])



# Special cases
class Spike_phase(base._likelihood):
    """
    Renewal model base class
    """
    def __init__(self, tbin, neurons, inv_link, tensor_type, allow_duplicate, dequantize):
        super().__init__(tbin, neurons, neurons, inv_link, tensor_type)
        
    def set_Y(self, spikes, batch_size, filter_len=1):
        """
        Assumes at time zero, we start at global phase zero. At the end after the last spike, we do 
        not increment the phase here.
        """
        assert spikes.max() < 2 # only binary trains
        super().set_Y(spikes, batch_size, filter_len=filter_len)
        
        phases = [] # list of spike phases
        for spiketrain in self.all_spikes: # loop over trials
            cont = []
            dphase = torch.zeros(*spiketrain.shape, dtype=self.tensor_type)
            for k in range(self.neurons):
                locs = torch.nonzero(spiketrain).flatten()
                dlocs = torch.cat((locs[0:1]+1, locs[1:]-locs[:-1]))
                cur = 0 
                for dd in dlocs:
                    dphase[k, cur:cur+dd] = 1./dd
                
            phases.append(torch.cumsum(dphase, dim=-1)) # global spike phase
        self.phases = torch.stack(phases) # tr, n, time
            
    def geodesic(x, y):
        """
        Returns the geodesic displacement between x and y, (x-y).
        """
        xy = (x-y) % 1.
        xy[xy > 0.5] -= 1.
        return xy   
        
    def sample_helper(self, h, b, neuron, scale, samples):
        """
        MC estimator for NLL function.
        
        :param torch.tensor scale: additional scaling of the rate rescaling to preserve the ISI mean
        
        :returns: tuple of rates, spikes*log(rates*scale), rescaled ISIs
        :rtype: tuple
        """
        rates = self.f(h)
        spikes = self.spikes[b][:, neuron, self.filter_len-1:].to(self.tbin.device)
        
        if self.inv_link == 'exp': # bit masking seems faster than integer indexing using spiketimes
            l_rates = (spikes*(h+torch.log(scale))).sum(-1)
        else:
            l_rates = (spikes*torch.log(rates+1e-12)).sum(-1) # rates include scaling
        
        spiketimes = [s.to(self.tbin.device) for s in self.spiketimes[b]]
        rISI = self.rate_rescale(neuron, spiketimes, rates, self.duplicate[b])
        return rates
    
    def objective(self, F_mu, F_var, XZ, b, neuron, scale, samples=10, mode='MC'):
        """
        """
        if mode == 'MC':
            h = self.mc_gen(F_mu, F_var, samples, neuron)
            rates, l_rates, rISI = self.sample_helper(h, b, neuron, scale, samples)
            ws = torch.tensor(1./rates.shape[0])
        else:
            raise NotImplementedError
            
        return self.nll(l_rates, rISI, neuron), ws
    
    def nll(phase, tar_phase):
        """
        """
        lowest_loss = np.inf
        shift = Parameter(torch.zeros(1, device=dev))
        a = Parameter(torch.zeros(1, device=dev))

        XX = torch.tensor(x, device=dev)
        HD = torch.tensor(theta, device=dev)

        losses = []
        for k in range(iters):
            optimizer.zero_grad()
            X_ = 2*np.pi*XX*a + shift
            loss = (metric(X_, HD, 'torus')**2).mean()
            loss.backward()
            optimizer.step()
            losses.append(loss.cpu().item())

        l_ = loss.cpu().item()
        print(l_)
        shift_ = shift.cpu().item()
        a_ = a.cpu().item()

        return nll

    def sample(self, rate, neuron=None, XZ=None):
        """
        Sample spike trains from the modulated renewal process.
        
        :param numpy.array rate: input rate of shape (trials, neuron, timestep)
        :returns: spike train of shape (trials, neuron, timesteps)
        :rtype: np.array
        """
        neuron = self._validate_neuron(neuron)
        spiketimes = point_process.gen_IRP(self.ISI_dist(neuron), rate[:, neuron, :], self.tbin.item())
            
        tr_t_spike = []
        for sp in spiketimes:
            tr_t_spike.append(self.ind_to_train(torch.tensor(sp), rate.shape[-1]).numpy())

        return np.array(tr_t_spike).reshape(rate.shape[0], -1, rate.shape[-1])
    
    
    
class Universal(base._likelihood):
    """
    Universal count distribution with finite cutoff at max_count.
    
    """
    def __init__(self, neurons, C, inv_link, max_count, mapping_net, tensor_type=torch.float):
        super().__init__(1., neurons*C, neurons, inv_link, tensor_type) # dummy tbin
        self.K = max_count+1
        self.C = C
        self.neurons = neurons
        self.lsoftm = torch.nn.LogSoftmax(dim=-1)
        self.add_module('mapping_net', mapping_net) # maps from NxC to NxK
    
    
    def set_Y(self, spikes, batch_size, filter_len=1):
        """
        Get all the activity into batches useable format for fast log-likelihood evaluation. 
        Batched spikes will be a list of tensors of shape (trials, neurons, time) with trials 
        set to 1 if input has no trial dimension (e.g. continuous recording).
        
        :param np.array spikes: becomes a list of [neuron_dim, batch_dim]
        :param int/list batch_size: 
        :param int filter_len: history length of the GLM couplings (1 indicates no history coupling)
        """
        if self.K <= spikes.max():
            raise ValueError('Maximum count is exceeded in the spike count data')
        super().set_Y(spikes, batch_size, filter_len)
                
        
    def onehot_to_counts(self, onehot):
        """
        Convert one-hot vector representation of counts. Assumes the event dimension is the last.
        
        :param torch.tensor onehot: one-hot vector representation of shape (..., event)
        :returns: spike counts
        :rtype: torch.tensor
        """
        counts = torch.zeros(*onehot.shape[:-1], device=onehot.device)
        inds = torch.where(onehot)
        counts[inds[:-1]] = inds[-1].float()
        return counts

    
    def counts_to_onehot(self, counts):
        """
        Convert counts to one-hot vector representation. Adds the event dimension at the end.
        
        :param torch.tensor counts: spike counts of some tensor shape
        :param int max_counts: size of the event dimension (max_counts + 1)
        :returns: one-hot representation of shape (counts.shape, event)
        :rtype: torch.tensor
        """
        onehot = torch.zeros(*counts.shape, self.K, device=counts.device)
        onehot_ = onehot.view(-1, self.K)
        g = onehot_.shape[0]
        onehot_[np.arange(g), counts.flatten()[np.arange(g)].long()] = 1
        return onehot_.view(*onehot.shape)
    
    
    def get_logp(self, F_mu, neuron):
        """
        Compute count probabilities from the rate model output.
        
        :param torch.tensor F_mu: the F_mu product output of the rate model (samples and/or trials, F_dims, time)
        :returns: log probability tensor
        :rtype: tensor of shape (samples and/or trials, n, t, c)
        """
        T = F_mu.shape[-1]
        samples = F_mu.shape[0]
        a = self.mapping_net(F_mu.permute(0, 2, 1).reshape(samples*T, -1), neuron) # samplesxtime, NxK
        log_probs = self.lsoftm(a.view(samples, T, -1, self.K).permute(0, 2, 1, 3))
        return log_probs
    
    
    def sample_helper(self, h, b, neuron, samples):
        """
        NLL helper function for sample evaluation. Note the F_mu dimensions here is equal to NxC.
        """
        logp = self.get_logp(h, neuron)
        spikes = self.spikes[b][:, neuron, self.filter_len-1:].to(self.tbin.device)
        if self.trials != 1 and samples > 1 and self.trials < h.shape[0]: # cannot rely on broadcasting
            spikes = spikes.repeat(samples, 1, 1)
        tar = self.counts_to_onehot(spikes)
        
        return logp, tar
    
    
    def _neuron_to_F(self, neuron):
        """
        Access subset of neurons in expanded space.
        """
        neuron = self._validate_neuron(neuron)
        if len(neuron) == self.neurons:
            F_dims = list(range(self.F_dims))
        else: # access subset of neurons
            F_dims = list(np.concatenate([np.arange(n*self.C, (n+1)*self.C) for n in neuron]))
            
        return F_dims
    
    
    def objective(self, F_mu, F_var, XZ, b, neuron, samples=10, mode='MC'):
        """
        Computes the terms for variational expectation :math:`\mathbb{E}_{q(f)q(z)}[]`, which 
        can be used to compute different likelihood objectives.
        The returned tensor will have sample dimension as MC over :math:`q(z)`, depending 
        on the evaluation mode will be MC or GH or exact over the likelihood samples. This 
        is all combined in to the same dimension to be summed over. The weights :math:`w_s` 
        are the quadrature weights or equal weights for MC, with appropriate normalization.
        
        :param int samples: number of MC samples or GH points (exact will ignore and give 1)
        
        :returns: negative likelihood term of shape (samples, timesteps)
        """
        F_dims = self._neuron_to_F(neuron)
        
        if mode == 'MC':
            h = self.mc_gen(F_mu, F_var, samples, F_dims) # h has only observed neurons (from neuron)
            logp, tar = self.sample_helper(h, b, neuron, samples)
            ws = torch.tensor(1./logp.shape[0])
        elif mode == 'GH':
            h, ws = self.gh_gen(F_mu, F_var, samples, F_dims)
            logp, tar = self.sample_helper(h, b, neuron, samples)
            ws = ws[:, None]
        else:
            raise NotImplementedError

        nll = -(tar*logp).sum(-1)
        return nll.sum(1), ws

    
    def sample(self, F_mu, neuron, XZ=None):
        """
        Sample from the categorical distribution.
        
        :param numpy.array log_probs: log count probabilities (trials, neuron, timestep, counts), no 
                                      need to be normalized
        :returns: spike train of shape (trials, neuron, timesteps)
        :rtype: np.array
        """
        F_dims = self._neuron_to_F(neuron)
        log_probs = self.get_logp(torch.tensor(F_mu[:, F_dims, :], dtype=self.tensor_type))
        c_dist = mdl.distributions.Categorical(logits=log_probs)
        cnt_prob = torch.exp(log_probs)
        return c_dist.sample().numpy()

    
    
# count distributions
class Bernoulli(count_model):
    """
    Inhomogeneous Bernoulli likelihood, limits the count to binary trains.
    """
    def __init__(self, tbin, neurons, inv_link, tensor_type=torch.float):
        super().__init__(tbin, neurons, None, inv_link, tensor_type)
        
        
    def set_Y(self, spikes, batch_size, filter_len=1):
        """
        Get all the activity into batches useable format for quick log-likelihood evaluation
        Tensor shapes: self.spikes (neuron_dim, batch_dim)
        
        tfact is the log of time_bin times the spike count
        """
        assert spikes.max() < 2
        super().set_Y(spikes, batch_size, filter_len=filter_len)
        
        
    def get_saved_factors(self, b, neuron, spikes):
        """
        Get saved factors for proper likelihood values and perform broadcasting when needed.
        """
        if self.strict_likelihood:
            tfact = self.tfact[b][:, neuron, :].to(self.tbin.device)
            if tfact.shape[0] != 1 and tfact.shape[0] < spikes.shape[0]: # cannot rely on broadcasting
                tfact = tfact.repeat(spikes.shape[0]//tfact.shape[0], 1, 1)
        else:
            tfact = 0
        return tfact
        
        
    def nll(self, b, rates, l_rates, spikes, neuron, disper_param=None):
        tfact = self.get_saved_factors(b, neuron, spikes)
        nll = -(l_rates + tfact + (1-spikes)*torch.log(1-rates*self.tbin))
        return nll.sum(1)
    
    
    def sample(self, rate, neuron=None, XZ=None):
        """
        Takes into account the quantization bias if we sample IPP with dilation factor.
        
        :param numpy.array rate: input rate of shape (trials, neuron, timestep)
        :returns: spike train of shape (trials, neuron, timesteps)
        :rtype: np.array
        """
        neuron = self._validate_neuron(neuron)
        return point_process.gen_IBP(rate[:, neuron, :]*self.tbin.item())
    
    

class Poisson(count_model):
    """
    Poisson count likelihood.
    """
    def __init__(self, tbin, neurons, inv_link, tensor_type=torch.float):
        super().__init__(tbin, neurons, None, inv_link, tensor_type)
        
        
    def get_saved_factors(self, b, neuron, spikes):
        """
        Get saved factors for proper likelihood values and perform broadcasting when needed.
        """
        if self.strict_likelihood:
            tfact = self.tfact[b][:, neuron, :].to(self.tbin.device)
            lfact = self.lfact[b][:, neuron, :].to(self.tbin.device)
            if tfact.shape[0] != 1 and tfact.shape[0] < spikes.shape[0]: # cannot rely on broadcasting
                tfact = tfact.repeat(spikes.shape[0]//tfact.shape[0], 1, 1)
                lfact = lfact.repeat(spikes.shape[0]//lfact.shape[0], 1, 1)
        else:
            tfact, lfact = 0, 0
        return tfact, lfact
        
        
    def nll(self, b, rates, l_rates, spikes, neuron, disper_param=None):
        tfact, lfact = self.get_saved_factors(b, neuron, spikes)
        T = rates*self.tbin
        nll = (-l_rates + T - tfact + lfact)
        return nll.sum(1)
    
    
    def objective(self, F_mu, F_var, XZ, b, neuron, samples=10, mode='MC'):
        """
        The Poisson likelihood with the log Cox process has an exact variational likelihood term.
        """
        if self.inv_link == 'exp': # exact
            if isinstance(F_var, numbers.Number) is False and len(F_var.shape) == 4: # diagonalize
                F_var = F_var.view(*F_var.shape[:2], -1)[:, :, ::F_var.shape[-1]+1]
            
            spikes = self.spikes[b][:, neuron, self.filter_len-1:].to(self.tbin.device)
            rates = self.f(F_mu + F_var/2.)[:, neuron, :] # watch out for underflow or overflow here
            l_rates = (spikes*F_mu[:, neuron, :])
            
            tfact, lfact = self.get_saved_factors(b, neuron, spikes)
            T = rates*self.tbin
            nll = (-l_rates + T - tfact + lfact)
            
            ws = torch.tensor(1./rates.shape[0])
            return nll.sum(1), ws # first dimension is summed over later (MC over Z), hence divide by shape[0]
        else:
            return super().objective(F_mu, F_var, XZ, b, neuron, 
                                     samples=samples, mode=mode)

        
    def sample(self, rate, neuron=None, XZ=None):
        """
        Takes into account the quantization bias if we sample IPP with dilation factor.
        
        :param numpy.array rate: input rate of shape (trials, neuron, timestep)
        :returns: spike train of shape (trials, neuron, timesteps)
        :rtype: np.array
        """
        neuron = self._validate_neuron(neuron)
        return torch.poisson(torch.tensor(rate[:, neuron, :]*self.tbin.item())).numpy()
    
    
    
class ZI_Poisson(count_model):
    """
    Zero-inflated Poisson (ZIP) count likelihood. [1]
    
    References:
    
    [1] `Untethered firing fields and intermittent silences: Why grid‐cell discharge is so variable`, 
        Johannes Nagele  Andreas V.M. Herz  Martin B. Stemmler
    
    """
    def __init__(self, tbin, neurons, inv_link, alpha=None, tensor_type=torch.float, dispersion_mapping=None):
        super().__init__(tbin, neurons, dispersion_mapping, inv_link, tensor_type)
        if alpha is not None:
            self.register_parameter('alpha', Parameter(torch.tensor(alpha, dtype=self.tensor_type)))
        else:
            self.alpha = None
            
            
    def set_params(self, alpha=None, strict_ll=None):
        super().set_params(strict_ll)
        assert self.strict_likelihood is True
        if alpha is not None:
            self.alpha.data = torch.tensor(alpha, device=self.tbin.device, dtype=self.tensor_type)
            
            
    def constrain(self):
        if self.alpha is not None:
            self.alpha.data = torch.clamp(self.alpha.data, min=0., max=1.)
            
            
    def get_saved_factors(self, b, neuron, spikes):
        """
        Get saved factors for proper likelihood values and perform broadcasting when needed.
        """
        tfact = self.tfact[b][:, neuron, :].to(self.tbin.device)
        lfact = self.lfact[b][:, neuron, :].to(self.tbin.device)
        if tfact.shape[0] != 1 and tfact.shape[0] < spikes.shape[0]: # cannot rely on broadcasting
            tfact = tfact.repeat(spikes.shape[0]//tfact.shape[0], 1, 1)
            lfact = lfact.repeat(spikes.shape[0]//lfact.shape[0], 1, 1)
 
        return tfact, lfact
        
    
    def nll(self, b, rates, l_rates, spikes, neuron, disper_param=None):
        if disper_param is None:
            alpha_ = self.alpha.expand(1, self.neurons)[:, neuron, None]
        else:
            alpha_ = disper_param
            
        tfact, lfact = self.get_saved_factors(b, neuron, spikes)
        T = rates*self.tbin
        zero_spikes = (spikes == 0) # mask
        nll_ = (-l_rates + T - tfact + lfact - torch.log(1.-alpha_)) # -log (1-alpha)*p(N)
        p = torch.exp(-nll_) # stable as nll > 0
        nll_0 = -torch.log(alpha_ + p)
        nll = zero_spikes*nll_0 + (~zero_spikes)*nll_
        return nll.sum(1)

    
    def sample(self, rate, neuron=None, XZ=None):
        """
        Sample from ZIP process.
        
        :param numpy.array rate: input rate of shape (trials, neuron, timestep)
        :returns: spike train of shape (trials, neuron, timesteps)
        :rtype: np.array
        """
        neuron = self._validate_neuron(neuron)
        rate_ = rate[:, neuron, :]
        
        if self.dispersion_mapping is None:
            alpha_ = self.alpha[None, :, None].expand(rate.shape[0], self.neurons, 
                                                      rate_.shape[-1]).data.cpu().numpy()[:, neuron, :]
        else:
            samples = rate.shape[0]
            alpha_ = self.eval_dispersion_mapping(XZ, samples, 
                                                  neuron)[None, ...].expand(rate.shape[0], 
                                                                            *rate_.shape[1:]).data.cpu().numpy()
            
        zero_mask = point_process.gen_IBP(alpha_)
        return (1.-zero_mask)*torch.poisson(torch.tensor(rate_*self.tbin.item())).numpy()

        
        
class Negative_binomial(count_model):
    """
    Gamma-Poisson mixture.
    
    :param np.array r_inv: :math:`r^{-1}` parameter of the NB likelihood, if left to None this value is 
                           expected to be provided by the heteroscedastic model in the inference class.
    """
    def __init__(self, tbin, neurons, inv_link, r_inv=None, tensor_type=torch.float, dispersion_mapping=None):
        super().__init__(tbin, neurons, dispersion_mapping, inv_link, tensor_type)
        if r_inv is not None:
            self.register_parameter('r_inv', Parameter(torch.tensor(r_inv, dtype=self.tensor_type)))
        else:
            self.r_inv = None
        
        
    def set_params(self, r_inv=None, strict_ll=None):
        super().set_params(strict_ll)
        if r_inv is not None:
            self.r_inv.data = torch.tensor(r_inv, device=self.tbin.device, dtype=self.tensor_type)
            
            
    def constrain(self):
        if self.r_inv is not None:
            self.r_inv.data = torch.clamp(self.r_inv.data, min=0.) # effective r_inv > 1e-6 stabilized in NLL
        
        
    def get_saved_factors(self, b, neuron, spikes):
        """
        Get saved factors for proper likelihood values and perform broadcasting when needed.
        """
        if self.strict_likelihood:
            tfact = self.tfact[b][:, neuron, :].to(self.tbin.device)
            lfact = self.lfact[b][:, neuron, :].to(self.tbin.device)
            if tfact.shape[0] != 1 and tfact.shape[0] < spikes.shape[0]: # cannot rely on broadcasting
                tfact = tfact.repeat(spikes.shape[0]//tfact.shape[0], 1, 1)
                lfact = lfact.repeat(spikes.shape[0]//lfact.shape[0], 1, 1)
        else:
            tfact, lfact = 0, 0
        return tfact, lfact
        
        
    def nll(self, b, rates, l_rates, spikes, neuron, disper_param=None):
        """
        The negative log likelihood function. Note that if disper_param is not None, it will use those values for 
        the dispersion parameter rather than its own dispersion parameters.
        
        :param int b: batch index to evaluate
        :param torch.tensor rates: rates of shape (trial, neuron, time)
        :param torch.tensor l_rates: spikes*log(rates)
        :param torch.tensor spikes: spike counts of shape (trial, neuron, time)
        :param list neuron: list of neuron indices to evaluate
        :param torch.tensor disper_param: input for heteroscedastic NB likelihood of shape (trial, neuron, time), 
                                          otherwise uses fixed :math:`r_{inv}`
        :returns: NLL of shape (trial, time)
        :rtype: torch.tensor
        """
        if disper_param is None:
            r_ = 1./(self.r_inv.expand(1, self.neurons)[:, neuron, None] + 1e-6)
        else:
            r_ = 1./(disper_param + 1e-6)
        
        tfact, lfact = self.get_saved_factors(b, neuron, spikes) 
        lambd = rates*self.tbin
        fac_lgamma = (-torch.lgamma(r_+spikes) + torch.lgamma(r_))
        fac_power = ((spikes+r_)*torch.log(r_+lambd) - r_*torch.log(r_))
        
        nll = (-l_rates + fac_power + fac_lgamma - tfact + lfact)
        return nll.sum(1)

    
    def sample(self, rate, neuron=None, XZ=None):
        """
        Sample from the Gamma-Poisson mixture.
        
        :param numpy.array rate: input rate of shape (trials, neuron, timestep)
        :param int max_count: maximum number of spike counts per time bin
        :returns: spike train of shape (trials, neuron, timesteps)
        :rtype: np.array
        """
        neuron = self._validate_neuron(neuron)
        rate_ = rate[:, neuron, :]
        
        if self.dispersion_mapping is None:
            r_ = 1./(self.r_inv[None, :, None].expand(rate.shape[0], self.neurons, 
                                                      rate_.shape[-1]).data.cpu().numpy()
                     + 1e-6)[:, neuron, :]
        else:
            samples = rate.shape[0]
            disp = self.eval_dispersion_mapping(XZ, samples, neuron)[None, ...].expand(rate.shape[0], 
                                                                                       *rate_.shape[1:])
            r_ = 1./(disp.data.cpu().numpy() + 1e-6)
            
        s = np.random.gamma(r_, rate_*self.tbin.item()/r_)
        return torch.poisson(torch.tensor(s)).numpy()

        
        
class COM_Poisson(count_model):
    """
    Conway-Maxwell-Poisson, as described in 
    https://en.wikipedia.org/wiki/Conway%E2%80%93Maxwell%E2%80%93Poisson_distribution.
    
    """
    def __init__(self, tbin, neurons, inv_link, nu=None, tensor_type=torch.float, J=100, dispersion_mapping=None):
        super().__init__(tbin, neurons, dispersion_mapping, inv_link, tensor_type)
        if nu is not None:
            self.register_parameter('log_nu', Parameter(torch.tensor(nu, dtype=self.tensor_type)))
        else:
            self.log_nu = None
            
        self.J = J
        self.register_buffer('powers', torch.tensor(np.arange(self.J+1), dtype=self.tensor_type).to(self.tbin.device))
        self.register_buffer('jfact', torch.lgamma(self.powers+1.).to(self.tbin.device))
        
        
    def set_params(self, log_nu=None, strict_ll=None):
        super().set_params(strict_ll)
        if log_nu is not None:
            self.log_nu.data = torch.tensor(log_nu, device=self.tbin.device, dtype=self.tensor_type)
            
            
    def get_saved_factors(self, b, neuron, spikes):
        """
        Get saved factors for proper likelihood values and perform broadcasting when needed.
        """
        if self.strict_likelihood:
            tfact = self.tfact[b][:, neuron, :].to(self.tbin.device)
            if tfact.shape[0] != 1 and tfact.shape[0] < spikes.shape[0]: # cannot rely on broadcasting
                tfact = tfact.repeat(spikes.shape[0]//tfact.shape[0], 1, 1)
        else:
            tfact = 0
            
        lfact = self.lfact[b][:, neuron, :].to(self.tbin.device)
        if lfact.shape[0] != 1 and lfact.shape[0] < spikes.shape[0]: # cannot rely on broadcasting
            lfact = lfact.repeat(spikes.shape[0]//lfact.shape[0], 1, 1)
            
        return tfact, lfact
    
    
    def log_Z(self, log_lambda, nu):
        """
        Partition function.
        
        :param torch.tensor lambd: lambda of shape (samples, neurons, timesteps)
        :param torch.tensor nu: nu of shape (samples, neurons, timesteps)
        :returns: log Z of shape (samples, neurons, timesteps)
        :rtype: torch.tensor
        """
        #indx = torch.where((self.powers*lambd.max() - nu_.min()*self.j) < -1e1) # adaptive
        #if len(indx) == 0:
        #    indx = self.J+1
        log_Z_term = (self.powers[:, None, None, None]*log_lambda[None, ...] - \
            nu[None, ...]*self.jfact[:, None, None, None])
        return torch.logsumexp(log_Z_term, dim=0)
        
        
    def nll(self, b, rates, l_rates, spikes, neuron, disper_param=None):
        """
        :param int b: batch index to evaluate
        :param torch.tensor rates: rates of shape (trial, neuron, time)
        :param torch.tensor l_rates: spikes*log(rates)
        :param torch.tensor spikes: spike counts of shape (neuron, time)
        :param list neuron: list of neuron indices to evaluate
        :param torch.tensor disper_param: input for heteroscedastic NB likelihood of shape (trial, neuron, time), 
                                          otherwise uses fixed :math:`\nu`
        :returns: NLL of shape (trial, time)
        :rtype: torch.tensor
        """
        if disper_param is None:
            nu_ = torch.exp(self.log_nu).expand(1, self.neurons)[:, neuron, None]
        else:
            nu_ = torch.exp(disper_param) # nn.functional.softplus
        
        tfact, lfact = self.get_saved_factors(b, neuron, spikes)
        log_lambda = torch.log(rates*self.tbin+1e-12)
        
        l_Z = self.log_Z(log_lambda, nu_)
        
        nll = (-l_rates + l_Z - tfact + nu_*lfact)
        return nll.sum(1)

    
    def sample(self, rate, neuron=None, XZ=None):
        """
        Sample from the CMP distribution.
        
        :param numpy.array rate: input rate of shape (neuron, timestep)
        :returns: spike train of shape (trials, neuron, timesteps)
        :rtype: np.array
        """
        neuron = self._validate_neuron(neuron)
        mu_ = rate[:, neuron, :]*self.tbin.item()
        
        if self.dispersion_mapping is None:
            nu_ = torch.exp(self.log_nu)[None, :, None].expand(
                rate.shape[0], self.neurons, mu_.shape[-1]).data.cpu().numpy()[:, neuron, :]
        else:
            samples = rate.shape[0]
            disp = self.eval_dispersion_mapping(XZ, samples, neuron)[None, ...].expand(rate.shape[0], 
                                                                                       *mu_.shape[1:])
            nu_ = torch.exp(disp.data).cpu().numpy()
        
        return point_process.gen_CMP(mu_, nu_)



# renewal distributions
class Gamma(renewal_model):
    """
    Gamma renewal process
    """
    def __init__(self, tbin, neurons, inv_link, shape, tensor_type=torch.float, allow_duplicate=True, 
                 dequantize=True):
        """
        Renewal parameters shape can be shared for all neurons or independent.
        """
        super().__init__(tbin, neurons, inv_link, tensor_type, allow_duplicate, dequantize)
        self.register_parameter('shape', Parameter(torch.tensor(shape, dtype=self.tensor_type)))
          
            
    def set_params(self, shape=None):
        if shape is not None:
            self.shape.data = torch.tensor(shape, device=self.shape.device, dtype=self.tensor_type)
          
        
    def constrain(self):
        self.shape.data = torch.clamp(self.shape.data, min=1e-5, max=2.5)
    
    
    def nll(self, l_rates, rISI, neuron):
        """
        Gamma case, approximates the spiketrain NLL (takes tbin into account for NLL).
        
        :param np.array neuron: fit over given neurons, must be an array
        :param torch.tensor F_mu: F_mu product with shape (samples, neurons, timesteps)
        :param torch.tensor F_var: variance of the F_mu values, same shape
        :param int b: batch number
        :param np.array neuron: neuron indices that are used
        :param int samples: number of MC samples for likelihood evaluation
        :returns: NLL array over sample dimensions
        :rtype: torch.tensor
        """
        samples_ = l_rates.shape[0] # ll_samplesxcov_samples, in case of trials trial_num=cov_samples
        shape_ = self.shape.expand(1, self.F_dims)[:, neuron]
        
        # Ignore the end points of the spike train
        # d_Lambda_i = rates[:self.spiketimes[0]].sum()*self.tbin
        # d_Lambda_f = rates[self.spiketimes[ii]:].sum()*self.tbin
        # l_start = torch.empty((len(neuron)), device=self.tbin.device)
        # l_end = torch.empty((len(neuron)), device=self.tbin.device)
        # l_start[n_enu] = torch.log(sps.gammaincc(self.shape.item(), d_Lambda_i))
        # l_end[n_enu] = torch.log(sps.gammaincc(self.shape.item(), d_Lambda_f))
        
        intervals = torch.zeros((samples_, len(neuron)), device=self.tbin.device)
        T = torch.empty((samples_, len(neuron)), device=self.tbin.device) # MC samples, neurons
        l_Lambda = torch.empty((samples_, len(neuron)), device=self.tbin.device)
        for tr, isis in enumerate(rISI): # over trials
            for n_enu, isi in enumerate(isis): # over neurons
                if len(isi) > 0: # nonzero number of ISIs
                    intervals[tr::self.trials, n_enu] = isi.shape[-1]
                    T[tr::self.trials, n_enu] = isi.sum(-1)
                    l_Lambda[tr::self.trials, n_enu] = torch.log(isi+1e-12).sum(-1)
                else:
                    T[tr::self.trials, n_enu] = 0 # TODO: minibatching ISIs approximate due to b.c.
                    l_Lambda[tr::self.trials, n_enu] = 0

        nll = -(shape_-1)*l_Lambda - l_rates + T + intervals[None, :]*torch.lgamma(shape_)
        return nll.sum(1, keepdims=True) # sum over neurons, keep as dummy time index
    
    
    def objective(self, F_mu, F_var, XZ, b, neuron, samples=10, mode='MC'):
        return super().objective(F_mu, F_var, XZ, b, neuron, 
                                 self.shape, samples=samples, mode=mode)
    
    
    def ISI_dist(self, n):
        shape = self.shape[n].data.cpu().numpy()
        return point_process.ISI_gamma(shape, scale=1./shape)

        
        
class logNormal(renewal_model):
    """
    Log-normal ISI distribution
    Ignores the end points of the spike train in each batch
    """
    def __init__(self, tbin, neurons, inv_link, sigma, tensor_type=torch.float, allow_duplicate=True, 
                 dequantize=True):
        """
        :param np.array sigma: :math:`$sigma$` parameter which is > 0
        """
        super().__init__(tbin, neurons, inv_link, tensor_type, allow_duplicate, dequantize)
        #self.register_parameter('mu', Parameter(torch.tensor(mu, dtype=self.tensor_type)))
        self.register_parameter('sigma', Parameter(torch.tensor(sigma, dtype=self.tensor_type)))
        self.register_buffer('twopi_fact', 0.5*torch.tensor(2*np.pi, dtype=self.tensor_type).log())
         
            
    def set_params(self, sigma=None):
        if sigma is not None:
            self.sigma.data = torch.tensor(sigma, device=self.sigma.device)
         
        
    def constrain(self):
        self.sigma.data = torch.clamp(self.sigma.data, min=1e-5)
          
            
    def set_Y(self, spikes, batch_size, filter_len=1):
        super().set_Y(spikes, batch_size, filter_len=filter_len)
        
        
    def nll(self, l_rates, rISI, neuron):        
        """
        Computes the log Normal distribution
        
        .. math:: p(f^* \mid X_{new}, X, y, k, X_u, u_{loc}, u_{scale\_tril})
            = \mathcal{N}(loc, cov).
        
        :param torch.tensor l_rates: log rates at spike times (samples, neurons, timesteps)
        :param torch.tensor rISI: modified rate rescaled ISIs
        :param np.array neuron: neuron indices that are used
        :returns: spike train negative log likelihood of shape (timesteps, samples (dummy dimension))
        :rtype: torch.tensor
        """
        sigma_ = self.sigma.expand(1, self.F_dims)[:, neuron]
        samples_ = l_rates.shape[0]
        
        l_Lambda = torch.empty((samples_, len(neuron)), device=self.tbin.device)
        quad_term = torch.empty((samples_, len(neuron)), device=self.tbin.device)
        norm_term = torch.empty((samples_, len(neuron)), device=self.tbin.device)
        for tr, isis in enumerate(rISI):
            for n_enu, isi in enumerate(isis):
                if len(isi) > 0: # nonzero number of ISIs
                    intervals = isi.shape[1]
                    l_Lambda[tr::self.trials, n_enu] = torch.log(isi+1e-12).sum(-1)
                    quad_term[tr::self.trials, n_enu] = 0.5*((torch.log(isi+1e-12)/sigma_[:, n_enu:n_enu+1])**2).sum(-1) # -mu_[:, n_enu:n_enu+1]
                    norm_term[tr::self.trials, n_enu] = intervals*(torch.log(sigma_[0, n_enu]) + self.twopi_fact)
                else:
                    l_Lambda[tr::self.trials, n_enu] = 0
                    quad_term[tr::self.trials, n_enu] = 0
                    norm_term[tr::self.trials, n_enu] = 0
    
        nll = -l_rates + norm_term + l_Lambda + quad_term
        return nll.sum(1, keepdims=True)
    
    
    def objective(self, F_mu, F_var, XZ, b, neuron, samples=10, mode='MC'):
        return super().objective(F_mu, F_var, XZ, b, neuron, 
                                 torch.exp(-self.sigma**2/2.), samples=samples, mode=mode)
    
    
    def ISI_dist(self, n):
        sigma = self.sigma[n].data.cpu().numpy()
        return point_process.ISI_logNormal(sigma, scale=np.exp(sigma**2/2.))

    
    
class invGaussian(renewal_model):
    """
    Inverse Gaussian ISI distribution
    Ignores the end points of the spike train in each batch
    """
    def __init__(self, tbin, neurons, inv_link, mu, tensor_type=torch.float, allow_duplicate=True, 
                 dequantize=True):
        """
        :param np.array mu: :math:`$mu$` parameter which is > 0
        """
        super().__init__(tbin, neurons, inv_link, tensor_type, allow_duplicate, dequantize)
        self.register_parameter('mu', Parameter(torch.tensor(mu, dtype=self.tensor_type)))
        #self.register_parameter('lambd', Parameter(torch.tensor(lambd, dtype=self.tensor_type)))
        self.register_buffer('twopi_fact', 0.5*torch.tensor(2*np.pi, dtype=self.tensor_type).log())
       
    
    def set_params(self, mu=None):
        if mu is not None:
            self.mu.data = torch.tensor(mu, device=self.mu.device)
          
        
    def constrain(self):
        self.mu.data = torch.clamp(self.mu.data, min=1e-5)
         
            
    def set_Y(self, spikes, batch_size, filter_len=1):
        super().set_Y(spikes, batch_size, filter_len=filter_len)
        
        
    def nll(self, l_rates, rISI, neuron):        
        """
        :param torch.tensor F_mu: F_mu product with shape (samples, neurons, timesteps)
        :param torch.tensor F_var: variance of the F_mu values, same shape
        :param int b: batch number
        :param np.array neuron: neuron indices that are used
        :param int samples: number of MC samples for likelihood evaluation
        """
        mu_ = self.mu.expand(1, self.F_dims)[:, neuron]
        samples_ = l_rates.shape[0]
   
        l_Lambda = torch.empty((samples_, len(neuron)), device=self.tbin.device)
        quad_term = torch.empty((samples_, len(neuron)), device=self.tbin.device)
        norm_term = torch.empty((samples_, len(neuron)), device=self.tbin.device)
        for tr, isis in enumerate(rISI):
            for n_enu, isi in enumerate(isis):
                if len(isi) > 0: # nonzero number of ISIs
                    intervals = isi.shape[1]
                    l_Lambda[tr::self.trials, n_enu] = torch.log(isi+1e-12).sum(-1)
                    quad_term[tr::self.trials, n_enu] = 0.5*(((isi - mu_[:, n_enu:n_enu+1])/ \
                                                               mu_[:, n_enu:n_enu+1])**2 / isi).sum(-1) # (lambd_[:, n_enu:n_enu+1])
                    norm_term[tr::self.trials, n_enu] = intervals*(self.twopi_fact) # - 0.5*torch.log(lambd_[0, n_enu])
                else:
                    l_Lambda[tr::self.trials, n_enu] = 0
                    quad_term[tr::self.trials, n_enu] = 0
                    norm_term[tr::self.trials, n_enu] = 0
    
        nll = -l_rates + norm_term + 1.5*l_Lambda + quad_term
        return nll.sum(1, keepdims=True)
    
    
    def objective(self, F_mu, F_var, XZ, b, neuron, samples=10, mode='MC'):
        return super().objective(F_mu, F_var, XZ, b, neuron, 
                                 1./self.mu, samples=samples, mode=mode)
    
    
    def ISI_dist(self, n):
        """
        Note the scale parameter here is the inverse of the scale parameter in nll(), as the scale 
        parameter here is :math:`\tau/s` while in nll() is refers to :math:`d\tau = s*r(t) \, \mathrm{d}t`
        """
        # self.lambd[n].data.cpu().numpy()
        mu = self.mu[n].data.cpu().numpy()
        return point_process.ISI_invGauss(mu, scale=mu)



# noise distribution
class Gaussian(base._likelihood):
    """
    Gaussian noise likelihood.
    Analogous to Factor Analysis.
    """
    def __init__(self, neurons, inv_link, log_var, tensor_type=torch.float):
        """
        :param np.array log_var: log observation noise of shape (neuron,) or (1,) if parameters tied
        """
        super().__init__(1., neurons, inv_link, tensor_type) # dummy tbin
        self.register_parameter('log_var', Parameter(torch.tensor(log_var, dtype=self.tensor_type)))
          
            
    def set_params(self, log_var=None):
        if log_var is not None:
            self.log_var.data = torch.tensor(log_var, device=self.tbin.device)
          
        
    def sample_helper(self, h, b, neuron, samples):
        """
        NLL helper function for MC sample evaluation.
        """ 
        rates = self.f(h) # watch out for underflow or overflow here
        spikes = self.spikes[b][:, neuron, self.filter_len-1:].to(self.tbin.device)
        return rates, spikes
    
    
    def nll(self, rates, spikes, noise_var):
        """
        Gaussian likelihood for activity train
        samples introduces a sample dimension from the left
        F_mu has shape (samples, neuron, timesteps)
        if F_var = 0, we don't expand by samples in the sample dimension
        """
        nll = .5*(torch.log(noise_var) + ((spikes - rates)**2)/noise_var) + \
              .5*torch.log(torch.tensor(2*np.pi))
        return nll.sum(1)
    
    
    def objective(self, F_mu, F_var, XZ, b, neuron, samples=10, mode='MC'):
        """
        Computes the terms for variational expectation :math:`\mathbb{E}_{q(f)q(z)}[]`, which 
        can be used to compute different likelihood objectives.
        The returned tensor will have sample dimension as MC over :math:`q(z)`, depending 
        on the evaluation mode will be MC or GH or exact over the likelihood samples. This 
        is all combined in to the same dimension to be summed over. The weights :math:`w_s` 
        are the quadrature weights or equal weights for MC, with appropriate normalization.
        
        :param int samples: number of MC samples or GH points (exact will ignore and give 1)
        
        :returns: negative likelihood term of shape (samples, timesteps), sample weights (samples, 1
        :rtype: tuple of torch.tensors
        """
        if disper is None:
            if self.log_var.shape[0] == 1:
                log_var = self.log_var.expand(1, len(neuron))[..., None]
            else:
                log_var = self.log_var[None, neuron, None]
        else:
            dh = self.mc_gen(disper, disper_var, samples, neuron)
            log_var = disp_f(dh)
            
        if self.inv_link == 'identity': # exact
            spikes = self.spikes[b][:, neuron, self.filter_len-1:].to(self.tbin.device)
            if isinstance(F_var, numbers.Number):
                F_var = 0
            else:
                F_var = F_var[:, neuron, :]
            noise_var = (torch.exp(log_var) + F_var)
            
            nll = .5*(torch.log(noise_var) + ((spikes - F_mu)**2)/noise_var + F_var/noise_var) + \
                  .5*torch.log(torch.tensor(2*np.pi))
            ws = torch.tensor(1/F_mu.shape[0])
            return nll.sum(1), ws
        #elif self.inv_link == 'exp' # exact
        
        if mode == 'MC':
            h = self.mc_gen(F_mu, F_var, samples, neuron)
            rates, spikes = self.sample_helper(h, b, neuron, samples)
            ws = torch.tensor(1./rates.shape[0])
        elif mode == 'GH':
            h, ws = self.gh_gen(F_mu, F_var, samples, neuron)
            rates, spikes = self.sample_helper(h, b, neuron, samples)
            ws = ws[:, None]
        else:
            raise NotImplementedError
        
        return self.nll(rates, spikes, torch.exp(log_var)), ws

    
    def sample(self, rate, neuron=None, XZ=None):
        """
        Sample activity trains [trial, neuron, timestep]
        """
        neuron = self._validate_neuron(neuron)
        rate_ = rate[:, neuron, :]
        
        if self.log_var.shape[0] == 1:
            log_var = self.log_var.expand(1, len(neuron)).data[..., None].cpu().numpy()
        else:
            log_var = self.log_var.data[None, neuron, None].cpu().numpy()
        act = rate_ + np.exp(log_var/2.)*np.random.randn((rate.shape[0], len(neuron), rate.shape[-1])), rate_
        return act



class MultivariateGaussian(base._likelihood):
    r"""
    Account for noise correlations as in [1]. The covariance over neuron dimension is introduced.
    
    [1] `Learning a latent manifold of odor representations from neural responses in piriform cortex`,
        Anqi Wu, Stan L. Pashkovski, Sandeep Robert Datta, Jonathan W. Pillow, 2018
    """
    def __init__(self, neurons, inv_link, log_var, tensor_type=torch.float):
        """
        log_var can be shared or independent for neurons depending on the shape
        """
        super().__init__(neurons, inv_link, tensor_type)
        self.register_parameter('log_var', Parameter(torch.tensor(log_var, dtype=self.tensor_type)))
          
            
    def set_params(self, log_var=None, jitter=1e-6):
        if log_var is not None:
            self.log_var.data = torch.tensor(log_var, device=self.tbin.device)
        
        
    def objective(self, F_mu, F_var, XZ, b, neuron, samples, mode='MC'):        
        """
        Gaussian likelihood for activity train
        samples introduces a sample dimension from the left
        F_mu has shape (samples, neuron, timesteps)
        if F_var = 0, we don't expand by samples in the sample dimension
        """
        spikes = self.spikes[b][None, neuron, self.filter_len-1:].to(self.tbin.device) # activity
        batch_size = F_mu.shape[-1]
        
        if self.inv_link == 'identity': # exact
            noise_var = (self.L @ self.L.t())[None, neuron, None] + F_var[:, neuron, :]
            
            nll = .5*(torch.log(noise_var) + ((spikes - F_mu)**2)/noise_var + F_var/noise_var).sum(-1) + \
                  .5*torch.log(torch.tensor(2*np.pi))*batch_size
        else:
            if F_var != 0: # MC samples
                h = dist.Rn_Normal(F_mu, F_var)((samples,)).view(-1, *F_mu.shape[1:])[:, neuron, :]
                F_var = F_var.repeat(samples, 1, 1)
            else:
                h = F_mu[:, neuron, :]

            rates = self.f(h)
            noise_var = (torch.exp(self.log_var)[None, neuron, None] + F_var[:, neuron, :])
            
            nll = .5*(torch.log(noise_var) + ((spikes - rates)**2)/noise_var).sum(-1) + \
                  .5*torch.log(torch.tensor(2*np.pi))*batch_size

        ws = torch.tensor(1./nll.shape[0])
        return nll.sum(1), ws

    
    def sample(self, rate, neuron=None, XZ=None):
        """
        Sample activity trains [trial, neuron, timestep]
        """
        neuron = self._validate_neuron(neuron)
        act = rate + torch.exp(self.log_var).data.sqrt().cpu().numpy()*np.random.randn((rate.shape[0], len(neuron), rate.shape[-1]))
        return act
        
