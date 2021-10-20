import numpy as np
import scipy.stats as scstats
import scipy.special as scsps
import numbers

import torch
from torch.nn.parameter import Parameter
from torch.distributions import Bernoulli

from tqdm.autonotebook import tqdm




# ISI distribution classes
class ISI_base():
    """
    Base class for ISI distributions in statistical goodness-of-fit tests.
    Takes in time-rescaled ISIs.
    """
    def __init__(self, loc=0., scale=1.):
        self.loc = loc
        self.scale = scale
        
    def prob_dens(self, tau):
        return self.rv.pdf((tau-self.loc)/self.scale)/self.scale
        
    def cum_prob(self, tau, loc=0., scale=1.):
        return self.rv.cdf((tau-self.loc)/self.scale)
    
    def sample(self, shape):
        return self.rv.rvs(size=shape+self.event_shape)*self.scale + self.loc
    
    def intensity(self, tau, prev_tau):
        """
        Evaluates :math:'\lambda(\tau|\tau')'
        """
        return prob_dens(tau)/(1. - cum_prob(tau-prev_tau))



class ISI_gamma(ISI_base):
    r"""
    Gamma distribution
    
    ..math::
    
    """
    def __init__(self, shape, loc=0., scale=1.):
        super().__init__(loc, scale)
        self.rv = scstats.gamma(shape)
        if isinstance(shape, numbers.Number):
            self.event_shape = ()
        else:
            self.event_shape = shape.shape
            

            
class ISI_invgamma(ISI_base):
    r"""
    Inverse Gamma distribution
    
    ..math::
    
    """
    def __init__(self, shape, loc=0., scale=1.):
        super().__init__(loc, scale)
        self.rv = scstats.invgamma(shape)
        if isinstance(shape, numbers.Number):
            self.event_shape = ()
        else:
            self.event_shape = shape.shape
    

    
class ISI_invGauss(ISI_base):
    r"""
    Inverse Gaussian distribution
    Gives parameterization on wikipedia with lambda = 1
    
    ..math::
    
    """
    def __init__(self, mu, loc=0., scale=1.):
        super().__init__(loc, scale)
        self.rv = scstats.invgauss(mu)
        if isinstance(mu, numbers.Number):
            self.event_shape = ()
        else:
            self.event_shape = mu.shape


            
class ISI_logNormal(ISI_base):
    r"""
    Log normal renewal distribution
    Gives parameterization on wikipedia, mu = 0
    
    ..math::
    
    """
    def __init__(self, sigma, loc=0., scale=1.):
        super().__init__(loc, scale)
        self.rv = scstats.lognorm(sigma)
        if isinstance(sigma, numbers.Number):
            self.event_shape = ()
        else:
            self.event_shape = sigma.shape
        


def gen_IRP(interval_dist, rate, tbin, samples=100):
    """
    Sample event times from an Inhomogenous Renewal Process with a given rate function
    samples is an algorithm parameter, should be around the expect number of spikes
    Assumes piecewise constant rate function
    
    Samples intervals from :math:`q(\Delta)`, parallelizes sampling
    
    :param np.array rate: (trials, neurons, timestep)
    :param ISI_dist interval_dist: renewal interval distribution :math:`q(\tau)`
    :returns: event times as integer indices of the rate array time dimension
    :rtype: list of spike time indices as indexed by rate time dimension
    """
    sim_samples = rate.shape[2]
    N = rate.shape[1] # neurons
    trials = rate.shape[0]
    T = np.transpose(np.cumsum(rate, axis=-1), (2, 0, 1))*tbin # actual time to rescaled, (time, trials, neuron)
    
    psT = 0
    sT_cont = []
    while True:

        sT = psT + np.cumsum(interval_dist.sample((samples, trials,)), axis=0)
        sT_cont.append(sT)
        
        if not (T[-1, ...] >= sT[-1, ...]).any(): # all False
            break
            
        psT = np.tile(sT[-1:, ...], (samples, 1, 1))
        
    sT_cont = np.stack(sT_cont, axis=0).reshape(-1, trials, N)
    samples_tot = sT_cont.shape[0]
    st = []
    
    iterator = tqdm(range(samples_tot), leave=False)
    for ss in iterator: # AR assignment
        comp = np.tile(sT_cont[ss:ss+1, ...], (sim_samples, 1, 1))
        st.append(np.argmax((comp < T), axis=0)) # convert to rescaled time indices

    st = np.array(st) # (samples_tot, trials, neurons)
    st_new = []
    for st_ in st.reshape(samples_tot, -1).T:
        if not st_.any(): # all zero
            st_new.append(np.array([]).astype(int))
        else: # first zeros in this case counts as a spike indices
            for k in range(samples_tot):
                if st_[-1-k] != 0:
                    break
            if k == 0:
                st_new.append(st_)
            else:
                st_new.append(st_[:-k])            
            
    return st_new # list of len trials x neurons



def gen_IBP(intensity):
    """
    Sample of the Inhomogenous Bernoulli Process
    
    :param numpy.array intensity: intensity of the Bernoulli process at array index
    :returns: sample of binary variables with same shape as intensity
    :rtype: numpy.array
    """
    b = Bernoulli(torch.tensor(intensity))
    return b.sample().numpy()



def gen_CMP(mu, nu, max_rejections=1000):
    """
    Use rejection sampling to sample from the COM-Poisson count distribution. [1]
    
    References:
    
    [1] `Bayesian Inference, Model Selection and Likelihood Estimation using Fast Rejection 
         Sampling: The Conway-Maxwell-Poisson Distribution`, Alan Benson, Nial Friel (2021)
    
    :param numpy.array rate: input rate of shape (..., time)
    :param float tbin: time bin size
    :param float eps: order of magnitude of P(N>1)/P(N<2) per dilated Bernoulli bin
    :param int max_count: maximum number of spike counts per bin possible
    :returns: inhomogeneous Poisson process sample
    :rtype: numpy.array
    """
    trials = mu.shape[0]
    neurons = mu.shape[1]
    Y = np.empty(mu.shape)
    
    for tr in range(trials):
        for n in range(neurons):
            mu_, nu_ = mu[tr, n, :], nu[tr, n, :]

            # Poisson
            k = 0
            left_bins = np.where(nu_ >= 1)[0]
            while len(left_bins) > 0:
                mu__, nu__ = mu_[left_bins], nu_[left_bins]
                y_dash = torch.poisson(torch.tensor(mu__)).numpy()
                _mu_ = np.floor(mu__)
                alpha = (mu__**(y_dash-_mu_) / scsps.factorial(y_dash) * scsps.factorial(_mu_))**(nu__-1)
                
                u = np.random.rand(*mu__.shape)
                selected = (u <= alpha)
                Y[tr, n, left_bins[selected]] = y_dash[selected]
                left_bins = left_bins[~selected]
                if k >= max_rejections:
                    raise ValueError('Maximum rejection steps exceeded')
                else:
                    k += 1
                
                
            # geometric
            k = 0
            left_bins = np.where(nu_ < 1)[0]
            while len(left_bins) > 0:
                mu__, nu__ = mu_[left_bins], nu_[left_bins]
                p = 2*nu__ / (2*mu__*nu__ + 1 + nu__)
                u_0 = np.random.rand(*p.shape)
                
                y_dash = np.floor(np.log(u_0) / np.log(1-p))
                a = np.floor(mu__ / (1-p)**(1/nu__))
                alpha = (1-p)**(a - y_dash) * (mu__**(y_dash-a) / scsps.factorial(y_dash) * scsps.factorial(a))**nu__
                
                u = np.random.rand(*mu__.shape)
                selected = (u <= alpha)
                Y[tr, n, left_bins[selected]] = y_dash[selected]
                left_bins = left_bins[~selected]
                if k >= max_rejections:
                    raise ValueError('Maximum rejection steps exceeded')
                else:
                    k += 1
                    
    return Y
    
    """
    if rate.max() == 0:
        return np.zeros_like(rate)
        
    dt_ = np.sqrt(eps)/rate.max()
    dilation = max(int(np.ceil(tbin/dt_)), 1) # number of counts to allow per original bin
    if dilation > max_count:
        raise ValueError('Maximum count ({}, requested {}) exceeded for Poisson process sampling'.format(max_count, dilation))
    tbin_ = tbin / dilation
    rate_ = np.repeat(rate, dilation, axis=-1) # repeat to allow IBP to sum to counts > 1
    return gen_IBP(rate_*tbin_).reshape(*rate.shape[:-1], -1, dilation).sum(-1)
    """