import numpy as np
import scipy
import torch
import random

from tqdm.autonotebook import tqdm



def random_W(neuron_data, mode, rv_list, strict_balance=False):
    """
    :param list rv_list: list of scipy.stats random variable distribution for matrix
    
    References:
    
    [1] `Eigenvalue Spectra of Random Matrices for Neural Networks`, 
        Kanaka Rajan and L. F. Abbott (2006)
        
    """
    neurons = int(torch.sum(neuron_data[:, 0]).item())
    groups = neuron_data.shape[0]
    w_ij = np.zeros((neurons, neurons))
    
    if mode == 'full' or mode == 'symmetric': # Girko's circle law
        R2 = 0
        cur_n = 0
        for g in range(groups):
            rv = rv_list[g]
            mean, var, skew, kurt = rv.stats(a, moments='mvsk')
            N = int(neuron_data[g, 0])
            w_ij[:, cur_n:cur_n+N] = rv.rvs(neurons, N)
            
            R2 += N*var
            cur_n += N
            
        R = np.sqrt(R2)
            
        if mode == 'symmetric': # Wigner semicircle law
            w_ij = np.tril(w_ij, k=1) + np.tril(w_ij).T
    
    elif mode == 'sparse': # subtract row means
        assert groups == 2
        N_E = int(neuron_data[0, 0])
        N_I = int(neuron_data[1, 0])
        
        # Dale modification
        p, w_0, g = rv_list

        R = w_0*np.sqrt(p*(1-p)*(1+g**2)/2)
        lambda_0 = (1-g)*np.sqrt(neurons)*p*w_0/2

        r_E = (np.random.rand(neurons, N_E) < p)
        r_I = (np.random.rand(neurons, N_I) < p)
        w_ij[:, :N_E][r_E] = w_0/np.sqrt(neurons)
        w_ij[:, N_E:][r_I] = -g*w_0/np.sqrt(neurons)

    if strict_balance: # individual input balance, subtract row
        w_ij -= w_ij.mean(1, keepdims=True)
    
    return w_ij, R




def SOC_W(ini_W, N_E, B=0.2, C=1.5, inhib_syn_ratio=0.4, gamma=3., 
          loss_margin=0, stop_iters=10, max_iters=1000, eta=1e-2):
    """
    Optimization procedure taking in initial chaotic sparse matrix.
    """
    N = ini_W.shape[0]
    W = ini_W
    
    I = np.eye(N)
    N_I = N - N_E
    
    # initial inhibitory connections
    loc_inhib = np.where(w_ij < 0.)
    tot_syn = N_I*N
    exist_inhib = (loc_inhib[1]-N_E) + N_I*loc_inhib[0]
    
    tot_inhibs = int(tot_syn*inhib_syn_ratio)
    new_inhibs = int(tot_inhibs-len(loc_inhib[0]))
    if new_inhibs > 0:
        new_inhib = random.sample(list(np.delete(np.arange(tot_syn), exist_inhib)), new_inhibs)
        exist_inhib = np.concatenate((exist_inhib, new_inhib))
        
    ee = exist_inhib[tot_inhibs:]
    W[ee//N_I, ee%N_I + N_E] = 0 # set excess inhibitory connections to 0
    
    exist_inhib = exist_inhib[:tot_inhibs]
    minloss = np.inf
    cnt = 0
    iterator = tqdm(range(max_iters))          
    for k in iterator:
        # get the approximation of step 1
        alpha = np.max(np.real(np.linalg.eigvals(W)))
        alpha_tilde = min(C*alpha, alpha+B)
        
        s = alpha_tilde
        Q = scipy.linalg.solve_continuous_lyapunov((W - s*I).T, -2*I)
        P = scipy.linalg.solve_continuous_lyapunov((W - s*I), -2*I)

        mp = Q @ P
        grad_W = mp / np.trace(mp)
        
        W[exist_inhib//N_I, exist_inhib%N_I + N_E] -= eta*grad_W[exist_inhib//N_I, exist_inhib%N_I + N_E]
        remove_inhib = np.where(W[exist_inhib//N_I, exist_inhib%N_I + N_E] >= 0)[0]
        nonexist_inhib = np.delete(np.arange(tot_inhibs), exist_inhib)
        
        scale = -gamma * W[:N_E, :N_E].mean()/W[:N_E, N_E:].mean()
        W[:N_E, N_E:] *= scale
        scale = -gamma * W[N_E:, :N_E].mean()/W[N_E:, N_E:].mean()
        W[N_E:, N_E:] *= scale
        
        # prune zero inhibitory connections and reassign
        exist_inhib = np.delete(exist_inhib, remove_inhib)
        new_inhib = random.sample(list(nonexist_inhib), len(remove_inhib))
        exist_inhib = np.concatenate((exist_inhib, new_inhib))
        
        # premature stopping if loss doesn't improve according to margin
        if alpha_tilde <= minloss + loss_margin:
            cnt = 0
        else:
            cnt += 1

        iterator.set_postfix(loss=alpha_tilde)
        if alpha_tilde < minloss:
            minloss = alpha_tilde
        if cnt > stop_iters:
            break
    
    return W