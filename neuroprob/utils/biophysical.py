import torch
import torch.nn as nn

import numpy as np

from tqdm.autonotebook import tqdm



# Continuous models
class Hodgkin_Huxley():
    r"""
    Hodgkin-Huxley model via Euler integration
    """
    def __init__(self, G_na=120, G_k=36, G_l=0.3, E_na=50, E_k=-77, E_l=-54.4):
        r"""
        units are in mV, microS, nF, mA, ms
        """
        self.G_na = G_na
        self.G_k = G_k
        self.G_l = G_l
        self.E_na = E_na
        self.E_k = E_k
        self.E_l = E_l
    
    def euler_int(self, T, runs, I_ext, ic, dt=0.001, prin=1000):
        r"""
        Integrate the HH dynamics, the state array (v, m, h, n) is represented by 4 floating-point values 
        
        :param int T: timesteps to run the simulation for
        :param int runs: number of trials to run (I_ext and i.c. can differ per run)
        :param np.array I_ext: external input current, with shape (runs, timesteps)
        :param np.array ic: neuron initial conditions, with shape (runs, 4)
        :returns: neuron state over the simulation
        :rtype: np.array
        """
        alpha_m = lambda V: (2.5-0.1*(V+65)) / (np.exp(2.5-0.1*(V+65)) -1)
        beta_m = lambda V: 4.0 * np.exp(-(V+65)/18)
        alpha_h = lambda V: 0.07 * np.exp(-(V+65)/20)
        beta_h = lambda V: 1.0 / (np.exp(3.0-0.1*(V+65)) + 1)
        alpha_n = lambda V: (0.1-0.01*(V+65)) / (np.exp(1-0.1*(V+65)) - 1)
        beta_n = lambda V: 0.125 * np.exp(-(V+65)/80)

        state = np.zeros((runs, T, 4)) # vector v, m, h, n
        for k in range(runs):
            state[k, 0, :] = ic[k, :]#[-6.49997224e+01, 5.29342176e-02, 5.96111046e-01, 3.17681168e-01]

        ds = np.zeros((runs, 4))
        iterator = tqdm(range(T-1))
        for t in iterator:
            ds[:, 0] = -(G_l*(state[:, t, 0] - E_l) + \
                   G_k*np.power(state[:, t, 3], 4)*(state[:, t, 0] - E_k) + \
                   G_na*np.power(state[:, t, 1], 3)*state[:, t, 2]*(state[:, t, 0] - E_na)) + I_ext[:, t]
            ds[:, 1] = alpha_m(state[:, t, 0]) * (1 - state[:, t, 1]) - beta_m(state[:, t, 0]) * state[:, t, 1]
            ds[:, 2] = alpha_h(state[:, t, 0]) * (1 - state[:, t, 2]) - beta_h(state[:, t, 0]) * state[:, t, 2]
            ds[:, 3] = alpha_n(state[:, t, 0]) * (1 - state[:, t, 3]) - beta_n(state[:, t, 0]) * state[:, t, 3]

            state[:, t+1] = state[:, t] + ds * dt

        return state
    
    
    
class FitzHugh_Nagumo():
    r"""
    A 2D reduction of the Hodgkin-Huxley model to the phase plane.
    """
    def __init__(self, b_0, b_1, tau_u, tau_w):
        r"""
        units are in mV, microS, nF, mA, ms
        """
        self.b_0 = b_0
        self.b_1 = b_1
        self.tau_u = tau_u
        self.tau_w = tau_w
        
    def euler_int(self, T, runs, I_ext, ic, dt=0.001, prin=1000):
        r"""
        Integrate the HH dynamics, the state array (v, m, h, n) is represented by 4 floating-point values 
        
        :param int T: timesteps to run the simulation for
        :param int runs: number of trials to run (I_ext and i.c. can differ per run)
        :param np.array I_ext: external input current, with shape (runs, timesteps)
        :param np.array ic: neuron initial conditions, with shape (runs, 4)
        :returns: neuron state over the simulation
        :rtype: np.array
        """
        state = np.zeros((runs, T, 2)) # vector u, w
        for k in range(runs):
            state[k, 0, :] = ic[k, :]#[-6.49997224e+01, 5.29342176e-02, 5.96111046e-01, 3.17681168e-01]

        ds = np.zeros((runs, 2))
        iterator = tqdm(range(T-1))
        for t in iterator:
            ds[:, 0] = 1/self.tau_u * (state[:, t, 0] - state[:, t, 0]**3/3. - state[:, t, 1] + I_ext)
            ds[:, 1] = 1/self.tau_w * (self.b_0 + self.b_1*state[:, t, 0] - state[:, t, 1])

            state[:, t+1] = state[:, t] + ds * dt

        return state
    
    
    
class Morris_Lecar():
    r"""
    A 2D reduction of the Hodgkin-Huxley model to the phase plane.
    """
    def __init__(self, G_na=120, G_k=36, G_l=0.3, E_na=50, E_k=-77, E_l=-54.4):
        r"""
        units are in mV, microS, nF, mA, ms
        """
        self.G_na = G_na
        self.G_k = G_k
        self.G_l = G_l
        self.E_na = E_na
        self.E_k = E_k
        self.E_l = E_l

    def euler_int(self, T, runs, I_ext, ic, dt=0.001, prin=1000):
        r"""
        Integrate the HH dynamics, the state array (v, m, h, n) is represented by 4 floating-point values 
        
        :param int T: timesteps to run the simulation for
        :param int runs: number of trials to run (I_ext and i.c. can differ per run)
        :param np.array I_ext: external input current, with shape (runs, timesteps)
        :param np.array ic: neuron initial conditions, with shape (runs, 4)
        :returns: neuron state over the simulation
        :rtype: np.array
        """
        alpha_m = lambda V: (2.5-0.1*(V+65)) / (np.exp(2.5-0.1*(V+65)) -1)
        beta_m = lambda V: 4.0 * np.exp(-(V+65)/18)
        alpha_h = lambda V: 0.07 * np.exp(-(V+65)/20)
        beta_h = lambda V: 1.0 / (np.exp(3.0-0.1*(V+65)) + 1)
        alpha_n = lambda V: (0.1-0.01*(V+65)) / (np.exp(1-0.1*(V+65)) - 1)
        beta_n = lambda V: 0.125 * np.exp(-(V+65)/80)

        state = np.zeros((runs, T, 4)) # vector v, m, h, n
        for k in range(runs):
            state[k, 0, :] = ic[k, :]#[-6.49997224e+01, 5.29342176e-02, 5.96111046e-01, 3.17681168e-01]

        ds = np.zeros((runs, 4))
        iterator = tqdm(range(T-1))
        for t in iterator:
            ds[:, 0] = -(G_l*(state[:, t, 0] - E_l) + \
                   G_k*np.power(state[:, t, 3], 4)*(state[:, t, 0] - E_k) + \
                   G_na*np.power(state[:, t, 1], 3)*state[:, t, 2]*(state[:, t, 0] - E_na)) + I_ext[:, t]
            ds[:, 1] = alpha_m(state[:, t, 0]) * (1 - state[:, t, 1]) - beta_m(state[:, t, 0]) * state[:, t, 1]
            ds[:, 2] = alpha_h(state[:, t, 0]) * (1 - state[:, t, 2]) - beta_h(state[:, t, 0]) * state[:, t, 2]
            ds[:, 3] = alpha_n(state[:, t, 0]) * (1 - state[:, t, 3]) - beta_n(state[:, t, 0]) * state[:, t, 3]

            state[:, t+1] = state[:, t] + ds * dt

        return state
    
    
def count_APs(V, lim=20.0):
    r"""
    Action potential counter
    """
    idx = (V > lim).astype(float)
    idf = np.diff(idx) == 1
    return idf.sum()



# Integrate-and-fire models    
class Izhikevich():
    r""" 
    Biophysically inspired Izhikevich model (2003/2004) [1], a nonlinear integrate-and-fire model.
    
    References:
    
    [1] 
    
    """
    def __init__(self, a, b, c, d):
        self.a = a
        self.b = b
        self.c = c
        self.d = d
    
    def euler_int(self, T, runs, I_ext, ic, dt=0.1, prin=1000):
        r"""
        Euler integration of the dynamics, with state array (v, u)
        """
        state = np.zeros((runs, T, 2)) # vector v, u
        spiketrain = np.zeros((runs, T))
        reset_state = np.empty((runs, 2))
        reset_state[:, 0].fill(self.c)
        
        for k in range(runs):
            state[k, 0, :] = ic[k, :]

        ds = np.zeros((runs, 2))
        iterator = tqdm(range(T-1))
        for t in iterator:
            ds[:, 0] = 0.04*state[:, t, 0]**2 + 5.*state[:, t, 0] + 140. - state[:, t, 1] + I_ext[:, t]
            ds[:, 1] = self.a*(self.b*state[:, t, 0] - state[:, t, 1])

            reset = (state[:, t, 0] >= 30.)
            if reset.sum() > 0:
                reset_state[:, 1] = (state[:, t, 1] + self.d)
                state[:, t+1] = reset[:, None]*reset_state + (1-reset)[:, None]*(state[:, t] + ds * dt)
                spiketrain[:, t+1] = reset
            else:
                state[:, t+1] = state[:, t] + ds * dt

        return state, spiketrain
    
    
    
class AdExIF():
    r"""
    Adaptive exponential integrate-and-fire model. [1]
    
    References:
    
    [1] `Neuronal Dynamics`, Wulfram Gerstner, Werner M. Kistler, Richard Naud and Liam Paninski.
    
    """
    def __init__(self, a, b, c, d):
        self.a = a
        self.b = b
        self.c = c
        self.d = d
        
    def euler_int(self, T, runs, I_ext, ic, dt=0.001, prin=1000):
        r"""
        Euler integration of the dynamics, with state array (v, u)
        """
        state = np.zeros((runs, T, 2)) # vector v, u
        spiketrain = np.zeros((runs, T))
        reset_state = np.empty((runs, 2))
        reset_state[:, 0].fill(self.c)
        
        for k in range(runs):
            state[k, 0, :] = ic[k, :]

        ds = np.zeros((runs, 2))
        iterator = tqdm(range(T-1))
        for t in iterator:
            ds[:, 0] = 0.04*state[:, t, 0]**2 + 5.*state[:, t, 0] + 140. - state[:, t, 1] + I_ext[:, t]
            ds[:, 1] = self.a*(self.b*state[:, t, 0] - state[:, t, 1])

            reset = (state[:, t, 0] >= 30.)
            if reset.sum() > 0:
                reset_state[:, 1] = (state[:, t, 1] + self.d)
                state[:, t+1] = reset[:, None]*reset_state + (1-reset)[:, None]*(state[:, t] + ds * dt)
                spiketrain[:, t+1] = reset
            else:
                state[:, t+1] = state[:, t] + ds * dt

        return state, spiketrain
    
    
    
def neuron_model(dynamics, model_type):
    r"""
    Neuronal dynamics library of parameter values.
    Izhikevich parameters from [1].
    
    References:
    
    [1] `Capturing the Dynamical Repertoire of Single Neurons with Generalized Linear Models`,
        Alison I. Weber & Jonathan W. Pillow
        
    """
    
    if model_type == 'Izhikevich': # dt in ms
        if dynamics == 'tonic_spiking':
            model = Izhikevich(0.02, 0.2, -65, 6)
            I = 14
            dt = 0.1
        elif dynamics == 'phasic_spiking':
            model = Izhikevich(0.02, 0.2, -65, 6)
            I = 0.5
            dt = 0.1
        elif dynamics == 'tonic_bursting':
            model = Izhikevich(0.02, 0.2, -50, 2)
            I = 10
            dt = 0.1
        elif dynamics == 'phasic_bursting':
            model = Izhikevich(0.02, 0.25, -55, 0.05)
            I = 0.6
            dt = 0.1
        elif dynamics == 'mixed':
            model = Izhikevich(0.02, 0.2, -55, 4)
            I = 10
            dt = 0.1
        elif dynamics == 'frequency_adaptation':
            model = Izhikevich(0.01, 0.2, -65, 5)
            I = 20
            dt = 0.1
        elif dynamics == 'type_I':
            model = Izhikevich(0.02, -0.1, -55, 6)
            I = 25
            dt = 1.
        elif dynamics == 'type_II':
            model = Izhikevich(0.2, 0.26, -65, 0)
            I = 0.5
            dt = 1.
        elif dynamics == 'spike_latency':
            model = Izhikevich(0.02, 0.2, -65, 6)
            I = 3.49
            dt = 0.1
        elif dynamics == 'resonator':
            model = Izhikevich(0.1, 0.26, -60, -1)
            I = 0.3
            dt = 0.5
        elif dynamics == 'integrator':
            model = Izhikevich(0.02, -0.1, -66, 6)
            I = 27.4
            dt = 0.5
        elif dynamics == 'rebound_spike':
            model = Izhikevich(0.03, 0.25, -60, 4)
            I = -5.
            dt = 0.1
        elif dynamics == 'rebound_burst':
            model = Izhikevich(0.03, 0.25, -52, 0)
            I = -5.
            dt = 0.1
        elif dynamics == 'threshold_variability':
            model = Izhikevich(0.03, 0.25, -60, 4)
            I = 2.3
            dt = 1.
        elif dynamics == 'bistability_I':
            model = Izhikevich(1., 1.5, -60, 0)
            I = 30.
            dt = 0.05
        elif dynamics == 'bistability_II':
            model = Izhikevich(1., 1.5, -60, 0)
            I = 40.
            dt = 0.05
        else:
            raise NotImplementedError
        
        return model, I, dt
    
    else:
        raise NotImplementedError