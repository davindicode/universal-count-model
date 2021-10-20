import numpy as np
import torch

from .parametrics import GLM, custom_wrapper



# parameterizations
def w_to_gaussian(w):
    """
    Get Gaussian and orthogonal theta parameterization from the GLM parameters.
    
    :param np.array w: input GLM parameters of shape (neurons, dims), dims labelling (w_1, w_x,
                       w_y, w_xx, w_yy, w_xy, w_cos, w_sin)
    """
    neurons = mu.shape[0]
    w_spat = w[:, 0:6]
    prec = np.empty((neurons, 3)) # xx, yy and xy/yx
    mu = np.empty((neurons, 2)) # x and y
    prec[:, 0] = -2*w_spat[:, 3]
    prec[:, 1] = -2*w_spat[:, 4]
    prec[:, 2] = -w_spat[:, 5]
    prec_mat = []
    for n in range(neurons):
        prec_mat.append([[prec[n, 0], prec[n, 2]], [prec[n, 2], prec[n, 1]]])
    prec_mat = np.array(prec_mat)
    denom = prec[:, 0]*prec[:, 1] - prec[:, 2]**2
    mu[:, 0] = (w_spat[:, 1]*prec[:, 1] - w_spat[:, 2]*prec[:, 2])/denom
    mu[:, 1] = (w_spat[:, 2]*prec[:, 0] - w_spat[:, 1]*prec[:, 2])/denom
    rate_0 = np.exp(w_spat[:, 0] + 0.5*(mu * np.einsum('nij,nj->ni', prec_mat, mu)).sum(1))

    w_theta = w[:, 6:]
    theta_0 = np.angle(w_theta[:, 0] + w_theta[:, 1]*1j)
    beta = np.sqrt(w_theta[:, 0]**2 + w_theta[:, 1]**2)

    return mu, prec, rate_0, np.concatenate((beta[:, None], theta_0[:, None]), axis=1)


def gaussian_to_w(mu, prec, rate_0, theta_p):
    """
    Get GLM parameters from Gaussian and orthogonal theta parameterization
    
    :param np.array mu: mean of the Gaussian field of shape (neurons, 2)
    :param np.array prec: precision matrix elements xx, yy, and xy of shape (neurons, 3)
    :param np.array rate_0: rate amplitude of shape (neurons)
    :param np.array theta_p: theta modulation parameters beta and theta_0 of shape (neurons, 2)
    """
    neurons = mu.shape[0]
    prec_mat = []
    for n in range(neurons):
        prec_mat.append([[prec[n, 0], prec[n, 2]], [prec[n, 2], prec[n, 1]]])
    prec_mat = np.array(prec_mat)
    w = np.empty((neurons, 8))
    w[:, 0] = np.log(rate_0) - 0.5*(mu * np.einsum('nij,nj->ni', prec_mat, mu)).sum(1)
    w[:, 1] = mu[:, 0]*prec[:, 0] + mu[:, 1]*prec[:, 2]
    w[:, 2] = mu[:, 1]*prec[:, 1] + mu[:, 0]*prec[:, 2]
    w[:, 3] = -0.5*prec[:, 0]
    w[:, 4] = -0.5*prec[:, 1]
    w[:, 5] = -prec[:, 2]
    w[:, 6] = theta_p[:, 0]*np.cos(theta_p[:, 1])
    w[:, 7] = theta_p[:, 0]*np.sin(theta_p[:, 1])
    return w


def w_to_vonmises(w):
    """
    :param np.array w: parameters of the GLM of shape (neurons, 3)
    """
    rate_0 = w[:, 0]
    theta_0 = np.angle(w[:, 1] + w[:, 2]*1j)
    kappa = np.sqrt(w[:, 1]**2 + w[:, 2]**2)
    return rate_0, kappa, theta_0


def vonmises_to_w(rate_0, kappa, theta_0):
    """
    :param np.array rate_0: rate amplitude of shape (neurons)
    :param np.array theta_p: von Mises parameters kappa and theta_0 of shape (neurons, 2) 
    """
    neurons = rate_0.shape[0]
    w = np.empty((neurons, 3))
    w[:, 0] = np.log(rate_0)
    w[:, 1] = kappa*np.cos(theta_0)
    w[:, 2] = kappa*np.sin(theta_0)
    return w



# GLM
class vonMises_GLM(GLM):
    """
    Angular (head direction/theta) variable GLM
    """
    def __init__(self, neurons, inv_link='exp', tens_type=torch.float, active_dims=None):
        super().__init__(1, neurons, 3, inv_link, tens_type=tens_type, active_dims=active_dims)
        
    def set_params(self, tbin=None, w=None):
        if tbin is not None:
            super().set_params(tbin)
        if w is not None:
            self.w.data = torch.tensor(w, device=self.dummy.device, dtype=self.tensor_type)

    def compute_F(self, XZ):
        XZ = self._XZ(XZ)
        theta = XZ[..., 0]
        g = torch.stack((torch.ones_like(theta, device=theta.device), torch.cos(theta), torch.sin(theta)), dim=-1)
        return (g[:, None, :, :]*self.w[None, :, None, :]).sum(-1), 0



class Gauss_GLM(GLM):
    """
    Quadratic GLM for position
    """
    def __init__(self, neurons, inv_link='exp', tens_type=torch.float, active_dims=None):
        super().__init__(2, neurons, 6, inv_link, tens_type=tens_type, active_dims=active_dims)
        
    def set_params(self, w=None):
        if w is not None:
            self.w.data = torch.tensor(w, device=self.dummy.device, dtype=self.tensor_type)
      
    def compute_F(self, XZ):
        XZ = self._XZ(XZ)
        x = XZ[..., 0]
        y = XZ[..., 1]
        g = torch.stack((torch.ones_like(x, device=x.device), x, y, x**2, y**2, x*y), dim=-1)
        return (g[:, None, :, :]*self.w[None, :, None, :]).sum(-1), 0
    
    
    
# Joint AV tuning HDC model
class ATI_HDC(custom_wrapper):
    """
    Angular (head direction/theta) variable GLM, with anticipatory time intervals
    """
    def __init__(self, neurons, inv_link='exp', tens_type=torch.float, active_dims=None):
        super().__init__(2, neurons, inv_link, tensor_type=tens_type, active_dims=active_dims)
        self.register_parameter('field', Parameter(torch.zeros((neurons, 2), dtype=self.tensor_type))) 
        self.register_parameter('ATI', Parameter(torch.zeros((neurons, 1), dtype=self.tensor_type)))
        
    def set_params(self, field=None, ATI=None):
        """
        :param np.array w: GLM parameters with additional ATI parameter, shape (neurons, 4)
        """
        if field is not None: # mu, prec
            self.field.data = torch.tensor(field, device=self.dummy.device, dtype=self.tensor_type)
        if ATI is not None: # ATI
            self.ATI.data = torch.tensor(ATI, device=self.dummy.device, dtype=self.tensor_type)

    def compute_F(self, XZ):
        XZ = self._XZ(XZ)
        kappa = self.field[None, :, None, 1]
        lrate_0 = self.field[None, :, None, 2]
        ATI = self.ATI[None, :, None, 0]
        
        omega = XZ[:, None, :, 1]
        theta_0 = self.field[None, :, None, 0] - ATI*omega
        theta = XZ[:, None, :, 0]-theta_0
        return lrate_0 - kappa*torch.cos(theta), 0
            


# Position-Theta-Phase models
class PTP_1D(custom_wrapper):
    """
    1D phase precession model
    """
    def __init__(self, neurons, inv_link='exp', tens_type=torch.float, active_dims=None):
        super().__init__(2, neurons, inv_link, tensor_type=tens_type, active_dims=active_dims)
        self.register_parameter('field', Parameter(torch.zeros((neurons, 3), dtype=self.tensor_type))) 
        self.register_parameter('modulation', Parameter(torch.zeros((neurons, 2), dtype=self.tensor_type)))
        self.register_parameter('precession', Parameter(torch.zeros((neurons, 2), dtype=self.tensor_type)))
            
    def set_params(self, field=None, modulation=None, precession=None):
        if field is not None: # mu, prec, lrate_0
            self.field.data = torch.tensor(field, device=self.dummy.device, dtype=self.tensor_type)
        if modulation is not None: # beta, theta_0
            self.modulation.data = torch.tensor(modulation, device=self.dummy.device, dtype=self.tensor_type)
        if precession is not None: # theta_A, w_theta
            self.precession.data = torch.tensor(precession, device=self.dummy.device, dtype=self.tensor_type)
            
    def compute_F(self, XZ):
        """
        Theta precession is sigmoidal.
        """
        XZ = self._XZ(XZ)
        
        x = (XZ[:, None, :, 0] - self.field[None, :, None, 0])
        lrate_0 = self.field[None, :, None, 2]
        prec = self.field[None, :, None, 1]
        wp = self.precession[None, :, None, 1]
        Ap = self.precession[None, :, None, 0]
        beta = self.modulation[None, :, None, 0]
        theta_0 = self.modulation[None, :, None, 1]
        theta = (XZ[:, None, :, 1] - Ap*torch.sigmoid(theta_0 + wp*x))
        return lrate_0 - .5*prec*x**2 + beta*torch.cos(theta), 0



class PTP_GLM(GLM):
    """
    Position-theta-phase model (in 2D), factorized modulation in both covariates.
    
    References:
    
    [1] `Position–theta-phase model of hippocampal place cell activity applied 
     to quantification of running speed modulation of firing rate`,
     
    """
    def __init__(self, neurons, inv_link='exp', decoding=False, tens_type=torch.float, active_dims=None):
        self.decoding = decoding
        if decoding:
            super().__init__(3, neurons, 8, inv_link, tens_type=tens_type, active_dims=active_dims)
        else:
            super().__init__(8, neurons, 8, inv_link, tens_type=tens_type, active_dims=active_dims)
            
            
    def set_params(self, w=None):
        if w is not None:
            self.w.data = torch.tensor(w, device=self.dummy.device, dtype=self.tensor_type)
    
    
    def compute_F(self, XZ):
        XZ = self._XZ(XZ)
        
        #if self.decoding is True:
        x = XZ[..., 0]
        y = XZ[..., 1]
        theta = XZ[..., 2]
        g = torch.stack((torch.ones_like(x, device=x.device), x, y, x**2, y**2, x*y, torch.cos(theta), torch.sin(theta)), dim=-1)
        return (g[:, None, :, :]*self.w[None, :, None, :]).sum(-1), 0
        #else:
        #    return super().compute_F(XZ)
    
    """
    def set_Y(self, covariates, timesteps, batch_size):
        if self.decoding is False:
            assert len(covariates) == 3 # x, y, theta

            x = covariates[0]
            y = covariates[1]
            theta = covariates[2]
            super().preprocess([np.ones_like(x), x, y, x**2, \
                                y**2, x*y, np.cos(theta), np.sin(theta)], timesteps, batch_size)
        
        else:
            super().preprocess(covariates, timesteps, batch_size)
    """
    
            

class PTP_field(custom_wrapper):
    """
    Gaussian fields with von Mises theta modulation, linear speed modulation, von Mises head direction modulation, 
    von Mises heading direction modulation. The model is factorized over separate task dimensions.
    """
    def __init__(self, neurons, inv_link='exp', tens_type=torch.float, active_dims=None):
        super().__init__(10, neurons, inv_link, tens_type=tens_type, active_dims=active_dims)
        self.register_parameter('mu', Parameter(torch.zeros((neurons, 2), dtype=self.tensor_type)))
        self.register_parameter('prec', Parameter(torch.zeros((neurons, 3), dtype=self.tensor_type)))
        self.register_parameter('lrate', Parameter(torch.zeros((neurons, 1), dtype=self.tensor_type)))
        self.register_parameter('w_t', Parameter(torch.zeros((neurons, 2), dtype=self.tensor_type)))
        self.register_parameter('w_hd', Parameter(torch.zeros((neurons, 2), dtype=self.tensor_type)))
        self.register_parameter('w_dir', Parameter(torch.zeros((neurons, 2), dtype=self.tensor_type)))
        self.register_parameter('speed', Parameter(torch.zeros((neurons, 1), dtype=self.tensor_type)))
            
    def set_params(self, mu=None, prec=None, rate_0=None, t_params=None, hd_params=None, \
                   dir_params=None, speed=None):
        if mu is not None:
            self.mu.data = torch.tensor(mu, device=self.dummy.device, dtype=self.tensor_type)
        if prec is not None:
            self.prec.data = torch.tensor(prec, device=self.dummy.device, dtype=self.tensor_type)
        if rate_0 is not None:
            self.lrate.data[:, 0] = torch.tensor(np.log(rate_0), device=self.dummy.device, dtype=self.tensor_type)
        if t_params is not None:
            self.w_t.data[:, 0] = torch.tensor(t_params[:, 0]*np.cos(t_params[:, 1]), 
                                               device=self.dummy.device, dtype=self.tensor_type)
            self.w_t.data[:, 1] = torch.tensor(t_params[:, 0]*np.sin(t_params[:, 1]), 
                                               device=self.dummy.device, dtype=self.tensor_type)
        if hd_params is not None:
            self.w_hd.data[:, 0] = torch.tensor(hd_params[:, 0]*np.cos(hd_params[:, 1]), 
                                               device=self.dummy.device, dtype=self.tensor_type)
            self.w_hd.data[:, 1] = torch.tensor(hd_params[:, 0]*np.sin(hd_params[:, 1]), 
                                               device=self.dummy.device, dtype=self.tensor_type)
        if dir_params is not None:
            self.w_dir.data[:, 0] = torch.tensor(dir_params[:, 0]*np.cos(dir_params[:, 1]), 
                                               device=self.dummy.device, dtype=self.tensor_type)
            self.w_dir.data[:, 1] = torch.tensor(dir_params[:, 0]*np.sin(dir_params[:, 1]), 
                                               device=self.dummy.device, dtype=self.tensor_type)
        if speed is not None:
            self.speed.data[:, 0] = torch.tensor(speed, device=self.dummy.device, dtype=self.tensor_type)
            
    """def preprocess(self, covariates, timesteps, batch_size):
        assert len(covariates) == 6 # x, y, s, theta, hd, dir
        
        x = covariates[0]
        y = covariates[1]
        s = covariates[2]
        theta = covariates[3]
        hd = covariates[4]
        mdir = covariates[5]
        super().preprocess([x, y, np.ones_like(x), s, np.cos(theta), np.sin(theta), 
                            np.cos(hd), np.sin(hd), np.cos(mdir), np.sin(mdir)], timesteps, batch_size)"""

    def compute_F(self, XZ):
        """
        Speed modulation is linear
        """
        XZ = self._XZ(XZ)
        
        x = XZ[..., 0]
        y = XZ[..., 1]
        s = XZ[..., 2]
        theta = XZ[..., 3]
        hd = XZ[..., 4]
        mdir = XZ[..., 5]
        cov = torch.stack((x, y, torch.ones_like(x, device=x.device), s, torch.cos(theta), torch.sin(theta), 
                            torch.cos(hd), torch.sin(hd), torch.cos(mdir), torch.sin(mdir)), dim=-1)
        
        dp = (cov[:, None, :, :2] - self.mu[None, :, None, :])
        p = torch.cat((dp[..., 0:1]**2, dp[..., 1:2]**2, 2*dp[..., 0:1]*dp[..., 1:2]), dim=-1)
        
        return ((cov[:, None, :, 4:6]*self.w_t[None, :, None, :] + cov[:, None, :, 6:8]*self.w_hd[None, :, None, :] \
                + cov[:, None, :, 8:10]*self.w_dir[None, :, None, :]).sum(-1) + cov[:, None, :, 3]*self.speed[None, :, None, 0] \
                + cov[:, None, :, 2]*self.lrate[None, :, None, 0] - 0.5*(p*self.prec[None, :, None, :]).sum(-1)), 0

    
    
# Grid cell class
class grid_cells(custom_wrapper):
    """
    Grid cells activity maps as presented in:
    - Blair et al. (2007), equation (1)
    - Almeida et al. (2009), equation (1)
    Incorporates theta modulation
    """
    def __init__(self, neurons, decoding=False, tensor_type=torch.float, active_dims=None):
        r"""
        theta (float)             : Grid rotation (assume to be either 0°, 20°, or 40°, in degrees)
        phase (tuple of int)      : Spatial phase of the grid     
        lamb (int)                : Distance between firing fields
        """
        self.decoding = decoding
        if decoding:
            super().__init__(3, neurons, 'identity', tensor_type=tensor_type, active_dims=active_dims)
        else:
            super().__init__(5, neurons, 'identity', tensor_type=tensor_type, active_dims=active_dims)
        self.register_parameter('w', Parameter(torch.zeros((neurons, 4), dtype=self.tensor_type))) # theta, phase, lamb
        self.register_parameter('w_t', Parameter(torch.zeros((neurons, 3), dtype=self.tensor_type))) # sinusoidal, log_rate
        
    def set_params(self, w=None, w_t=None, lrate=None):
        if w is not None:
            self.w.data = torch.tensor(w, device=self.dummy.device, dtype=self.tensor_type)
        if w_t is not None:
            self.w_t.data[:, :-1] = torch.tensor(w_t, device=self.dummy.device, dtype=self.tensor_type)
        if lrate is not None:
            self.w_t.data[:, -1:] = torch.tensor(lrate, device=self.dummy.device, dtype=self.tensor_type)
        
    def compute_F(self, XZ):
        """
        :param torch.tensor cov: covariates with shape (samples, timesteps, dims)
        """
        cov = XZ[..., self.active_dims]
        if self.decoding is True:
            x = cov[..., 0]
            y = cov[..., 1]
            theta = cov[..., 2]
            g = torch.stack((x, y, torch.cos(theta), torch.sin(theta), torch.ones_like(x, device=x.device)), dim=-1)
        else:
            g = cov
            
        theta = self.w[:, :1]
        phase = self.w[:, 1:3]
        lamb = self.w[None, :, None, -1]
        
        a = 0.3
        b = 3./2.
        lambV = (4*np.pi)/(torch.sqrt(3*lamb))

        tmp_g = 0
        for i in np.deg2rad(np.linspace(-30, 90, 3)):
            u_f = torch.cat((torch.cos(i+theta[None, :, None :]), torch.sin(i+theta[None, :, None :])), dim=-1)
            dist = cov[:, None, :, :2] - phase[None, :, None :]
            tmp_g += torch.cos(lambV * (u_f*dist).sum(-1))

        h = torch.exp(a*(tmp_g+b))-1
        return torch.exp((cov[:, None, :, 2:]*self.w_t[None, :, None, :]).sum(-1)) * h, 0
    
    def set_XZ(self, covariates, timesteps, batch_size):
        if self.decoding is False:
            assert len(covariates) == 3 # x, y, theta

            x = covariates[0]
            y = covariates[1]
            theta = covariates[2]
            super().preprocess([x, y, np.cos(theta), np.sin(theta), np.ones_like(x)], timesteps, batch_size)
        
        else:
            super().preprocess(covariates, batch_size)
            