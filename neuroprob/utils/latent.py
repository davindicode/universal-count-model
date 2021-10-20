import numpy as np

import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.optim as optim

from sklearn.cross_decomposition import CCA



# metrics
def metric(x, y, topology='euclid'):
    """
    Returns the geodesic displacement between x and y, (x-y).
    
    :param torch.tensor x: input x of any shape
    :param torch.tensor y: input y of same shape as x
    :returns: x-y tensor of geodesic distances
    :rtype: torch.tensor
    """
    if topology == 'euclid':
        xy = (x-y)
    elif topology == 'torus':
        xy = (x-y) % (2*np.pi)
        xy[xy > np.pi] -= 2*np.pi
    elif topology == 'circ':
        xy = 2*(1-torch.cos(x-y))
    else:
        raise NotImplementedError
    #xy[xy < 0] = -xy[xy < 0] # abs
    return xy



# non-Euclidean
def S_n_coord(input, to_cartesian=False):
    """
    Hyperspherical coordinates here have :math:`theta_{-1}` being 0 to :math:`2\pi`, :math:`theta_{1+n}` 
    corresponds to :math:`x_{d-n}`. Angles are between :math:`[0, \pi]`, the last angle :math:`[-pi, \pi]`.
    
    :param np.array input: input of coordinates (r, theta_1, ...) or (x, y, ...), shape (..., dim)
    :param bool to_cartesian: indicator whether we convert to cartesian or from
    """
    dims = input.shape[-1]
    output = np.ones(input.shape)
    
    if to_cartesian:
        r = input[..., 0:1]
        thetas = input[..., 1:]
        output *= r
        for d in range(dims-1, 1, -1):
            output[..., d] *= np.cos(thetas[..., dims-1-d])
            output[..., :d] *= np.sin(thetas[..., dims-1-d:dims-d])
            
        output[..., 0] = output[..., 1]*np.cos(thetas[..., dims-2])
        output[..., 1] = output[..., 1]*np.sin(thetas[..., dims-2])
        
    else:
        output[..., 0] = np.linalg.norm(input, axis=-1)
        input /= output[..., 0:1]
        for d in range(dims-1, 1, -1):
            output[..., dims-d] = np.arccos(input[..., d])
            input /= np.sin(output[..., dims-d:dims-d+1])
            
        output[..., dims-1] = np.angle(input[..., 0] + 1j*input[..., 1])
            
    return output



# align latent
def signed_scaled_shift(x, x_ref, dev='cpu', topology='torus', iters=1000, lr=1e-2, learn_scale=True):
    """
    Shift trajectory, with scaling, reflection and translation.
    
    Shift trajectories to be as close as possible to each other, including 
    switches in sign.
    
    :param np.array theta: circular input array of shape (timesteps,)
    :param np.array theta_ref: reference circular input array of shape (timesteps,)
    :param string dev:
    :param int iters:
    :param float lr:
    :returns:
    :rtype: tuple
    """
    XX = torch.tensor(x, device=dev)
    XR = torch.tensor(x_ref, device=dev)
        
    lowest_loss = np.inf
    for sign in [1, -1]: # select sign automatically
        shift = Parameter(torch.zeros(1, device=dev))
        p = [shift]
        
        if learn_scale:
            scale = Parameter(torch.ones(1, device=dev))
            p += [scale]
        else:
            scale = torch.ones(1, device=dev)
        
        optimizer = optim.Adam(p, lr=lr)
        losses = []
        for k in range(iters):
            optimizer.zero_grad()
            X_ = XX*sign*scale + shift
            loss = (metric(X_, XR, topology)**2).mean()
            loss.backward()
            optimizer.step()
            losses.append(loss.cpu().item())
            
        l_ = loss.cpu().item()
        
        if l_ < lowest_loss:
            lowest_loss = l_
            shift_ = shift.cpu().item()
            scale_ = scale.cpu().item()
            sign_ = sign
            losses_ = losses

    return x*sign_*scale_ + shift_, shift_, sign_, scale_, losses_



def align_CCA(X, X_tar):
    """
    :param np.array X: input variables of shape (time, dimensions)
    """
    d = X.shape[-1]
    cca = CCA(n_components=d)
    cca.fit(X, X_tar)
    X_c = cca.transform(X)
    return X_c, cca
    
    

def align_affine_2D(lats, target, scales_arr, dev='cpu', iters=20000, lr=1e-2):
    """
    Align 2D latent space through affine (rotation and scaling and translation) transform.
    Assumes the input latents are roughly unit normal distributed.
    """
    rx_t = target[0]
    ry_t = target[1]
    
    class rot_W(nn.Module):
        def __init__(self, theta=0.0):
            super().__init__()
            self.theta = nn.Parameter(torch.tensor(theta))

        def W(self):
            """ 
            assemble rotation matrix.
            """
            C = torch.cos(self.theta)[None]
            S = torch.sin(self.theta)[None]
            top = torch.cat((C, -S))
            down = torch.cat((S, C))
            return torch.cat((top[None], down[None]), dim=0)
    
    lowest_loss = np.inf

    lats = torch.tensor(lats, device=dev)
    x_tar = torch.tensor(target, device=dev)
    for scales in scales_arr: # try out different reflections, makes optimization more robust

        scales = nn.Parameter(scales)
        bias = nn.Parameter(lats.mean(-1)*torch.ones(2, device=dev))
        T = rot_W().to(dev)

        optimizer = optim.Adam(list(T.parameters()) + [scales, bias], lr=lr)

        losses = []
        for k in range(iters):
            optimizer.zero_grad()
            lats_ = scales[:, None]*(T.W() @ lats + bias[:, None])
            loss = ((x_tar-lats_)**2).mean()
            loss.backward()
            optimizer.step()
            losses.append(loss.cpu().item())

        l_ = loss.cpu().item()
        
        if l_ < lowest_loss:
            lowest_loss = l_
            W_ = T.W().data.cpu().numpy()
            bias_ = bias.data.cpu().numpy()
            scale_ = scales.data.cpu().numpy()
            losses_ = losses

    return W_, bias_, scale_, losses_
