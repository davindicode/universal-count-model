import torch
from torch import Tensor
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.optim as optim
from torch.optim.optimizer import Optimizer, required
from torch.nn.utils.convert_parameters import vector_to_parameters, parameters_to_vector
from torch.utils.data import Dataset

import copy
from functools import reduce

import math






### PyTorch ###
def get_device(gpu=0):
    """
    Enable PyTorch with CUDA if available.
    
    :param int gpu: device number for CUDA
    :returns: device name for CUDA
    :rtype: string
    """
    print("PyTorch version: %s" % torch.__version__)
    dev = torch.device("cuda:{}".format(gpu)) if torch.cuda.is_available() else torch.device("cpu")
    print("Using device: %s" % dev)
    return dev



class Siren(nn.Module):
    """
    Activation function class for SIREN
    
    `Implicit Neural Representations with Periodic Activation Functions`,
    Vincent Sitzmann, Julien N. P. Martel, Alexander W. Bergman, David B. Lindell, Gordon Wetzstein
    """
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return torch.sin(input)
    
    
    
class Parallel_Linear(nn.Module):
    """
    Linear layers that separate different operations in one dimension.
    """
    __constants__ = ['in_features', 'out_features', 'channels']
    in_features: int
    out_features: int
    weight: Tensor

    def __init__(self, in_features: int, out_features: int, channels: int, bias: bool = True) -> None:
        """
        If channels is 1, then we share all the weights over the channel dimension.
        """
        super(Parallel_Linear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.channels = channels
        self.weight = Parameter(torch.Tensor(channels, out_features, in_features))
        if bias:
            self.bias = Parameter(torch.Tensor(channels, out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input: Tensor, channels: list) -> Tensor:
        """
        :param torch.tensor input: input of shape (batch, channels, in_dims)
        """
        if self.channels > 1: # separate weight matrices per channel
            W = self.weight.expand(1, self.channels, self.out_features, self.in_features)[:, channels, ...]
            B = 0 if self.bias is None else self.bias.expand(1, self.channels, self.out_features)[:, channels, :]
        else:
            W = self.weight[None, ...]
            B = self.bias[None, ...]
            
        return (W*input[..., None, :]).sum(-1) + B

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, channels={}, bias={}'.format(
            self.in_features, self.out_features, self.channels, self.bias is not None
        )
    
    
    
class Parallel_MLP(nn.Module):
    """
    Multi-layer perceptron class with parallel layers.
    """
    def __init__(self, layers, in_dims, out_dims, channels, nonlin=nn.ReLU(), 
                 out=None, bias=True, shared_W=False):
        super().__init__()
        self.in_dims = in_dims
        self.channels = channels
        c = 1 if shared_W else channels
        self.net = nn.ModuleList([])
        if len(layers) == 0:
            self.net.append(Parallel_Linear(in_dims, out_dims, c, bias=bias))
        else:
            self.net.append(Parallel_Linear(in_dims, layers[0], c, bias=bias))
            self.net.append(nonlin)
            for k in range(len(layers)-1):
                self.net.append(Parallel_Linear(layers[k], layers[k+1], c, bias=bias))
                self.net.append(nonlin)
            self.net.append(Parallel_Linear(layers[-1:][0], out_dims, c, bias=bias))
        if out is not None:
            self.net.append(out)
        
    def forward(self, input, channel_dims=None):
        """
        :param torch.tensor input: input of shape (samplesxtime, channelsxin_dims)
        """
        if channel_dims is None:
            channel_dims = list(range(self.channels))
        input = input.view(input.shape[0], -1, self.in_dims)
        
        for en, net_part in enumerate(self.net): # run through network list
            if en % 2 == 0:
                input = net_part(input, channel_dims)
            else:
                input = net_part(input)

        return input.view(input.shape[0], -1) # t, NxK


    
class MLP(nn.Module):
    """
    Multi-layer perceptron class
    """
    def __init__(self, layers, in_dims, out_dims, nonlin=nn.ReLU(), out=None, bias=True):
        super().__init__()
        net = nn.ModuleList([])
        if len(layers) == 0:
            net.append(nn.Linear(in_dims, out_dims, bias=bias))
        else:
            net.append(nn.Linear(in_dims, layers[0], bias=bias))
            net.append(nonlin)
            for k in range(len(layers)-1):
                net.append(nn.Linear(layers[k], layers[k+1], bias=bias))
                net.append(nonlin)
                #net.append(nn.BatchNorm1d())
            net.append(nn.Linear(layers[-1:][0], out_dims, bias=bias))
        if out is not None:
            net.append(out)
        self.net = nn.Sequential(*net)
        
    def forward(self, input):
        return self.net(input)
    
    

### Autograd ###
def compute_nat_grads(param_pairs):
    """
    Compute the natural gradients and replace the .grad of the relevant parameters. [1]
    Note compared to the Newton method, the gradient here is computed in parallel for each output dimension
    while the Newton method uses the exact Hessian involving cross-terms.

    References:

    [1] `Natural Gradients in Practice: Non-Conjugate Variational Inference in Gaussian Process Models`, 
        Hugh Salimbeni, Stefanos Eleftheriadis, James Hensman (2018)

    """
    for k in range(len(param_pairs) // 2):
        m = param_pairs[2*k]
        L = param_pairs[2*k+1]
        grad_m = m.grad # N x M
        grad_L = torch.tril(L.grad) # N x M x M
        
        M = L.shape[-1]
        N = L.shape[0]
        
        # convert to grad_S
        A = (L.permute(0, 2, 1) @ grad_L)
        A.view(N, -1)[:, ::M+1] -= 0.5*A[:, range(M), range(M)] # apply phi()
        b = torch.eye(M, device=L.device)
        L_inv = torch.triangular_solve(b, L, upper=False)[0]
        S = L_inv.permute(0, 2, 1) @ A @ L_inv
        grad_S = S + S.permute(0, 2, 1)
        grad_S.view(N, -1)[:, ::M+1] -= 0.5*grad_S[:, range(M), range(M)] # apply phi()
        
        dL_dxi = torch.cat((grad_m.flatten(), grad_L.flatten())).double()

        # compute expectation parameters to xi
        eta_m = m.data
        eta_S = torch.matmul(L.data, L.data.permute(0, 2, 1)) + m.data[..., None] * m.data[:, None, :]
        eta_params = torch.cat((eta_m.flatten(), eta_S.flatten())).double()
        eta_params.requires_grad = True

        eta_m = eta_params[:N*M].view(N, M)
        eta_S = eta_params[N*M:].view(N, M, M)
        chsk_term = eta_S - eta_m[..., None]*eta_m[:, None, :]
        chsk_term.data.view(N, -1)[:, ::M+1] += 1e-1
        L_back = torch.cholesky(chsk_term, upper=False)
        #print(((L.data-L_back)**2).sum())
        xi_back = torch.cat((eta_m.flatten(), L_back.flatten()))
        dL_deta, = torch.autograd.grad(xi_back, eta_params, dL_dxi, retain_graph=False, create_graph=False)

        # compute theta (natural) to xi transform
        b = torch.eye(M, device=L.device).double()
        S_inv = torch.cholesky_solve(b, L.data.double(), upper=False)
        nat_m = (S_inv @ m.data[..., None].double())[..., 0]
        nat_S = -0.5*S_inv.contiguous()
        nat_params = torch.cat((nat_m.flatten(), nat_S.flatten()))
        nat_params.requires_grad = True

        nat_m = nat_params[:N*M].view(N, M)
        nat_S = nat_params[N*M:].view(N, M, M)
        chsk_term = -2.*nat_S
        #chsk_term.data.view(N, -1)[:, ::M+1] += 1e-3
        L_inv = torch.cholesky(chsk_term, upper=False)
        L__ = torch.triangular_solve(b, L_inv, upper=False)[0]
        S = L__.permute(0, 2, 1) @ L__
        #S = torch.inverse(-2.*nat_S)
        m_ = (S @ nat_m[..., None])[..., 0]
        L_ = torch.cholesky(S, upper=False)
        xi = torch.cat((m_.flatten(), L_.flatten()))

        # Jacobian-vector product trick
        v = torch.ones(len(xi), device=L.device, requires_grad=True)
        dvxi_dnat, = torch.autograd.grad(xi, nat_params, v, retain_graph=True, create_graph=True)
        dJ_dv, = torch.autograd.grad(dvxi_dnat, v, dL_deta, retain_graph=False, create_graph=False)

        # assign gradients
        m.grad = dJ_dv[:N*M].view(N, M).clone().type(self.rate_model[0].tensor_type)
        L.grad = torch.tril(dJ_dv[N*M:].view(N, M, M).clone()).type(self.rate_model[0].tensor_type)



def compute_newton_grads(params, loss, mode='inverse'):
    """
    Newton's method for optimization.
    Compute the inverse Hessian gradient product (jvp) to replace the gradients for a chosen 
    subset of second-order optimized parameters.
    """
    #grad_p = torch.cat([p.grad.flatten() for p in self.newton_grad])
    grad_p = torch.cat([torch.autograd.grad(loss, p, retain_graph=True, create_graph=True)[0].flatten() 
                        for p in params])

    # compute Hessian vector product
    v = torch.ones(len(grad_p), device=grad_p.device, requires_grad=True)
    a = torch.dot(grad_p, v)
    Hv = torch.cat([torch.autograd.grad(grad_p, p, v, retain_graph=True, create_graph=True)[0].flatten() 
                     for p in params])
    H = torch.stack([torch.autograd.grad(h, v, retain_graph=True, create_graph=True)[0] 
                     for h in Hv])

    if mode == 'inverse': # invert the Hessian matrix
        L = torch.cholesky(H, upper=False)
        b = torch.eye(M, device=L.device)
        L_ = torch.triangular_solve(b, L_inv, upper=False)[0]
        H_inv = L_ @ L_.permute(1, 0)
        #H_inv = torch.inverse(H)
    elif mode == 'Hessian_free': # conjugate gradients (Hessian free algorithm)
        pass
    else:
        raise NotImplementedError

    params_grad = H_inv @ grad_p.data

    accum_l = 0
    for p in params:
        l = len(p.flatten())
        p.grad = params_grad[accum_l:accum_l+l].view(*p.shape).clone() # shape back
        accum_l += l

            
            
def jacobian(y, x, create_graph=True):
    """
    Compute the Jacobian :math:`\frac{\partial y}{\partial x}`, via the Jacobian-vector product
    
    :param torch.tensor y: variables differentiated w.r.t
    :param torch.tensor x: variables to differentiate
    :param bool create_graph: create graph for the derivates useful for higher order derivates
    :returns: Jacobian
    :rtype: torch.tensor
    """
    jac = []
    flat_y = y.reshape(-1)
    grad_y = torch.zeros_like(flat_y)   
   
    for i in range(len(flat_y)):
        grad_y[i] = 1.
        grad_x, = torch.autograd.grad(flat_y, x, grad_y, retain_graph=True, create_graph=create_graph)
        jac.append(grad_x.reshape(x.shape))
        grad_y[i] = 0.
        
    return torch.stack(jac).reshape(y.shape + x.shape)                                                



def hessian(y, x):
    """
    Compute the Hessian matrix
    
    :param torch.tensor y: variables differentiated w.r.t
    :param torch.tensor x: variables to differentiate
    :returns: Hessian
    :rtype: torch.tensor
    """
    return jacobian(jacobian(y, x, create_graph=True), x)
    
    
    
### networks ###
class BatchNorm(nn.Module):
    def __init__(self, eps=1.e-10):
        super().__init__()
        self.eps_cpu = torch.tensor(eps)
        self.register_buffer('eps', self.eps_cpu)

    def forward(self, z):
        """
        Do batch norm over batch and sample dimension
        """
        mean = torch.mean(z, dim=0, keepdims=True)
        std = torch.std(z, dim=0, keepdims=True)
        z_ = (z - mean) / (std + self.eps)
        return z_
    
class WeightNormConv2d(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size, stride=1, padding=0, bias=True, weight_norm=1.0, scale=False):
        super(WeightNormConv2d, self).__init__()
        
        self.conv = nn.utils.weight_norm(
            nn.Conv2d(in_dim, out_dim, kernel_size, 
                stride=stride, padding=padding, bias=bias))
        if not scale:
            self.conv.weight_g.data = torch.ones_like(self.conv.weight_g.data)
            self.conv.weight_g.requires_grad = False

    def forward(self, x):
        return self.conv(x)
    
class WeightNormConvTranspose2d(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size, stride=1, padding=0, bias=True, weight_norm=1.0, scale=False):
        super(WeightNormConvTranspose2d, self).__init__()
        
        self.convT = nn.utils.weight_norm(
            nn.ConvTranspose2d(in_dim, out_dim, kernel_size, 
                stride=stride, padding=padding, bias=bias))
        if not scale: # normalization is per filter
            self.convT.weight_g.data = torch.ones_like(self.convT.weight_g.data)
            self.convT.weight_g.requires_grad = False

    def forward(self, x):
        return self.convT(x)
    
class MaskedLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__(in_features, out_features, bias)        
        self.register_buffer('mask', torch.ones(out_features, in_features))
        
    def set_mask(self, mask):
        self.mask.data.copy_(torch.tensor(mask.T).uint8())
        
    def forward(self, input):
        return F.linear(input, self.mask * self.weight, self.bias)
    
    

### optimizers ###
class HessianFree(torch.optim.Optimizer):
    """
    Implements the Hessian-free algorithm presented in `Training Deep and
    Recurrent Networks with Hessian-Free Optimization`_.
    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1)
        delta_decay (float, optional): Decay of the previous result of
            computing delta with conjugate gradient method for the
            initialization of the next conjugate gradient iteration
        damping (float, optional): Initial value of the Tikhonov damping
            coefficient. (default: 0.5)
        max_iter (int, optional): Maximum number of Conjugate-Gradient
            iterations (default: 50)
        use_gnm (bool, optional): Use the generalized Gauss-Newton matrix:
            probably solves the indefiniteness of the Hessian (Section 20.6)
        verbose (bool, optional): Print statements (debugging)
    .. _Training Deep and Recurrent Networks with Hessian-Free Optimization:
        https://doi.org/10.1007/978-3-642-35289-8_27
    """

    def __init__(self, params,
                 lr=1,
                 damping=0.5,
                 delta_decay=0.95,
                 cg_max_iter=100,
                 use_gnm=True,
                 verbose=False):

        if not (0.0 < lr <= 1):
            raise ValueError("Invalid lr: {}".format(lr))

        if not (0.0 < damping <= 1):
            raise ValueError("Invalid damping: {}".format(damping))

        if not cg_max_iter > 0:
            raise ValueError("Invalid cg_max_iter: {}".format(cg_max_iter))

        defaults = dict(alpha=lr,
                        damping=damping,
                        delta_decay=delta_decay,
                        cg_max_iter=cg_max_iter,
                        use_gnm=use_gnm,
                        verbose=verbose)
        super(HessianFree, self).__init__(params, defaults)

        if len(self.param_groups) != 1:
            raise ValueError(
                "HessianFree doesn't support per-parameter options (parameter groups)")

        self._params = self.param_groups[0]['params']

    def _gather_flat_grad(self):
        views = list()
        for p in self._params:
            if p.grad is None:
                view = p.data.new(p.data.numel()).zero_()
            elif p.grad.data.is_sparse:
                view = p.grad.data.to_dense().view(-1)
            else:
                view = p.grad.contiguous().view(-1)
            views.append(view)
        return torch.cat(views, 0)

    def step(self, closure, b=None, M_inv=None):
        """
        Performs a single optimization step.
        Arguments:
            closure (callable): A closure that re-evaluates the model
                and returns a tuple of the loss and the output.
            b (callable, optional): A closure that calculates the vector b in
                the minimization problem x^T . A . x + x^T b.
            M (callable, optional): The INVERSE preconditioner of A
        """
        assert len(self.param_groups) == 1

        group = self.param_groups[0]
        alpha = group['alpha']
        delta_decay = group['delta_decay']
        cg_max_iter = group['cg_max_iter']
        damping = group['damping']
        use_gnm = group['use_gnm']
        verbose = group['verbose']

        state = self.state[self._params[0]]
        state.setdefault('func_evals', 0)
        state.setdefault('n_iter', 0)

        loss_before, output = closure()
        current_evals = 1
        state['func_evals'] += 1

        # Gather current parameters and respective gradients
        flat_params = parameters_to_vector(self._params)
        flat_grad = self._gather_flat_grad()

        # Define linear operator
        if use_gnm:
            # Generalized Gauss-Newton vector product
            def A(x):
                return self._Gv(loss_before, output, x, damping)
        else:
            # Hessian-vector product
            def A(x):
                return self._Hv(flat_grad, x, damping)

        if M_inv is not None:
            m_inv = M_inv()

            # Preconditioner recipe (Section 20.13)
            if m_inv.dim() == 1:
                m = (m_inv + damping) ** (-0.85)

                def M(x):
                    return m * x
            else:
                m = torch.inverse(m_inv + damping * torch.eye(*m_inv.shape))

                def M(x):
                    return m @ x
        else:
            M = None

        b = flat_grad.detach() if b is None else b().detach().flatten()

        # Initializing Conjugate-Gradient (Section 20.10)
        if state.get('init_delta') is not None:
            init_delta = delta_decay * state.get('init_delta')
        else:
            init_delta = torch.zeros_like(flat_params)

        eps = torch.finfo(b.dtype).eps

        # Conjugate-Gradient
        deltas, Ms = self._CG(A=A, b=b.neg(), x0=init_delta,
                              M=M, max_iter=cg_max_iter,
                              tol=1e1 * eps, eps=eps, martens=True)

        # Update parameters
        delta = state['init_delta'] = deltas[-1]
        M = Ms[-1]

        vector_to_parameters(flat_params + delta, self._params)
        loss_now = closure()[0]
        current_evals += 1
        state['func_evals'] += 1

        # Conjugate-Gradient backtracking (Section 20.8.7)
        if verbose:
            print("Loss before CG: {}".format(float(loss_before)))
            print("Loss before BT: {}".format(float(loss_now)))

        for (d, m) in zip(reversed(deltas[:-1][::2]), reversed(Ms[:-1][::2])):
            vector_to_parameters(flat_params + d, self._params)
            loss_prev = closure()[0]
            if float(loss_prev) > float(loss_now):
                break
            delta = d
            M = m
            loss_now = loss_prev

        if verbose:
            print("Loss after BT:  {}".format(float(loss_now)))

        # The Levenberg-Marquardt Heuristic (Section 20.8.5)
        reduction_ratio = (float(loss_now) -
                           float(loss_before)) / M if M != 0 else 1

        if reduction_ratio < 0.25:
            group['damping'] *= 3 / 2
        elif reduction_ratio > 0.75:
            group['damping'] *= 2 / 3
        if reduction_ratio < 0:
            group['init_delta'] = 0

        # Line Searching (Section 20.8.8)
        beta = 0.8
        c = 1e-2
        min_improv = min(c * torch.dot(b, delta), 0)

        for _ in range(60):
            if float(loss_now) <= float(loss_before) + alpha * min_improv:
                break

            alpha *= beta
            vector_to_parameters(flat_params + alpha * delta, self._params)
            loss_now = closure()[0]
        else:  # No good update found
            alpha = 0.0
            loss_now = loss_before

        # Update the parameters (this time fo real)
        vector_to_parameters(flat_params + alpha * delta, self._params)

        if verbose:
            print("Loss after LS:  {0} (lr: {1:.3f})".format(
                float(loss_now), alpha))
            print("Tikhonov damping: {0:.3f} (reduction ratio: {1:.3f})".format(
                group['damping'], reduction_ratio), end='\n\n')

        return loss_now

    def _CG(self, A, b, x0, M=None, max_iter=50, tol=1.2e-6, eps=1.2e-7,
            martens=False):
        """
        Minimizes the linear system x^T.A.x - x^T b using the conjugate
            gradient method
        Arguments:
            A (callable): An abstract linear operator implementing the
                product A.x. A must represent a hermitian, positive definite
                matrix.
            b (torch.Tensor): The vector b.
            x0 (torch.Tensor): An initial guess for x.
            M (callable, optional): An abstract linear operator implementing
            the product of the preconditioner (for A) matrix with a vector.
            tol (float, optional): Tolerance for convergence.
            martens (bool, optional): Flag for Martens' convergence criterion.
        """

        x = [x0]
        r = A(x[0]) - b

        if M is not None:
            y = M(r)
            p = -y
        else:
            p = -r

        res_i_norm = r @ r

        if martens:
            m = [0.5 * (r - b) @ x0]

        for i in range(max_iter):
            Ap = A(p)

            alpha = res_i_norm / ((p @ Ap) + eps)

            x.append(x[i] + alpha * p)
            r = r + alpha * Ap

            if M is not None:
                y = M(r)
                res_ip1_norm = y @ r
            else:
                res_ip1_norm = r @ r

            beta = res_ip1_norm / (res_i_norm + eps)
            res_i_norm = res_ip1_norm

            # Martens' Relative Progress stopping condition (Section 20.4)
            if martens:
                m.append(0.5 * A(x[i + 1]) @ x[i + 1] - b @ x[i + 1])

                k = max(10, int(i / 10))
                if i > k:
                    stop = (m[i] - m[i - k]) / (m[i] + eps)
                    if stop < 1e-4:
                        break

            if res_i_norm < tol or torch.isnan(res_i_norm):
                break

            if M is not None:
                p = - y + beta * p
            else:
                p = - r + beta * p

        return (x, m) if martens else (x, None)

    def _Hv(self, gradient, vec, damping):
        """
        Computes the Hessian vector product.
        """
        Hv = self._Rop(gradient, self._params, vec)

        # Tikhonov damping (Section 20.8.1)
        return Hv.detach() + damping * vec

    def _Gv(self, loss, output, vec, damping):
        """
        Computes the generalized Gauss-Newton vector product.
        """
        Jv = self._Rop(output, self._params, vec)

        gradient = torch.autograd.grad(loss, output, create_graph=True)
        HJv = self._Rop(gradient, output, Jv)

        JHJv = torch.autograd.grad(
            output, self._params, grad_outputs=HJv.reshape_as(output), retain_graph=True)

        # Tikhonov damping (Section 20.8.1)
        return parameters_to_vector(JHJv).detach() + damping * vec

    @staticmethod
    def _Rop(y, x, v, create_graph=False):
        """
        Computes the product (dy_i/dx_j) v_j: R-operator
        """
        if isinstance(y, tuple):
            ws = [torch.zeros_like(y_i, requires_grad=True) for y_i in y]
        else:
            ws = torch.zeros_like(y, requires_grad=True)

        jacobian = torch.autograd.grad(
            y, x, grad_outputs=ws, create_graph=True)

        Jv = torch.autograd.grad(parameters_to_vector(
            jacobian), ws, grad_outputs=v, create_graph=create_graph)

        return parameters_to_vector(Jv)


# The empirical Fisher diagonal (Section 20.11.3)
def empirical_fisher_diagonal(net, xs, ys, criterion):
    grads = list()
    for (x, y) in zip(xs, ys):
        fi = criterion(net(x), y)
        grads.append(torch.autograd.grad(fi, net.parameters(),
                                         retain_graph=False))

    vec = torch.cat([(torch.stack(p) ** 2).mean(0).detach().flatten()
                     for p in zip(*grads)])
    return vec


# The empirical Fisher matrix (Section 20.11.3)
def empirical_fisher_matrix(net, xs, ys, criterion):
    grads = list()
    for (x, y) in zip(xs, ys):
        fi = criterion(net(x), y)
        grad = torch.autograd.grad(fi, net.parameters(),
                                   retain_graph=False)
        grads.append(torch.cat([g.detach().flatten() for g in grad]))

    grads = torch.stack(grads)
    n_batch = grads.shape[0]
    return torch.einsum('ij,ik->jk', grads, grads) / n_batch



class AccSGD(Optimizer):
    r"""Implements the algorithm proposed in https://arxiv.org/pdf/1704.08227.pdf, which is a provably accelerated method 
    for stochastic optimization. This has been employed in https://openreview.net/forum?id=rJTutzbA- for training several 
    deep learning models of practical interest. This code has been implemented by building on the construction of the SGD 
    optimization module found in pytorch codebase.
    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float): learning rate (required)
        kappa (float, optional): ratio of long to short step (default: 1000)
        xi (float, optional): statistical advantage parameter (default: 10)
        smallConst (float, optional): any value <=1 (default: 0.7)
    Example:
        >>> from AccSGD import *
        >>> optimizer = AccSGD(model.parameters(), lr=0.1, kappa = 1000.0, xi = 10.0)
        >>> optimizer.zero_grad()
        >>> loss_fn(model(input), target).backward()
        >>> optimizer.step()
    """

    def __init__(self, params, lr=required, kappa = 1000.0, xi = 10.0, smallConst = 0.7, weight_decay=0):
        defaults = dict(lr=lr, kappa=kappa, xi=xi, smallConst=smallConst,
                        weight_decay=weight_decay)
        super(AccSGD, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(AccSGD, self).__setstate__(state)

    def step(self, closure=None):
        """ Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            large_lr = (group['lr']*group['kappa'])/(group['smallConst'])
            Alpha = 1.0 - ((group['smallConst']*group['smallConst']*group['xi'])/group['kappa'])
            Beta = 1.0 - Alpha
            zeta = group['smallConst']/(group['smallConst']+Beta)
            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                if weight_decay != 0:
                    d_p.add_(weight_decay, p.data)
                param_state = self.state[p]
                if 'momentum_buffer' not in param_state:
                    param_state['momentum_buffer'] = copy.deepcopy(p.data)
                buf = param_state['momentum_buffer']
                buf.mul_((1.0/Beta)-1.0)
                buf.add_(-large_lr,d_p)
                buf.add_(p.data)
                buf.mul_(Beta)

                p.data.add_(-group['lr'],d_p)
                p.data.mul_(zeta)
                p.data.add_(1.0-zeta,buf)

        return loss
    
    
### datasets ###
class CustomDataSet(Dataset):
    """
    PyTorch dataset class.
    
        >>> transform = transforms.Compose([transforms.ToTensor()])
        >>> my_dataset = CustomDataSet(img_folder_path, transform=transform)
        >>> train_loader = DataLoader(my_dataset , batch_size=batch_size, shuffle=False, 
        >>>                           num_workers=4, drop_last=True)

        >>> for idx, img in enumerate(train_loader):
        >>>     #do your training now
        
    """
    def __init__(self, main_dir, transform):
        self.main_dir = main_dir
        self.transform = transform
        all_imgs = os.listdir(main_dir)
        self.total_imgs = natsort.natsorted(all_imgs)

    def __len__(self):
        return len(self.total_imgs)

    def __getitem__(self, idx):
        img_loc = os.path.join(self.main_dir, self.total_imgs[idx])
        image = Image.open(img_loc).convert("RGB")
        tensor_image = self.transform(image)
        return tensor_image
    
