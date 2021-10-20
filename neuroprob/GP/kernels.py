import torch
from torch.nn.parameter import Parameter
import torch.nn.functional as F



class Kernel(torch.nn.Module):
    """
    Base class for multi-lengthscale kernels used in Gaussian Processes.
    Inspired by pyro GP kernels.
    """
    def __init__(self, input_dims, track_dims=None, f='exp'):
        """
        :param int input_dims: number of input dimensions
        :param list track_dims: list of dimension indices used as input from X
        :param string f: inverse link function for positive constraints in hyperparameters
        """
        super().__init__()

        if track_dims is None:
            track_dims = list(range(input_dims))
        elif input_dims != len(track_dims):
            raise ValueError("Input size and the length of active dimensionals should be equal.")
        self.input_dims = input_dims
        self.track_dims = track_dims
        
        if f == 'exp':
            self.f = lambda x : torch.exp(x)
            self.f_inv = lambda x: torch.log(x)
        elif f == 'softplus':
            self.f = lambda x : F.softplus(x)
            self.f_inv = lambda x: torch.where(x > 30, x, torch.log(torch.exp(x) - 1))
        elif f == 'relu':
            self.f = lambda x : torch.clamp(x, min=0)
            self.f_inv = lambda x: x
        else:
            raise NotImplementedError("Link function is not supported.")

    def forward(self, X, Z=None, diag=False):
        r"""
        Calculates covariance matrix of inputs on active dimensionals.
        """
        raise NotImplementedError

    def _slice_input(self, X):
        r"""
        Slices :math:`X` according to ``self.track_dims``.

        :param torch.tensor X: input with shape (neurons, samples, timesteps, dimensions)
        :returns: a 2D slice of :math:`X`
        :rtype: torch.tensor
        """
        if X.dim() == 4:
            return X[..., self.track_dims]
        else:
            raise ValueError("Input X must be of shape (N x K x T x D).")
            
    def _XZ(self, X, Z=None):
        """
        Slice the input of :math:`X` and :math:`Z` into correct shapes.
        """
        if Z is None:
            Z = X
            
        X = self._slice_input(X)
        Z = self._slice_input(Z)
        if X.size(-1) != Z.size(-1):
            raise ValueError("Inputs must have the same number of features.")
            
        return X, Z
            
            
        
class Combination(Kernel):
    """
    Base class for kernels derived from a combination of kernels.

    :param Kernel kern0: First kernel to combine.
    :param kern1: Second kernel to combine.
    :type kern1: Kernel or numbers.Number
    """
    def __init__(self, kern0, kern1):
        if not isinstance(kern0, Kernel) or not isinstance(kern1, Kernel):
            raise TypeError("The components of a combined kernel must be "
                            "Kernel instances.")

        track_dims = (set(kern0.track_dims) | set(kern1.track_dims)) # | OR operator, & is AND
        track_dims = sorted(track_dims)
        input_dims = max(track_dims)+1 # cover all dimensions to it
        super().__init__(input_dims)

        self.kern0 = kern0
        self.kern1 = kern1

        
        
class Sum(Combination):
    """
    Returns a new kernel which acts like a sum/direct sum of two kernels.
    The second kernel can be a constant.
    """
    def forward(self, X, Z=None, diag=False):
        return self.kern0(X, Z, diag=diag) + self.kern1(X, Z, diag=diag)
                    
                    
            
class Product(Combination):
    """
    Returns a new kernel which acts like a product/tensor product of two kernels.
    The second kernel can be a constant.
    """
    def forward(self, X, Z=None, diag=False):
        return self.kern0(X, Z, diag=diag) * self.kern1(X, Z, diag=diag)
        


def _torch_sqrt(x, eps=1e-12):
    """
    A convenient function to avoid the NaN gradient issue of :func:`torch.sqrt`
    at 0.
    """
    # Ref: https://github.com/pytorch/pytorch/issues/2421
    return (x + eps).sqrt()



class Constant(Kernel):
    r"""
    Constant kernel (functions independent of input).
    """
    def __init__(self, variance, f='exp'):
        super().__init__(0, None, f)
        self._variance = Parameter(self.f_inv(variance)) # N
        
    @property
    def variance(self):
        return self.f(self._variance)[:, None, None, None] # N, K, T, T
    
    @variance.setter
    def variance(self):
        self._variance.data = self.f_inv(variance)

    def forward(self, X, Z=None, diag=False):
        if Z is None:
            Z = X
            
        if diag:
            return self.variance[..., 0].expand(self._variance.shape[0], X.size(1), X.size(2)) # N, K, T
        else:
            return self.variance.expand(self._variance.shape[0], X.size(1), X.size(2), Z.size(2)) # N, K, T, T
        

    
### Full kernels
class DotProduct(Kernel):
    r"""
    Base class for kernels which are functions of :math:`x \cdot z`.
    """
    def __init__(self, input_dims, track_dims=None, f='exp'):
        super().__init__(input_dims, track_dims, f)

    def _dot_product(self, X, Z=None, diag=False):
        r"""
        Returns :math:`X \cdot Z`.
        """
        if diag:
            return (self._slice_input(X) ** 2).sum(-1)

        if Z is None:
            Z = X
            
        X = self._slice_input(X)
        Z = self._slice_input(Z)
        if X.size(-1) != Z.size(-1):
            raise ValueError("Inputs must have the same number of features.")

        return X.matmul(Z.permute(0, 1, 3, 2))


class Linear(DotProduct):
    r"""
    Implementation of Linear kernel:
        :math:`k(x, z) = \sigma^2 x \cdot z.`
    Doing Gaussian Process regression with linear kernel is equivalent to doing a
    linear regression.
    .. note:: Here we implement the homogeneous version. To use the inhomogeneous
        version, consider using :class:`Polynomial` kernel with ``degree=1`` or making
        a :class:`.Sum` with a :class:`.Constant` kernel.
    """

    def __init__(self, input_dims, track_dims=None, f='exp'):
        super().__init__(input_dims, track_dims, f)

    def forward(self, X, Z=None, diag=False):
        return self._dot_product(X, Z, diag)


class Polynomial(DotProduct):
    r"""
    Implementation of Polynomial kernel:
        :math:`k(x, z) = \sigma^2(\text{bias} + x \cdot z)^d.`
    :param torch.Tensor bias: Bias parameter of this kernel. Should be positive.
    :param int degree: Degree :math:`d` of the polynomial.
    """

    def __init__(self, input_dims, bias, degree=1, track_dims=None, f='exp'):
        super().__init__(input_dims, track_dims, f)
        self._bias = Parameter(self.f_inv(bias)) # N

        if not isinstance(degree, int) or degree < 1:
            raise ValueError("Degree for Polynomial kernel should be a positive integer.")
        self.degree = degree
        
    @property
    def bias(self):
        return self.f(self._bias)[:, None, None, None] # N, K, T, T
    
    @bias.setter
    def bias(self):
        self._bias.data = self.f_inv(bias)

    def forward(self, X, Z=None, diag=False):
        if diag:
            bias = self.bias[..., 0]
        else:
            bias = self.bias
            
        return ((bias + self._dot_product(X, Z, diag)) ** self.degree)

    
    
class DSE(Kernel):
    r"""
    Implementation of Decaying Squared Exponential kernel:

        :math:`k(x, z) = \exp\left(-0.5 \times \frac{|x-\beta|^2 + |z-\beta|^2} {l_{\beta}^2}\right) \, 
        \exp\left(-0.5 \times \frac{|x-z|^2}{l^2}\right).`

    :param torch.Tensor scale_mixture: Scale mixture (:math:`\alpha`) parameter of this
        kernel. Should have size 1.
    """
    def __init__(self, input_dims, lengthscale, lengthscale_beta, beta, 
                 track_dims=None, f='exp'):
        super().__init__(input_dims, track_dims, f)
        assert self.input_dims == lengthscale.shape[0] # lengthscale per input dimension
        
        self.beta = Parameter(beta.t()) # N, D
        self._lengthscale_beta = Parameter(self.f_inv(lengthscale_beta).t()[:, None, None, :]) # N, K, T, D
        self._lengthscale = Parameter(self.f_inv(lengthscale).t()) # N, D
        
    @property
    def lengthscale(self):
        return self.f(self._lengthscale)[:, None, None, :] # N, K, T, D
    
    @lengthscale.setter
    def lengthscale(self):
        self._lengthscale.data = self.f_inv(lengthscale)
        
    @property
    def lengthscale_beta(self):
        return self.f(self._lengthscale_beta)[:, None, None, :] # N, K, T, D
    
    @lengthscale_beta.setter
    def lengthscale_beta(self):
        self._lengthscale_beta.data = self.f_inv(lengthscale_beta)

    def forward(self, X, Z=None, diag=False):
        X, Z = self._XZ(X, Z)

        if diag:
            return torch.exp(-(((X-self.beta)/self.lengthscale_beta)**2).sum(-1))
        
        scaled_X = X / self.lengthscale # N, K, T, D
        scaled_Z = Z / self.lengthscale
        X2 = (scaled_X ** 2).sum(-1, keepdim=True)
        Z2 = (scaled_Z ** 2).sum(-1, keepdim=True)
        XZ = scaled_X.matmul(scaled_Z.permute(0, 1, 3, 2))
        r2 = X2 - 2 * XZ + Z2.permute(0, 1, 3, 2)

        return torch.exp(-0.5 * (r2 + (((X-self.beta)/self.lengthscale_beta)**2).sum(-1)[..., None] +
                                (((Z-self.beta)/self.lengthscale_beta)**2).sum(-1)[..., None, :]))



class Isotropy(Kernel):
    """
    Base class for a family of isotropic covariance kernels which are functions of the
    distance :math:`|x-z|/l`, where :math:`l` is the length-scale parameter.

    :param torch.Tensor variance: variance parameter of shape (neurons)
    :param torch.tensor lengthscale: length-scale parameter of shape (dims, neurons)
    """
    def __init__(self, input_dims, topology, lengthscale, track_dims=None, f='exp'):
        super().__init__(input_dims, track_dims, f)

        self.n = lengthscale.shape[1] # separate lengthscale per output dimension
        assert self.input_dims == lengthscale.shape[0] # lengthscale per input dimension
        
        if topology == 'euclid':
            self.scaled_dist = self._scaled_dist
            self.square_scaled_dist = self._square_scaled_dist
        elif topology == 'torus_wrap':
            self.scaled_dist = self._scaled_dist_Tn
            self.square_scaled_dist = self._square_scaled_dist_Tn
        elif topology == 'torus':
            self.scaled_dist = self._scaled_dist_torus
            self.square_scaled_dist = self._square_scaled_dist_torus
        elif topology == 'sphere':
            self.scaled_dist = self._scaled_dist_sphere
            self.square_scaled_dist = self._square_scaled_dist_sphere
            lengthscale = lengthscale[:1, :] # dummy dimensions
        else:
            raise NotImplementedError('Topology is not supported.')
            
        self._lengthscale = Parameter(self.f_inv(lengthscale).t()) # N, D
        
    @property
    def lengthscale(self):
        return self.f(self._lengthscale)[:, None, None, :] # N, K, T, D
    
    @lengthscale.setter
    def lengthscale(self):
        self._lengthscale.data = self.f_inv(lengthscale)
        
    # Euclidean
    def _square_scaled_dist(self, X, Z=None):
        r"""
        Returns :math:`\|\frac{X-Z}{l}\|^2`.
        :param torch.tensor X: input of shape (neurons, samples, points, dims)
        
        """
        X, Z = self._XZ(X, Z)

        scaled_X = X / self.lengthscale # N, K, T, D
        scaled_Z = Z / self.lengthscale
        X2 = (scaled_X ** 2).sum(-1, keepdim=True)
        Z2 = (scaled_Z ** 2).sum(-1, keepdim=True)
        XZ = scaled_X.matmul(scaled_Z.permute(0, 1, 3, 2))
        r2 = X2 - 2 * XZ + Z2.permute(0, 1, 3, 2)
        return r2.clamp(min=0)

    def _scaled_dist(self, X, Z=None):
        r"""
        Returns :math:`\|\frac{X-Z}{l}\|`.
        """
        return _torch_sqrt(self._square_scaled_dist(X, Z))
    
    # Torus wrapped
    def _square_scaled_dist_Tn(self, X, Z=None):
        r"""
        Returns :math:`\|\frac{X-Z}{l}\|^2`.
        """
        X, Z = self._XZ(X, Z)
            
        U = (X-Z) % (2*math.pi)
        n = torch.arange(-3, 3, device=X.device)[:, None, None, None]*2*math.pi # not correct for multidimensions
        return ((torch.min(U, 2*math.pi-U).unsqueeze(0)+n)/self.lengthscale).pow(2).sum(-1)
    
    def _scaled_dist_Tn(self, X, Z=None):
        r"""
        Returns :math:`\|\frac{X-Z}{l}\|`.
        """
        return _torch_sqrt(self._square_scaled_dist(X, Z))
    
    # Torus cosine
    def _square_scaled_dist_torus(self, X, Z=None):
        r"""
        Returns :math:`\|\frac{X-Z}{l}\|^2`.
        """         
        X, Z = self._XZ(X, Z)
        U = (X[..., None, :]-Z[..., None, :].permute(0, 1, 3, 2, 4))
        
        return 2*((1 - torch.cos(U))/(self.lengthscale[:, :, None, ...])**2).sum(-1)
    
    def _scaled_dist_torus(self, X, Z=None):
        r"""
        Returns :math:`\|\frac{X-Z}{l}\|`.
        """         
        return _torch_sqrt(self._square_scaled_dist_torus(X, Z))
    
    # n-sphere S(n), input dimension is n
    def _square_scaled_dist_sphere(self, X, Z=None):
        r"""
        Returns :math:`\|\frac{X-Z}{l}\|^2`.
        """         
        X, Z = self._XZ(X, Z)
        
        return 2*((1 - (X[..., None, :]*Z[..., None, :].permute(0, 1, 3, 2, 4)).sum(-1))/(self.lengthscale[..., None, 0])**2)
    
    def _scaled_dist_sphere(self, X, Z=None):
        r"""
        Returns :math:`\|\frac{X-Z}{l}\|`.
        """         
        return _torch_sqrt(self._square_scaled_dist_sphere(X, Z))



class RBF(Isotropy):
    r"""
    Implementation of Radial Basis Function kernel:

        :math:`k(x,z) = \exp\left(-0.5 \times \frac{|x-z|^2}{l^2}\right).`

    .. note:: This kernel also has name `Squared Exponential` in literature.
    """
    def __init__(self, input_dims, lengthscale, track_dims=None, topology='euclid', f='exp'):
        super().__init__(input_dims, topology, lengthscale, track_dims, f)

    def forward(self, X, Z=None, diag=False):
        if diag:
            return X.new_ones(self.n, X.size(1), X.size(2))

        r2 = self.square_scaled_dist(X, Z)
        return torch.exp(-0.5 * r2)
                         


class RationalQuadratic(Isotropy):
    r"""
    Implementation of RationalQuadratic kernel:

        :math:`k(x, z) = \left(1 + 0.5 \times \frac{|x-z|^2}{\alpha l^2}
        \right)^{-\alpha}.`

    :param torch.Tensor scale_mixture: Scale mixture (:math:`\alpha`) parameter of this
        kernel. Should have size 1.
    """
    def __init__(self, input_dims, lengthscale, scale_mixture,
                 track_dims=None, topology='euclid', f='exp'):
        super().__init__(input_dims, topology, lengthscale, track_dims, f)

        self._scale_mixture = Parameter(self.f_inv(scale_mixture)) # N
        
    @property
    def scale_mixture(self):
        return self.f(self._scale_mixture)[:, None, None] # N, K, T
    
    @scale_mixture.setter
    def scale_mixture(self):
        self._scale_mixture.data = self.f_inv(scale_mixture)

    def forward(self, X, Z=None, diag=False):
        if diag:
            return X.new_ones(self.n, X.size(1), X.size(2))

        r2 = self.square_scaled_dist(X, Z)
        return (1 + (0.5 / self.scale_mixture) * r2).pow(-self.scale_mixture)



class Exponential(Isotropy):
    r"""
    Implementation of Exponential kernel:

        :math:`k(x, z) = \sigma^2\exp\left(-\frac{|x-z|}{l}\right).`
    """
    def __init__(self, input_dims, lengthscale, track_dims=None, topology='euclid', f='exp'):
        super().__init__(input_dims, topology, lengthscale, track_dims, f)

    def forward(self, X, Z=None, diag=False):
        if diag:
            return X.new_ones(self.n, X.size(1), X.size(2))

        r = self.scaled_dist(X, Z)
        return torch.exp(-r)



class Matern32(Isotropy):
    r"""
    Implementation of Matern32 kernel:

        :math:`k(x, z) = \sigma^2\left(1 + \sqrt{3} \times \frac{|x-z|}{l}\right)
        \exp\left(-\sqrt{3} \times \frac{|x-z|}{l}\right).`
    """
    def __init__(self, input_dims, lengthscale, track_dims=None, topology='euclid', f='exp'):
        super().__init__(input_dims, topology, lengthscale, track_dims, f)

    def forward(self, X, Z=None, diag=False):
        if diag:
            return X.new_ones(self.n, X.size(1), X.size(2))

        r = self.scaled_dist(X, Z)
        sqrt3_r = 3**0.5 * r
        return (1 + sqrt3_r) * torch.exp(-sqrt3_r)



class Matern52(Isotropy):
    r"""
    Implementation of Matern52 kernel:

        :math:`k(x,z)=\sigma^2\left(1+\sqrt{5}\times\frac{|x-z|}{l}+\frac{5}{3}\times
        \frac{|x-z|^2}{l^2}\right)\exp\left(-\sqrt{5} \times \frac{|x-z|}{l}\right).`
    """
    def __init__(self, input_dims, lengthscale, track_dims=None, topology='euclid', f='exp'):
        super().__init__(input_dims, topology, lengthscale, track_dims, f)

    def forward(self, X, Z=None, diag=False):
        if diag:
            return X.new_ones(self.n, X.size(1), X.size(2))

        r2 = self.square_scaled_dist(X, Z)
        r = _torch_sqrt(r2)
        sqrt5_r = 5**0.5 * r
        return (1 + sqrt5_r + (5/3) * r2) * torch.exp(-sqrt5_r)
    

    
### Pyro style kernels, are faster and smaller but shared kernel parameters
class Kernel_light(Kernel):
    r"""
    Base class for multi-lengthscale kernels used in Gaussian Processes
    """
    def _slice_input(self, X):
        r"""
        Slices :math:`X` according to ``self.track_dims``.

        :param torch.tensor X: input with shape (timesteps, dimensions)
        :returns: a 2D slice of :math:`X`
        :rtype: torch.tensor
        """
        if X.dim() == 2:
            return X[:, self.track_dims]
        else:
            raise ValueError("Input X must be of shape (T x D).")


            
class Constant_light(Kernel_light):
    r"""
    Constant kernel (functions independent of input).
    """
    def __init__(self, variance, f='exp'):
        super().__init__(0, None, f)
        self._variance = Parameter(self.f_inv(variance))
        
    @property
    def variance(self):
        return self.f(self._variance)[None, None] # T, T
    
    @variance.setter
    def variance(self):
        self._variance.data = self.f_inv(variance)

    def forward(self, X, Z=None, diag=False):
        if Z is None:
            Z = X
            
        if diag:
            return self.variance[:, 0].expand(X.size(0)) # T
        else:
            return self.variance.expand(X.size(0), Z.size(0)) # T, T

        
        
class Isotropy_light(Kernel_light):
    """
    Base class for a family of isotropic covariance kernels which are functions of the
    distance :math:`|x-z|/l`, where :math:`l` is the length-scale parameter.

    :param torch.Tensor variance: variance parameter of shape (neurons)
    :param torch.tensor lengthscale: length-scale parameter of shape (dims, neurons)
    """
    def __init__(self, input_dims, topology, lengthscale, track_dims=None, f='exp'):
        super().__init__(input_dims, track_dims, f)

        assert self.input_dims == lengthscale.shape[0] # lengthscale per input dimension
        
        if topology == 'euclid':
            self.scaled_dist = self._scaled_dist
            self.square_scaled_dist = self._square_scaled_dist
        elif topology == 'torus_wrap':
            self.scaled_dist = self._scaled_dist_Tn
            self.square_scaled_dist = self._square_scaled_dist_Tn
        elif topology == 'torus':
            self.scaled_dist = self._scaled_dist_torus
            self.square_scaled_dist = self._square_scaled_dist_torus
        elif topology == 'sphere':
            self.scaled_dist = self._scaled_dist_sphere
            self.square_scaled_dist = self._square_scaled_dist_sphere
            lengthscale = lengthscale[:1, :] # dummy dimensions
        else:
            raise NotImplementedError('Topology is not supported.')
            
        self._lengthscale = Parameter(self.f_inv(lengthscale)) # D
        
    @property
    def lengthscale(self):
        return self.f(self._lengthscale)[None, :] # T, D
    
    @lengthscale.setter
    def lengthscale(self):
        self._lengthscale.data = self.f_inv(lengthscale)
        
    # Euclidean
    def _square_scaled_dist(self, X, Z=None):
        r"""
        Returns :math:`\|\frac{X-Z}{l}\|^2`.
        :param torch.tensor X: input of shape (neurons, samples, points, dims)
        
        """
        X, Z = self._XZ(X, Z)

        scaled_X = X / self.lengthscale # T, D
        scaled_Z = Z / self.lengthscale
        X2 = (scaled_X ** 2).sum(-1, keepdim=True)
        Z2 = (scaled_Z ** 2).sum(-1, keepdim=True)
        XZ = scaled_X.matmul(scaled_Z.permute(1, 0))
        r2 = X2 - 2 * XZ + Z2.permute(1, 0)
        return r2.clamp(min=0)

    def _scaled_dist(self, X, Z=None):
        r"""
        Returns :math:`\|\frac{X-Z}{l}\|`.
        """
        return _torch_sqrt(self._square_scaled_dist(X, Z))
    
    # Torus wrapped
    def _square_scaled_dist_Tn(self, X, Z=None):
        r"""
        Returns :math:`\|\frac{X-Z}{l}\|`.
        """
        X, Z = self._XZ(X, Z)
            
        U = (X-Z) % (2*math.pi)
        n = torch.arange(-3, 3, device=X.device)[:, None, None, None]*2*math.pi # not correct for multidimensions
        return ((torch.min(U, 2*math.pi-U).unsqueeze(0)+n)/self.lengthscale).pow(2).sum(-1)
    
    def _scaled_dist_Tn(self, X, Z=None):
        r"""
        Returns :math:`\|\frac{X-Z}{l}\|^2`.
        """
        return _torch_sqrt(self._square_scaled_dist(X, Z))
    
    # Torus cosine
    def _square_scaled_dist_torus(self, X, Z=None):
        r"""
        Returns :math:`\|\frac{X-Z}{l}\|`.
        """         
        X, Z = self._XZ(X, Z)
        U = (X[:, None, :]-Z[:, None, :].permute(1, 0, 2))
        
        return 2*((1 - torch.cos(U))/(self.lengthscale[None, ...])**2).sum(-1)
    
    def _scaled_dist_torus(self, X, Z=None):
        r"""
        Returns :math:`\|\frac{X-Z}{l}\|`.
        """         
        return _torch_sqrt(self._square_scaled_dist_torus(X, Z))
    
    # n-sphere S(n), input dimension is n
    def _square_scaled_dist_sphere(self, X, Z=None):
        r"""
        Returns :math:`\|\frac{X-Z}{l}\|`.
        """         
        X, Z = self._XZ(X, Z)
        
        return 2*((1 - (X[..., None, :]*Z[..., None, :].permute(1, 0, 2)).sum(-1))/(self.lengthscale[:, None, 0])**2)
    
    def _scaled_dist_sphere(self, X, Z=None):
        r"""
        Returns :math:`\|\frac{X-Z}{l}\|`.
        """         
        return _torch_sqrt(self._square_scaled_dist_sphere(X, Z))



class RBF_light(Isotropy_light):
    r"""
    Implementation of Radial Basis Function kernel:

        :math:`k(x,z) = \exp\left(-0.5 \times \frac{|x-z|^2}{l^2}\right).`

    .. note:: This kernel also has name `Squared Exponential` in literature.
    """
    def __init__(self, input_dims, lengthscale, track_dims=None, topology='euclid', f='exp'):
        super().__init__(input_dims, topology, lengthscale, track_dims, f)

    def forward(self, X, Z=None, diag=False):
        if diag:
            return X.new_ones(X.size(0))

        r2 = self.square_scaled_dist(X, Z)
        return torch.exp(-0.5 * r2)
    
    
    
class Exponential_light(Isotropy_light):
    r"""
    Implementation of Exponential kernel:

        :math:`k(x, z) = \sigma^2\exp\left(-\frac{|x-z|}{l}\right).`
    """
    def __init__(self, input_dims, lengthscale, track_dims=None, topology='euclid', f='exp'):
        super().__init__(input_dims, topology, lengthscale, track_dims, f)

    def forward(self, X, Z=None, diag=False):
        if diag:
            return X.new_ones(X.size(0))

        r = self.scaled_dist(X, Z)
        return torch.exp(-r)



class DSE_light(Kernel_light):
    r"""
    Implementation of Decaying Squared Exponential kernel:

        :math:`k(x, z) = \exp\left(-0.5 \times \frac{|x-\beta|^2 + |z-\beta|^2} {l_{\beta}^2}\right) \, 
        \exp\left(-0.5 \times \frac{|x-z|^2}{l^2}\right).`

    :param torch.Tensor scale_mixture: Scale mixture (:math:`\alpha`) parameter of this
        kernel. Should have size 1.
    """
    def __init__(self, input_dims, lengthscale, lengthscale_beta, beta, 
                 track_dims=None, f='exp'):
        super().__init__(input_dims, track_dims, f)
        assert self.input_dims == lengthscale.shape[0] # lengthscale per input dimension
        
        self.beta = Parameter(beta[None, :]) # T, D
        self._lengthscale_beta = Parameter(self.f_inv(lengthscale_beta)) # D
        self._lengthscale = Parameter(self.f_inv(lengthscale)) # D
        
    @property
    def lengthscale(self):
        return self.f(self._lengthscale)[None, :] # T, D
    
    @lengthscale.setter
    def lengthscale(self):
        self._lengthscale.data = self.f_inv(lengthscale)
        
    @property
    def lengthscale_beta(self):
        return self.f(self._lengthscale_beta)[None, :] # T, D
    
    @lengthscale_beta.setter
    def lengthscale_beta(self):
        self._lengthscale_beta.data = self.f_inv(lengthscale_beta)

    def forward(self, X, Z=None, diag=False):
        X, Z = self._XZ(X, Z)

        if diag:
            return torch.exp(-(((X-self.beta)/self.lengthscale_beta)**2).sum(-1))
        
        scaled_X = X / self.lengthscale # T, D
        scaled_Z = Z / self.lengthscale
        X2 = (scaled_X ** 2).sum(-1, keepdim=True)
        Z2 = (scaled_Z ** 2).sum(-1, keepdim=True)
        XZ = scaled_X.matmul(scaled_Z.permute(1, 0))
        r2 = X2 - 2 * XZ + Z2.permute(1, 0)

        return torch.exp(-0.5 * (r2 + (((X-self.beta)/self.lengthscale_beta)**2).sum(-1)[:, None] +
                                (((Z-self.beta)/self.lengthscale_beta)**2).sum(-1)[None, :]))
    
    
    
    
### functions ###
def create_kernel(kernel_tuples, kern_f, tensor_type):
    """
    """
    track_dims = 0
    kernel = 0

    constrain_dims = ()
    """
    if shared_kernel_params:
        for k, k_tuple in enumerate(kernel_tuples):

            if k_tuple[0] is not None:

                if k_tuple[0] == 'variance':
                    krn = Constant_light(variance=torch.tensor(k_tuple[1], dtype=tensor_type))

                else:
                    kernel_type = k_tuple[0]
                    topology = k_tuple[1]
                    lengthscales = k_tuple[2]

                    if topology == 'sphere':
                        constrain_dims += ((track_dims, len(lengthscales)),)

                    act = []
                    for _ in lengthscales:
                        act += [track_dims]
                        track_dims += 1

                    if kernel_type == 'RBF':
                        krn = RBF_light(input_dims=len(lengthscales), \
                                           lengthscale=torch.tensor(lengthscales[:, 0], dtype=tensor_type), \
                                           track_dims=act, topology=topology, f=kern_f)
                    elif kernel_type == 'OU':
                        krn = Exponential_light(input_dims=len(lengthscales), \
                                                   lengthscale=torch.tensor(lengthscales[:, 0], dtype=tensor_type), \
                                                   track_dims=act, topology=topology, f=kern_f)
                    elif kernel_type == 'DSE':
                        assert topology == 'euclid'
                        lengthscale_beta = k_tuple[3]
                        beta = k_tuple[4]
                        krn = DSE_light(input_dims=len(lengthscales), \
                                          lengthscale=torch.tensor(lengthscales[:, 0], dtype=tensor_type), \
                                          lengthscale_beta=torch.tensor(lengthscale_beta[:, 0], dtype=tensor_type), \
                                          beta=torch.tensor(beta[:, 0], dtype=tensor_type), \
                                          track_dims=act, f=kern_f)
                    else:
                        raise NotImplementedError('Kernel type is not supported.')

                    kernel = Product(kernel, krn) if kernel != 0 else krn

            else:
                track_dims += 1

    else:"""
    for k, k_tuple in enumerate(kernel_tuples):

        if k_tuple[0] is not None:

            if k_tuple[0] == 'variance':
                krn = Constant(variance=torch.tensor(k_tuple[1], dtype=tensor_type))

            else:
                kernel_type = k_tuple[0]
                topology = k_tuple[1]
                lengthscales = k_tuple[2]

                if topology == 'sphere':
                    constrain_dims += ((track_dims, len(lengthscales)),)

                act = []
                for _ in lengthscales:
                    act += [track_dims]
                    track_dims += 1

                if kernel_type == 'RBF':
                    krn = RBF(input_dims=len(lengthscales), \
                                      lengthscale=torch.tensor(lengthscales, dtype=tensor_type), \
                                      track_dims=act, topology=topology, f=kern_f)
                elif kernel_type == 'DSE':
                    assert topology == 'euclid'
                    lengthscale_beta = k_tuple[3]
                    beta = k_tuple[4]
                    krn = DSE(input_dims=len(lengthscales), \
                                      lengthscale=torch.tensor(lengthscales, dtype=tensor_type), \
                                      lengthscale_beta=torch.tensor(lengthscale_beta, dtype=tensor_type), \
                                      beta=torch.tensor(beta, dtype=tensor_type), \
                                      track_dims=act, f=kern_f)
                elif kernel_type == 'OU':
                    krn = Exponential(input_dims=len(lengthscales), \
                                              lengthscale=torch.tensor(lengthscales, dtype=tensor_type), \
                                              track_dims=act, topology=topology, f=kern_f)
                elif kernel_type == 'RQ':
                    scale_mixture = k_tuple[3]
                    krn = RationalQuadratic(input_dims=len(lengthscales), \
                                            lengthscale=torch.tensor(lengthscales, dtype=tensor_type), \
                                            scale_mixture=torch.tensor(scale_mixture, dtype=tensor_type), \
                                            track_dims=act, topology=topology, f=kern_f)
                elif kernel_type == 'Matern32':
                    krn = Matern32(input_dims=len(lengthscales), \
                                           lengthscale=torch.tensor(lengthscales, dtype=tensor_type), \
                                           track_dims=act, topology=topology, f=kern_f)
                elif kernel_type == 'Matern52':
                    krn = Matern52(input_dims=len(lengthscales), \
                                           lengthscale=torch.tensor(lengthscales, dtype=tensor_type), \
                                           track_dims=act, topology=topology, f=kern_f)
                elif kernel_type == 'linear':
                    assert topology == 'euclid'
                    krn = Linear(input_dims=len(lengthscales), \
                                         track_dims=act, f=kern_f)
                elif kernel_type == 'polynomial':
                    assert topology == 'euclid'
                    degree = k_tuple[3]
                    krn = Polynomial(input_dims=len(lengthscales), \
                                      bias=torch.tensor(lengthscales, dtype=tensor_type), \
                                      degree=degree, track_dims=act, f=kern_f)
                else:
                    raise NotImplementedError('Kernel type is not supported.')

            kernel = Product(kernel, krn) if kernel != 0 else krn

        else:
            track_dims += 1

    return kernel, track_dims, constrain_dims