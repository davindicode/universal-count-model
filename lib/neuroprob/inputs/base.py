import torch.nn as nn


class _prior(nn.Module):
    """ """

    def __init__(self, p, tensor_type, dims):
        super().__init__()
        self.tensor_type = tensor_type
        self.AR_p = p
        self.dims = dims

    def validate(self, tsteps, trials):
        raise NotImplementedError

    def log_p(self, x, initial):
        raise NotImplementedError
        
        
        
class _variational(nn.Module):
    """ """

    def __init__(self, tensor_type, tsteps, dims):
        super().__init__()
        self.tensor_type = tensor_type
        self.dims = dims
        self.tsteps = tsteps

    def validate(self, tsteps, trials):
        return

    def eval_moments(self, t_lower, t_upper, net_input):
        raise NotImplementedError

    def sample(self, t_lower, t_upper, offs, samples, net_input):
        raise NotImplementedError


        
class _VI_object(nn.Module):
    """
    input VI objects
    """

    def __init__(self, dims, tensor_type):
        super().__init__()
        # self.register_buffer('dummy', torch.empty(0)) # keeping track of device
        self.dims = dims
        self.tensor_type = tensor_type

    def validate(self, tsteps, trials, batches):
        """ """
        raise NotImplementedError

    def sample(self, b, batch_info, samples, lv_input, importance_weighted):
        """
        :returns: tuple of (samples of :math:`q(z)`, KL terms)
        """
        raise NotImplementedError