import torch
import torch.nn as nn
from torch.nn.parameter import Parameter


# GP means
class decaying_exponential(nn.Module):
    def __init__(self, dims, a, b, learnable=[True, True]):
        """
        :param int neurons: the number of output dimensios
        :param float a: initial value for all :math:`a` tensor entries
        :param float b: initial value for all :math:`b` tensor entries
        :param list learnable: list of booleans indicating whether :math:`a` and :math:`b` are learnable
        """
        super().__init__()
        if learnable[0]:
            self.register_parameter("a", Parameter(a * torch.ones(1, dims, 1)))
        else:
            self.register_buffer("a", a * torch.ones(1, dims, 1))
        if learnable[1]:
            self.register_parameter("b", Parameter(b * torch.ones(1, dims, 1)))
        else:
            self.register_buffer("b", b * torch.ones(1, dims, 1))

    def forward(self, input):
        """
        :param torch.Tensor input: input of shape (K, T, D), only the first dimension in D is considered as time
        :returns: output of shape (K, N, T)
        :rtype: torch.Tensor
        """
        return self.a * torch.exp(-input[:, None, :, 0] / self.b)
