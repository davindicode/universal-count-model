import math
import types

from numbers import Number

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.parameter import Parameter

from tqdm.autonotebook import tqdm

from . import distributions


# Link functions
_link_functions = {
    "exp": torch.log,
    "softplus": lambda x: torch.where(
        x > 30, x, torch.log(torch.exp(x) - 1.0)
    ),  # | log(1+exp(30)) - 30 | < 1e-10, numerically stabilized
    "relu": lambda x: x,
    "identity": lambda x: x,
    "sigmoid": lambda x: torch.log(x) - torch.log(1.0 - x),
}

_inv_link_functions = {
    "exp": torch.exp,
    "softplus": F.softplus,
    "relu": lambda x: torch.clamp(x, min=0),
    "identity": lambda x: x,
    "sigmoid": torch.sigmoid,
}


def safe_sqrt(x, eps=1e-8):
    """
    A convenient function to avoid the NaN gradient issue of sqrt() at 0.
    Ref: https://github.com/pytorch/pytorch/issues/2421
    """
    return torch.sqrt(x + eps)


def safe_log(x, eps=1e-8):
    """
    A convenient function to avoid NaN at small x
    """
    return torch.log(x + eps)


class _data_object(nn.Module):
    """
    Object that will take in data or generates data (leaf node objects in model graph).
    """

    def __init__(self):
        super().__init__()

    def setup_batching(self, batch_info, tsteps, trials):
        """
        :param int/List batch_info: batch size if scalar, else list of tuple (batch size, batch link), where the
                                    batch link is a boolean indicating continuity from previous batch
        :param int tsteps: total time steps of data to be batched
        :param int trials: total number of trials in the spike data
        """
        self.tsteps = tsteps
        self.trials = trials

        ### batching ###
        if type(batch_info) == list:  # not continuous data
            self.batches = len(batch_info)
            batch_size = [b["size"] for b in batch_info]
            batch_link = [b["linked"] for b in batch_info]
            batch_initial = [b["initial"] for b in batch_info]

        else:  # number
            batch_size = batch_info
            self.batches = int(np.ceil(self.tsteps / batch_size))
            n = self.batches - 1
            fin_bs = self.tsteps - n * batch_size
            batch_size = [batch_size] * n + [fin_bs]
            batch_link = [True] * self.batches
            batch_initial = [False] * self.batches

        batch_link[0] = False  # first batch
        batch_initial[0] = True
        batch_edge = list(np.cumsum(np.array([0] + batch_size)))
        self.batch_info = (batch_edge, batch_link, batch_initial)
