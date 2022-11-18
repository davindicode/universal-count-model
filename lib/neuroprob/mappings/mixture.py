import numbers

import numpy as np
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter

from .. import base


### Mixtures ###
class _mappings(base._input_mapping):
    """ """

    def __init__(self, input_dims, mappings, inv_link):
        """
        Additive fields, so exponential inverse link function.
        All models should have same the input and output structure.

        :params list models: list of base models, each initialized separately
        """
        self.maps = len(mappings)
        if self.maps < 2:
            raise ValueError("Need to have more than one component mapping")

        self.inv_link = inv_link

        # covar_type = None # intially no uncertainty in model
        for m in range(len(mappings)):  # consistency check
            if m < len(mappings) - 1:  # consistency check
                if mappings[m].out_dims != mappings[m + 1].out_dims:
                    raise ValueError("Mappings do not match in output dimensions")
                if mappings[m].tensor_type != mappings[m + 1].tensor_type:
                    raise ValueError("Tensor types of mappings do not match")

        super().__init__(input_dims, mappings[0].out_dims, mappings[0].tensor_type)

        self.mappings = nn.ModuleList(mappings)

    def constrain(self):
        for m in self.mappings:
            m.constrain()

    def KL_prior(self):
        KL_prior = 0
        for m in self.mappings:
            KL_prior = KL_prior + m.KL_prior()
        return KL_prior


class mixture_model(_mappings):
    """
    Takes in identical base models to form a mixture model.
    """

    def __init__(self, input_dims, mappings, inv_link="relu"):
        super().__init__(input_dims, mappings, inv_link)

    def compute_F(self, XZ):
        """
        Note that the rates used for addition in the mixture model are evaluated as the
        quantities :math:`f(\mathbb{E}[a])`, different to :math:`\mathbb{E}[f(a)]` that is the
        posterior mean. The difference between these quantities is small when the posterior
        variance is small.

        """
        var = 0
        r_ = 0
        for m in self.mappings:
            F_mu, F_var = m.compute_F(XZ)
            if isinstance(F_var, numbers.Number) is False:
                var = var + (
                    base._inv_link_deriv[self.inv_link](F_mu) ** 2 * F_var
                )  # delta method
            r_ = r_ + m.f(F_mu)

        return r_, var


class product_model(_mappings):
    """
    Takes in identical base models to form a product model.
    """

    def __init__(self, input_dims, mappings, inv_link="relu"):
        super().__init__(input_dims, mappings, inv_link)

    def compute_F(self, XZ):
        """
        Note that the rates used for multiplication in the product model are evaluated as the
        quantities :math:`f(\mathbb{E}[a])`, different to :math:`\mathbb{E}[f(a)]` that is the
        posterior mean. The difference between these quantities is small when the posterior
        variance is small.

        The exact method would be to use MC sampling.

        :param torch.Tensor cov: input covariates of shape (sample, time, dim)
        """
        rate_ = []
        var_ = []
        for m in self.mappings:
            F_mu, F_var = m.compute_F(XZ)
            rate_.append(m.f(F_mu))
            if isinstance(F_var, numbers.Number):
                var_.append(0)
                continue
            var_.append(
                base._inv_link_deriv[self.inv_link](F_mu) ** 2 * F_var
            )  # delta method

        tot_var = 0
        rate_ = torch.stack(rate_, dim=0)
        for m, var in enumerate(var_):
            ind = torch.tensor([i for i in range(rate_.shape[0]) if i != m])
            if isinstance(var, numbers.Number) is False:
                tot_var = tot_var + (rate_[ind]).prod(dim=0) ** 2 * var

        return rate_.prod(dim=0), tot_var


class mixture_composition(_mappings):
    """
    Takes in identical base models to form a mixture model with custom functions.
    Does not support models with variational uncertainties, ignores those.
    """

    def __init__(self, input_dims, mappings, comp_func, inv_link="relu"):
        super().__init__(input_dims, mappings, inv_link)
        self.comp_func = comp_func

    def compute_F(self, XZ):
        """
        Note that the rates used for addition in the mixture model are evaluated as the
        quantities :math:`f(\mathbb{E}[a])`, different to :math:`\mathbb{E}[f(a)]` that is the
        posterior mean. The difference between these quantities is small when the posterior
        variance is small.

        """
        r_ = [base._inv_link[self.inv_link](m.compute_F(XZ)[0]) for m in self.mappings]
        return self.comp_func(r_), 0
