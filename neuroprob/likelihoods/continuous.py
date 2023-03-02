import numbers

import numpy as np
import torch
from torch.nn.parameter import Parameter

from .. import distributions as dist

from . import base


# noise distribution
class Gaussian(base._likelihood):
    """
    Gaussian noise likelihood.
    Analogous to Factor Analysis.
    """

    def __init__(self, out_dims, inv_link, log_var, tensor_type=torch.float):
        """
        :param np.ndarray log_var: log observation noise of shape (out_inds,) or (1,) if parameters tied
        """
        super().__init__(1.0, out_dims, out_dims, inv_link, tensor_type)  # dummy tbin
        if log_var is not None:
            self.register_parameter(
                "log_var", Parameter(log_var.type(self.tensor_type))
            )
            self.dispersion_mapping = None

    def eval_dispersion_mapping(self, XZ, samples, out_inds):
        """
        Posterior predictive mean of the dispersion model.
        """
        if self.dispersion_mapping.MC_only:
            dh = self.dispersion_mapping.sample_F(XZ, samples, out_inds)[:, out_inds, :]
        else:
            disp, disp_var = self.dispersion_mapping.compute_F(XZ)
            dh = base.mc_gen(disp, disp_var, samples, out_inds)
        return self.dispersion_mapping_f(dh).mean(
            0
        )  # watch out for underflow or overflow here

    def sample_helper(self, h, b, out_inds, samples):
        """
        NLL helper function for MC sample evaluation.
        """
        batch_edge, _, _ = self.batch_info
        rates = self.f(h)  # watch out for underflow or overflow here
        spikes = self.all_spikes[:, out_inds, batch_edge[b] : batch_edge[b + 1]].to(
            self.tbin.device
        )
        # self.spikes[b][:, out_inds, self.filter_len-1:].to(self.tbin.device)
        return rates, spikes

    def nll(self, rates, spikes, noise_var):
        """
        Gaussian likelihood for activity train
        samples introduces a sample dimension from the left
        F_mu has shape (samples, out_dims, timesteps)
        if F_var = 0, we don't expand by samples in the sample dimension
        """
        nll = 0.5 * (
            torch.log(noise_var) + ((spikes - rates) ** 2) / noise_var
        ) + 0.5 * torch.log(torch.tensor(2 * np.pi))
        return nll.sum(1)

    def objective(self, F_mu, F_var, XZ, b, out_inds, samples=10, mode="MC"):
        """
        Computes the terms for variational expectation :math:`\mathbb{E}_{q(f)q(z)}[]`, which
        can be used to compute different likelihood objectives.
        The returned tensor will have sample dimension as MC over :math:`q(z)`, depending
        on the evaluation mode will be MC or GH or exact over the likelihood samples. This
        is all combined in to the same dimension to be summed over. The weights :math:`w_s`
        are the quadrature weights or equal weights for MC, with appropriate normalization.

        :param int samples: number of MC samples or GH points (exact will ignore and give 1)

        :returns: negative likelihood term of shape (samples, timesteps), sample weights (samples, 1
        :rtype: tuple of torch.tensors
        """
        if self.dispersion_mapping is None:
            if self.log_var.shape[0] == 1:
                log_var = self.log_var.expand(1, len(out_inds))[..., None]
            else:
                log_var = self.log_var[None, out_inds, None]
        else:
            log_var = self.eval_dispersion_mapping(XZ, samples, out_inds)

        if self.inv_link == "identity" and F_var is not None:  # exact
            batch_edge = self.batch_info[0]
            spikes = self.all_spikes[:, out_inds, batch_edge[b] : batch_edge[b + 1]].to(
                self.tbin.device
            )
            if isinstance(F_var, numbers.Number):
                F_var = 0
            else:
                F_var = F_var[:, out_inds, :]
            noise_var = torch.exp(log_var) + F_var

            nll = 0.5 * (
                torch.log(noise_var)
                + ((spikes - F_mu) ** 2) / noise_var
                + F_var / noise_var
            ) + 0.5 * torch.log(torch.tensor(2 * np.pi))
            ws = torch.tensor(1 / F_mu.shape[0])
            return nll.sum(1), ws
        # elif self.inv_link == 'exp' # exact

        if mode == "MC":
            h = base.mc_gen(F_mu, F_var, samples, out_inds)
            rates, spikes = self.sample_helper(h, b, out_inds, samples)
            ws = torch.tensor(1.0 / rates.shape[0])
        elif mode == "GH":
            h, ws = base.gh_gen(F_mu, F_var, samples, out_inds)
            rates, spikes = self.sample_helper(h, b, out_inds, samples)
            ws = ws[:, None]
        else:
            raise NotImplementedError

        return self.nll(rates, spikes, torch.exp(log_var)), ws

    def sample(self, rate, out_inds=None, XZ=None, rng=None):
        """
        Sample activity trains [trial, out_inds, timestep]
        """
        if rng is None:
            rng = np.random.default_rng()

        out_inds = self._validate_out_inds(out_inds)
        rate_ = rate[:, out_inds, :]

        if self.dispersion_mapping is None:
            if self.log_var.shape[0] == 1:
                log_var = (
                    self.log_var.expand(1, len(out_inds)).data[..., None].cpu().numpy()
                )
            else:
                log_var = self.log_var.data[None, out_inds, None].cpu().numpy()
        else:
            samples = rate.shape[0]
            log_var = self.eval_dispersion_mapping(XZ, samples, out_inds)[
                None, ...
            ].expand(rate.shape[0], *rate_.shape[1:])

        act = rate_ + np.exp(log_var / 2.0) * rng.normal(
            size=(rate.shape[0], len(out_inds), rate.shape[-1])
        )
        return act


class hGaussian(Gaussian):
    """
    Gaussian noise likelihood.
    Analogous to Factor Analysis.
    """

    def __init__(self, out_dims, inv_link, dispersion_mapping, tensor_type=torch.float):
        """
        :param _input_mapping dispersion_mapping: function mapping input to log observation noise
        """
        super().__init__(out_dims, inv_link, None, tensor_type)
        self.dispersion_mapping = dispersion_mapping
        self.dispersion_mapping_f = lambda x: x
