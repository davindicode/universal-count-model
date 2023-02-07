import torch
from torch.distributions.distribution import Distribution


class TorchDistribution(Distribution):
    """
    Module to interface with PyTorch distributions. (Similar to Pyro distribution mixin)

    You should instead use `TorchDistribution` for new distribution classes.

    This is mainly useful for wrapping existing PyTorch distributions.
    Derived classes must first inherit from
    :class:`torch.distributions.distribution.Distribution` and then inherit
    from :class:`TorchDistributionMixin`.
    """

    def __call__(self, sample_shape=torch.Size()):
        """
        Samples a random value.

        This is reparameterized whenever possible, calling
        :meth:`~torch.distributions.distribution.Distribution.rsample` for
        reparameterized distributions and
        :meth:`~torch.distributions.distribution.Distribution.sample` for
        non-reparameterized distributions.

        :param sample_shape: the size of the iid batch to be drawn from the
            distribution.
        :type sample_shape: torch.Size
        :return: A random value or batch of random values (if parameters are
            batched). The shape of the result should be `self.shape()`.
        :rtype: torch.Tensor
        """
        return (
            self.rsample(sample_shape)
            if self.has_rsample
            else self.sample(sample_shape)
        )

    @property
    def event_dim(self):
        """
        :return: Number of dimensions of individual events.
        :rtype: int
        """
        return len(self.event_shape)

    def shape(self, sample_shape=torch.Size()):
        """
        The tensor shape of samples from this distribution.

        Samples are of shape::

          d.shape(sample_shape) == sample_shape + d.batch_shape + d.event_shape

        :param sample_shape: the size of the iid batch to be drawn from the
            distribution.
        :type sample_shape: torch.Size
        :return: Tensor shape of samples.
        :rtype: torch.Size
        """
        return sample_shape + self.batch_shape + self.event_shape