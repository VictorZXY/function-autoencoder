import torch
import torch.nn as nn

from util import MLP


class Decoder(nn.Module):
    """The Decoder."""

    def __init__(self, x_dim, y_dim, out_dims, latent_dim=0, deterministic_dim=0):
        """
        (C)NP decoder.

        :param out_dims: An iterable containing the output sizes of the encoding MLP.
            The output size of the last layer must be 2 * y_dim.
        :param latent_dim: Integer >= 1 for the dimensions of the latent variable.
        :param deterministic_dim: Integer >= 1 for the dimensions of the deterministic
            representation.
        """

        super().__init__()

        assert y_dim == out_dims[-1] // 2
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.out_dims = out_dims
        self.latent_dim = latent_dim
        self.deterministic_dim = deterministic_dim
        self.f = MLP(latent_dim + deterministic_dim + x_dim, out_dims)

    def forward(self, representation, x):
        """
        Decodes the individual targets.

        :param representation: The representation of the context for target predictions.
            Tensor of shape [batch_size, num_targets, latent_dim + deterministic_dim].
        :param x: The x values for the target query. Tensor of shape
            [batch_size, num_targets, x_dim].

        :returns:
            dist: A multivariate Gaussian over the target points. A distribution
                over tensors of shape [batch_size, num_targets, y_dim].
            μ: The mean of the multivariate Gaussian. Tensor of shape
                [batch_size, num_targets, y_dim].
            σ: The standard deviation of the multivariate Gaussian. Tensor of shape
                [batch_size, num_targets, y_dim].
        """

        # Concatenate representation and x
        decoder_input = torch.cat([representation, x], dim=-1)

        # Get the shapes of the input and reshape to parallelise across observations
        batch_size, num_observations, input_dim = decoder_input.size()
        decoder_input = torch.reshape(decoder_input, (-1, input_dim))
        assert decoder_input.size(dim=0) == batch_size * num_observations

        # Pass the input through MLP
        μσ = self.f(decoder_input)

        # Get the mean and the standard deviation
        μσ = torch.reshape(μσ, (batch_size, -1, self.out_dims[-1]))
        μ, σ_raw = torch.split(μσ, self.y_dim, dim=-1)

        # Bound the standard deviation
        σ = 0.1 + 0.9 * torch.nn.functional.softplus(σ_raw)

        # Get the distribution
        dist = torch.distributions.independent.Independent(
            torch.distributions.normal.Normal(loc=μ, scale=σ), 1)

        return dist, μ, σ
