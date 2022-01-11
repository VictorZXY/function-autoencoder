import torch
import torch.nn as nn

from util import MLP


class DeterministicEncoder(nn.Module):
    """The Deterministic Encoder."""

    def __init__(self, x_dim, y_dim, out_dims, decoder=None):
        """
        (C)NP deterministic encoder.

        :param x_dim: Integer >= 1 for the dimensions of the x values.
        :param y_dim: Integer >= 1 for the dimensions of the y values.
        :param out_dims: An iterable containing the output sizes of the encoding MLP.
        :param decoder: The decoder model, which is only passed to the deterministic
            encoder when it acts as a stand-alone CNP encoder (i.e. no latent
            encoder involved).
        """

        super().__init__()

        if decoder:
            self.d = decoder.deterministic_dim
            self.f = decoder
        else:
            self.d = out_dims[-1]

        self.h = MLP(x_dim + y_dim, out_dims)

    def forward(self, x, y):
        """
        Encodes the inputs into one representation.

        :param x: Tensor of shape [batch_size, num_observations, x_dim]. For this
            1D regression task this corresponds to the x-values.
        :param y: Tensor of shape [batch_size, num_observations, y_dim]. For this
            1D regression task this corresponds to the y-values.
        :return: The encoded representation. Tensor of shape [batch_size, deterministic_dim]
        """

        # Concatenate x and y along the filter axes
        encoder_input = torch.cat([x, y], dim=-1)

        # Get the shapes of the input and reshape to parallelise across observations
        batch_size, num_observations, input_dim = encoder_input.size()
        encoder_input = torch.reshape(encoder_input, (-1, input_dim))
        assert encoder_input.size(dim=0) == batch_size * num_observations

        # Pass the input through MLP
        output = self.h(encoder_input)

        # Aggregator: take the mean over all points
        output = torch.reshape(output, (batch_size, -1, self.d))
        return torch.mean(output, dim=1)

    def loglik(self, context_x, context_y, num_targets, target_x, target_y=None):
        """
        Returns the log likelihood of the predicted target points. Only to be
        called when this acts as a stand-alone CNP encoder (i.e. no latent encoder
        involved).

        :param context_x: Tensor of shape [batch_size, num_contexts, x_dim]. Contains
            the x values of the context points.
        :param context_y: Tensor of shape [batch_size, num_contexts, y_dim]. Contains
            the y values of the context points.
        :param num_targets: Number of target points.
        :param target_x: Tensor of shape [batch_size, num_targets, x_dim]. Contains
            the x values of the target points.
        :param target_y: Tensor of shape [batch_size, num_targets, y_dim]. Contains
            the ground truth y values of the target y.
        :return: At training time when target_y is available, return the log likelihood
            of target_y given the predicted distribution, which is a tensor of shape
            [batch_size, num_targets].
            At test time when target_y is unavailable, return None.
        """

        if self.f is not None and target_y is not None:
            r = self(context_x, context_y)
            r = torch.unsqueeze(r, dim=1).repeat(1, num_targets, 1)
            dist, μ, σ = self.f(r, target_x)
            return dist.log_prob(target_y)
        else:
            return None
