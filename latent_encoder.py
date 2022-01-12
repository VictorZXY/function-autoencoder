import torch
import torch.nn as nn

from util import MLP


class LatentEncoder(nn.Module):
    """The Latent Encoder."""

    def __init__(self, x_dim, y_dim, out_dims, decoder,
                 deterministic_encoder=None):
        """
        NP latent encoder.

        :param x_dim: Integer >= 1 for the dimensions of the x values.
        :param y_dim: Integer >= 1 for the dimensions of the y values.
        :param out_dims: An iterable containing the output sizes of the encoding
            MLP, excluding the final layer for the latent output.
        :param decoder: The decoder model.
        :param deterministic_encoder: The deterministic encoder model (if any).
        """

        super().__init__()

        self.out_dims = out_dims
        self.d = decoder.latent_dim
        self.f = decoder
        self.h = deterministic_encoder

        # Internal MLP layers
        self.g = MLP(x_dim + y_dim, out_dims)

        # Have further MLP layers that map to the parameters of the Gaussian latent
        # First apply intermediate Linear + ReLU layer
        self.g_latent = nn.Sequential(
            nn.Linear(out_dims[-1], (out_dims[-1] + self.d) // 2),
            nn.ReLU()
        )

        # Then apply further Linear layers to output latent μ and σ
        self.g_μ = nn.Linear((out_dims[-1] + self.d) // 2, self.d)
        self.g_σ = nn.Linear((out_dims[-1] + self.d) // 2, self.d)

    def forward(self, x, y):
        """
        Encodes the inputs into one latent distribution.

        :param x: Tensor of shape [batch_size, num_observations, x_dim]. For this
            1D regression task this corresponds to the x-values.
        :param y: Tensor of shape [batch_size, num_observations, y_dim]. For this
            1D regression task this corresponds to the y-values.
        :return: A normal distribution over tensors of shape [batch_size, latent_dim].
        """

        # Concatenate x and y along the filter axes
        encoder_input = torch.cat([x, y], dim=-1)

        # Get the shapes of the input and reshape to parallelise across observations
        batch_size, num_observations, input_dim = encoder_input.size()
        encoder_input = torch.reshape(encoder_input, (-1, input_dim))
        assert encoder_input.size(dim=0) == batch_size * num_observations

        # Pass the input through MLP
        output = self.g(encoder_input)

        # Aggregator: take the mean over all points
        output = torch.reshape(output, (batch_size, -1, self.out_dims[-1]))
        output = torch.mean(output, dim=1)

        # Pass the output through the further MLP layers that map to the parameters
        # of the Gaussian latent
        output = self.g_latent(output)
        μ = self.g_μ(output)
        σ_raw = self.g_σ(output)

        # Bound the standard deviation
        σ = torch.nn.functional.softplus(σ_raw)

        return torch.distributions.normal.Normal(loc=μ, scale=σ)

    def loglik_lb(self, context_x, context_y, num_targets, target_x,
                  target_y=None):
        """
        Returns the lower bound of the log likelihood of the predicted target points.

        :param context_x: Tensor of shape [batch_size, num_contexts, x_dim]. Contains
            the x values of the context points.
        :param context_y: Tensor of shape [batch_size, num_contexts, y_dim]. Contains
            the y values of the context points.
        :param num_targets: Number of target points.
        :param target_x: Tensor of shape [batch_size, num_targets, x_dim]. Contains
            the x values of the target points.
        :param target_y: Tensor of shape [batch_size, num_targets, y_dim]. Contains
            the ground truth y values of the target y.
        :return: At training time when target_y is available, return the lower bound
            of the log likelihood of target_y given the predicted distribution,
            which is a tensor of shape [batch_size, num_targets].
            At test time when target_y is unavailable, return None.
        """

        # Pass query through the encoder and the decoder
        Z = self(context_x, context_y)

        # For training where target_y is available, use the targets for latent encoder.
        # Note that targets contain contexts by design.
        if target_y is not None:
            Ztilde = self(target_x, target_y)
            z = Ztilde.sample()
        # For testing where target_y is unavailable, use the contexts for latent encoder.
        else:
            z = Z.sample()
        z = torch.unsqueeze(z, dim=1).repeat(1, num_targets, 1)

        # Concatenate latent variable and the deterministic representation, if
        # a deterministic encoder is used.
        if self.h is not None:
            r = self.h(context_x, context_y)
            r = torch.unsqueeze(r, dim=1).repeat(1, num_targets, 1)
            representation = torch.cat([r, z], dim=-1)
        else:
            representation = z

        # Get the generated distribution from the decoder
        dist, μ, σ = self.f(representation, target_x)

        # If we want to calculate the log likelihood for training, we will make
        # use of the target_y. At test time, the target_y is not available so we
        # return None.
        if target_y is not None:
            Ztilde = self(target_x, target_y)
            kl = torch.sum(
                torch.distributions.kl.kl_divergence(Ztilde, Z),
                dim=-1, keepdim=True)
            kl = kl.repeat(1, num_targets)
            ll = dist.log_prob(target_y)
            return ll - kl
        else:
            return None
