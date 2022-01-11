import collections

import numpy as np
import torch

# The neural process (NP) takes as input a `NPRegressionDescription` namedtuple
# with fields:
#     context_x: A tensor containing the context observations.
#     context_y: A tensor containing the evaluations of the context observations.
#     target_x: A tensor containing the target points to be predicted.
#     target_y: A tensor containing the ground truth for the targets to be predicted.
#     num_total_points: A vector containing a scalar that describes the total
#         number of datapoints used (context + target).
#     num_context_points: A vector containing a scalar that describes the number
#         of datapoints used as context.

NPRegressionDescription = collections.namedtuple(
    'NPRegressionDescription',
    ('context_x', 'context_y', 'target_x', 'target_y', 'num_total_points', 'num_context_points')
)


# The GPCurvesReader returns the newly sampled data in this format at each iteration

class GPCurvesReader(object):
    """
    Generates curves using a Gaussian Process (GP).

    Supports vector inputs (x) and vector outputs (y). Kernel is mean-squared
    exponential, using the x-value l2 coordinate distance scaled by some factor
    chosen randomly in a range. Outputs are independent gaussian processes.
    """

    def __init__(self,
                 batch_size,
                 max_num_context,
                 x_dim=1,
                 y_dim=1,
                 l1_scale=0.6,
                 sigma_scale=1.0,
                 random_kernel_parameters=True,
                 testing=False):
        """
        Creates a regression dataset of functions sampled from a GP.

        :param batch_size: An integer for the size of the batch.
        :param max_num_context: The max number of observations in the context.
        :param x_dim: Integer >= 1 for the dimensions of the x values.
        :param y_dim: Integer >= 1 for the dimensions of the y values.
        :param l1_scale: Float; typical scale for kernel distance function.
        :param sigma_scale: Float; typical scale for standard deviation.
        :param random_kernel_parameters: If `True`, the kernel parameters
            (l1 and sigma) will be sampled uniformly within [0.1, l1_scale] and
            [0.1, sigma_scale].
        :param testing: Boolean that indicates whether we are testing. If so
            there are more targets for visualization.
        """

        self._batch_size = batch_size
        self._max_num_context = max_num_context
        self._x_dim = x_dim
        self._y_dim = y_dim
        self._l1_scale = l1_scale
        self._sigma_scale = sigma_scale
        self._random_kernel_parameters = random_kernel_parameters
        self._testing = testing

    def _gaussian_kernel(self, xdata, l1, sigma_f, sigma_noise=2e-2):
        """
        Applies the Gaussian kernel to generate curve data.

        :param xdata: Tensor of shape [batch_size, num_total_points, x_dim] with
            the values of the x-axis data.
        :param l1: Tensor of shape [batch_size, y_dim, x_dim], the scale parameter
            of the Gaussian kernel.
        :param sigma_f: Tensor of shape [batch_size, y_dim], the magnitude of the
            standard deviation.
        :param sigma_noise: Float, standard deviation of the noise added for stability.
        :return: The kernel, a float tensor of shape
            [batch_size, y_dim, num_total_points, num_total_points].
        """

        num_total_points = xdata.size(dim=1)

        # Expand and take the difference
        # [batch_size, 1, num_total_points, x_dim]
        xdata1 = torch.unsqueeze(xdata, dim=1)
        # [batch_size, num_total_points, 1, x_dim]
        xdata2 = torch.unsqueeze(xdata, dim=2)
        # [batch_size, num_total_points, num_total_points, x_dim]
        diff = xdata1 - xdata2

        # [batch_size, y_dim, num_total_points, num_total_points, x_dim]
        norm = torch.square(diff[:, None, :, :, :] / l1[:, :, None, None, :])

        # [batch_size, data_dim, num_total_points, num_total_points]
        norm = torch.sum(norm, dim=-1)

        # [batch_size, y_dim, num_total_points, num_total_points]
        kernel = torch.square(sigma_f)[:, :, None, None] * torch.exp(-0.5 * norm)

        # Add some noise to the diagonal to make the Cholesky work.
        kernel += (sigma_noise ** 2) * torch.eye(num_total_points)

        return kernel

    def generate_curves(self):
        """
        Builds the op delivering the data.
        Generated functions are float32 with x values between -2 and 2.

        :return: An `NPRegressionDescription` namedtuple.
        """

        num_context = torch.randint(3, self._max_num_context, ())

        # If we are testing we want to have more targets and have them evenly
        # distributed in order to plot the function.
        if self._testing:
            num_target = 400
            num_total_points = num_target
            x_values = torch.unsqueeze(torch.arange(-2., 2., 0.01), dim=0)
            x_values = x_values.repeat(self._batch_size, 1)
            x_values = torch.unsqueeze(x_values, dim=-1)
        # During training the number of target points and their x-positions are
        # selected at random
        else:
            num_target = torch.randint(0, self._max_num_context - num_context, ())
            num_total_points = num_context + num_target
            x_values = (2 - (-2)) * torch.rand(self._batch_size, num_total_points, self._x_dim) - 2

        # Set kernel parameters
        # Either choose a set of random parameters for the mini-batch
        if self._random_kernel_parameters:
            l1 = (self._l1_scale - 0.1) * torch.rand(self._batch_size, self._y_dim, self._x_dim) + 0.1
            sigma_f = (self._sigma_scale - 0.1) * torch.rand(self._batch_size, self._y_dim) + 0.1
        # Or use the same fixed parameters for all mini-batches
        else:
            l1 = torch.ones(self._batch_size, self._y_dim, self._x_dim) * self._l1_scale
            sigma_f = torch.ones(self._batch_size, self._y_dim) * self._sigma_scale

        # Pass the x_values through the Gaussian kernel
        # [batch_size, y_dim, num_total_points, num_total_points]
        kernel = self._gaussian_kernel(x_values, l1, sigma_f)

        # Calculate Cholesky, using double precision for better stability:
        cholesky = torch.linalg.cholesky(kernel.double()).float()

        # Sample a curve
        # [batch_size, y_dim, num_total_points, 1]
        y_values = torch.matmul(
            cholesky,
            torch.randn(self._batch_size, self._y_dim, num_total_points, 1))

        # [batch_size, num_total_points, y_dim]
        y_values = torch.transpose(torch.squeeze(y_values, 3), 1, 2)

        if self._testing:
            # Select the targets
            target_x = x_values
            target_y = y_values

            # Select the observations
            idx = torch.arange(num_target).numpy()
            np.random.shuffle(idx)
            idx = torch.tensor(idx)
            context_x = torch.index_select(x_values, dim=1, index=idx[:num_context])
            context_y = torch.index_select(y_values, dim=1, index=idx[:num_context])
        else:
            # Select the targets which will consist of the context points as well as
            # some new target points
            target_x = x_values[:, :num_target + num_context, :]
            target_y = y_values[:, :num_target + num_context, :]

            # Select the observations
            context_x = x_values[:, :num_context, :]
            context_y = y_values[:, :num_context, :]

        return NPRegressionDescription(
            context_x=context_x,
            context_y=context_y,
            target_x=target_x,
            target_y=target_y,
            num_total_points=target_x.size(dim=1),
            num_context_points=num_context)
