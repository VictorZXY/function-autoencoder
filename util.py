import torch.nn as nn
from matplotlib import pyplot as plt


def MLP(input_dim, out_dims):
    """
    Creates an MLP for the models.

    :param input_dim: Integer containing the dimensions of the input (= x_dim + y_dim).
    :param out_dims: An iterable containing the output sizes of the layers of the MLP.
    :return: The MLP, defined as a PyTorch neural network module.
    """

    # The MLP (last layer without a ReLU)
    layers = [nn.Linear(input_dim, out_dims[0])]

    if len(out_dims) > 1:
        layers.append(nn.ReLU())

        for i in range(1, len(out_dims) - 1):
            layers.append(nn.Linear(out_dims[i - 1], out_dims[i]))
            layers.append(nn.ReLU())

        layers.append(nn.Linear(out_dims[-2], out_dims[-1]))

    return nn.Sequential(*layers)


def plot_functions(target_x, target_y, context_x, context_y, pred_y, σ_y):
    """
    Plots the predicted mean and variance and the context points.

    :param target_x: An array of shape [batch_size, num_targets, 1] that contains
        the x values of the target points.
    :param target_y: An array of shape [batch_size, num_targets, 1] that contains
    the y values of the target points.
    :param context_x: An array of shape [batch_size, num_contexts, 1] that contains
        the x values of the context points.
    :param context_y: An array of shape [batch_size, num_contexts, 1] that contains
        the y values of the context points.
    :param pred_y: An array of shape [batch_size, num_targets, 1] that contains the
        predicted means of the y values at the target points in target_x.
    :param σ: An array of shape [batch_size, num_targets, 1] that contains the
        predicted std. dev. of the y values at the target points in target_x.
    """

    # Plot everything
    plt.plot(target_x[0], pred_y[0], 'tab:blue', linewidth=2)
    plt.plot(target_x[0], target_y[0], 'k', linewidth=2, alpha=0.25)
    plt.plot(context_x[0], context_y[0], 'kP', markersize=6)
    plt.fill_between(
        target_x[0, :, 0],
        pred_y[0, :, 0] - 1.96 * σ_y[0, :, 0],
        pred_y[0, :, 0] + 1.96 * σ_y[0, :, 0],
        alpha=0.2,
        facecolor='tab:blue',
        interpolate=True)

    # Make the plot pretty
    plt.yticks([-2, 0, 2], fontsize=12)
    plt.xticks([-2, 0, 2], fontsize=12)
    plt.ylim([-2, 2])
    ax = plt.gca()
    plt.show()
