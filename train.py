import torch
import torch.optim as optim

from data_generator import GPCurvesReader
from decoder import Decoder
from deterministic_encoder import DeterministicEncoder
from latent_encoder import LatentEncoder
from util import plot_functions

TOTAL_EPOCHS = 100000
REFRESH_DATA_AFTER = 100
PLOT_AFTER = 10000
MAX_CONTEXT_POINTS = 50
HIDDEN_SIZE = 128
ENCODER_LAYERS = 4
DECODER_LAYERS = 2
MODEL_TYPE = 'Latent + Deterministic'  # ['Latent', 'Deterministic', 'Latent + Deterministic']
RANDOM_KERNEL_PARAMS = True

# Train dataset
dataset_train = GPCurvesReader(
    batch_size=16, max_num_context=MAX_CONTEXT_POINTS,
    random_kernel_parameters=RANDOM_KERNEL_PARAMS)

# Test dataset
dataset_test = GPCurvesReader(
    batch_size=1, max_num_context=MAX_CONTEXT_POINTS, testing=True,
    random_kernel_parameters=RANDOM_KERNEL_PARAMS)
data_test = dataset_test.generate_curves()

# Sizes of the layers of the MLPs for the encoders and decoder
# The final output layer of the decoder outputs two values, one for the mean and
# one for the standard deviation of the prediction at the target location
latent_encoder_out_dims = [HIDDEN_SIZE] * ENCODER_LAYERS
representation_dim = HIDDEN_SIZE
deterministic_encoder_out_dims = [HIDDEN_SIZE] * ENCODER_LAYERS
decoder_out_dims = [HIDDEN_SIZE] * DECODER_LAYERS + [2]

if __name__ == '__main__':
    # Define the model and the loss:
    # Latent encoder only
    if MODEL_TYPE == 'Latent':
        model = LatentEncoder(
            x_dim=1,
            y_dim=1,
            out_dims=latent_encoder_out_dims,
            decoder=Decoder(
                x_dim=1,
                y_dim=1,
                out_dims=decoder_out_dims,
                latent_dim=representation_dim
            )
        )
    # Deterministic encoder only
    elif MODEL_TYPE == 'Deterministic':
        model = DeterministicEncoder(
            x_dim=1,
            y_dim=1,
            out_dims=latent_encoder_out_dims,
            decoder=Decoder(
                x_dim=1,
                y_dim=1,
                out_dims=decoder_out_dims,
                deterministic_dim=representation_dim
            )
        )
    # Latent encoder + deterministic encoder
    elif MODEL_TYPE == 'Latent + Deterministic':
        model = LatentEncoder(
            x_dim=1,
            y_dim=1,
            out_dims=latent_encoder_out_dims,
            decoder=Decoder(
                x_dim=1,
                y_dim=1,
                out_dims=decoder_out_dims,
                latent_dim=representation_dim,
                deterministic_dim=representation_dim
            ),
            deterministic_encoder=DeterministicEncoder(
                x_dim=1,
                y_dim=1,
                out_dims=deterministic_encoder_out_dims
            )
        )
    else:
        raise NameError("MODEL_TYPE not among ['Latent', 'Deterministic', 'Latent + Deterministic']")

    # Set up the optimizer
    optimiser = optim.Adam(model.parameters(), lr=1e-4)

    # Train and plot
    for epoch in range(TOTAL_EPOCHS):
        optimiser.zero_grad()

        # Get training datapoints
        if epoch % REFRESH_DATA_AFTER == 0:
            data_train = dataset_train.generate_curves()
            train_context_x = data_train.context_x
            train_context_y = data_train.context_y
            train_target_x = data_train.target_x
            train_target_y = data_train.target_y
            train_num_targets = data_train.num_total_points

        if MODEL_TYPE == 'Latent' or MODEL_TYPE == 'Latent + Deterministic':
            loss = -torch.mean(model.loglik_lb(context_x=train_context_x,
                                               context_y=train_context_y,
                                               num_targets=train_num_targets,
                                               target_x=train_target_x,
                                               target_y=train_target_y))
        else:
            loss = -torch.mean(model.loglik(context_x=train_context_x,
                                            context_y=train_context_y,
                                            num_targets=train_num_targets,
                                            target_x=train_target_x,
                                            target_y=train_target_y))

        loss.backward()
        optimiser.step()

        # Plot the predictions in `PLOT_AFTER` intervals
        if epoch % PLOT_AFTER == 0:
            # Get testing datapoints
            test_context_x = data_test.context_x
            test_context_y = data_test.context_y
            test_target_x = data_test.target_x
            test_target_y = data_test.target_y
            test_num_targets = data_test.num_total_points

            # Get the log likelihood and representation
            if MODEL_TYPE == 'Latent':
                loglik = torch.mean(model.loglik_lb(context_x=test_context_x,
                                                    context_y=test_context_y,
                                                    num_targets=test_num_targets,
                                                    target_x=test_target_x,
                                                    target_y=test_target_y).detach())
                Z = model(test_context_x, test_context_y)
                z = Z.sample().detach()
                z = torch.unsqueeze(z, dim=1).repeat(1, test_num_targets, 1)
                representation = z
            elif MODEL_TYPE == 'Latent + Deterministic':
                loglik = torch.mean(model.loglik_lb(context_x=test_context_x,
                                                    context_y=test_context_y,
                                                    num_targets=test_num_targets,
                                                    target_x=test_target_x,
                                                    target_y=test_target_y).detach())
                Z = model(test_context_x, test_context_y)
                z = Z.sample().detach()
                z = torch.unsqueeze(z, dim=1).repeat(1, test_num_targets, 1)
                r = model.h(test_context_x, test_context_y).detach()
                r = torch.unsqueeze(r, dim=1).repeat(1, test_num_targets, 1)
                representation = torch.cat([r, z], dim=-1)
            else:
                loglik = torch.mean(model.loglik(context_x=test_context_x,
                                                 context_y=test_context_y,
                                                 num_targets=test_num_targets,
                                                 target_x=test_target_x,
                                                 target_y=test_target_y).detach())
                representation = model(test_context_x, test_context_y).detach()

            dist, μ, σ = model.f(representation, test_target_x)

            print(f'Iteration: {epoch}, log likelihood: {loglik}')

            # Plot the prediction and the context
            plot_functions(target_x=test_target_x.detach(),
                           target_y=test_target_y.detach(),
                           context_x=test_context_x.detach(),
                           context_y=test_context_y.detach(),
                           pred_y=μ.detach(),
                           σ_y=σ.detach())
