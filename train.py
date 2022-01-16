import pickle

import numpy as np
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from data_generator import GPCurvesReader
from decoder import Decoder
from deterministic_encoder import DeterministicEncoder
from latent_encoder import LatentEncoder
from util import plot_functions

TOTAL_EPOCHS = 100001
REFRESH_DATA_AFTER = {
    'Latent': 5000,  # 1000000 if no data refresh for latent encoder
    'Deterministic': 100,
    'LatentDeterministic': 500
}
PLOT_AFTER = 10000
MAX_CONTEXT_POINTS = [10, 50, 100]
TRAIN_BATCH_SIZES = {
    'Latent': 16,  # 256 for no data refresh
    'Deterministic': 16,
    'LatentDeterministic': 16
}
TEST_BATCH_SIZE = 1
LAYER_DIM = 128
ENCODER_NUM_LAYERS = 4
DECODER_NUM_LAYERS = 2
MODEL_TYPES = ['Latent', 'Deterministic', 'LatentDeterministic']
RANDOM_KERNEL_PARAMS = True


def train(total_epochs, refresh_data_after, plot_after, max_context_points,
          train_batch_size, test_batch_size, layer_dim, encoder_num_layers,
          decoder_num_layers, model_type, random_kernel_params, seed=2022):
    # initialise seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Training dataset
    dataset_train = GPCurvesReader(
        batch_size=train_batch_size, max_num_context=max_context_points,
        random_kernel_parameters=random_kernel_params)

    # Test dataset
    dataset_test = GPCurvesReader(
        batch_size=test_batch_size, max_num_context=max_context_points,
        testing=True, random_kernel_parameters=random_kernel_params)
    data_test = dataset_test.generate_curves()

    # Sizes of the layers of the MLPs for the encoders and decoder
    # The final output layer of the decoder outputs two values, one for the mean and
    # one for the standard deviation of the prediction at the target location
    latent_encoder_out_dims = [layer_dim] * encoder_num_layers
    representation_dim = layer_dim
    deterministic_encoder_out_dims = [layer_dim] * encoder_num_layers
    decoder_out_dims = [layer_dim] * decoder_num_layers + [2]

    # Define the model:
    # Latent encoder only
    if model_type == 'Latent':
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
    elif model_type == 'Deterministic':
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
    elif model_type == 'LatentDeterministic':
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
        raise NameError("model_type not among ['Latent', 'Deterministic', 'LatentDeterministic']")

    # Set up the optimizer
    optimiser = optim.Adam(model.parameters(), lr=1e-4)

    # Set up TensorBoard logs
    tensorboard_comment = f'{model_type}_{refresh_data_after}_{max_context_points}'
    tensorboard_writer = SummaryWriter(log_dir='results/TensorBoard',
                                       comment=tensorboard_comment)

    # Train and plot
    loglik_curve = []

    for epoch in range(total_epochs):
        optimiser.zero_grad()

        # Get training datapoints
        if epoch % refresh_data_after == 0:
            data_train = dataset_train.generate_curves()
            train_context_x = data_train.context_x
            train_context_y = data_train.context_y
            train_target_x = data_train.target_x
            train_target_y = data_train.target_y
            train_num_targets = data_train.num_total_points

        if model_type == 'Latent' or model_type == 'LatentDeterministic':
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

        tensorboard_writer.add_scalar('Log likelihood', -loss, epoch)
        loglik_curve.append(-loss)
        loss.backward()
        optimiser.step()

        # Plot the predictions in `PLOT_AFTER` intervals
        if epoch % plot_after == 0:
            # Get testing datapoints
            test_context_x = data_test.context_x
            test_context_y = data_test.context_y
            test_target_x = data_test.target_x
            test_target_y = data_test.target_y
            test_num_targets = data_test.num_total_points

            # Get the log likelihood and representation
            if model_type == 'Latent':
                loglik = torch.mean(model.loglik_lb(context_x=test_context_x,
                                                    context_y=test_context_y,
                                                    num_targets=test_num_targets,
                                                    target_x=test_target_x,
                                                    target_y=test_target_y).detach())
                Z = model(test_context_x, test_context_y)
                z = Z.sample().detach()
                z = torch.unsqueeze(z, dim=1).repeat(1, test_num_targets, 1)
                representation = z
            elif model_type == 'LatentDeterministic':
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
                r = model(test_context_x, test_context_y).detach()
                r = torch.unsqueeze(r, dim=1).repeat(1, test_num_targets, 1)
                representation = r

            dist, μ, σ = model.f(representation, test_target_x)

            print(f'Iteration: {epoch}, log likelihood: {loglik}')

            # Plot the prediction and the context
            plot_functions(target_x=test_target_x.detach(),
                           target_y=test_target_y.detach(),
                           context_x=test_context_x.detach(),
                           context_y=test_context_y.detach(),
                           pred_y=μ.detach(),
                           σ_y=σ.detach(),
                           save_to_filepath=f'results/{model_type}_{max_context_points}/{epoch}.png')

    tensorboard_writer.flush()
    torch.save(model.state_dict(), f'models/{model_type}_{max_context_points}.pt')
    with open(f'results/{model_type}_{max_context_points}/loglik_curve.pickle', 'wb') as f:
        pickle.dump(loglik_curve, f)


if __name__ == '__main__':
    for model_type in MODEL_TYPES:
        for max_context_points in MAX_CONTEXT_POINTS:
            print(f'Model type: {model_type}, max context points: {max_context_points}')
            print()
            train(total_epochs=TOTAL_EPOCHS,
                  refresh_data_after=REFRESH_DATA_AFTER[model_type],
                  plot_after=PLOT_AFTER,
                  max_context_points=max_context_points,
                  train_batch_size=TRAIN_BATCH_SIZES[model_type],
                  test_batch_size=TEST_BATCH_SIZE,
                  layer_dim=LAYER_DIM,
                  encoder_num_layers=ENCODER_NUM_LAYERS,
                  decoder_num_layers=DECODER_NUM_LAYERS,
                  model_type=model_type,
                  random_kernel_params=RANDOM_KERNEL_PARAMS)
            print()
            print('========================================================')
