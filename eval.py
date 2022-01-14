import pickle

import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt

from data_generator import GPCurvesReader
from decoder import Decoder
from deterministic_encoder import DeterministicEncoder
from latent_encoder import LatentEncoder
from util import plot_functions

TOTAL_EPOCHS = 100001
MAX_CONTEXT_POINTS = [10, 50, 100]
TRAIN_BATCH_SIZE = 16
TEST_BATCH_SIZE = 1
LAYER_DIM = 128
ENCODER_NUM_LAYERS = 4
DECODER_NUM_LAYERS = 2
MODEL_TYPES = ['Latent', 'Deterministic', 'LatentDeterministic']


def eval_model(model_path, total_epochs, max_context_points, train_batch_size,
               test_batch_size, layer_dim, encoder_num_layers, decoder_num_layers,
               model_type, seed=2022):
    # initialise seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Training dataset
    dataset_train = GPCurvesReader(
        batch_size=train_batch_size, max_num_context=max_context_points,
        random_kernel_parameters=True)

    # Test dataset
    dataset_test = GPCurvesReader(
        batch_size=test_batch_size, max_num_context=max_context_points,
        testing=True, random_kernel_parameters=True)
    data_test = dataset_test.generate_curves()

    # Get testing datapoints
    test_context_x = data_test.context_x
    test_context_y = data_test.context_y
    test_target_x = data_test.target_x
    test_target_y = data_test.target_y
    test_num_targets = data_test.num_total_points

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

    # Load model's saved state_dict
    model.load_state_dict(torch.load(model_path))

    with torch.no_grad():
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
            representation = model(test_context_x, test_context_y).detach()

        dist, μ, σ = model.f(representation, test_target_x)

        print(f'Model type: {model_type}, max context points: {max_context_points}')
        print(f'Iteration: {total_epochs}, log likelihood: {loglik}')

        # Plot the prediction and the context
        plot_functions(target_x=test_target_x.detach(),
                       target_y=test_target_y.detach(),
                       context_x=test_context_x.detach(),
                       context_y=test_context_y.detach(),
                       pred_y=μ.detach(),
                       σ_y=σ.detach(),
                       save_to_filepath=f'results/{model_type}_{max_context_points}/{total_epochs}.png')


def plot_loglik_curves(total_epochs, max_context_points, model_types):
    fig, axes = plt.subplots(1, len(max_context_points), figsize=(20, 4.8), sharey='all')

    df = pd.DataFrame({})
    for i, max_context_points in enumerate(max_context_points):
        for model_type in model_types:
            model_name = f'{model_type}_{max_context_points}'
            with open(f'results/{model_name}/loglik_curve.pickle', 'rb') as f:
                loglik_curve = pickle.load(f)
                df[model_name] = torch.stack(loglik_curve).detach().numpy()
                df[model_name] = df[model_name].ewm(alpha=0.001).mean()

    epochs = np.arange(total_epochs)
    for i, max_context_points in enumerate(max_context_points):
        for model_type in model_types:
            model_name = f'{model_type}_{max_context_points}'
            axes[i].plot(epochs, df[model_name], label=model_type)
        axes[i].legend()
        axes[i].set_title(f'Max context points = {max_context_points}')

    axes[0].set_ylabel('Log likelihood')
    plt.show()


if __name__ == '__main__':
    plot_loglik_curves(total_epochs=TOTAL_EPOCHS,
                       max_context_points=MAX_CONTEXT_POINTS,
                       model_types=MODEL_TYPES)

    # eval_model(model_path='models/LatentDeterministic_50.pt',
    #            total_epochs=TOTAL_EPOCHS,
    #            max_context_points=50,
    #            train_batch_size=TRAIN_BATCH_SIZE,
    #            test_batch_size=TEST_BATCH_SIZE,
    #            layer_dim=LAYER_DIM,
    #            encoder_num_layers=ENCODER_NUM_LAYERS,
    #            decoder_num_layers=DECODER_NUM_LAYERS,
    #            model_type='LatentDeterministic')
