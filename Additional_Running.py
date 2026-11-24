"""Continue training a saved QuInt-Net model."""
import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import optim

from Running_QuInt import ENVIRONMENTS, Making_Dataset, device, load_model_environment, train


def main():
    parser = argparse.ArgumentParser(description="Train a QuInt-Net model.")
    parser.add_argument('--model_type', type=str, default='DQC1', choices=['QNN', 'QSP', 'DQC1'], help='Model type')
    parser.add_argument('--env', type=str, default='Ideal', choices=ENVIRONMENTS, help='Environment')
    parser.add_argument('--layers', type=int, default=10, help='Number of layers in the circuit')
    parser.add_argument('--test_func', type=str, default='CPF', choices=['CPF', 'Step', 'BW'], help='Test function')
    parser.add_argument('--sample_type', type=str, default='Uni', choices=['Uni', 'GS', 'HMC'], help='Sample type')
    parser.add_argument('--loss_type', type=str, default='MSE', choices=['MSE', 'Log_Cosh', 'Chisqr', 'MSE_KL'], help='Loss function')
    parser.add_argument('--num_epochs', type=int, default=1000, help = 'Number of Epochs')
    parser.add_argument('--lr', type =float, default=1e-3, help = 'Learning Rate')
    parser.add_argument('--model', type=str, default='.pt', help = 'Model State')

    args = parser.parse_args()
    if args.env.endswith('_Layer') and args.model_type != 'QNN':
        parser.error('Layerwise noise is defined for QNN.')

    model_type = args.model_type
    env = args.env
    layers = args.layers
    test_func = args.test_func
    sample_type = args.sample_type
    loss_type = args.loss_type
    n_epochs = args.num_epochs
    lr = args.lr
    model_state = args.model


    ### Data preparation ####################################
    file_path = Path(__file__).resolve().parent / 'Data File' / f'{test_func}_Data({sample_type}, 5000).npz'
    dataset, X_norm, y_norm = Making_Dataset(file_path)

    save_dir = f"./Model/{model_type}/{env}/{test_func}/{sample_type}/{loss_type}/{lr}/Additional_Run"
    os.makedirs(save_dir, exist_ok=True)

    print("--------------------------------------------")
    print("Directory Path :", save_dir)
    print("--------------------------------------------")

    model_path = f"./Model/{model_type}/{env}/{test_func}/{sample_type}/{loss_type}/{lr}/{model_state}"

    print("--------------------------------------------")
    print("Model Path :", model_path)
    print("--------------------------------------------")

    QNN_model = load_model_environment(model_type, env, layers).to(device)
    QNN_model.load_state_dict(torch.load(model_path, map_location=device))

    ### Optimizer ####################################
    optimizer = optim.Adam(QNN_model.parameters(), lr=lr)

    optimized_model, train_losses, val_losses = train(QNN_model, dataset, n_epochs, loss_type, optimizer, 256)

    val_str = f"{val_losses[-1]:.3e}".replace('.', '_').replace('e-', '_e')

    ### Training Result Plot #######################
    plt.figure(figsize=(5,4))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.yscale('log')
    plt.legend()
    plt.title('Training & Validation Loss over Epochs')
    plt.savefig(f"{save_dir}/{model_type}_{env}_loss_curve_{val_str}.png", dpi=300)  # 저장

    def derivative(y, t) : 
        return torch.autograd.grad(y, t, create_graph=True, grad_outputs=torch.ones_like(y))[0]

    optimized_model = optimized_model.cpu()
    optimized_model.eval()
    X_norm = X_norm.requires_grad_()
    Qx = optimized_model(X_norm)
    dQdx = derivative(Qx, X_norm)

    plt.figure(figsize=(5,4))
    plt.scatter(X_norm.detach().numpy(), dQdx.detach().numpy(), s=1, label='Model derivative')
    plt.scatter(X_norm.detach().numpy(), y_norm.detach().numpy(), s=1, label='Target function')
    plt.legend()
    # plt.title('Model Derivative vs Target')
    plt.savefig(f"{save_dir}/{model_type}_{env}_derivative_{val_str}.png", dpi=300)  # 저장

    torch.save(optimized_model.state_dict(), f"{save_dir}/{model_type}_{env}_L{layers}_addN{n_epochs}_{val_str}.pt")
    np.savez(f'{save_dir}/Training_Graph_{val_str}.npz', train_graph = train_losses, val_graph = val_losses)

if __name__ == "__main__":
    main()
