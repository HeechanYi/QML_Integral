"""Train a 62-parameter BW antiderivative using the original PyTorch workflow."""

import argparse
import copy
from pathlib import Path
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, TensorDataset

from classic_utils import (CLASSIC_DIR, Data_loader, Data_Normalization, data_file,
                           file_hash, Reference_Results, Evaluate_Antiderivative,
                           Save_Results, M_Z, GAMMA)


### Model ####################################
class MLP_Model(nn.Module):
    def __init__(self):
        super().__init__()
        # 18 + 40 + 4 = 62 parameters, matching QNN at L=10 (6L+2).
        # An output bias would only shift Q and cannot be learned from Q'.
        self.network = nn.Sequential(nn.Linear(1, 9), nn.Tanh(),
                                     nn.Linear(9, 4), nn.Tanh(),
                                     nn.Linear(4, 1, bias=False))
        self.num_training_parameters = sum(p.numel() for p in self.parameters())
        assert self.num_training_parameters == 6 * 10 + 2

    def forward(self, data_point):
        return self.network(data_point.reshape(-1, 1)).reshape(data_point.shape)


### Loss functions: same definitions as Running_QuInt.py ########
def MSE_loss(pred, target):
    return torch.mean((target - pred)**2)


def Log_Cosh_loss(pred, target):
    return torch.mean(torch.log(torch.cosh(pred - target)))


def Chisqr_loss(pred, target):
    return torch.mean((target - pred)**2 / torch.abs(target))


def MSE_KL_loss(pred, target):
    logsoft_target = F.log_softmax(target, dim=-1)
    logsoft_pred = F.log_softmax(pred, dim=-1)
    kl = torch.sum(F.softmax(target, dim=-1) * (logsoft_target - logsoft_pred))
    return MSE_loss(pred, target) + kl


def derivative(y, t, create_graph=True):
    return torch.autograd.grad(y, t, create_graph=create_graph,
                               grad_outputs=torch.ones_like(y))[0]


### Train ####################################
def train(model, dataset, n_epochs, loss_type, optimizer, batch_size, seed):
    loss_fn = {"MSE": MSE_loss, "Log_Cosh": Log_Cosh_loss,
               "Chisqr": Chisqr_loss, "MSE_KL": MSE_KL_loss}[loss_type]
    train_size = int(0.8 * len(dataset))
    train_set, val_set = random_split(
        dataset, [train_size, len(dataset) - train_size],
        generator=torch.Generator().manual_seed(seed))
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              generator=torch.Generator().manual_seed(seed + 1))
    val_loader = DataLoader(val_set, batch_size=max(1, batch_size // 2))
    train_losses, val_losses = [], []
    best_val_loss, best_epoch, best_model_state = float("inf"), 0, None
    min_delta, patience_limit, patient = 1e-5, 15, 0

    for epoch in range(n_epochs):
        model.train()
        epoch_train_loss = 0.0
        for xb, yb in train_loader:
            xb = xb.requires_grad_()
            loss = loss_fn(derivative(model(xb), xb), yb)
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite training loss; no loss stabilization was added.")
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_train_loss += loss.item() * len(xb)
        train_losses.append(epoch_train_loss / len(train_set))

        model.eval()
        epoch_val_loss = 0.0
        for xb, yb in val_loader:
            xb = xb.requires_grad_()
            pred = derivative(model(xb), xb, create_graph=False)
            loss = loss_fn(pred, yb)
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite validation loss.")
            epoch_val_loss += loss.item() * len(xb)
        val_losses.append(epoch_val_loss / len(val_set))

        if val_losses[-1] < best_val_loss:
            best_val_loss = val_losses[-1]
            best_epoch = epoch + 1
            best_model_state = copy.deepcopy(model.state_dict())
        if epoch % 50 == 0:
            print(f"Epoch {epoch:3d} | Train Loss: {train_losses[-1]:.6f} | "
                  f"Val Loss: {val_losses[-1]:.6f}")
        if len(val_losses) > 5:
            current_avg = np.mean(val_losses[-2:])
            previous_avg = np.mean(val_losses[-3:-1])
            patient = patient + 1 if previous_avg - current_avg < min_delta else 0
            if patient >= patience_limit:
                print("Early stopping triggered.")
                break

    model.load_state_dict(best_model_state)
    info = dict(best_epoch=best_epoch, best_val_loss=best_val_loss,
                epochs_completed=len(train_losses), n_train=len(train_set),
                n_validation=len(val_set), min_delta=min_delta,
                patience_limit=patience_limit)
    indices = dict(train_indices=np.array(train_set.indices),
                   val_indices=np.array(val_set.indices))
    return model, train_losses, val_losses, info, indices


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Train the BW MLP baseline (62 parameters).")
    parser.add_argument('--sample_type', default='Uni', choices=['Uni', 'GS', 'HMC'])
    parser.add_argument('--loss_type', default='MSE', choices=['MSE', 'Log_Cosh', 'Chisqr', 'MSE_KL'])
    parser.add_argument('--num_epochs', type=int, default=1000)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--dtype', default='float32', choices=['float32', 'float64'])
    parser.add_argument('--data_file', type=Path)
    parser.add_argument('--output_dir', type=Path)
    args = parser.parse_args(argv)
    if args.num_epochs < 1 or args.batch_size < 2 or args.lr <= 0 or args.seed < 0:
        parser.error("Use positive epochs/lr, batch_size >= 2, and seed >= 0.")
    return args


def run(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.set_num_threads(1)
    dtype = getattr(torch, args.dtype)
    sample_type = args.sample_type
    source = (args.data_file or data_file(sample_type)).resolve()
    output_dir = (args.output_dir or CLASSIC_DIR / 'Results' / 'MLP' / 'BW'
                  / sample_type / args.loss_type / str(args.lr) / f'seed_{args.seed}').resolve()
    if output_dir.exists():
        raise FileExistsError(f"Results already exist: {output_dir}; choose a new --output_dir.")

    X, y, lower, upper = Data_loader(source)
    X_norm, y_norm = Data_Normalization(X, y, lower, upper)
    amplitude = float(np.max(np.abs(y)))
    dataset = TensorDataset(torch.tensor(X_norm, dtype=dtype),
                            torch.tensor(y_norm, dtype=dtype))
    model = MLP_Model().to(dtype=dtype)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    print('Running Device : cpu')
    print(f'BW | {sample_type} | {args.loss_type} | '
          f'{len(dataset)} data points | {model.num_training_parameters} parameters')

    start = time.perf_counter()
    model, train_losses, val_losses, info, indices = train(
        model, dataset, args.num_epochs, args.loss_type, optimizer, args.batch_size, args.seed)
    training_seconds = time.perf_counter() - start
    start = time.perf_counter()
    reference = Reference_Results(lower, upper)
    reference_seconds = time.perf_counter() - start
    start = time.perf_counter()
    prediction, estimation = Evaluate_Antiderivative(model, amplitude, lower, upper, reference)
    inference_seconds = time.perf_counter() - start

    metadata = dict(method='MLP', test_func='BW', sample_type=sample_type,
                    loss_type=args.loss_type, seed=args.seed, dtype=args.dtype,
                    data_file=str(source), data_sha256=file_hash(source),
                    n_data=len(dataset), **info, amplitude=amplitude,
                    lower=lower, upper=upper, M_Z=M_Z, Gamma=GAMMA,
                    x_mid=(lower + upper) / 2, s_scale=(upper - lower) / 2,
                    hidden_widths=[9, 4], activation='tanh', output_bias=False,
                    num_training_parameters=model.num_training_parameters,
                    matched_qnn_layers=10, lr=args.lr, batch_size=args.batch_size,
                    optimizer='Adam', training_seconds=training_seconds,
                    inference_seconds=inference_seconds, reference_seconds=reference_seconds,
                    cumulative_normalization='none', integral_units='GeV^-3',
                    data_generation_cost='archived dataset; not timed',
                    source_validation_evaluations=len(X),
                    split_note='Original QNN split indices were not saved; this split is reproducible, not a recovered historical split.',
                    initialization='PyTorch Linear defaults; torch.manual_seed(seed)',
                    torch_version=str(torch.__version__), numpy_version=np.__version__)
    Save_Results(output_dir, prediction, estimation, metadata)
    torch.save(model.state_dict(), output_dir / 'MLP_BW.pt')
    np.savez(output_dir / 'Training_Graph.npz', train_graph=train_losses,
             val_graph=val_losses, **indices)

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(train_losses, label='Train Loss')
    ax.plot(val_losses, label='Validation Loss')
    ax.axvline(info['best_epoch'] - 1, color='gray', linestyle=':', label='Saved model')
    ax.set(xlabel='Epoch', ylabel='Loss', yscale='log')
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / 'Training_Loss.png', dpi=200)
    plt.close(fig)
    print(f"Saved best epoch {info['best_epoch']} to {output_dir}")
    print(f"30-bin integral MAE: {float(estimation['int_mae']):.6e} GeV^-3")
    return output_dir


if __name__ == '__main__':
    run(parse_args())

