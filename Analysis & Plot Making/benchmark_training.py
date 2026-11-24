"""Seeded QuInt-Net training and checkpoint evaluation."""

import os

os.environ.setdefault('MPLBACKEND', 'Agg')
import time
from pathlib import Path
import numpy as np
import torch
import pennylane as qml
from Running_QuInt import (QNN_Model, QSP_Model, DQC1_Model, QNN_circuit, QSP_circuit, DQC1_circuit,
                           QNN_CHANNELS, load_model_environment, MSE_loss, Log_Cosh_loss, Chisqr_loss, MSE_KL_loss)
from qnn_results import ROOT, load_data, sha256, json_save, evaluate, evaluate_gate_error

CLASSES = dict(QNN=QNN_Model, QSP=QSP_Model, DQC1=DQC1_Model)
SOURCES = ('Running_QuInt.py', 'Analysis & Plot Making/qnn_results.py', 'Analysis & Plot Making/benchmark_training.py')
LOSSES = dict(MSE=MSE_loss, Log_Cosh=Log_Cosh_loss, Chisqr=Chisqr_loss, MSE_KL=MSE_KL_loss)


def flatten(model):
    return torch.cat([p.detach().reshape(-1) for p in model.parameters()])


def initial_parameters(seeds, layers, model_type):
    rows = []
    for seed in seeds:
        torch.manual_seed(seed)
        rows.append(flatten(CLASSES[model_type](layers)))
    result = torch.stack(rows)
    return result


def model_from_flat(model_type, layers, packed):
    model = CLASSES[model_type](layers)
    offset = 0
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.copy_(packed[offset : offset + parameter.numel()].reshape(parameter.shape))
            offset += parameter.numel()
    assert offset == packed.numel()
    return model


def ensemble_forward(packed, x, layers, model_type):
    n = x.shape[1]
    # One independent parameter row per seed; only circuit evaluations broadcast.
    weights = packed.T.repeat_interleave(n, dim=1)
    flat_x = x.reshape(-1)
    if model_type == 'QNN':
        y = QNN_circuit(layers, flat_x, weights)
    elif model_type == 'QSP':
        y = QSP_circuit(layers, flat_x, weights)
    else:
        split = 6 * (layers + 1)
        phi = weights[:split].reshape(layers + 1, 6, -1)
        theta = weights[split:].reshape(layers, 2, -1)
        y = DQC1_circuit(layers, flat_x, theta, phi)
    return y.reshape(x.shape)


def derivative(packed, x, args, create_graph, generators):
    if args.env == 'Ideal':
        y = ensemble_forward(packed, x, args.layers, args.model_type)
    else:
        circuit = QNN_CHANNELS.get(args.env, QNN_circuit)
        if args.env == 'GateError':
            noise = torch.stack([torch.randn(packed.shape[1], generator=g, dtype=packed.dtype) for g in generators])
            packed = packed + 0.001 * noise.to(packed.device)
        weights = packed.T.repeat_interleave(x.shape[1], dim=1)
        y = circuit(args.layers, x.reshape(-1), weights).reshape_as(x)
    return torch.autograd.grad(y, x, torch.ones_like(y), create_graph=create_graph)[0]


def save_run(args, seed, theta, initial, histories, info, indices, data_info):
    x, y, lower, upper, amplitude, source = data_info
    folder = args.result_dir / f'seed_{seed}'
    folder.mkdir(parents=True, exist_ok=True)
    model = model_from_flat(args.model_type, args.layers, theta.detach().cpu())
    if args.env in QNN_CHANNELS:
        channel_model = load_model_environment('QNN', args.env, args.layers)
        channel_model.load_state_dict(model.state_dict())
        model = channel_model
    start = time.perf_counter()
    if args.env == 'GateError':
        prediction, estimation = evaluate_gate_error(model, args.test_func, amplitude, lower, upper, seed, args.device, QNN_circuit)
    else:
        prediction, estimation = evaluate(model, args.test_func, amplitude, lower, upper, args.model_type)
        if args.env in QNN_CHANNELS:
            prediction['within_model_bin_sd'] = np.zeros_like(prediction['prediction'])
            prediction['within_model_interval_sd'] = np.zeros_like(prediction['interval_prediction'])
    inference_seconds = time.perf_counter() - start
    torch.save(model.state_dict(), folder / f'{args.model_type}_{args.env}_L{args.layers}.pt')
    np.savez(
        folder / 'Training_Graph.npz',
        train_graph=histories[0],
        val_graph=histories[1],
        train_indices=indices[0],
        val_indices=indices[1],
        initial_theta=initial.cpu().numpy(),
    )
    np.savez(folder / 'Model_Prediction.npz', **prediction)
    np.savez(folder / 'Model_Estimation.npz', **estimation)
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(histories[0], label='Train')
    ax.plot(histories[1], label='Validation')
    ax.axvline(info['best_epoch'] - 1, color='gray', linestyle=':', label='Saved model')
    ax.set(xlabel='Epoch', ylabel='Loss', yscale='log')
    ax.legend()
    fig.tight_layout()
    fig.savefig(folder / 'Training_Loss.png', dpi=200)
    plt.close(fig)
    gpu_name = torch.cuda.get_device_name(torch.device(args.device)) if args.device.startswith('cuda') else 'CPU'
    meta = dict(
        method=args.model_type,
        env=args.env,
        test_func=args.test_func,
        sample_type=args.sample_type,
        loss_type=args.loss_type,
        seed=seed,
        layers=args.layers,
        num_training_parameters=theta.numel(),
        num_qubits={'QNN': 2, 'QSP': 4, 'DQC1': 3}[args.model_type],
        n_data=len(x),
        n_train=4000,
        n_validation=1000,
        lr=args.lr,
        batch_size=args.batch_size,
        max_epochs=args.num_epochs,
        optimizer='Adam',
        parameter_dtype='float32',
        input_dtype='float64',
        amplitude=amplitude,
        lower=lower,
        upper=upper,
        x_mid=(lower + upper) / 2,
        s_scale=(upper - lower) / 2,
        data_file=str(source),
        data_sha256=sha256(source),
        device=args.device,
        gpu_name=gpu_name,
        backend='default.qubit' if args.env in ('GateError', 'Ideal') else 'default.mixed',
        diff_method='backprop',
        pennylane_version=qml.__version__,
        torch_version=str(torch.__version__),
        numpy_version=np.__version__,
        cuda_version=torch.version.cuda,
        min_delta=1e-5,
        patience_limit=15,
        seeds_batched=args.seeds,
        seed_variation='Fixed archived dataset; initialization, split and minibatch order vary.',
        split_seed=seed,
        minibatch_seed=seed + 1,
        minibatch_order='CPU torch.randperm(4000) each epoch, dedicated seed+1 generator.',
        cumulative_normalization='none',
        normalization='x=(2s-a-b)/(b-a); Qprime=f/A; integral=A*(b-a)/2*(Q(x2)-Q(x1))',
        integral_units='GeV^-3' if args.test_func == 'BW' else 'dimensionless',
        relative_error_note='Undefined when absolute reference <=1e-12*integral(abs(f)); stored as NaN with validity masks.',
        checkpoint_note='Independent cloned minimum-validation-loss checkpoint per seed.',
        timing_note='Training time is elapsed wall time within the concurrently trained seed group; not isolated per-seed cost.',
        inference_seconds=inference_seconds,
        source_sha256={n: sha256(ROOT / n) for n in SOURCES},
        **info,
    )
    meta.update(
        noise_probability=0.001 if args.env in QNN_CHANNELS else None,
        perturbation_scale=0.001 if args.env == 'GateError' else None,
        channel_placement='layer' if args.env.endswith('_Layer') else 'terminal' if args.env in QNN_CHANNELS else None,
        noise_training_seed=2_000_000 + seed if args.env == 'GateError' else None,
        gate_evaluation_seed=3_000_000 + seed if args.env == 'GateError' else None,
        evaluation_realizations=1000 if args.env == 'GateError' else 1,
        uncertainty_note='Independently trained seeds; gate realization variability is saved separately within each trained seed.',
        metric_note='Gate metrics average per-realization errors over 1000 perturbations; channel predictions are deterministic for each trained model.',
    )
    json_save(folder / 'metadata.json', meta)
    (folder / 'Run.log').write_text(
        f"Completed {args.test_func}/{args.sample_type}/{args.loss_type}/seed_{seed}\n"
        f"Best epoch: {info['best_epoch']}\nEpochs: {info['epochs_completed']}\n"
        f"Integral MAE: {float(estimation['int_mae']):.12g}\n"
    )
    print(
        f'SAVED seed={seed} epochs={info["epochs_completed"]} MAE={float(estimation["int_mae"]):.6e}', flush=True
    )


def train_group(args):
    import fcntl

    args.result_dir = args.results_dir / args.model_type / args.env / args.test_func / args.sample_type / args.loss_type / str(args.lr)
    args.result_dir.mkdir(parents=True, exist_ok=True)
    with (args.result_dir / 'training.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        train_seeds(args)


def train_seeds(args):
    torch.set_num_threads(1)
    # No AMP/TF32 or precision change: second derivatives use the original dtypes.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    data = load_data(args.test_func, args.sample_type)
    x, y, lower, upper, amplitude, source = data
    device = torch.device(args.device)
    if device.type == 'cuda':
        torch.cuda.set_device(device)
    x_gpu = torch.tensor((2 * x - upper - lower) / (upper - lower), device=device, dtype=torch.float64)
    y_gpu = torch.tensor(y / amplitude, device=device, dtype=torch.float64)
    initial = initial_parameters(args.seeds, args.layers, args.model_type)
    theta = torch.nn.Parameter(initial.to(device))
    optimizer = torch.optim.Adam([theta], lr=args.lr)
    splits, generators = [], []
    for seed in args.seeds:
        perm = torch.randperm(5000, generator=torch.Generator().manual_seed(seed))
        splits.append((perm[:4000], perm[4000:]))
        generators.append(torch.Generator().manual_seed(seed + 1))
    train_idx = torch.stack([v[0] for v in splits]).to(device)
    val_idx = torch.stack([v[1] for v in splits]).to(device)
    noise_generators = [torch.Generator().manual_seed(2_000_000 + s) for s in args.seeds]
    nseeds = len(args.seeds)
    train_history = [[] for _ in args.seeds]
    val_history = [[] for _ in args.seeds]
    best = np.full(nseeds, np.inf)
    best_epoch = np.zeros(nseeds, dtype=int)
    patience = np.zeros(nseeds, dtype=int)
    active = np.ones(nseeds, dtype=bool)
    best_theta = initial.clone()
    start_epoch, elapsed_offset = 0, 0.0
    ckpt = args.result_dir / 'checkpoint.pt'
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    signature = dict(
        env=args.env,
        test_func=args.test_func,
        sample_type=args.sample_type,
        loss_type=args.loss_type,
        layers=args.layers,
        seeds=args.seeds,
        lr=args.lr,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        data_sha256=sha256(source),
        source_sha256={n: sha256(ROOT / n) for n in SOURCES},
    )
    if not ckpt.exists() and any((args.result_dir / f'seed_{seed}' / 'metadata.json').exists() for seed in args.seeds):
        raise FileExistsError('Existing results without a matching checkpoint. Select another --results_dir.')
    if ckpt.exists():
        state = torch.load(ckpt, map_location='cpu', weights_only=False)
        assert state['signature'] == signature, 'Checkpoint protocol/source changed; do not mix runs.'
        theta.data.copy_(state['theta'].to(device))
        optimizer.load_state_dict(state['optimizer'])
        train_history, val_history = state['train_history'], state['val_history']
        best, best_epoch, patience, active = (state[k] for k in ('best', 'best_epoch', 'patience', 'active'))
        best_theta = state['best_theta']
        start_epoch = state['next_epoch']
        elapsed_offset = state['elapsed']
        for g, s in zip(generators, state['rng']):
            g.set_state(s)
        for g, s in zip(noise_generators, state['noise_rng']):
            g.set_state(s)
        print('RESUMED', start_epoch, flush=True)
    start = time.perf_counter()

    def checkpoint(next_epoch):
        state = dict(
            signature=signature,
            theta=theta.detach().cpu(),
            optimizer=optimizer.state_dict(),
            train_history=train_history,
            val_history=val_history,
            best=best,
            best_epoch=best_epoch,
            patience=patience,
            active=active,
            best_theta=best_theta,
            next_epoch=next_epoch,
            elapsed=elapsed_offset + time.perf_counter() - start,
            rng=[g.get_state() for g in generators],
            noise_rng=[g.get_state() for g in noise_generators],
        )
        temp = ckpt.with_suffix('.tmp')
        torch.save(state, temp)
        temp.replace(ckpt)

    if not ckpt.exists():
        checkpoint(0)
    loss_fn = LOSSES[args.loss_type]
    for epoch in range(start_epoch, args.num_epochs):
        if not active.any():
            break
        members = np.flatnonzero(active).tolist()
        member_gpu = torch.tensor(members, device=device)
        permutations = torch.stack([torch.randperm(4000, generator=generators[k]) for k in members]).to(device)
        epoch_indices = torch.gather(train_idx[member_gpu], 1, permutations)
        tr = torch.zeros(len(members), device=device, dtype=torch.float64)
        va = torch.zeros_like(tr)
        for offset in range(0, 4000, args.batch_size):
            idx = epoch_indices[:, offset : offset + args.batch_size]
            xb = x_gpu[idx].detach().requires_grad_()
            yb = y_gpu[idx]
            optimizer.zero_grad(set_to_none=True)
            pred = derivative(
                theta[member_gpu], xb, args, True, [noise_generators[k] for k in members]
            )
            losses = torch.stack([loss_fn(pred[j], yb[j]) for j in range(len(members))])
            # SUM, not mean, across seeds keeps each seed's original gradient.
            losses.sum().backward()
            optimizer.step()
            tr += losses.detach() * xb.shape[1]
        vbatch = max(1, args.batch_size // 2)
        for offset in range(0, 1000, vbatch):
            idx = val_idx[member_gpu, offset : offset + vbatch]
            xb = x_gpu[idx].detach().requires_grad_()
            yb = y_gpu[idx]
            pred = derivative(
                theta[member_gpu], xb, args, False, [noise_generators[k] for k in members]
            )
            losses = torch.stack([loss_fn(pred[j], yb[j]) for j in range(len(members))])
            va += losses.detach() * xb.shape[1]
        tr_np = (tr / 4000).cpu().numpy()
        va_np = (va / 1000).cpu().numpy()
        if not (np.isfinite(tr_np).all() and np.isfinite(va_np).all()):
            raise FloatingPointError('Non-finite loss; original losses were not stabilized.')
        for j, k in enumerate(members):
            train_history[k].append(float(tr_np[j]))
            val_history[k].append(float(va_np[j]))
            if va_np[j] < best[k]:
                best[k] = va_np[j]
                best_epoch[k] = epoch + 1
                best_theta[k] = theta[k].detach().cpu().clone()
            if len(val_history[k]) > 5:
                current = np.mean(val_history[k][-2:])
                previous = np.mean(val_history[k][-3:-1])
                patience[k] = patience[k] + 1 if previous - current < 1e-5 else 0
            if patience[k] >= 15 or epoch + 1 == args.num_epochs:
                active[k] = False
                info = dict(
                    best_epoch=int(best_epoch[k]),
                    best_val_loss=float(best[k]),
                    epochs_completed=len(train_history[k]),
                    training_seconds=elapsed_offset + time.perf_counter() - start,
                )
                save_run(
                    args,
                    args.seeds[k],
                    best_theta[k],
                    initial[k],
                    (train_history[k], val_history[k]),
                    info,
                    splits[k],
                    data,
                )
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(
                f'EPOCH {epoch+1} active={int(active.sum())} elapsed={elapsed_offset+time.perf_counter()-start:.1f}s '
                f'mean_val={float(va_np.mean()):.7g}',
                flush=True,
            )
        if (epoch + 1) % 25 == 0 or not active.any():
            checkpoint(epoch + 1)
    print('GROUP_COMPLETE', args.env, args.test_func, args.sample_type, args.loss_type, flush=True)


