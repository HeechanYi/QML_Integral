"""Physical normalization and raw integral errors for the archived benchmarks."""

from pathlib import Path
import hashlib
import json
import numpy as np
from scipy.integrate import quad

ROOT = Path(__file__).resolve().parents[1]
M_Z, GAMMA = 91.18, 2.49


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def function(name, x):
    x = np.asarray(x)
    if name == 'BW':
        return 1 / ((x * x - M_Z * M_Z) ** 2 + M_Z * M_Z * GAMMA * GAMMA)
    if name == 'CPF':
        return np.cos(x + 0.5 * np.sin(4 * x))
    if name == 'Step':
        return np.heaviside(x, 0) - 0.5
    raise ValueError(name)


def load_data(name, sample):
    path = ROOT / 'Data File' / f'{name}_Data({sample}, 5000).npz'
    with np.load(path, allow_pickle=False) as z:
        x, y = z['x_data'].copy(), z['y_data'].copy()
        lower, upper = float(z['x_init']), float(z['x_final'])
    assert x.shape == y.shape == (5000,)
    assert np.isfinite(x).all() and np.isfinite(y).all()
    assert lower < upper and np.all((x >= lower) & (x <= upper))
    np.testing.assert_allclose(y, function(name, x), rtol=1e-10, atol=1e-14)
    amplitude = float(np.max(np.abs(y)))
    assert amplitude > 0
    return x, y, lower, upper, amplitude, path


def reference_results(name, lower, upper):
    edges = np.linspace(lower, upper, 31)
    if name == 'BW':
        bounds = np.array([[M_Z - k * GAMMA, M_Z + k * GAMMA] for k in (3, 5, 10)] + [[lower, upper]])
        names = ['M +/- 3 Gamma', 'M +/- 5 Gamma', 'M +/- 10 Gamma', 'Full domain']
    elif name == 'CPF':
        bounds = np.array([[0, np.pi / 2], [-np.pi / 2, np.pi / 2], [-np.pi, np.pi], [lower, upper]])
        names = ['[0, pi/2]', '[-pi/2, pi/2]', '[-pi, pi]', 'Full domain']
    else:
        bounds = np.array([[0, 0.5], [-0.5, 0], [-0.5, 0.5], [lower, upper]])
        names = ['[0, 0.5]', '[-0.5, 0]', '[-0.5, 0.5]', 'Full domain']

    def integral(a, b, absolute=False):
        if name == 'Step':
            return 0.5 * (b - a) if absolute else 0.5 * (abs(b) - abs(a))
        kwargs = dict(epsabs=1e-14, epsrel=1e-10, limit=300)
        f = (lambda x: abs(function(name, x))) if absolute else (lambda x: function(name, x))
        return quad(f, a, b, **kwargs)[0]

    bins = np.array([integral(a, b) for a, b in zip(edges[:-1], edges[1:])])
    s = np.linspace(lower, upper, 100)
    return dict(
        evaluation_s=s,
        xlin=(2 * s - upper - lower) / (upper - lower),
        target_physical=function(name, s),
        slin=edges[:-1],
        bin_end=edges[1:],
        integral=bins,
        cum_idx=edges[1:],
        cum_true=np.cumsum(bins),
        interval_bounds=bounds,
        interval_names=np.array(names),
        interval_reference=np.array([integral(a, b) for a, b in bounds]),
        absolute_integral_reference=integral(lower, upper, absolute=True),
    )


def evaluate(model, name, amplitude, lower, upper, model_type):
    import torch

    reference = reference_results(name, lower, upper)
    model = model.cpu().eval()
    valid_derivative = np.ones(len(reference['xlin']), dtype=bool)
    if model_type == 'QSP':
        valid_derivative = np.abs(reference['xlin']) < 1
    x = torch.tensor(reference['xlin'][valid_derivative], dtype=torch.float64, requires_grad=True)
    y = model(x)
    derivative = np.full(valid_derivative.shape, np.nan)
    derivative[valid_derivative] = torch.autograd.grad(y, x, torch.ones_like(y))[0].detach().numpy()
    h, c = (upper - lower) / 2, (upper + lower) / 2

    def integrate(a, b):
        with torch.no_grad():
            qa = model(torch.tensor((np.asarray(a) - c) / h, dtype=torch.float64)).numpy()
            qb = model(torch.tensor((np.asarray(b) - c) / h, dtype=torch.float64)).numpy()
        return amplitude * h * (qb - qa)

    pred = integrate(reference['slin'], reference['bin_end'])
    bounds = reference['interval_bounds']
    ipred = integrate(bounds[:, 0], bounds[:, 1])
    result = dict(
        reference,
        target=reference['target_physical'] / amplitude,
        dQdx=derivative,
        derivative_valid=valid_derivative,
        derivative_physical=amplitude * derivative,
        prediction=pred,
        cum_pred=np.cumsum(pred),
        interval_prediction=ipred,
    )
    err = pred - reference['integral']
    ierr = ipred - reference['interval_reference']
    scale = float(reference['absolute_integral_reference'])
    threshold = 1e-12 * scale
    valid = abs(reference['integral']) > threshold
    ivalid = abs(reference['interval_reference']) > threshold
    relative = np.full(ipred.shape, np.nan)
    np.divide(abs(ierr), abs(reference['interval_reference']), out=relative, where=ivalid)
    absolute_cum_error = abs(result['cum_pred'] - reference['cum_true'])
    cum_x = np.concatenate(([lower], reference['cum_idx']))
    cum_error = np.concatenate(([0.0], absolute_cum_error))
    phys_derivative_error = result['derivative_physical'] - reference['target_physical']
    metrics = dict(
        int_mae=np.mean(abs(err)),
        int_rmse=np.sqrt(np.mean(err * err)),
        int_mre=np.mean(abs(err[valid] / reference['integral'][valid])),
        bin_relative_error_valid=valid,
        interval_absolute_error=abs(ierr),
        interval_relative_error=relative,
        interval_relative_error_valid=ivalid,
        total_integral_absolute_error=abs(float(ierr[-1])),
        cumulative_mae=np.mean(absolute_cum_error),
        cumulative_integrated_absolute_error=np.trapz(cum_error, cum_x),
        diff_R2=1
        - np.sum(phys_derivative_error[1:-1] ** 2)
        / np.sum((reference['target_physical'][1:-1] - np.mean(reference['target_physical'][1:-1])) ** 2),
    )
    for k in ('prediction', 'cum_pred', 'interval_prediction'):
        assert np.isfinite(result[k]).all(), k
    assert np.isfinite(derivative[valid_derivative]).all()
    for k in ('int_mae', 'int_rmse', 'int_mre', 'total_integral_absolute_error', 'diff_R2'):
        assert np.isfinite(metrics[k]), k
    # Endpoint differences must telescope without separate cumulative normalization.
    np.testing.assert_allclose(np.sum(pred), ipred[-1], rtol=1e-10, atol=1e-12 * scale)
    return result, metrics


def json_save(path, data):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def evaluate_gate_error(model, name, amplitude, lower, upper, seed, device, circuit):
    import torch
    # Each realization uses one shared parameter perturbation for ALL endpoints:
    # I=A*h*(Q(xb;theta+delta*phi)-Q(xa;theta+delta*phi)).
    # This preserves the correlations and exact telescoping within a realization.
    reference = reference_results(name, lower, upper)
    boundaries = np.unique(
        np.r_[reference['slin'], reference['bin_end'], reference['interval_bounds'].reshape(-1)]
    )
    normalized = (2 * boundaries - upper - lower) / (upper - lower)
    coordinates = np.r_[reference['xlin'], normalized]
    idx_a = np.searchsorted(boundaries, reference['slin'])
    idx_b = np.searchsorted(boundaries, reference['bin_end'])
    idx_ia = np.searchsorted(boundaries, reference['interval_bounds'][:, 0])
    idx_ib = np.searchsorted(boundaries, reference['interval_bounds'][:, 1])
    generator = torch.Generator().manual_seed(3_000_000 + seed)
    theta = model.theta.detach().to(device)
    bin_runs, interval_runs, derivatives = [], [], []
    h = (upper - lower) / 2
    for offset in range(0, 1000, 100):
        noise = torch.randn(100, theta.numel(), generator=generator, dtype=theta.dtype).to(device)
        perturbed = theta[None, :] + 0.001 * noise
        x = torch.tensor(coordinates, device=device, dtype=torch.float64).expand(100, -1).clone().requires_grad_()
        y = circuit(model.num_layers, x.reshape(-1), perturbed.T.repeat_interleave(x.shape[1], dim=1)).reshape(
            x.shape
        )
        d = torch.autograd.grad(y, x, torch.ones_like(y))[0]
        values = y[:, 100:].detach().cpu().numpy()
        bin_runs.append(amplitude * h * (values[:, idx_b] - values[:, idx_a]))
        interval_runs.append(amplitude * h * (values[:, idx_ib] - values[:, idx_ia]))
        derivatives.append(d[:, :100].detach().cpu().numpy())
    bins, intervals, d = map(np.concatenate, (bin_runs, interval_runs, derivatives))
    pred, ipred, derivative_mean = bins.mean(axis=0), intervals.mean(axis=0), d.mean(axis=0)
    result = dict(
        reference,
        target=reference['target_physical'] / amplitude,
        dQdx=derivative_mean,
        derivative_physical=amplitude * derivative_mean,
        derivative_valid=np.ones(100, dtype=bool),
        prediction=pred,
        cum_pred=np.cumsum(pred),
        interval_prediction=ipred,
        within_model_bin_sd=bins.std(axis=0, ddof=1),
        within_model_interval_sd=intervals.std(axis=0, ddof=1),
        gate_bin_realizations=bins,
        gate_interval_realizations=intervals,
        gate_evaluation_seed=np.array(3_000_000 + seed),
    )
    error = bins - reference['integral']
    ierr = intervals - reference['interval_reference']
    relative = np.abs(ierr / reference['interval_reference'])
    target = reference['target_physical'][1:-1]
    r2 = 1 - np.sum((amplitude * d[:, 1:-1] - target) ** 2, axis=1) / np.sum((target - target.mean()) ** 2)
    cerr = np.abs(np.cumsum(error, axis=1))
    metrics = dict(
        int_mae=float(np.abs(error).mean()),
        int_rmse=float(np.sqrt((error * error).mean(axis=1)).mean()),
        int_mre=float(np.abs(error / reference['integral']).mean()),
        bin_relative_error_valid=np.ones(30, dtype=bool),
        interval_absolute_error=np.abs(ierr).mean(axis=0),
        interval_relative_error=relative.mean(axis=0),
        interval_relative_error_valid=np.ones(4, dtype=bool),
        total_integral_absolute_error=float(np.abs(ierr[:, -1]).mean()),
        cumulative_mae=float(cerr.mean()),
        cumulative_integrated_absolute_error=float(
            np.trapz(np.column_stack((np.zeros(1000), cerr)), np.r_[lower, reference['cum_idx']], axis=1).mean()
        ),
        diff_R2=float(r2.mean()),
        int_mae_of_mean_prediction=float(np.abs(pred - reference['integral']).mean()),
        gate_evaluation_realizations=np.array(1000),
    )
    for key in ('prediction', 'cum_pred', 'interval_prediction', 'dQdx'):
        assert np.isfinite(result[key]).all(), key
    assert np.isfinite(metrics['int_mae']) and np.isfinite(metrics['diff_R2'])
    np.testing.assert_allclose(
        bins.sum(axis=1), intervals[:, -1], rtol=1e-10, atol=1e-12 * reference['absolute_integral_reference']
    )
    return result, metrics
