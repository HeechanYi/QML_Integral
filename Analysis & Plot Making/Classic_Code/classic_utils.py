"""BW data, reference integrals and evaluation shared by the classical baselines."""

import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.integrate import quad


CLASSIC_DIR = Path(__file__).resolve().parent
DATA_DIR = CLASSIC_DIR.parents[1] / "Data File"
M_Z = 91.18       # GeV; same values as Code/Data File/Sampling.ipynb
GAMMA = 2.49      # GeV


### Data and normalization ####################################
def Breit_Wigner_Distribution(s, M=M_Z, Gamma=GAMMA):
    """Toy line shape in GeV^-4, with s an energy in GeV."""
    return 1.0 / ((np.asarray(s)**2 - M**2)**2 + M**2 * Gamma**2)


def data_file(sample_type="Uni"):
    return DATA_DIR / f"BW_Data({sample_type}, 5000).npz"


def Data_loader(file_path):
    with np.load(file_path, allow_pickle=False) as data:
        X = np.array(data["x_data"], dtype=float)
        y = np.array(data["y_data"], dtype=float)
        lower = float(data["x_init"])
        upper = float(data["x_final"])
    if X.ndim != 1 or X.shape != y.shape or len(X) < 2:
        raise ValueError("Expected equal, one-dimensional x_data and y_data arrays.")
    if not (np.isfinite(X).all() and np.isfinite(y).all() and lower < upper):
        raise ValueError("Non-finite data or invalid integration domain.")
    if np.any((X < lower) | (X > upper)):
        raise ValueError("Data points lie outside the saved domain.")
    if not np.allclose(y, Breit_Wigner_Distribution(X), rtol=1e-10, atol=0):
        raise ValueError("Saved labels do not match M_Z=91.18 GeV, Gamma=2.49 GeV.")
    return X, y, lower, upper


def Data_Normalization(X, y, lower, upper):
    # Same mapping and sampled-label amplitude as Running_QuInt.py.
    X_norm = (2 * X - upper - lower) / (upper - lower)
    amplitude = float(np.max(np.abs(y)))
    if amplitude <= 0:
        raise ValueError("A positive normalization amplitude is required.")
    return X_norm, y / amplitude


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


### Reference integrals ####################################
def Reference_Results(lower, upper, samples=100, subintervals=30):
    if samples < 3 or subintervals < 1:
        raise ValueError("Use at least 3 evaluation points and 1 integration bin.")
    edges = np.linspace(lower, upper, subintervals + 1)
    intervals = np.array([[M_Z - k * GAMMA, M_Z + k * GAMMA]
                          for k in (3, 5, 10)] + [[lower, upper]])
    if np.any(intervals[:, 0] < lower) or np.any(intervals[:, 1] > upper):
        raise ValueError("The domain must contain all three resonance intervals.")

    def integral(a, b):
        return quad(Breit_Wigner_Distribution, a, b,
                    epsabs=1e-14, epsrel=1e-10)[0]

    reference = np.array([integral(a, b) for a, b in zip(edges[:-1], edges[1:])])
    evaluation_s = np.linspace(lower, upper, samples)
    return dict(evaluation_s=evaluation_s,
                xlin=(2 * evaluation_s - upper - lower) / (upper - lower),
                target_physical=Breit_Wigner_Distribution(evaluation_s),
                slin=edges[:-1], bin_end=edges[1:], integral=reference,
                cum_idx=edges[1:], cum_true=np.cumsum(reference),
                interval_bounds=intervals,
                interval_names=np.array(["M +/- 3 Gamma", "M +/- 5 Gamma",
                                         "M +/- 10 Gamma", "Full domain"]),
                interval_reference=np.array([integral(a, b) for a, b in intervals]))


def R2_estimation(pred, target):
    denominator = np.sum((target - np.mean(target))**2)
    return float(1 - np.sum((pred - target)**2) / denominator)


def Integral_Errors(prediction, reference, interval_prediction, interval_reference):
    error = np.asarray(prediction) - reference
    return dict(int_mae=np.mean(np.abs(error), axis=-1),
                int_mre=np.mean(np.abs(error / reference), axis=-1),
                interval_absolute_error=np.abs(interval_prediction - interval_reference),
                interval_relative_error=np.abs((interval_prediction - interval_reference)
                                               / interval_reference))


def Evaluate_Antiderivative(model, amplitude, lower, upper, reference, input_dtype=None):
    """Q'(x) ~ f(s)/A, s=c+h*x => I=A*h*(Q(x2)-Q(x1))."""
    import torch

    dtype = input_dtype or next(model.parameters()).dtype
    model.eval()
    x = torch.tensor(reference["xlin"], dtype=dtype, requires_grad=True)
    output = model(x)
    gradient = torch.autograd.grad(output, x, grad_outputs=torch.ones_like(output))[0]
    gradient = gradient.detach().cpu().numpy().astype(float)
    h = (upper - lower) / 2
    c = (upper + lower) / 2

    def integrate(a, b):
        a = torch.tensor((np.asarray(a) - c) / h, dtype=dtype)
        b = torch.tensor((np.asarray(b) - c) / h, dtype=dtype)
        with torch.no_grad():
            qa = model(a).cpu().numpy().astype(float)
            qb = model(b).cpu().numpy().astype(float)
        return amplitude * h * (qb - qa)

    pred = integrate(reference["slin"], reference["bin_end"])
    intervals = reference["interval_bounds"]
    interval_pred = integrate(intervals[:, 0], intervals[:, 1])
    result = dict(reference, target=reference["target_physical"] / amplitude,
                  dQdx=gradient, derivative_physical=amplitude * gradient,
                  prediction=pred, cum_pred=np.cumsum(pred),
                  interval_prediction=interval_pred)
    metrics = Integral_Errors(pred, reference["integral"], interval_pred,
                             reference["interval_reference"])
    metrics["diff_R2"] = R2_estimation(result["derivative_physical"][1:-1],
                                      reference["target_physical"][1:-1])
    return result, metrics


def Save_Results(output_dir, prediction, estimation, metadata):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    np.savez(output_dir / "Model_Prediction.npz", **prediction)
    np.savez(output_dir / "Model_Estimation.npz", **estimation)
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")

