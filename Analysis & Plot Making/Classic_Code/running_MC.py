"""Plain MC on a finite interval: a common uniform sample estimates all requested intervals."""

import argparse
from pathlib import Path
import time

import numpy as np

from classic_utils import CLASSIC_DIR, Reference_Results, Integral_Errors, Save_Results, M_Z, GAMMA


### Test functions (Sampling.ipynb) ####################################
def Complex_Phase_function(x):
    return np.cos(x + 0.5*np.sin(4*x))


def Step_function(x, x_0=0):
    return np.heaviside(x, x_0) - 0.5


def Breit_Wigner_Distribution(x, M=M_Z, Gamma=GAMMA):
    val = (x**2 - M**2)**2 + M**2 * Gamma**2
    return 1 / val


def Benchmark_Settings(name):
    if name == 'CPF':
        return (Complex_Phase_function, -2*np.pi, 2*np.pi,
                [(0, np.pi/2), (-np.pi/2, np.pi/2), (-np.pi, np.pi)],
                ['[0, pi/2]', '[-pi/2, pi/2]', '[-pi, pi]'], None)
    if name == 'STEP':
        return (Step_function, -1., 1., [(0., .5), (-.5, 0.), (-.5, .5)],
                ['[0, 0.5]', '[-0.5, 0]', '[-0.5, 0.5]'], lambda x: abs(x)/2)
    if name == 'BW':
        return (Breit_Wigner_Distribution, 60., 120.,
                [(M_Z-k*GAMMA, M_Z+k*GAMMA) for k in (3, 5, 10)],
                ['M +/- 3 Gamma', 'M +/- 5 Gamma', 'M +/- 10 Gamma'], None)
    raise ValueError(f'Unknown benchmark: {name}')


def Benchmark_Reference(name, lower, upper, intervals=None):
    fn, _, _, bounds, labels, primitive = Benchmark_Settings(name)
    if intervals:
        bounds = [(a, b) for a, b in intervals if (a, b) != (lower, upper)]
        labels = [f'[{a:g}, {b:g}]' for a, b in bounds]
    if any(not lower <= a < b <= upper for a, b in bounds):
        raise ValueError('Evaluation intervals must lie inside the dataset domain.')
    return Reference_Results(fn, lower, upper, np.array(bounds + [(lower, upper)]),
                             labels + ['Full domain'], primitive)


### Monte Carlo integration ####################################
def Integration_with_MC(x, y, starts, ends, lower, upper):
    """I(u,v)=(b-a)*mean[f(X)*1_{u<=X<v}], X~Uniform(a,b)."""
    values, standard_errors = [], []
    for a, b in zip(starts, ends):
        weighted = np.where((x >= a) & (x < b), y, 0.0)
        values.append((upper - lower) * np.mean(weighted))
        standard_errors.append((upper - lower) * np.std(weighted, ddof=1) / np.sqrt(len(x)))
    return np.array(values), np.array(standard_errors)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description='Plain MC integration of a one-dimensional function.')
    parser.add_argument('--function', '--test_func', type=str.upper, choices=['CPF', 'STEP', 'BW'], default='BW')
    parser.add_argument('--n_samples', type=int, default=5000)
    parser.add_argument('--interval', type=float, nargs=2, action='append')
    parser.add_argument('--n_trials', type=int, default=1)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--output_dir', type=Path)
    args = parser.parse_args(argv)
    if args.n_trials < 1 or args.seed < 0 or (args.n_samples is not None and args.n_samples < 2):
        parser.error('Use n_trials >= 1, n_samples >= 2, and seed >= 0.')
    return args


def run(args):
    name = args.function
    fn, lower, upper, _, _, _ = Benchmark_Settings(name)
    units = 'GeV^-3' if name == 'BW' else ''
    n_samples = args.n_samples
    output_dir = (args.output_dir or CLASSIC_DIR / 'Results' / 'MC' / name
                  / f'N{n_samples}' / f'seed_{args.seed}_trials_{args.n_trials}').resolve()
    if output_dir.exists():
        raise FileExistsError(f'Results already exist: {output_dir}; choose a new --output_dir.')
    start = time.perf_counter()
    reference = Benchmark_Reference(name, lower, upper, args.interval)
    reference_seconds = time.perf_counter() - start
    samples, predictions, errors, cumulative_errors, intervals, interval_errors = [], [], [], [], [], []
    # Separate child streams make the first trials independent of n_trials.
    streams = np.random.SeedSequence(args.seed).spawn(args.n_trials)
    start = time.perf_counter()
    for stream in streams:
        rng = np.random.default_rng(stream)
        x = rng.uniform(lower, upper, n_samples)
        y = fn(x)
        pred, se = Integration_with_MC(x, y, reference['slin'], reference['bin_end'], lower, upper)
        _, cum_se = Integration_with_MC(x, y, np.full(len(pred), lower),
                                         reference['cum_idx'], lower, upper)
        bounds = reference['interval_bounds']
        target, target_se = Integration_with_MC(x, y, bounds[:, 0], bounds[:, 1], lower, upper)
        samples.append((x, y))
        predictions.append(pred)
        errors.append(se)
        cumulative_errors.append(cum_se)
        intervals.append(target)
        interval_errors.append(target_se)
    mc_seconds = time.perf_counter() - start
    predictions, intervals = np.array(predictions), np.array(intervals)
    prediction = dict(reference, prediction=predictions,
                      prediction_standard_error=np.array(errors),
                      cum_pred=np.cumsum(predictions, axis=1),
                      cumulative_standard_error=np.array(cumulative_errors),
                      interval_prediction=intervals,
                      interval_standard_error=np.array(interval_errors))
    estimation = Integral_Errors(predictions, reference['integral'], intervals,
                                 reference['interval_reference'])
    metadata = dict(method='MC', test_func=name, seed=args.seed, n_trials=args.n_trials,
                    n_samples=n_samples, lower=lower, upper=upper, 
                    integrand_evaluations_per_trial=n_samples,
                    integrand_evaluations_total=n_samples * args.n_trials,
                    sampling_and_integration_seconds=mc_seconds,
                    reference_seconds=reference_seconds,
                    cumulative_normalization='none', integral_units=units,
                    estimator='One uniform sample on the full domain, reused with interval indicators.',
                    standard_error='Per-trial sample standard deviation / sqrt(N), including zero contributions outside each interval.',
                    numpy_version=np.__version__)
    if name == 'BW':
        metadata.update(M_Z=M_Z, Gamma=GAMMA)
    Save_Results(output_dir, prediction, estimation, metadata)
    np.savez(output_dir / 'MC_Samples.npz',
             x_data=np.array([x for x, y in samples]),
             y_data=np.array([y for x, y in samples]))
    print(f'{name} | Plain MC | {n_samples} points/trial | {args.n_trials} trials')
    print(f"Mean 30-bin integral MAE: {np.mean(estimation['int_mae']):.6e} {units}")
    print(f'Saved to {output_dir}')
    return output_dir


if __name__ == '__main__':
    run(parse_args())
