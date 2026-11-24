"""Plain MC for BW: a common uniform sample estimates all requested intervals."""

import argparse
from pathlib import Path
import time

import numpy as np

from classic_utils import (CLASSIC_DIR, Data_loader, data_file, file_hash,
                           Breit_Wigner_Distribution, Reference_Results,
                           Integral_Errors, Save_Results, M_Z, GAMMA)


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
    parser = argparse.ArgumentParser(description='Plain MC integration of BW.')
    parser.add_argument('--data_file', type=Path, default=data_file('Uni'),
                        help='Use saved bounds and point count; draw fresh uniform MC samples.')
    parser.add_argument('--n_samples', type=int, default=None,
                        help='Default: all 5000 points in the source dataset; use 4000 to match optimizer points only.')
    parser.add_argument('--n_trials', type=int, default=1)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--output_dir', type=Path)
    args = parser.parse_args(argv)
    if args.n_trials < 1 or args.seed < 0 or (args.n_samples is not None and args.n_samples < 2):
        parser.error('Use n_trials >= 1, n_samples >= 2, and seed >= 0.')
    return args


def run(args):
    source = args.data_file.resolve()
    X, _, lower, upper = Data_loader(source)
    n_samples = len(X) if args.n_samples is None else args.n_samples
    output_dir = (args.output_dir or CLASSIC_DIR / 'Results' / 'MC' / 'BW'
                  / f'N{n_samples}' / f'seed_{args.seed}_trials_{args.n_trials}').resolve()
    if output_dir.exists():
        raise FileExistsError(f'Results already exist: {output_dir}; choose a new --output_dir.')
    start = time.perf_counter()
    reference = Reference_Results(lower, upper)
    reference_seconds = time.perf_counter() - start
    samples, predictions, errors, cumulative_errors, intervals, interval_errors = [], [], [], [], [], []
    # Separate child streams make the first trials independent of n_trials.
    streams = np.random.SeedSequence(args.seed).spawn(args.n_trials)
    start = time.perf_counter()
    for stream in streams:
        rng = np.random.default_rng(stream)
        x = rng.uniform(lower, upper, n_samples)
        y = Breit_Wigner_Distribution(x)
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
    metadata = dict(method='MC', test_func='BW', seed=args.seed, n_trials=args.n_trials,
                    n_samples=n_samples, source_dataset_points=len(X),
                    source_training_points=int(0.8 * len(X)),
                    sample_count_basis='full saved dataset' if args.n_samples is None else 'explicit --n_samples',
                    data_file=str(source), data_sha256=file_hash(source),
                    lower=lower, upper=upper, M_Z=M_Z, Gamma=GAMMA,
                    integrand_evaluations_per_trial=n_samples,
                    integrand_evaluations_total=n_samples * args.n_trials,
                    source_validation_evaluations=len(X),
                    sampling_and_integration_seconds=mc_seconds,
                    reference_seconds=reference_seconds,
                    cumulative_normalization='none', integral_units='GeV^-3',
                    estimator='One uniform sample on the full domain, reused with interval indicators.',
                    standard_error='Per-trial sample standard deviation / sqrt(N), including zero contributions outside each interval.',
                    numpy_version=np.__version__)
    Save_Results(output_dir, prediction, estimation, metadata)
    np.savez(output_dir / 'MC_Samples.npz',
             x_data=np.array([x for x, y in samples]),
             y_data=np.array([y for x, y in samples]))
    print(f'BW | Plain MC | {n_samples} points/trial | {args.n_trials} trials')
    print(f"Mean 30-bin integral MAE: {np.mean(estimation['int_mae']):.6e} GeV^-3")
    print(f'Saved to {output_dir}')
    return output_dir


if __name__ == '__main__':
    run(parse_args())
