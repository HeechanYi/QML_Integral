"""Reproduce Table V from independent HMC chains and plot the fixed datasets."""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


### Benchmark functions and leapfrog update from Data File/Sampling.ipynb

def U(test_func, x):
    fx = test_func(x)
    return -np.log(np.abs(fx) + 1e-10)


def grad_U(test_func, x, h=1e-5):
    return (U(test_func, x + h) - U(test_func, x - h)) / (2*h)


def leapfrog(test_func, x, p, step_size, n_steps):
    x_new = x
    p_new = p - 0.5 * step_size * grad_U(test_func, x_new)
    for _ in range(n_steps):
        x_new = x_new + step_size * p_new
        if _ != n_steps - 1:
            p_new = p_new - step_size * grad_U(test_func, x_new)
    p_new = p_new - 0.5 * step_size * grad_U(test_func, x_new)
    return x_new, p_new


def Complex_Phase_function(x):
    return np.cos(x + 0.5*np.sin(4*x))


def Step_function(x, x_0 = 0):
    return np.heaviside(x, x_0) - 0.5


def Breit_Wigner_Distribution(x, M = 91.18, Gamma = 2.49):
    k = 1
    val = (x**2 - M**2)**2 + M**2 * Gamma**2
    return  k / val


def sample_chain(test_func, lower, upper, seed):
    rng = np.random.RandomState(seed)
    x = (lower + upper) / 2
    positions = np.empty(2500)
    accepted = 0
    for i in range(len(positions)):
        p = rng.normal()
        current_H = U(test_func, x) + 0.5 * p**2
        x_new, p_new = leapfrog(test_func, x, p, 0.1, 20)
        if lower <= x_new <= upper:
            proposed_H = U(test_func, x_new) + 0.5 * p_new**2
            if rng.rand() < np.exp(min(0., current_H - proposed_H)):
                x = x_new
                accepted += 1
        positions[i] = x
    return positions, 100 * accepted / len(positions)


def position_stats(x):
    centered = x - x.mean()
    cov = np.correlate(centered, centered, mode='full')[len(x)-1:]
    if cov[0] == 0:
        raise ValueError('Cannot estimate position ESS for a constant chain.')
    rho = cov / cov[0]
    pairs = rho[::2] + rho[1::2]
    stop = next((i for i, pair in enumerate(pairs) if pair <= 0), len(pairs))
    tau = max(1., 2 * pairs[:stop].sum() - 1)
    return rho[:101], len(x) / tau


def main():
    functions = {'CPF': Complex_Phase_function, 'Step': Step_function, 'BW': Breit_Wigner_Distribution}
    fig, axes = plt.subplots(2, 3, figsize=(10, 4), sharex='col', layout='constrained')
    diag, ax = plt.subplots(figsize=(5, 3))
    print('Benchmark,Acceptance mean (%),Acceptance SD (%),Lag-1 mean,Lag-1 SD,Median position ESS')
    for col, (name, test_func) in enumerate(functions.items()):
        for method in ('Uni', 'GS', 'HMC'):
            with np.load(ROOT / 'Data File' / f'{name}_Data({method}, 5000).npz') as data:
                lower, upper = float(data['x_init']), float(data['x_final'])
                axes[1, col].hist(data['x_data'], bins=30, range=(lower, upper), density=True,
                                  histtype='step', label=method)
        grid = np.linspace(lower, upper, 1201)
        axes[0, col].plot(grid, test_func(grid), color='black')
        axes[0, col].set(title=name, ylabel='$f(s)$')
        axes[1, col].set(xlabel='$s$', ylabel='Density')

        # Independent chains start at the midpoint, with no warmup or thinning.
        rates, acfs, ess = [], [], []
        for seed in range(10):
            positions, acceptance = sample_chain(test_func, lower, upper, seed)
            rho, effective_size = position_stats(positions)
            rates.append(acceptance)
            acfs.append(rho)
            ess.append(effective_size)
        lag_one = np.asarray(acfs)[:, 1]
        print(f'{name},{np.mean(rates):.3f},{np.std(rates, ddof=1):.3f},'
              f'{lag_one.mean():.3f},{lag_one.std(ddof=1):.3f},{np.median(ess):.1f}')
        ax.plot(range(101), np.mean(acfs, axis=0), label=name)

    axes[1, 0].legend(frameon=False)
    ax.set(xlabel='Lag', ylabel='Mean position autocorrelation')
    ax.legend(frameon=False)
    diag.tight_layout()
    fig.savefig(ROOT / 'Figure' / 'Sampling_fixed_analysis.pdf')
    diag.savefig(ROOT / 'Figure' / 'HMC_autocorrelation_analysis.pdf')
    plt.close(fig)
    plt.close(diag)


if __name__ == '__main__':
    main()
