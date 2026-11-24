"""Reproduce the GS coordinate counts and validation overlap in Appendix A."""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]


def main():
    fig, ax = plt.subplots(figsize=(5, 3))
    print('Benchmark,Distinct,Duplicate draws (%),Max multiplicity,Overlap mean (%),Overlap SD (%)')
    for test_func in ('CPF', 'Step', 'BW'):
        with np.load(ROOT / 'Data File' / f'{test_func}_Data(GS, 5000).npz') as data:
            x = data['x_data']
        _, counts = np.unique(x, return_counts=True)
        overlap = []
        for seed in range(10):
            # Same CPU permutation and 80:20 split as benchmark_training.py.
            order = torch.randperm(len(x), generator=torch.Generator().manual_seed(seed)).numpy()
            train, val = np.split(order, [4000])
            overlap.append(100 * np.isin(x[val], x[train]).mean())

        duplicates = 100 * (1 - len(counts) / len(x))
        print(f'{test_func},{len(counts)},{duplicates:.2f},{counts.max()},{np.mean(overlap):.2f},{np.std(overlap, ddof=1):.2f}')
        ax.plot(range(10), overlap, 'o-', label=test_func)

    ax.set(xlabel='Split seed', ylabel='Validation coordinates in training (%)', xticks=range(10))
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(ROOT / 'Figure' / 'GS_validation_overlap_analysis.pdf')
    plt.close(fig)


if __name__ == '__main__':
    main()
