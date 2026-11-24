# Variational Quantum Integration for High Energy Physics

## Overview

Traditional Monte Carlo methods and numerical integration often struggle with slow convergence when evaluating integrals involving functions with singular features, such as rapidly varying regions or narrow peaks.
In this study, we present the impact of sampling strategies and loss functions on integration efficiency in the Quantum Integration Network (QuInt-Net).
We compare adaptive and non-adaptive sampling techniques and analyze how different loss functions influence integration accuracy and convergence rates.
Additionally, we explore quantum circuit architectures for numerical integration, comparing three different models: the data re-uploading model (QNN), the quantum signal processing protocol (QSP), and the deterministic quantum computation with one qubit (DQC1) model.
Our findings offer valuable insights into optimizing QuInt-Nets for applications in high-energy physics.

QuInt-Net learns an antiderivative with a parameterized quantum circuit. Training matches its input derivative to a normalized integrand. After training, two circuit evaluations give the integral over a chosen interval, with the input Jacobian and output scale restored.
The three one-dimensional benchmarks are a complex periodic function (CPF), a step function (Step), and a Breit–Wigner line shape (BW). The archived ideal-simulator comparisons use ten layers, with 62 parameters for QNN, 21 for QSP, and 86 for DQC1.

***

## Key Features

* **Quantum Circuit-Based Integration**: Designs quantum circuits using PennyLane to perform integral calculations with the QNN, QSP and DQC1 architectures.
* **PyTorch Integration**: Uses PyTorch to optimize the parameters of the quantum circuit and train the model on CPU or GPU.
* **Noise Studies**: Trains and evaluates QNN under gate-parameter errors and under terminal or layerwise bit-flip and depolarizing channels.
* **Seeded Benchmarks**: Repeats training over independent seeds with resumable checkpoints and per-seed error metrics.
* **Classical Baselines**: Provides plain Monte Carlo and MLP baselines for the BW benchmark.
* **Data Analysis and Visualization**: Analyzes experimental results using Jupyter Notebooks and visualizes them clearly with matplotlib.

***

## Code Structure
```text
QML_Integral
├── Analysis & Plot Making/        # Analysis notebook, benchmark helpers and classical baselines
│   ├── Analysis.ipynb             # Comparison of the published seed results
│   ├── benchmark_training.py      # Seeded training and checkpoint evaluation
│   ├── qnn_results.py             # Physical normalization and integral errors
│   └── Classic_Code/              # Plain MC and MLP baselines
├── Data File/                     # Fixed datasets, sampling notebook and archived benchmark CSVs
│   └── Benchmark Results/         # Published per-seed, interval and summary CSVs
├── Figure/                        # Manuscript figures
├── Additional_Running.py          # Continue training from saved single-model weights
├── Running_QuInt.py               # Main execution code
└── environment.yml                # Conda environment
```
***

## Tech Stack

* **Languages**: Python 3.11, Jupyter Notebook
* **Libraries**:
    * PennyLane: For building and simulating quantum circuits
    * PyTorch: For model training and optimization
    * NumPy and SciPy: For numerical operations and reference integrals
    * Matplotlib: For data visualization
    * tqdm: For training progress bars

***

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/HeechanYi/QML_Integral.git
cd QML_Integral
```

### 2. Install the necessary libraries

From the repository root, create the Python 3.11 environment.

```bash
conda env create -f environment.yml
conda activate QuInt_Net
```

The environment file pins the local CPU setup. The archived GPU runs used PyTorch 2.3.0 with CUDA 12.1, PennyLane 0.41.1 and NumPy 1.26.4. For GPU execution, install a CUDA-enabled PyTorch build and pass `--device cuda:0` for training.

-----

## Usage

The main script (`Running_QuInt.py`) uses command-line arguments to configure experiments. Run it from the repository root.

### Basic Execution

To run the script with default settings, use the following command:

```bash
python Running_QuInt.py
```

  * **Default Configuration**: This trains a single `QNN` model in the `Ideal` environment with `10` layers on the `BW` test function, using `Uni` samples and the `MSE` loss.

### Custom Execution

You can specify your own settings by providing arguments after the script name.

**Command Format:**

```bash
python Running_QuInt.py [--arg1 value1] [--arg2 value2] ...
```

**Example 1: Train the QSP model on the 'CPF' function with 'GS' samples and the 'Log\_Cosh' loss function.**

```bash
python Running_QuInt.py --model_type QSP --layers 10 --test_func CPF --sample_type GS --loss_type Log_Cosh
```

**Example 2: Train the QNN model on 'BW' with layerwise bit-flip noise over ten seeds.**

```bash
python Running_QuInt.py --model_type QNN --env BitFlib_Layer --test_func BW --seeds 0 1 2 3 4 5 6 7 8 9
```

**Example 3: Train the QNN model on 'BW' with layerwise depolarizing noise on the CPU.**

```bash
python Running_QuInt.py --model_type QNN --env Depolarizing_Layer --test_func BW --seeds 0 --device cpu
```

### Available Arguments

  * `--model_type`: The model type to use.
      * **Choices**: `QNN`, `QSP`, `DQC1`
      * **Default**: `QNN`
  * `--env`: The noise environment.
      * **Choices**: `Ideal`, `GateError`, `BitFlip`, `Depolarizing`, `BitFlib_Layer`, `Depolarizing_Layer`
      * **Default**: `Ideal`
  * `--layers`: The number of layers in the circuit.
      * **Default**: `10`
  * `--test_func`: The test function data to use.
      * **Choices**: `CPF`, `Step`, `BW`
      * **Default**: `BW`
  * `--sample_type`: The data sampling method.
      * **Choices**: `Uni`, `GS`, `HMC`
      * **Default**: `Uni`
  * `--loss_type`: The loss function for training.
      * **Choices**: `MSE`, `Log_Cosh`, `Chisqr`, `MSE_KL`
      * **Default**: `MSE`
  * `--num_epochs`: The number of training epochs.
      * **Default**: `1000`
  * `--lr`: The learning rate for the optimizer.
      * **Default**: `0.001`
  * `--batch_size`: The batch size for training.
      * **Default**: `128`
  * `--device`: The PyTorch device for training.
      * **Default**: `cuda:0` if CUDA is available, otherwise `cpu` (overridable with the `QUINT_DEVICE` environment variable)
  * `--seeds`: One or more unique integer seeds. Supplying this option selects seeded benchmark training.
      * **Default**: None (single run)
  * `--results_dir`: The output directory for seeded benchmark runs.
      * **Default**: `Results/`

### Noise Environments

| QNN environment | Operation |
| --- | --- |
| `Ideal` | Unitary circuit |
| `GateError` | Gaussian parameter perturbation with standard deviation 0.001 |
| `BitFlip`, `Depolarizing` | Independent channels on both wires after the final rotations |
| `BitFlib_Layer`, `Depolarizing_Layer` | Independent channels on both wires immediately after the CZ in each re-uploading layer |

All four Pauli-channel environments use probability 0.001 per wire. Training and evaluation use the same channel placement. The two layerwise environments apply to QNN only.

### Seeded Benchmark Runs

Supplying `--seeds` selects benchmark training. All three ideal models support CPF, Step and BW. The seeded noise study uses QNN on BW. Each seed independently initializes the model and splits the fixed 5,000 points into 4,000 training and 1,000 validation points. Defaults are ten layers, Adam with learning rate 0.001, batch size 128 and up to 1,000 epochs. The minimum-validation-loss weights are evaluated.

Seeded runs save per-seed weights, split indices, histories, interval and bin predictions, error metrics and metadata under `Results/<model>/<env>/<benchmark>/<sampler>/<loss>/<lr>/seed_<n>/`. Repeat the same command to resume from the group checkpoint. Use `--results_dir` to keep different settings or seed groups separate.

### Additional Training

Single runs save weights, training histories and plots under `Model/`. To continue from those weights, run `Additional_Running.py` with the same model, environment, dataset, layer count and learning rate, and pass the saved weight filename with `--model`. Note that its defaults are `--model_type DQC1` and `--test_func CPF`, so set both explicitly.

```bash
python Additional_Running.py --model_type QNN --test_func BW --model <saved_weights>.pt
```

### Classical Baselines

The classical BW baselines can be run separately.

```bash
python "Analysis & Plot Making/Classic_Code/running_MC.py" --help
python "Analysis & Plot Making/Classic_Code/running_MLP.py" --help
```

***

## Results

Open `Analysis & Plot Making/Analysis.ipynb` to compare the published seed results under `Data File/Benchmark Results/`. The `Figure` folder contains the manuscript figures saved as PDF files.

The nine fixed NPZ datasets in `Data File/` (CPF, Step and BW, each with `Uni`, `GS` and `HMC` samples) contain coordinates, function values and domain endpoints. The original HMC chain settings for the archived CPF and Step datasets remain unconfirmed.

* Integral MAE is the mean absolute integral error over 30 equal subintervals. BW integral errors are in GeV⁻³. CPF and Step integral errors are dimensionless.
* Seed variation measures training variability on a fixed dataset.
* GS weights depend on secant slopes between successive candidates in their stored order.
* The QNN code measures the mean of the two single-qubit Pauli Z observables. A terminal independent bit-flip channel on each measured wire scales this readout by `1 − 2p`, and the depolarizing channel scales it by `1 − 4p/3`. Both use `p = 0.001`.

***


## Citation

If our code benefits your research, please acknowledge our efforts by citing the following paper:

```bibtex
@article{Yi:2025QuIntNet,
  title         = {Quantum Integration Networks for Efficient Monte Carlo in High-Energy Physics},
  author        = {Yi, Heechan and Ban, Kayoung and Park, Myeonghun and Kong, Kyoungchul},
  journal       = {arXiv preprint arXiv:2510.10501},
  year          = {2025},
  eprint        = {2510.10501},
  archivePrefix = {arXiv},
  primaryClass  = {quant-ph}
}
```

## Reference

* H. Yi, K. Ban, M. Park and K. Kong, *Quantum Integration Networks for Efficient Monte Carlo in High-Energy Physics*, [arXiv:2510.10501](https://arxiv.org/abs/2510.10501).


## License

This project is licensed under the MIT License. See the `LICENSE` file for details.
