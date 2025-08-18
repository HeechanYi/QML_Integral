# Variational Quantum Integration for High Energy Physics

## Overview

Traditional Monte Carlo methods and numerical integration often struggle with slow convergence when evaluating integrals involving functions with singular features, such as rapidly varying regions or narrow peaks.
In this study, we present the impact of sampling strategies and loss functions on integration efficiency in the Quantum Integration Network (QuInt-Net).
We compare adaptive and non-adaptive sampling techniques and analyze how different loss functions influence integration accuracy and convergence rates.
Additionally, we explore quantum circuit architectures for numerical integration, comparing three different models: the data re-uploading model, the quantum signal processing protocol, and the deterministic quantum computation with one qubit(DQC1) model.
Our findings offer valuable insights into optimizing QuInt-Nets for applications in high-energy physics.
***

## Key Features

* **Quantum Circuit-Based Integration**: Designs quantum circuits using Qiskit to perform integral calculations.
* **PyTorch Integration**: Uses PyTorch to optimize the parameters of the quantum circuit and train the model.
* **Data Analysis and Visualization**: Analyzes experimental results using Jupyter Notebooks and visualizes them clearly with matplotlib.

***

## Code Structure
```

QML Integral
├── Analysis & Plot Making/ # Notebooks for data analysis and visualization
├── Data File/              # Experimental data files
├── Figure/                 # Result graphs and images
├── Additional_Running.py   # Additional experimental code
└── Running_QuInt.py        # Main execution code
```
***

## Tech Stack

* **Languages**: Python, Jupyter Notebook
* **Libraries**:
    * Pennylane: For building and simulating quantum circuits
    * PyTorch: For model training and optimization
    * NumPy: For numerical operations
    * Matplotlib: For data visualization

***

## Getting Started

### 1. Clone the repository

```bash
git clone [https://github.com/HeechanYi/QML_Integral.git](https://github.com/HeechanYi/QML_Integral.git)
cd QML_Integral
````

### 2\. Install the necessary libraries

```bash
pip install pennylane torch numpy matplotlib
```

-----

## Usage

The main script (`Running_QuInt.py`) uses command-line arguments to configure experiments.

### Basic Execution

To run the script with default settings, use the following command:

```bash
python Running_QuInt.py
```

  * **Default Configuration**: This will run the `QNN` model in the `Ideal` environment with `10` layers, using the `BW` test function, etc.

### Custom Execution

You can specify your own settings by providing arguments after the script name.

**Command Format:**

```bash
python Running_QuInt.py [--arg1 value1] [--arg2 value2] ...
```

**Example 1: Train the QSP model in a BitFlip noise environment with 20 layers.**

```bash
python Running_QuInt.py --model_type QSP --env BitFlip --layers 20
```

**Example 2: Train the DQC1 model on the 'Step' function data, using the 'Log\_Cosh' loss function and a learning rate of 0.05.**

```bash
python Running_QuInt.py --model_type DQC1 --test_func Step --loss_type Log_Cosh --lr 0.05
```

### Available Arguments

  * `--model_type`: The model type to use.
      * **Choices**: `QNN`, `QSP`, `DQC1`
      * **Default**: `QNN`
  * `--env`: The noise environment.
      * **Choices**: `Ideal`, `GateError`, `BitFlip`, `Depolarizing`
      * **Default**: `Ideal`
  * `--layers`: The number of layers in the circuit.
      * **Default**: `10`
  * `--test_func`: The test function data to use.
      * **Choices**: `CPF`, `Step`, `BW`
      * **Default**: `BW`
  * `--sample_type`: The data sampling method.
      * **Choices**: `Uni`, `Imp`, `HMC`
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


***

## Results

You can find a detailed analysis and visualizations of the experimental results by running the Jupyter Notebooks in the `Analysis & Plot Making` folder. The `Figure` folder contains the main result graphs saved as image files.

***


## Citation

If our code benefits your research, please acknowledge our efforts by citing the following paper:

```bibtex

```

## Reference




## License

This project is licensed under the MIT License. See the `LICENSE` file for details.
