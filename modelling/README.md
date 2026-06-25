# Larval Brain Computational Model

This repository contains the code used to simulate a computational firing-rate model of the larval brain connectome.

## Contents

Source data should be downloaded in parent directory.

### Core Model

* **`ConnectomicFiringRateNetwork.py`**

  * Implements the connectome-based firing-rate network model of the larval brain.
  * Contains the network architecture, connectivity, and simulation dynamics.

### Simulation Scripts

* **`run-simulation.py`**

  * Runs network simulations with targeted stimulation of Basin neurons.
  * Provides an example workflow for executing and analyzing model behavior.

## Requirements

The code was developed and tested with the following software environment:

| Package | Version      |
| ------- | ------------ |
| Python  | 3.10.14      |
| PyTorch | 2.10.0+cu128 |
| NumPy   | 1.26.4       |
| Pandas  | 2.2.3        |

## Installation

Create a Python environment and install the required dependencies:

```bash
pip install torch==2.10.0+cu128 numpy==1.26.4 pandas==2.2.3
```

Alternatively, install the packages using your preferred environment manager (e.g., Conda).

## Usage

Run a simulation with Basin neuron stimulation:

```bash
python run-simulation.py
```

The simulation script loads the connectomic firing-rate network and executes the specified stimulation protocol.

## Repository Structure

```text
.
├── ConnectomicFiringRateNetwork.py   # Core firing-rate network model
├── run-simulation.py                 # Simulation runner with Basin neuron stimulation
└── README.md
```

## Notes

This repository contains the implementation of a connectome-based firing-rate model for investigating neural activity dynamics in the larval brain. The simulation workflow is centered around stimulation experiments targeting Basin neurons and the resulting network responses.
