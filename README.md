# An Efficient Max-Sum Algorithm for a Max-Margin Binary Perceptron Problem

This repository contains the full implementation of the Max-Sum algorithm described in the Bachelor's thesis *"An efficient Max-Sum algorithm for a max-margin binary perceptron problem"*. 
The project develops and tests an efficient message-passing scheme to solve the max-margin learning problem for binary-weight perceptrons.

## Structure

```bash
.
├── bperceptron.py                # General implementation of the inefficient max-sum updates
├── bperceptron_efficient.py      # Core implementation of the efficient max-sum updates
├── simulations/
│   ├── simulation.ipynb          # Implementation to run simulations
│   └── simulation_profiling.py   # Implementation for profiling of the message updates
├── results/                         # Output of simulations
│   ├── forced_linear.txt
│   ├── forced_squared.txt
│   ├── forced_exponential.txt
│   ├── non_forced_linear.txt
│   ├── non_forced_squared.txt
│   └── non_forced_exponential.txt
└── README.md              # This file
