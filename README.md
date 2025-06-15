# An Efficient Max-Sum Algorithm for a Max-Margin Binary Perceptron Problem

This repository contains the full implementation of the Max-Sum algorithm described in the Bachelor's thesis *"An efficient Max-Sum algorithm for a max-margin binary perceptron problem"*. 
The project develops and tests an efficient message-passing scheme to solve the max-margin learning problem for binary-weight perceptrons.

## Structure

```bash
.
├── results/                         # Output of simulations
│   ├── forced_exponential.csv       # Simulations with reinforcement, exponential external field csv
│   ├── forced_exponential.txt       # Simulations with reinforcement, exponential external field
│   ├── forced_linear.txt            # Simulations with reinforcement, linear external field
│   ├── forced_squared.csv           # Simulations with reinforcement, squared external field csv
│   ├── forced_squared.txt           # Simulations with reinforcement, squared external field
│   ├── non_forced_exponential.txt   # Simulations without reinforcement, exponential external field
│   ├── non_forced_linear.txt        # Simulations without reinforcement, linear external field
│   └── non_forced_squared.txt       # Simulations without reinforcement, squared external field
├── bperceptron_efficient.py      # Core implementation of the efficient max-sum updates
├── bperceptron.py                # General implementation of the max-sum updates
├── README.md
├── simulation_nb.ipynb           # Implementation to run the simulations in notebook version
├── simulation_plot.ipynb         # Implementation to plot the results of the simulations
├── simulation_profiling.py       # Implementation for profiling of the message updates
└── simulation_py.py              # Implementation to run simulations