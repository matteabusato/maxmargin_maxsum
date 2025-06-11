import pandas as pd
import matplotlib.pyplot as plt
import os

base_path = "results"
filenames = {
    "exponential": os.path.join(base_path, "results_forced_exponential.txt"),
    "squared": os.path.join(base_path, "results_forced_squared.txt"),
    "linear": os.path.join(base_path, "results_forced_linear.txt")
}


loaded_data = {}
for key, path in filenames.items():
    if os.path.exists(path):
        df = pd.read_csv(path, delim_whitespace=True)
        loaded_data[key] = df

