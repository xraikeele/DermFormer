import sys
import argparse
import torch
import torch.optim as optim
import torch_optimizer as optims
from torch.utils.data import DataLoader
import numpy as np
from torch.autograd import Variable
import os
import time
import itertools
from itertools import cycle
import json
import torch.nn.functional as F
import matplotlib.pyplot as plt

from derm7pt.eval_metrics import ConfusionMatrix, plot_metrics, plot_roc_curves
from derm7pt.dataloader import load_dataset, dataset #, train_data_transformation, test_data_transformation
from derm7pt.dataloader_corruptions import load_corrupt, dataset_corrupt
from derm7pt.dataloader_noise import load_noise, dataset_noise
from models.model_swinv2fusion import MM_Transformer
from models.TFormer import TFormer
from models.DermFormer import DermFormer, MMNestLoss
from models.NesT.nest_der import nest_der
from models.NesT.nest_cli import nest_cli
from models.NesT.nest_multimodalconcat import nest_MMC

def load_results(file_path):
    import json

    # Read the file
    with open(file_path, "r", encoding="utf-8") as file:
        lines = file.readlines()

    # Construct the JSON string
    json_string = "".join(
        line.strip() for line in lines if line.strip() and not line.lstrip().startswith("#")
    )

    # Debugging output
    print("Constructed JSON string (preview):")
    print("Start:", json_string[:500])  # First 500 characters
    print("End:", json_string[-500:])  # Last 500 characters

    # Attempt to parse JSON
    try:
        results = json.loads(json_string)
    except json.JSONDecodeError as e:
        print("Error parsing JSON content:")
        print(f"Details: {e}")
        print(f"Error location: Line {e.lineno}, Column {e.colno}")
        print(f"JSON Content (error vicinity): {json_string[max(0, e.pos - 50):e.pos + 50]}")
        raise

    return results

def prepare_plot_data(results, condition, metric_name="accuracy"):
    # Check available metrics
    available_metrics = set()
    for model_results in results.values():
        for condition_results in model_results.values():
            for entry in condition_results:
                available_metrics.update(entry['result'].keys())

    # Collect unique `der_noise` values
    unique_der_noise_values = set()

    # Extract plot data
    plot_data = {}
    baseline_data = {}

    for model_name, model_results in results.items():
        if condition in model_results:
            print(f"Found condition '{condition}' in model '{model_name}'")
            condition_results = model_results[condition]

            # Retrieve baseline data for the 'standard' condition
            if 'standard' in model_results:
                standard_results = model_results['standard']
                baseline_data[model_name] = [
                    entry['result'].get(metric_name, 0) for entry in standard_results
                ]

            # Collect unique `der_noise` values
            if condition == "der_noise":
                for entry in condition_results:
                    der_noise_value = entry['test_conditions'].get('der_noise')
                    if der_noise_value is not None:
                        unique_der_noise_values.add(der_noise_value)

            # Extract metrics for plotting
            condition_metrics = [
                entry['result'][metric_name] for entry in condition_results
            ]
            plot_data[model_name] = condition_metrics

    if condition == "der_noise":
        print("Unique `der_noise` values:", sorted(unique_der_noise_values))

    # Add baseline data points (standard condition) to the plot data
    for model_name, metrics in plot_data.items():
        if model_name in baseline_data:
            baseline = baseline_data[model_name]
            plot_data[model_name].insert(0, baseline[0])  # Insert baseline as the first data point

    return plot_data

def prepare_task_plot_data(results, condition, task_name, metric_name="accuracy"):
    plot_data = {}
    baseline_data = {}

    for model_name, model_results in results.items():
        if condition in model_results:
            condition_results = model_results[condition]
            avg_metrics = []

            # Retrieve baseline data for the 'standard' condition
            if 'standard' in model_results:
                standard_results = model_results['standard']
                baseline_data[model_name] = [
                    entry['result'].get(metric_name, 0) for entry in standard_results
                ]

            for entry in condition_results:
                metrics_list = entry['result'].get('metrics', [])
                
                # Extract metric values for the task
                task_metrics = [
                    metric[metric_name] for metric in metrics_list 
                    if metric['task'].lower() == task_name.lower() and metric_name in metric
                ]

                # Flatten if task_metrics contains lists
                flat_metrics = [value for metric in task_metrics for value in (metric if isinstance(metric, list) else [metric])]

                if flat_metrics:
                    avg_metrics.append(sum(flat_metrics) / len(flat_metrics))

            if avg_metrics:
                plot_data[model_name] = avg_metrics

    # Add baseline data points (standard condition) to the plot data
    for model_name, metrics in plot_data.items():
        if model_name in baseline_data:
            baseline = baseline_data[model_name]
            plot_data[model_name].insert(0, baseline[0])  # Insert baseline as the first data point

    return plot_data

def plot_results(plot_data, condition, x_values, metric_name="accuracy", output_dir="output_plots"):
    # Create subfolders
    condition_dir = os.path.join(output_dir, condition)
    metric_dir = os.path.join(condition_dir, metric_name)
    os.makedirs(metric_dir, exist_ok=True)

    # Define markers
    marker_cycle = itertools.cycle(('o', 's', 'D', '^', 'v', '>', '<', '*', 'p', 'h'))
    model_markers = {model_name: next(marker_cycle) for model_name in plot_data.keys()}

    # Create the plot
    plt.figure(figsize=(10, 6))
    for model_name, metrics in plot_data.items():
        plt.plot(x_values, metrics, label=model_name, marker=model_markers[model_name])

    plt.xticks(x_values, labels=[f"{val:.2f}" for val in x_values], rotation=45)
    plt.title(f"{metric_name.capitalize()} across Models ({condition})")
    plt.xlabel("Noise Level (0.0 - 1.0)")
    plt.ylabel(metric_name.capitalize())
    plt.legend()
    plt.grid(True)

    # Save the plot
    file_path = os.path.join(metric_dir, f"{condition}_{metric_name}.png")
    plt.savefig(file_path)
    plt.close()

    print(f"Plot saved to: {file_path}")

# Plotting task-specific results
def plot_task_results(plot_data, condition, task_name, x_values, metric_name="accuracy", output_dir="output_plots"):
    # Create a subfolder for the condition if it doesn't exist
    condition_dir = os.path.join(output_dir, condition)
    if not os.path.exists(condition_dir):
        os.makedirs(condition_dir)

    # Create a subfolder for the task and metric type (e.g., task-specific results)
    task_dir = os.path.join(condition_dir, task_name)
    if not os.path.exists(task_dir):
        os.makedirs(task_dir)

    metric_dir = os.path.join(task_dir, metric_name)
    if not os.path.exists(metric_dir):
        os.makedirs(metric_dir)

    # Define unique markers for models
    marker_cycle = itertools.cycle(('o', 's', 'D', '^', 'v', '>', '<', '*', 'p', 'h'))
    model_markers = {model_name: next(marker_cycle) for model_name in plot_data.keys()}

    # Create the plot
    plt.figure(figsize=(10, 6))
    for model_name, metrics in plot_data.items():
        plt.plot(x_values, metrics, label=model_name, marker=model_markers[model_name])

    # Set x-axis labels
    if condition in ["cli_noise", "der_noise", "combined"]:
        plt.xticks(x_values, labels=[f"{val:.2f}" for val in x_values])
        x_label = "Noise Level (0.05 - 1.0)"
    else:
        plt.xticks(x_values)
        x_label = "Corruption Severity (0-15)"

    plt.title(f"{metric_name.capitalize()} across Models ({condition})")
    plt.xlabel(x_label)  # Update the x-axis label
    plt.ylabel(metric_name.capitalize())
    plt.legend()
    plt.grid(True)

    # Save the plot to the appropriate subfolder
    file_path = os.path.join(metric_dir, f"{condition}_{task_name}_{metric_name}.png")
    plt.savefig(file_path)
    plt.close()  # Close the plot to free up memory

    print(f"Plot saved to: {file_path}")

def main():
    file_path = "/home/matthewcockayne/Documents/PhD/experiments/results/robustness/test/json_check3/Robustness_log.txt"
    results = load_results(file_path)

    tasks = ["diag", "pn", "bwv", "vs", "pig", "str", "dag", "rs"]
    conditions = ["corruptions", "cli_noise", "der_noise", "combined"]
    metrics = ["accuracy", "precision", "sensitivity", "f1_scores", "auc", "specificity"]

    # Define exact x-values for each condition
    x_values_map = {
        "corruptions": np.arange(16),  # Assuming 15 corruption severity levels
        "cli_noise": [round(0.05 * i, 2) for i in range(21)],  # From 0.0 to 1.0
        "der_noise": [round(0.05 * i, 2) for i in range(21)],  # From 0.0 to 1.0
        "combined":  [round(0.05 * i, 2) for i in range(21)]  # From 0.0 to 1.0
    }

    # Plot overall accuracy and loss across models
    for condition in conditions:
        x_values = x_values_map[condition]

        # Plot overall metrics (accuracy, loss, etc.)
        for metric_name in ["accuracy", "loss"]:
            plot_data = prepare_plot_data(results, condition, metric_name)
            if plot_data:
                plot_results(
                    plot_data, condition, x_values, metric_name,
                    output_dir="/home/matthewcockayne/Documents/PhD/experiments/results/robustness/test/json_check3/"
                )
                #print('plot')
            else:
                print(f"No data for condition '{condition}', metric '{metric_name}'.")

        # Plot individual task results
        for task in tasks:
            for metric_name in metrics:
                plot_data = prepare_task_plot_data(results, condition, task, metric_name)
                if plot_data:
                    plot_task_results(
                        plot_data, condition, task, x_values, metric_name,
                        output_dir="/home/matthewcockayne/Documents/PhD/experiments/results/robustness/test/json_check3/"
                    )
                    #print('plot')
                else:
                    print(f"No data for condition '{condition}', task '{task}', metric '{metric_name}'.")


# Run the visualization
if __name__ == "__main__":
    main()