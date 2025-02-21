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
from models.models_test.model_swinv2fusion import MM_Transformer
from models.TFormer import TFormer
from models.DermFormer import MM_nest, MMNestLoss
from models.NesT.nest_der import nest_der
from models.NesT.nest_cli import nest_cli
from models.NesT.nest_multimodalconcat import nest_MMC

def setup_logging(log_path):
    try:
        if not os.path.exists(log_path):
            os.makedirs(log_path)
        log_file = open(os.path.join(log_path, 'Robustness_log.txt'), 'w')
        return log_file
    except Exception as e:
        print(f"Error creating log file: {e}")
        return None  # Return None in case of an error

def log_params(options, log_file):
    print('===========Test Params===============')
    log_file.write('===========Test Params===============\n')
    for name, param in options.items():
        print(f'{name}: {param}')
        log_file.write(f'{name}: {param}\n')
    print('========================================')
    log_file.write('========================================\n')

def save_plot(fig, plot_name, log_path):
    if not os.path.exists(log_path):
        os.makedirs(log_path)  # Create the directory if it doesn't exist
    
    # Save the plot as a PNG file
    fig.savefig(os.path.join(log_path, plot_name))
    print(f"Plot saved as {plot_name} in {log_path}")

def setup_models(model_paths,class_num):
    return {
        'DermFormer': (MM_nest(class_num), model_paths['DermFormer']),
        'NEST-MMC': (nest_MMC(class_num), model_paths['NEST-MMC']),
        'NEST-DER': (nest_der(class_num), model_paths['NEST-DER']),
        'NEST-CLI': (nest_cli(class_num), model_paths['NEST-CLI']),
        'TFormer': (TFormer(class_num), model_paths['TFormer']),
    }

def setup_data_loaders(options, derm_data_group, num_distortions, der_noise, cli_noise):
    # Instantiate the datasets
    test_dataset = dataset(derm=derm_data_group, shape=(224, 224), mode='test')  # Ensure `dataset` is a valid class or function
    corrupt_dataset = dataset_corrupt(derm=derm_data_group, shape=(224, 224), mode='test', num_distortions=num_distortions)
    noise_dataset = dataset_noise(derm=derm_data_group, shape=(224, 224), mode='test', derm_noise=der_noise, clinical_noise=cli_noise)

    # Create data loaders from the datasets
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=1, pin_memory=True)
    corrupt_loader = DataLoader(corrupt_dataset, batch_size=1, shuffle=False, num_workers=1, pin_memory=True)
    noise_loader = DataLoader(noise_dataset, batch_size=1, shuffle=False, num_workers=1, pin_memory=True)
    
    return test_loader, corrupt_loader, noise_loader

def test(model, test_loader, options, log_file):
    if log_file is None:
        print("Warning: log_file is None. Skipping log file writing.")
    else:
        log_file.write(f"Model Test: \n")
    
    avg_test_loss = 0
    test_confusion_matrices = [ConfusionMatrix(num_classes=options['class_num'], labels=options['labels']) for _ in range(8)]
    pro_list = [[] for _ in range(8)]  # List to store predicted probabilities for each class
    lab_list = [[] for _ in range(8)]  # List to store true labels for each class
    total_correct = 0
    total_samples = 0

    # Define label names corresponding to your outputs
    label_names = ['diag', 'pn', 'bwv', 'vs', 'pig', 'str', 'dag', 'rs']
    
    # Get class weights from the dataset
    class_weights = test_loader.dataset.class_weights
    #weights = [class_weights[i].cuda() if options['cuda'] else class_weights[i] for i in range(8)]
    loss_fn = MMNestLoss(class_weights=None)

    with torch.no_grad():
        for der_data, cli_data, meta_data, meta_con, target in test_loader:
            # Extract individual target labels
            diagnosis_label = target[0].cuda().squeeze(1)
            pn_label = target[1].cuda().squeeze(1)
            bmv_label = target[2].cuda().squeeze(1)
            vs_label = target[3].cuda().squeeze(1)
            pig_label = target[4].cuda().squeeze(1)
            str_label = target[5].cuda().squeeze(1)
            dag_label = target[6].cuda().squeeze(1)
            rs_label = target[7].cuda().squeeze(1)

            # Gather all targets into a list
            targets = [diagnosis_label, pn_label, bmv_label, vs_label, pig_label, str_label, dag_label, rs_label]

            if options['cuda']:
                der_data, cli_data, meta_data = der_data.cuda(), cli_data.cuda(), meta_data.cuda().float()
                #der_data, cli_data, meta_data = Variable(der_data), Variable(cli_data), Variable(meta_data)
                meta_data = meta_data.long()
                meta_con = meta_con.long()

            # Forward pass
            outputs = model(meta_data, meta_con, cli_data, der_data)

            # Compute the loss for each output class and then average them
            loss = loss_fn(outputs, targets)
            avg_test_loss += loss.item()

            # Extract ensemble predictions from outputs
            predictions = [
                output[-1] for output in outputs.values()  # Get the last element (ensemble) from each classification head
            ]

            for i, (pred, true) in enumerate(zip(predictions, targets)):
                pred_np = pred.cpu().numpy()
                target_np = true.cpu().numpy()
                if len(target_np.shape) > 1:
                    target_np = target_np.squeeze()
                test_confusion_matrices[i].update(pred_np, target_np)

                # Correctly access softmax probabilities
                logits = outputs[label_names[i]][0]  # Use the logits for softmax (the first element)
                pro_list[i].extend(F.softmax(logits, dim=1).cpu().numpy())  # Store softmax probabilities
                lab_list[i].extend(true.cpu().numpy())

            # Calculate accuracy
            correct = sum([pred.eq(targets[i]).sum().item() for i, pred in enumerate(predictions)])
            total = sum([targets[i].size(0) for i in range(8)])
            total_correct += correct
            total_samples += total

    avg_test_loss /= len(test_loader)
    accuracy = total_correct / total_samples

    log_file.write(f"Test Loss: {avg_test_loss:.4f}, Accuracy: {accuracy:.4f}\n")

    # Convert pro_list and lab_list to numpy arrays and flatten appropriately
    for i in range(8):
        pro_list[i] = np.vstack(pro_list[i])  # Convert list of arrays to a single numpy array
        lab_list[i] = np.hstack(lab_list[i])  # Flatten the list of arrays

    # Call summary method for each confusion matrix with pro_list and lab_list
    fpr_dict_list = []
    tpr_dict_list = []
    for i, cm in enumerate(test_confusion_matrices):
        _, _, _, fpr_dict, tpr_dict = cm.summary(pro_list[i], lab_list[i], File=None)
        fpr_dict_list.append(fpr_dict)
        tpr_dict_list.append(tpr_dict)

    return accuracy

def test_standardloss(model, test_loader, options, log_file):
    if log_file is None:
        print("Warning: log_file is None. Skipping log file writing.")
    else:
        log_file.write(f"Model Test: \n")

    names = ('Diag', 'PN', 'BWV', 'VS', 'PIG', 'STR', 'DaG', 'RS')
    avg_test_loss = 0
    test_confusion_matrices = [ConfusionMatrix(num_classes=options['class_num'], labels=options['labels']) for _ in range(8)]
    pro_list = [[] for _ in range(8)]  # List to store predicted probabilities for each class
    lab_list = [[] for _ in range(8)]  # List to store true labels for each class
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        #for der_data, cli_data, meta_data, meta_con, target in test_loader:
        for der_data, cli_data, meta_data, meta_con, target in test_loader:
            # Extract individual target labels
            diagnosis_label = target[0].cuda().squeeze(1)
            pn_label = target[1].cuda().squeeze(1)
            bmv_label = target[2].cuda().squeeze(1)
            vs_label = target[3].cuda().squeeze(1)
            pig_label = target[4].cuda().squeeze(1)
            str_label = target[5].cuda().squeeze(1)
            dag_label = target[6].cuda().squeeze(1)
            rs_label = target[7].cuda().squeeze(1)

            # Gather all targets into a list
            targets = [diagnosis_label, pn_label, bmv_label, vs_label, pig_label, str_label, dag_label, rs_label]

            if options['cuda']:
                der_data, cli_data, meta_data = der_data.cuda(), cli_data.cuda(), meta_data.cuda().float()
                der_data, cli_data, meta_data = Variable(der_data), Variable(cli_data), Variable(meta_data)
                #meta_data = meta_data.long()
                #meta_con = meta_con.long()

            # Forward pass
            outputs = model(meta_data, meta_con, cli_data, der_data)

            # Get class weights from the dataset
            class_weights = test_loader.dataset.class_weights
            weights = [class_weights[i].cuda() if options['cuda'] else class_weights[i] for i in range(8)]

            # Compute the loss for each output class and then average them
            losses = [F.cross_entropy(output, target, weight=weight) for output, target, weight in zip(outputs, targets, weights)]
            loss = sum(losses) / len(losses)
            avg_test_loss += loss.item()

            # Get predictions and update confusion matrices
            predictions = [output.data.max(1)[1] for output in outputs]
            for i, (pred, true) in enumerate(zip(predictions, targets)):
                pred_np = pred.cpu().numpy()
                target_np = true.cpu().numpy()
                if len(target_np.shape) > 1:
                    target_np = target_np.squeeze()
                test_confusion_matrices[i].update(pred_np, target_np)
                pro_list[i].extend(F.softmax(outputs[i], dim=1).cpu().numpy())  # Store softmax probabilities
                lab_list[i].extend(true.cpu().numpy())

            # Calculate accuracy
            correct = sum([pred.eq(targets[i]).sum().item() for i, pred in enumerate(predictions)])
            total = sum([targets[i].size(0) for i in range(8)])
            total_correct += correct
            total_samples += total

    avg_test_loss /= len(test_loader)
    accuracy = total_correct / total_samples

    log_file.write(f"Test Loss: {avg_test_loss:.4f}, Accuracy: {accuracy:.4f}\n")

    # Convert pro_list and lab_list to numpy arrays and flatten appropriately
    for i in range(8):
        pro_list[i] = np.vstack(pro_list[i])  # Convert list of arrays to a single numpy array
        lab_list[i] = np.hstack(lab_list[i])  # Flatten the list of arrays

    # Call summary method for each confusion matrix with pro_list and lab_list
    fpr_dict_list = []
    tpr_dict_list = []
    for i, cm in enumerate(test_confusion_matrices):
        _, _, _, fpr_dict, tpr_dict = cm.summary(pro_list[i], lab_list[i], File=None)
        fpr_dict_list.append(fpr_dict)
        tpr_dict_list.append(tpr_dict)

    return accuracy

def test_MMconcat(model, test_loader, options, log_file):
    if log_file is None:
        print("Warning: log_file is None. Skipping log file writing.")
    else:
        log_file.write(f"Model Test: \n")

    names = ('Diag', 'PN', 'BWV', 'VS', 'PIG', 'STR', 'DaG', 'RS')
    avg_test_loss = 0
    test_confusion_matrices = [ConfusionMatrix(num_classes=options['class_num'], labels=options['labels']) for _ in range(8)]
    pro_list = [[] for _ in range(8)]  # List to store predicted probabilities for each class
    lab_list = [[] for _ in range(8)]  # List to store true labels for each class
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        #for der_data, cli_data, meta_data, meta_con, target in test_loader:
        for der_data, cli_data, meta_data, meta_con, target in test_loader:
            # Extract individual target labels
            diagnosis_label = target[0].cuda().squeeze(1)
            pn_label = target[1].cuda().squeeze(1)
            bmv_label = target[2].cuda().squeeze(1)
            vs_label = target[3].cuda().squeeze(1)
            pig_label = target[4].cuda().squeeze(1)
            str_label = target[5].cuda().squeeze(1)
            dag_label = target[6].cuda().squeeze(1)
            rs_label = target[7].cuda().squeeze(1)

            # Gather all targets into a list
            targets = [diagnosis_label, pn_label, bmv_label, vs_label, pig_label, str_label, dag_label, rs_label]

            if options['cuda']:
                der_data, cli_data, meta_data = der_data.cuda(), cli_data.cuda(), meta_data.cuda().float()
                der_data, cli_data, meta_data = Variable(der_data), Variable(cli_data), Variable(meta_data)
                meta_data = meta_data.long()
                meta_con = meta_con.long()

            # Forward pass
            outputs = model(meta_data, meta_con, cli_data, der_data)

            # Get class weights from the dataset
            class_weights = test_loader.dataset.class_weights
            weights = [class_weights[i].cuda() if options['cuda'] else class_weights[i] for i in range(8)]

            # Compute the loss for each output class and then average them
            losses = [F.cross_entropy(output, target, weight=weight) for output, target, weight in zip(outputs, targets, weights)]
            loss = sum(losses) / len(losses)
            avg_test_loss += loss.item()

            # Get predictions and update confusion matrices
            predictions = [output.data.max(1)[1] for output in outputs]
            for i, (pred, true) in enumerate(zip(predictions, targets)):
                pred_np = pred.cpu().numpy()
                target_np = true.cpu().numpy()
                if len(target_np.shape) > 1:
                    target_np = target_np.squeeze()
                test_confusion_matrices[i].update(pred_np, target_np)
                pro_list[i].extend(F.softmax(outputs[i], dim=1).cpu().numpy())  # Store softmax probabilities
                lab_list[i].extend(true.cpu().numpy())

            # Calculate accuracy
            correct = sum([pred.eq(targets[i]).sum().item() for i, pred in enumerate(predictions)])
            total = sum([targets[i].size(0) for i in range(8)])
            total_correct += correct
            total_samples += total

    avg_test_loss /= len(test_loader)
    accuracy = total_correct / total_samples

    log_file.write(f"Test Loss: {avg_test_loss:.4f}, Accuracy: {accuracy:.4f}\n")

    # Convert pro_list and lab_list to numpy arrays and flatten appropriately
    for i in range(8):
        pro_list[i] = np.vstack(pro_list[i])  # Convert list of arrays to a single numpy array
        lab_list[i] = np.hstack(lab_list[i])  # Flatten the list of arrays

    # Call summary method for each confusion matrix with pro_list and lab_list
    fpr_dict_list = []
    tpr_dict_list = []
    for i, cm in enumerate(test_confusion_matrices):
        _, _, _, fpr_dict, tpr_dict = cm.summary(pro_list[i], lab_list[i], File=None)
        fpr_dict_list.append(fpr_dict)
        tpr_dict_list.append(tpr_dict)

    return accuracy

def plot_results(results_all, distortions, cli_noises, der_noises, output_dir=None):
    """
    Plot results for each model across specific distortions, cli_noise, der_noise, and combined values (separate subplots).

    Args:
        results_all (dict): Results dictionary containing performance for each model and condition.
        distortions (list): List of distortion levels for corruptions.
        cli_noises (list): List of cli_noise values for cli_noise condition.
        der_noises (list): List of der_noise values for der_noise condition.
        output_dir (str): Directory to save plots. If None, plots are displayed but not saved.
    """
    for model_name, model_results in results_all.items():
        # Prepare subplots
        conditions = ['corruptions', 'combined', 'der_noise', 'cli_noise']
        fig, axs = plt.subplots(len(conditions), 1, figsize=(10, 4 * len(conditions)), sharex=False)

        # Extract the standard result for the baseline (dist_0_cli_0.0_der_0.0)
        standard_experiment = "dist_0_cli_0.0_der_0.0"
        standard_result = 0  # Default value if no data exists for the standard experiment
        if standard_experiment in model_results:
            standard_data = model_results[standard_experiment]
            if 'standard' in standard_data and standard_data['standard']:
                standard_result = standard_data['standard'][0]  # Extract the standard result

        # Plot the standard result at x = 0 for all conditions
        for ax, cond in zip(axs, conditions):
            x = [0]  # Plot x = 0 for the standard result
            y = [standard_result]  # Use the extracted standard result as the y value

            # Now plot the other experiments for each condition
            if cond == 'corruptions':
                x_axis = distortions
                # Plot the other results for corruptions
                for distortion in distortions:
                    experiment = f"dist_{distortion}_cli_0.0_der_0.0"
                    if experiment in model_results and cond in model_results[experiment] and model_results[experiment][cond]:
                        x.append(distortion)
                        y.append(model_results[experiment][cond][0])

            elif cond == 'combined':
                x_axis = cli_noises  # Use cli_noises as the x-axis
                # Plot the other results for combined
                for cli_noise in cli_noises:
                    for der_noise in der_noises:
                        experiment = f"dist_0_cli_{cli_noise}_der_{der_noise}"
                        if experiment in model_results and cond in model_results[experiment] and model_results[experiment][cond]:
                            x.append(cli_noise)
                            y.append(model_results[experiment][cond][0])

            elif cond == 'cli_noise':
                x_axis = cli_noises
                # Plot the other results for cli_noise
                for cli_noise in cli_noises:
                    experiment = f"dist_0_cli_{cli_noise}_der_0.0"
                    if experiment in model_results and cond in model_results[experiment] and model_results[experiment][cond]:
                        x.append(cli_noise)
                        y.append(model_results[experiment][cond][0])

            elif cond == 'der_noise':
                x_axis = der_noises
                # Plot the other results for der_noise
                for der_noise in der_noises:
                    experiment = f"dist_0_cli_0.0_der_{der_noise}"
                    if experiment in model_results and cond in model_results[experiment] and model_results[experiment][cond]:
                        x.append(der_noise)
                        y.append(model_results[experiment][cond][0])

            # Plot only if data exists
            if x and y:
                ax.plot(x, y, marker='o', label=f"{cond.capitalize()} Performance")
                ax.legend(loc="upper left", fontsize=10)

            ax.set_title(f"{cond.capitalize()} Performance", fontsize=14)
            ax.set_ylabel("Accuracy", fontsize=12)
            ax.set_xlabel(f"{cond.capitalize()} Levels", fontsize=12)
            ax.set_xticks(x_axis)  # Use the defined levels as x-axis ticks
            ax.grid(True)

        # Adjust layout
        plt.tight_layout()

        # Save or display the plot
        if output_dir:
            save_path = os.path.join(output_dir, f"{model_name}_results.png")
            plt.savefig(save_path)
            print(f"Plot saved at {save_path}")
        else:
            plt.show()

        plt.close(fig)

def plot_results_all_models(results_all, distortions, cli_noises, der_noises, output_dir=None):
    """
    Plot results for all models across specific distortions, cli_noise, der_noise, and combined values
    on the same set of subplots for comparison.

    Args:
        results_all (dict): Results dictionary containing performance for each model and condition.
        distortions (list): List of distortion levels for corruptions.
        cli_noises (list): List of cli_noise values for cli_noise condition.
        der_noises (list): List of der_noise values for der_noise condition.
        output_dir (str): Directory to save plots. If None, plots are displayed but not saved.
    """
    # Prepare subplots
    conditions = ['corruptions', 'combined', 'der_noise', 'cli_noise']
    fig, axs = plt.subplots(len(conditions), 1, figsize=(10, 4 * len(conditions)), sharex=False)

    # Marker styles for models
    markers = cycle(['o', 's', 'x', 'D', '^', 'v', '<', '>'])  # Cycle through different markers

    # For each condition, plot results for all models
    for ax, cond in zip(axs, conditions):
        # Set axis labels and title
        ax.set_title(f"{cond.capitalize()} Performance", fontsize=14)
        ax.set_ylabel("Accuracy", fontsize=12)
        ax.set_xlabel(f"{cond.capitalize()} Levels", fontsize=12)
        ax.grid(True)

        # Reset marker cycle for each condition
        model_markers = cycle(markers)

        # Iterate over each model in results_all
        for model_name, model_results in results_all.items():
            x = []  # Store x-values for the condition
            y = []  # Store y-values (accuracies)

            # Extract the standard result for the baseline (dist_0_cli_0.0_der_0.0) for this model
            standard_experiment = "dist_0_cli_0.0_der_0.0"
            standard_result = 0  # Default value if no data exists for the standard experiment
            if standard_experiment in model_results:
                standard_data = model_results[standard_experiment]
                if 'standard' in standard_data and standard_data['standard']:
                    standard_result = standard_data['standard'][0]

            # Plot the standard result at x = 0 for the current model
            x.append(0)
            y.append(standard_result)

            # Now plot the other experiments for each condition
            if cond == 'corruptions':
                x_axis = distortions
                for distortion in distortions:
                    experiment = f"dist_{distortion}_cli_0.0_der_0.0"
                    if experiment in model_results and cond in model_results[experiment] and model_results[experiment][cond]:
                        x.append(distortion)
                        y.append(model_results[experiment][cond][0])

            elif cond == 'combined':
                x_axis = cli_noises
                for cli_noise in cli_noises:
                    for der_noise in der_noises:
                        experiment = f"dist_0_cli_{cli_noise}_der_{der_noise}"
                        if experiment in model_results and cond in model_results[experiment] and model_results[experiment][cond]:
                            x.append(cli_noise)
                            y.append(model_results[experiment][cond][0])

            elif cond == 'cli_noise':
                x_axis = cli_noises
                for cli_noise in cli_noises:
                    experiment = f"dist_0_cli_{cli_noise}_der_0.0"
                    if experiment in model_results and cond in model_results[experiment] and model_results[experiment][cond]:
                        x.append(cli_noise)
                        y.append(model_results[experiment][cond][0])

            elif cond == 'der_noise':
                x_axis = der_noises
                for der_noise in der_noises:
                    experiment = f"dist_0_cli_0.0_der_{der_noise}"
                    if experiment in model_results and cond in model_results[experiment] and model_results[experiment][cond]:
                        x.append(der_noise)
                        y.append(model_results[experiment][cond][0])

            # Plot the results for the current model on the same axes
            if x and y:
                marker = next(model_markers)  # Get a unique marker for each model
                ax.plot(x, y, marker=marker, label=model_name)  # Plot with marker

        # Set the x-axis ticks based on the condition's axis (e.g., distortions, cli_noises, etc.)
        ax.set_xticks(x_axis)
        ax.legend(loc="upper right", bbox_to_anchor=(1.15, 1), fontsize=10)

    # Adjust layout
    plt.tight_layout()

    # Save or display the plot
    if output_dir:
        save_path = os.path.join(output_dir, "all_models_results.png")
        plt.savefig(save_path)
        print(f"Plot saved at {save_path}")
    else:
        plt.show()

    plt.close(fig)

def run_experiment(model_class_and_path, options, use_custom_loss, use_MMcat_loss, log_file, model_name):
    """
    Run an experiment for a single model and set of options.
    Returns accuracy for each test condition.
    """
    # Unpack the model class and model path
    model, model_path = model_class_and_path

    # Load the pre-trained weights (if any)
    if model_path:
        model.eval()
        model.load_state_dict(torch.load(model_path))
        print(f"Loaded model weights from {model_path}")
    
    derm_data_group, column_interest = load_dataset(dir_release=options['dir_release'])

    # Set up data loaders based on test conditions
    test_loader, corrupt_loader, noise_loader = setup_data_loaders(
        options,
        derm_data_group=derm_data_group,
        num_distortions=options['num_distortions'],
        der_noise=options['der_noise'],
        cli_noise=options['cli_noise']
    )
    
    # Select the test function
    if use_MMcat_loss:
        test_func = test_MMconcat  # Use the third loss function
    elif use_custom_loss:
        test_func = test  # Use the custom loss function
    else:
        test_func = test_standardloss 
    
    # Initialize results dictionary (using lists to store results for multiple test conditions)
    results = {
        'standard': [],
        'corruptions': [],
        'combined': [],
        'der_noise': [],
        'cli_noise': []
    }
    
    if log_file is not None:
        log_file.write(f"Starting experiment with model {model_name}...\n")
        log_file.write(f"Test conditions: num_distortions={options['num_distortions']}, cli_noise={options['cli_noise']}, der_noise={options['der_noise']}\n")

    if options['num_distortions'] == 0 and options['cli_noise'] == 0.0 and options['der_noise'] == 0.0:
        # Standard test: no distortions, no noise
        result = test_func(model, test_loader, options, log_file)
        results['standard'].append(result)

    if options['num_distortions'] > 0 and options['cli_noise'] == 0.0 and options['der_noise'] == 0.0:
        # Corruptions test: distortions only, no noise
        results['corruptions'].append(test_func(model, corrupt_loader, options, log_file))

    if options['num_distortions'] == 0 and options['cli_noise'] > 0.0 and options['der_noise'] == 0.0:
        # Clinical noise test: only cli_noise, no distortions or der_noise
        results['cli_noise'].append(test_func(model, noise_loader, {**options, 'cli_noise': options['cli_noise']}, log_file))

    if options['num_distortions'] == 0 and options['der_noise'] > 0.0 and options['cli_noise'] == 0.0:
        # Dermoscopy noise test: only der_noise, no distortions or cli_noise
        results['der_noise'].append(test_func(model, noise_loader, {**options, 'der_noise': options['der_noise']}, log_file))

    if options['num_distortions'] == 0 and options['cli_noise'] > 0.0 and options['der_noise'] > 0.0:
        # Combined noise test: both cli_noise and der_noise, no distortions
        if options['cli_noise'] == options['der_noise']:  # Ensure cli_noise and der_noise are the same
            results['combined'].append(test_func(model, noise_loader, {**options, 'cli_noise': options['cli_noise'], 'der_noise': options['der_noise']}, log_file))

    if log_file is not None:
        log_file.write("Completed experiments...\n")
        log_file.write(f"Results for model {model_name}: {results}\n")

    return results

def main(params):
    log_path = params.get('log_path', 'default_log_path')
    log_file = setup_logging(log_path)
    if log_file is None:
        print(f"Error: Failed to create log file at {log_path}.")
        return  # Exit early if logging fails

    model_paths = {
        'TFormer': params['tformer_model_path'],
        'NEST-DER': params['nestder_model_path'],
        'NEST-CLI': params['nestcli_model_path'],
        'NEST-MMC': params['nestmmc_model_path'],
        'DermFormer': params['dermformer_model_path'],
    }

    # Experiment settings
    models = setup_models(model_paths, params['class_num'])
    results_all = {}

    # Conditions to test
    distortions = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    cli_noises = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0]
    der_noises = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0]

    for model_name, model_tuple in models.items():
        print(f"Running experiments for model: {model_name}")
        results_all[model_name] = {}

        model_class, model_path = model_tuple

        # Instantiate the model
        model = model_class

        if model_name == 'DermFormer':
            use_custom_loss = True
            use_MMcat_loss = False
        if model_name == 'TFormer':
            use_custom_loss = False
            use_MMcat_loss = False
        elif model_name == 'NEST-DER':
            use_custom_loss = False
            use_MMcat_loss = False  
        elif model_name == 'NEST-CLI':
            use_custom_loss = False
            use_MMcat_loss = False 
        elif model_name == 'NEST-MMC':
            use_custom_loss = False
            use_MMcat_loss = True 

        if params['cuda']:
            model = model.cuda()

        # Running experiments for each combination of distortions, cli_noise, and der_noise
        for num_distortions in distortions:
            for cli_noise in cli_noises:
                for der_noise in der_noises:
                    # Build the options dictionary
                    options = {
                        'log_path': os.path.join(params['log_path'], f'{model_name}_distortions_{num_distortions}_cli_{cli_noise}_der_{der_noise}.txt'),
                        'model_path': f'/path/to/{model_name}.pth',
                        'cuda': params['cuda'],
                        'class_num': 5,
                        'labels': ['diag', 'pn', 'bwv', 'vs', 'pig', 'str', 'dag', 'rs'],
                        'num_distortions': num_distortions,
                        'der_noise': der_noise,
                        'cli_noise': cli_noise,
                        'use_custom_loss': use_custom_loss,
                        'dir_release': params['dir_release']
                    }

                    model_path = model_paths[model_name]
                    results = run_experiment((model, model_path), options, use_custom_loss, use_MMcat_loss, log_file, model_name)

                    # Save results for plotting, using a unique key for each condition
                    key = f"dist_{num_distortions}_cli_{cli_noise}_der_{der_noise}"
                    results_all[model_name][key] = results
                    print(results_all)
    # Plot results
    plot_results(results_all, distortions, cli_noises, der_noises, log_path)
    plot_results_all_models(results_all, distortions, cli_noises, der_noises, log_path)

if __name__ == "__main__":
    # Set up command-line argument parsing
    parser = argparse.ArgumentParser(description="Experiment setup and execution for models.")
    
    # Common experiment settings
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to run the experiment')
    parser.add_argument('--dir_release', dest='dir_release', type=str, default="/home/matthewcockayne/datasets/Derm7pt/release_v0/release_v0", help = 'Path to dataset directory')
    parser.add_argument('--log_path', type=str, default= '/home/matthewcockayne/Documents/PhD/experiments/results/robustness/test/morepoints/', help='Directory where logs will be saved')
    parser.add_argument('--cuda', type=bool, default=True, help='Whether to use CUDA')
    parser.add_argument('--class_num', type=int, default=5, help='Number of classes')
    parser.add_argument('--labels', type=str, default='["diag", "pn", "bwv", "vs", "pig", "str", "dag", "rs"]', help='Labels for the classes')

    # Add arguments for model paths
    parser.add_argument('--tformer_model_path', type=str, default= '/home/matthewcockayne/Documents/PhD/experiments/tformer (best)/models/best_model_6.pth', help='Path to TFormer model')
    parser.add_argument('--nestder_model_path', type=str, default= '/home/matthewcockayne/Documents/PhD/experiments/results/unimodal/der/20241118-145008/models/best_model_61.pth', help='Path to NEST-DER model')
    parser.add_argument('--nestcli_model_path', type=str, default= '/home/matthewcockayne/Documents/PhD/experiments/results/unimodal/cli/20241118-142556/models/best_model_37.pth', help='Path to NEST-CLI model')
    parser.add_argument('--nestmmc_model_path', type=str, default= '/home/matthewcockayne/Documents/PhD/experiments/results/multimodal_concat/cli_der_meta/20241129-170927/models/best_model_86.pth', help='Path to NEST-MMC model')
    parser.add_argument('--dermformer_model_path', type=str, default='/home/matthewcockayne/Documents/PhD/experiments/results/fusion/20241122-163959/models/best_model_132.pth', help='Path to DermFormer model')

    # Parse the command-line arguments
    args = parser.parse_args()
    
    # Convert the string representation of the labels back to a list
    args.labels = eval(args.labels)
    
    # Convert arguments to a dictionary
    params = vars(args)
    
    # Call main function with parsed arguments
    main(params)
