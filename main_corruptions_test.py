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
import ast
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
        log_file = open(os.path.join(log_path, 'robustness_log.txt'), 'w')
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

def test(model, test_loader, options, log_file):
    if log_file is None:
        print("Warning: log_file is None. Skipping log file writing.")

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

    #log_file.write(f"Test Loss: {avg_test_loss:.4f}, Accuracy: {accuracy:.4f}\n")

    # Convert pro_list and lab_list to numpy arrays and flatten appropriately
    for i in range(8):
        pro_list[i] = np.vstack(pro_list[i])  # Convert list of arrays to a single numpy array
        lab_list[i] = np.hstack(lab_list[i])  # Flatten the list of arrays

    # Call summary method for each confusion matrix with pro_list and lab_list
    fpr_dict_list = []
    tpr_dict_list = []
    for i, cm in enumerate(test_confusion_matrices):
        acc, f1_scores, aucs, precision, sensitivity, specificity, fpr_dict, tpr_dict = cm.summary(pro_list[i], lab_list[i], File=None)
        fpr_dict_list.append(fpr_dict)
        tpr_dict_list.append(tpr_dict)

    return accuracy

def test_standardloss(model, test_loader, options, log_file):
    if log_file is None:
        print("Warning: log_file is None. Skipping log file writing.")

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

    #log_file.write(f"Test Loss: {avg_test_loss:.4f}, Accuracy: {accuracy:.4f}\n")

    # Convert pro_list and lab_list to numpy arrays and flatten appropriately
    for i in range(8):
        pro_list[i] = np.vstack(pro_list[i])  # Convert list of arrays to a single numpy array
        lab_list[i] = np.hstack(lab_list[i])  # Flatten the list of arrays

    # Call summary method for each confusion matrix with pro_list and lab_list
    fpr_dict_list = []
    tpr_dict_list = []
    for i, cm in enumerate(test_confusion_matrices):
        acc, f1_scores, aucs, precision, sensitivity, specificity, fpr_dict, tpr_dict = cm.summary(pro_list[i], lab_list[i], File=None)
        fpr_dict_list.append(fpr_dict)
        tpr_dict_list.append(tpr_dict)

    return accuracy

def test_MMconcat(model, test_loader, options, log_file):
    if log_file is None:
        print("Warning: log_file is None. Skipping log file writing.")

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

    #log_file.write(f"Test Loss: {avg_test_loss:.4f}, Accuracy: {accuracy:.4f}\n")

    # Convert pro_list and lab_list to numpy arrays and flatten appropriately
    for i in range(8):
        pro_list[i] = np.vstack(pro_list[i])  # Convert list of arrays to a single numpy array
        lab_list[i] = np.hstack(lab_list[i])  # Flatten the list of arrays

    # Call summary method for each confusion matrix with pro_list and lab_list
    fpr_dict_list = []
    tpr_dict_list = []
    for i, cm in enumerate(test_confusion_matrices):
        acc, f1_scores, aucs, precision, sensitivity, specificity, fpr_dict, tpr_dict = cm.summary(pro_list[i], lab_list[i], File=None)
        fpr_dict_list.append(fpr_dict)
        tpr_dict_list.append(tpr_dict)

    return accuracy

def setup_models(model_paths,class_num):
    return {
        'NEST-MMC': (nest_MMC(class_num), model_paths['NEST-MMC']),
        'DermFormer': (MM_nest(class_num), model_paths['DermFormer']),
        'NEST-DER': (nest_der(class_num), model_paths['NEST-DER']),
        'NEST-CLI': (nest_cli(class_num), model_paths['NEST-CLI']),
        'TFormer': (TFormer(class_num), model_paths['TFormer']),
    }

def setup_data_loaders(options, derm_data_group, chosen_distortion, distort_target, der_noise, cli_noise):
    # Instantiate the datasets
    test_dataset = dataset(derm=derm_data_group, shape=(224, 224), mode='test')  # Ensure `dataset` is a valid class or function
    corrupt_dataset = dataset_corrupt(derm=derm_data_group, shape=(224, 224), mode='test', chosen_distortion=chosen_distortion, distort_target=distort_target)
    noise_dataset = dataset_noise(derm=derm_data_group, shape=(224, 224), mode='test', derm_noise=der_noise, clinical_noise=cli_noise)

    # Create data loaders from the datasets
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=1, pin_memory=True)
    corrupt_loader = DataLoader(corrupt_dataset, batch_size=1, shuffle=False, num_workers=1, pin_memory=True)
    noise_loader = DataLoader(noise_dataset, batch_size=1, shuffle=False, num_workers=1, pin_memory=True)
    
    return test_loader, corrupt_loader, noise_loader

def plot_results_all_models(results_all, distortions, output_dir=None):
    # Group models by modality
    modalities = ['derm', 'cli', 'both']
    
    for modality in modalities:
        # Filter models for the current modality
        modality_results = {key: value for key, value in results_all.items() if modality in key}
        
        if not modality_results:
            continue
        
        models = list(modality_results.keys())
        n_models = len(models)
        n_distortions = len(distortions)
        
        # Prepare data for plotting
        accuracies = np.zeros((n_distortions, n_models))
        for i, distortion in enumerate(distortions):
            for j, model in enumerate(models):
                accuracies[i, j] = modality_results[model][distortion]
        
        # Compute and print the average accuracy for each model across distortions
        average_accuracies = accuracies.mean(axis=0)  # Compute mean along distortions (axis 0)
        for model, avg_acc in zip(models, average_accuracies):
            print(f"Average accuracy for model '{model}' across distortions: {avg_acc:.4f}")
        
        # Plotting the bar chart
        fig, ax = plt.subplots(figsize=(15, 8))
        width = 0.8 / n_models  # Adjust the width so that there is enough space between bars
        ind = np.arange(n_distortions)
        
        bars = []
        for j, model in enumerate(models):
            bars.append(ax.bar(ind + j * width, accuracies[:, j], width, label=model))
        
        ax.set_xlabel('Distortions')
        ax.set_ylabel('Accuracy')
        ax.set_title(f'Accuracy of Models with Different Distortions applied to {modality.capitalize()} Modality')
        ax.set_xticks(ind + width * (n_models / 2 - 0.5))  # Shift the x-ticks to center them
        ax.set_xticklabels(distortions, rotation=45, ha='right')
        ax.legend()
        
        ax.set_ylim(0.5, 0.85)

        plt.tight_layout()
        
        # Save or display the plot for this modality
        if output_dir:
            plt.savefig(os.path.join(output_dir, f'{modality}_model_distortion_accuracies.png'))
        plt.show()

def plot_ratio_all_models(results_all, distortions, output_dir=None):
    modalities = ['derm', 'cli', 'both']
    
    for modality in modalities:
        modality_results = {key: value for key, value in results_all.items() if modality in key}
        
        if not modality_results:
            continue
        
        models = list(modality_results.keys())
        n_models = len(models)
        n_distortions = len(distortions)
        
        # Collect 'None' accuracies separately
        none_accuracies = np.array([modality_results[model].get(None, 0) for model in models])
        
        # Prepare data for plotting
        accuracies = np.zeros((n_distortions, n_models))
        
        # Collect accuracies for each distortion
        for i, distortion in enumerate(distortions):
            for j, model in enumerate(models):
                accuracies[i, j] = modality_results[model].get(distortion, 0)
        
        # Calculate the ratios (exclude 'None' from ratio calculation)
        ratios = np.zeros_like(accuracies)
        for i, distortion in enumerate(distortions):
            if distortion != 'None':  # Skip 'None' for ratio calculation
                for j in range(n_models):
                    if none_accuracies[j] != 0:
                        ratios[i, j] = accuracies[i, j] / none_accuracies[j]
                    else:
                        ratios[i, j] = 0
        
        print(f"Ratios for modality '{modality}':\n{ratios}")
        
        # Plotting the ratio bar chart
        fig, ax = plt.subplots(figsize=(15, 8))
        width = 0.8 / n_models
        ind = np.arange(n_distortions - 1)  # Exclude 'None' from x-axis
        
        bars = []
        for j, model in enumerate(models):
            bars.append(ax.bar(ind + j * width, ratios[1:, j], width, label=model))
        
        ax.set_xlabel('Distortions')
        ax.set_ylabel('Accuracy Ratio (Distortion / None)')
        ax.set_title(f'Accuracy Ratio for Models with {modality.capitalize()} Modality')
        ax.set_xticks(ind + width * (n_models / 2 - 0.5))
        ax.set_xticklabels(distortions[1:], rotation=45, ha='right')
        ax.legend()
        
        ax.set_ylim(0.5, 1.1)

        plt.tight_layout()
        
        if output_dir:
            plt.savefig(os.path.join(output_dir, f'{modality}_accuracy_ratio_distortions.png'))
        plt.show()

def run_experiment(model_class_and_path, options, use_custom_loss, use_MMcat_loss, log_file, model_name, modality):
    model, model_path = model_class_and_path

    if model_path:
        model.eval()
        model.load_state_dict(torch.load(model_path))
        print(f"Loaded model weights from {model_path}")

    derm_data_group, column_interest = load_dataset(dir_release=options['dir_release'])

    if use_MMcat_loss:
        test_func = test_MMconcat
    elif use_custom_loss:
        test_func = test
    else:
        test_func = test_standardloss

    results = {}
    for distortion_name in options['distortions']:
        corrupt_dataset = dataset_corrupt(
            derm=derm_data_group, shape=(224, 224), mode='test', 
            chosen_distortion=distortion_name, distort_target=modality
        )
        data_loader = DataLoader(corrupt_dataset, batch_size=1, shuffle=False, num_workers=1, pin_memory=True)

        accuracy = test_func(model, data_loader, options, log_file)
        results[distortion_name] = accuracy
        print(f"Model: {model_name}, Modality: {modality}, Distortion: {distortion_name}, Accuracy: {accuracy:.2f}")

        #if log_file:
        #    log_file.write(f"Distortion: {distortion_name}, Accuracy: {accuracy:.2f}\n")

    #if log_file:
        #log_file.write(f"Completed experiments for modality {modality}...\n")
        #log_file.write(f"Results for model {model_name} with modality {modality}: {results}\n")

    return results

def load_existing_results(log_path):
    results_dict = {}
    log_file_path = os.path.join(log_path, 'robustness_log.txt')
    
    try:
        with open(log_file_path, 'r') as file:
            for line in file:
                print(f"Reading line: {line.strip()}")  # Debugging line
                try:
                    key, value = line.split(':', 1)
                    value_dict = ast.literal_eval(value.strip())
                    results_dict[key.strip()] = value_dict
                except Exception as e:
                    print(f"Error parsing line: {line.strip()} - {e}")
    except Exception as e:
        print(f"Error opening file {log_file_path}: {e}")
    
    return results_dict

def save_results_to_log(log_path, results_all):
    # Save results to the log file
    log_file = os.path.join(log_path, 'robustness_log.txt')
    with open(log_file, 'a') as file:
        for key, value in results_all.items():
            file.write(f"{key}: {value}\n")

def main(params):
    log_path = params.get('log_path', 'default_log_path')
    log_file_path = os.path.join(log_path, 'robustness_log.txt')
    
    distortions = params.get('distortions', [
            None,'gaussian_noise', 'shot_noise', 'impulse_noise', 'speckle_noise',
            'gaussian_blur', 'glass_blur', 'defocus_blur', 'motion_blur',
            'zoom_blur', 'snow', 'spatter', 'contrast', 'brightness',
            'saturate', 'jpeg_compression', 'pixelate', 'elastic_transform'
        ])

    # Only create a new log file if the log file doesn't exist
    if os.path.exists(log_file_path):
        print(f"Loading existing results from {log_file_path}")
        existing_results = load_existing_results(log_path)
        if not existing_results:
            print(f"No valid data found in {log_file_path}, starting fresh.")
            existing_results = {}  # Start with an empty dictionary if no valid results are found
        print(existing_results)
        # Plot results for each modality
        plot_results_all_models(existing_results, distortions, log_path)
        plot_ratio_all_models(existing_results, distortions, log_path)
    else:
        # Set up logging if the log file doesn't exist
        log_file = setup_logging(log_path)
        if log_file is None:
            print(f"Error: Failed to create log file at {log_path}.")
            return  # Exit early if logging fails
        existing_results = {}  # No existing results, so start fresh

        model_paths = {
            'TFormer': params['tformer_model_path'],
            'NEST-DER': params['nestder_model_path'],
            'NEST-CLI': params['nestcli_model_path'],
            'NEST-MMC': params['nestmmc_model_path'],
            'DermFormer': params['dermformer_model_path'],
        }

        models = setup_models(model_paths, params['class_num'])
        results_all = {}

        modalities = ['derm','cli','both']  # Specify modalities here
        for modality in modalities:
            for model_name, model_tuple in models.items():
                print(f"Running experiments for model: {model_name} with modality: {modality}")
                
                # Skip experiments if results already exist
                if f"{model_name}_{modality}" in existing_results:
                    print(f"Skipping {model_name} with modality {modality}, results already exist.")
                    results_all[f"{model_name}_{modality}"] = existing_results[f"{model_name}_{modality}"]
                    continue  # Skip re-running if results already exist

                model_class, model_path = model_tuple

                if model_name == 'DermFormer':
                    use_custom_loss = True
                    use_MMcat_loss = False
                elif model_name == 'NEST-MMC':
                    use_custom_loss = False
                    use_MMcat_loss = True
                else:
                    use_custom_loss = False
                    use_MMcat_loss = False

                model = model_class.cuda() if params['cuda'] else model_class

                options = {
                    'log_path': os.path.join(params['log_path'], f'{model_name}_{modality}_results.txt'),
                    'model_path': model_path,
                    'cuda': params['cuda'],
                    'class_num': params['class_num'],
                    'labels': params['labels'],
                    'distortions': distortions,
                    'use_custom_loss': use_custom_loss,
                    'dir_release': params['dir_release'],
                    'modality': modality  # Add modality to options
                }

                # Run the experiment
                results = run_experiment((model, model_path), options, use_custom_loss, use_MMcat_loss, log_file, model_name, modality)

                # Store accuracy results with modality
                results_all[f"{model_name}_{modality}"] = results
                print(f"Results for model {model_name} with modality {modality}: {results}")
                save_results_to_log(log_path, results_all)

        # Plot results for each modality
        plot_results_all_models(results_all, distortions, log_path)
        plot_ratio_all_models(results_all, distortions, log_path)

if __name__ == "__main__":
    # Set up command-line argument parsing
    parser = argparse.ArgumentParser(description="Experiment setup and execution for models.")
    
    # Common experiment settings
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to run the experiment')
    parser.add_argument('--dir_release', dest='dir_release', type=str, default="/home/matthewcockayne/datasets/Derm7pt/release_v0/release_v0", help = 'Path to dataset directory')
    parser.add_argument('--log_path', type=str, default= '/home/matthewcockayne/Documents/PhD/experiments/results/robustness/test/corruptions/', help='Directory where logs will be saved')
    parser.add_argument('--cuda', type=bool, default=True, help='Whether to use CUDA')
    parser.add_argument('--class_num', type=int, default=5, help='Number of classes')
    parser.add_argument('--labels', type=str, default='["diag", "pn", "bwv", "vs", "pig", "str", "dag", "rs"]', help='Labels for the classes')

    # Add arguments for model paths
    parser.add_argument('--tformer_model_path', type=str, default= '/home/matthewcockayne/Documents/PhD/experiments/tformer (best)/models/best_model_6.pth', help='Path to TFormer model')
    parser.add_argument('--nestder_model_path', type=str, default= '/home/matthewcockayne/Documents/PhD/experiments/results/unimodal/der/20241218-172820/models/best_model_64.pth', help='Path to NEST-DER model')
    parser.add_argument('--nestcli_model_path', type=str, default= '/home/matthewcockayne/Documents/PhD/experiments/results/unimodal/cli/20241218-164526/models/best_model_38.pth', help='Path to NEST-CLI model')
    parser.add_argument('--nestmmc_model_path', type=str, default= '/home/matthewcockayne/Documents/PhD/experiments/results/multimodal_concat/cli_der_meta/20250108-143235/models/best_model_28.pth', help='Path to NEST-MMC model')
    #parser.add_argument('--nestmmc_model_path', type=str, default= '/home/matthewcockayne/Documents/PhD/experiments/results/multimodal_concat/cli_der_meta/20241129-170927/models/best_model_86.pth', help='Path to NEST-MMC model')
    parser.add_argument('--dermformer_model_path', type=str, default='/home/matthewcockayne/Documents/PhD/experiments/results/fusion/20241122-163959/models/best_model_132.pth', help='Path to DermFormer model')

    # Parse the command-line arguments
    args = parser.parse_args()
    
    # Convert the string representation of the labels back to a list
    args.labels = eval(args.labels)
    
    # Convert arguments to a dictionary
    params = vars(args)
    
    # Call main function with parsed arguments
    main(params)
