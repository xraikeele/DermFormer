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
import torch.nn.functional as F
import matplotlib.pyplot as plt
from derm7pt.eval_metrics import ConfusionMatrix, plot_metrics, plot_roc_curves
from derm7pt.dataloader import load_dataset, dataset #, train_data_transformation, test_data_transformation
from derm7pt.dataloader_corruptions import load_corrupt, dataset_corrupt
from derm7pt.dataloader_noise import load_noise, dataset_noise
#from models.model_swinv2concat import MM_Transformer
from models.TFormer import TFormer
from models.DermFormer import MM_nest, MMNestLoss
from models.NesT.nest_multimodalconcat import nest_MMC

def setup_logging(log_path):
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    log_file = open(os.path.join(log_path, 'Robustness_log.txt'), 'w')
    return log_file

def log_params(options, log_file):
    print('===========Test Params===============')
    log_file.write('===========Test Params===============\n')
    for name, param in options.items():
        print(f'{name}: {param}')
        log_file.write(f'{name}: {param}\n')
    print('========================================')
    log_file.write('========================================\n')

def setup_model(options, class_num):
    #model = MM_Transformer(class_num, cross_attention_depths=options['cross_attention_depths'])
    tformer = TFormer(class_num)
    dermformer = MM_nest(class_num)
    nestmmc = nest_MMC(class_num)
    if options['cuda']:
        dermformer = dermformer.cuda()
        tformer = tformer.cuda()
        nestmmc = nestmmc.cuda()
    return dermformer, tformer, nestmmc

def setup_data_loaders(options, derm_data_group, num_distortions, derm_noise, cli_noise):
    # Define how many distortions you want (e.g., 3 distortions per image)

    test_loader = DataLoader(dataset(derm=derm_data_group, shape=(224, 224), mode='test'),
                             batch_size=1, shuffle=False, num_workers=1, pin_memory=True)
    corrupt_loader = DataLoader(dataset_corrupt(derm=derm_data_group, shape=(224, 224), mode='test', num_distortions=num_distortions),
                               batch_size=1, shuffle=False, num_workers=1, pin_memory=True)
    noise_loader = DataLoader(dataset_noise(derm=derm_data_group, shape=(224, 224), mode='test', derm_noise=derm_noise, clinical_noise=cli_noise),
                               batch_size=1, shuffle=False, num_workers=1, pin_memory=True)
    return test_loader, corrupt_loader, noise_loader

def test(model, test_loader, options, log_file):
    avg_test_loss = 0
    log_file.write(f"Model Test: \n")
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
        _, _, _, fpr_dict, tpr_dict = cm.summary(pro_list[i], lab_list[i], log_file)
        fpr_dict_list.append(fpr_dict)
        tpr_dict_list.append(tpr_dict)

    # Plot ROC curves for each confusion matrix
    #plot_roc_curves(options['log_path'], fpr_dict_list, tpr_dict_list, options['labels'], label_names)

    return accuracy

def test_tformer(model, test_loader, options, log_file):
    names = ('Diag', 'PN', 'BWV', 'VS', 'PIG', 'STR', 'DaG', 'RS')
    avg_test_loss = 0
    log_file.write(f"Model Test: \n")
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
        _, _, _, fpr_dict, tpr_dict = cm.summary(pro_list[i], lab_list[i], log_file)
        fpr_dict_list.append(fpr_dict)
        tpr_dict_list.append(tpr_dict)

    # Plot ROC curves for each confusion matrix
    #plot_roc_curves(options['log_path'], fpr_dict_list, tpr_dict_list, options['labels'], names)

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
        _, _, _, fpr_dict, tpr_dict = cm.summary(pro_list[i], lab_list[i], log_file)
        fpr_dict_list.append(fpr_dict)
        tpr_dict_list.append(tpr_dict)

    return accuracy

def main(options):
    tformer_model_path = '/home/matthewcockayne/Documents/PhD/experiments/tformer (best)/models/best_model_6.pth'
    dermformer_model_path = '/home/matthewcockayne/Documents/PhD/experiments/results/fusion/20241122-163959/models/best_model_132.pth'
    nestMMC_model_path = '/home/matthewcockayne/Documents/PhD/experiments/results/multimodal_concat/cli_der_meta/20241129-170927/models/best_model_86.pth'
    log_file = setup_logging(options['log_path'])
    log_params(options, log_file)

    derm_data_group, column_interest = load_dataset(dir_release=options['dir_release'])
    dermformer, tformer, nestmmc = setup_model(options, options['class_num'])

    optimizer = optim.Adam(dermformer.parameters(), lr=options['learning_rate'], betas=(0.9, 0.999), weight_decay=options['weight_decay'])
    lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=options['epochs'])

    print(dermformer_model_path)
    dermformer.load_state_dict(torch.load(dermformer_model_path))
    dermformer.eval()
    print(tformer_model_path)
    tformer.load_state_dict(torch.load(tformer_model_path))
    tformer.eval()
    print(nestMMC_model_path)
    nestmmc.load_state_dict(torch.load(nestMMC_model_path))
    nestmmc.eval()
    
    # Define ranges for derm_noise and cli_noise
    derm_noise_values = np.linspace(0.0, 1.0, 10)  # e.g., 10 values from 0.0 to 1.0
    cli_noise_values = np.linspace(0.0, 1.0, 10)   # e.g., 10 values from 0.0 to 1.0

    accuracy_tformer_matrix = np.zeros((len(derm_noise_values), len(cli_noise_values)))
    accuracy_dermformer_matrix = np.zeros((len(derm_noise_values), len(cli_noise_values)))
    accuracy_nestmmc_matrix = np.zeros((len(derm_noise_values), len(cli_noise_values)))

    for i, derm_noise in enumerate(derm_noise_values):
        for j, cli_noise in enumerate(cli_noise_values):
            print(f"Testing with derm_noise={derm_noise}, cli_noise={cli_noise}")
            test_loader, _, noise_loader = setup_data_loaders(
                options, derm_data_group, num_distortions=1, derm_noise=derm_noise, cli_noise=cli_noise
            )
            accuracy_tformer = test_tformer(tformer, noise_loader, options, log_file)
            accuracy_dermformer = test(dermformer, noise_loader, options, log_file)
            accuracy_nestmmc = test_MMconcat(nestmmc,noise_loader,options,log_file)

            accuracy_tformer_matrix[i, j] = accuracy_tformer
            accuracy_dermformer_matrix[i, j] = accuracy_dermformer
            accuracy_nestmmc_matrix[i,j] = accuracy_nestmmc

    # Compute the global minimum and maximum for the color scale
    global_min = min(accuracy_tformer_matrix.min(), accuracy_dermformer_matrix.min(), accuracy_nestmmc_matrix.min())
    global_max = max(accuracy_tformer_matrix.max(), accuracy_dermformer_matrix.max(), accuracy_nestmmc_matrix.max())

    # Plot DermFormer heatmap
    plt.figure(figsize=(10, 8))
    plt.imshow(
        accuracy_dermformer_matrix,
        extent=[cli_noise_values[0], cli_noise_values[-1], derm_noise_values[0], derm_noise_values[-1]],
        origin='lower', aspect='auto', cmap='viridis', vmin=global_min, vmax=global_max
    )
    plt.colorbar(label='Accuracy')
    plt.xlabel('cli_noise')
    plt.ylabel('derm_noise')
    plt.title('DermFormer Model Accuracy under Different Modality Noise Levels')
    dermformer_heatmap_path = os.path.join(options['log_path'], 'accuracy_heatmap_dermformer.png')
    plt.savefig(dermformer_heatmap_path)
    plt.close()

    # Plot TFormer heatmap
    plt.figure(figsize=(10, 8))
    plt.imshow(
        accuracy_tformer_matrix,
        extent=[cli_noise_values[0], cli_noise_values[-1], derm_noise_values[0], derm_noise_values[-1]],
        origin='lower', aspect='auto', cmap='viridis', vmin=global_min, vmax=global_max
    )
    plt.colorbar(label='Accuracy')
    plt.xlabel('cli_noise')
    plt.ylabel('derm_noise')
    plt.title('TFormer Model Accuracy under Different Modality Noise Levels')
    tformer_heatmap_path = os.path.join(options['log_path'], 'accuracy_heatmap_tformer.png')
    plt.savefig(tformer_heatmap_path)
    plt.close()

    # Plot NesT_MMConcat heatmap
    plt.figure(figsize=(10, 8))
    plt.imshow(
        accuracy_nestmmc_matrix,
        extent=[cli_noise_values[0], cli_noise_values[-1], derm_noise_values[0], derm_noise_values[-1]],
        origin='lower', aspect='auto', cmap='viridis', vmin=global_min, vmax=global_max
    )
    plt.colorbar(label='Accuracy')
    plt.xlabel('cli_noise')
    plt.ylabel('derm_noise')
    plt.title('NesT_MMConcat Model Accuracy under Different Modality Noise Levels')
    nestmmc_heatmap_path = os.path.join(options['log_path'], 'accuracy_heatmap_nestmmc.png')
    plt.savefig(nestmmc_heatmap_path)
    plt.close()
        

if __name__ == "__main__":
    OPTIONS = argparse.ArgumentParser()

    OPTIONS.add_argument('--epochs', dest='epochs', type=int, default=100)
    OPTIONS.add_argument('--dir_release', dest='dir_release', type=str, default="/home/matthewcockayne/datasets/Derm7pt/release_v0/release_v0")
    OPTIONS.add_argument('--modal_path', dest='modal_path', type=str, default="/home/matthewcockayne/Documents/PhD/experiments/results/robustness/test/")
    OPTIONS.add_argument('--log_path', dest='log_path', type=str, default="/home/matthewcockayne/Documents/PhD/experiments/results/robustness/test/")
    OPTIONS.add_argument('--labels', dest='labels', default=[0, 1, 2, 3, 4])
    OPTIONS.add_argument('--class_num', dest='class_num', type=int, default=5)
    OPTIONS.add_argument('--patience', dest='patience', type=int, default=50)
    OPTIONS.add_argument('--cross_attention_depths', dest='cross_attention_depths', type=str, default='1, 1, 1, 1, 1')
    OPTIONS.add_argument('--batch_size', dest='batch_size', type=int, default=16)
    OPTIONS.add_argument('--weight_decay', dest='weight_decay', type=float, default=1e-4)
    OPTIONS.add_argument('--learning_rate', dest='learning_rate', type=float, default=1e-4)
    OPTIONS.add_argument('--cuda', dest='cuda', type=bool, default=True)
    OPTIONS.add_argument('--pretrained', dest='pretrained', type=bool, default=True)

    args = OPTIONS.parse_args()
    PARAMS = vars(args)
    PARAMS['cross_attention_depths'] = list(map(int, PARAMS['cross_attention_depths'].split(',')))
    main(PARAMS)
