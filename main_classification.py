"""
Train & Test Script:
- Code trains, tests and logs performance of the multi-modal transformer

    Performance logs includes:
    - accuracy
    - loss
    - precision
    - sensitivity
    - specificity 
    - f1-score
    - AUC
"""
import sys
import gc
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
from models.NesT.nest_cli import nest_cli
from models.NesT.nest_der import nest_der
#from models.NesT.nest_meta import MM_nest
#from models.NesT.nest_clider import MM_nest
from models.NesT.nest_multimodalconcat import nest_MMC
from models.DermFormer import DermFormer, MMNestLoss
from models.MM_resnet50 import resnet50_MMC
from models.MM_efficientnet import efficientnet_MMC

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def setup_logging(log_path):
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    log_file = open(os.path.join(log_path, 'train_log.txt'), 'w')
    return log_file

def log_params(options, log_file):
    print('===========Training Params===============')
    log_file.write('===========Training Params===============\n')
    for name, param in options.items():
        print(f'{name}: {param}')
        log_file.write(f'{name}: {param}\n')
    print('========================================')
    log_file.write('========================================\n')

def setup_model(options, class_num):
    model = DermFormer(class_num)
    if options['cuda']:
        model = model.cuda()
    # Create criterion separately
    criterion = MMNestLoss(class_weights=None)
    if options['cuda']:
        criterion = criterion.cuda()
    return model, criterion

def model_snapshot(model, new_model_path, old_model_path=None, only_best_model=False):
    if only_best_model and old_model_path:
        os.remove(old_model_path)
    torch.save(model.state_dict(), new_model_path)

def setup_data_loaders(options, derm_data_group):
    train_loader = DataLoader(dataset(derm=derm_data_group,shape=(224,224),mode='train'),
                                batch_size=options['batch_size'],shuffle=True,num_workers=1, pin_memory=True)
    valid_loader = DataLoader(dataset(derm= derm_data_group,shape=(224,224),mode='valid'),
                                batch_size=1,shuffle=False,num_workers=1, pin_memory=True)
    test_loader = DataLoader(dataset(derm=derm_data_group,shape=(224,224),mode='test'),
                               batch_size=1,shuffle=False,num_workers=1, pin_memory=True)

    return train_loader, valid_loader, test_loader

def train_one_epoch(model, criterion, train_loader, optimizer, options, log_file, epoch):
    model.train()
    train_loss = 0.0
    correct = [0] * 8 
    total = [0] * 8    
    print(f"Training epoch: {epoch}")
    log_file.write(f"Training epoch: {epoch}\n")
    
    label_names = ['DIAG', 'PN', 'BWV', 'VS', 'PIG', 'STR', 'DaG', 'RS']
    class_weights = train_loader.dataset.class_weights  # Extract class weights from dataset
    #print(f"class_weights: {class_weights}, type: {type(class_weights)}")

    for batch_idx, (der_data, cli_data, meta_data, meta_con, target) in enumerate(train_loader):
        # Prepare target labels for each task
        targets = [target[i].squeeze(1).cuda() if options['cuda'] else target[i].squeeze(1)
                   for i in range(8)]

        if options['cuda']:
            der_data, cli_data, meta_data = der_data.cuda(), cli_data.cuda(), meta_data.cuda().float()
            meta_data, meta_con = meta_data.long(), meta_con.long()

        optimizer.zero_grad()
        
        # Forward pass to get model outputs (returns dict)
        outputs = model(meta_data, meta_con, cli_data, der_data)
        
        # Use MMNestLoss criterion which expects dict outputs
        loss = criterion(outputs, targets)

        loss.backward()
        avg_loss = loss.item()
        train_loss += avg_loss
        optimizer.step()

        # Extract ensemble predictions (last index in each task output)
        task_keys = ['diag', 'pn', 'bwv', 'vs', 'pig', 'str', 'dag', 'rs']
        for i, (task_key, true) in enumerate(zip(task_keys, targets)):
            # Use the ensemble prediction (last element which is always the prediction)
            pred = outputs[task_key][-1]
            correct[i] += pred.eq(true).sum().item()
            total[i] += true.size(0)

        # Print intermediate results
        if batch_idx % 50 == 0 and batch_idx > 0:
            accuracy = np.mean([c / t if t > 0 else 0 for c, t in zip(correct, total)])
            print(f'Training epoch: {epoch} [{batch_idx * len(der_data)}/{len(train_loader.dataset)}], '
                  f'Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}, Learning rate: {optimizer.param_groups[0]["lr"]}')

    # Log final results for the epoch
    accuracy = np.mean([c / t if t > 0 else 0 for c, t in zip(correct, total)])
    log_file.write(f"Training Loss: {train_loss / len(train_loader):.4f}, Accuracy: {accuracy:.4f}\n")
    return train_loss / len(train_loader), accuracy

def validate(model, criterion, valid_loader, options, label_t, class_num, log_file):
    model.eval()
    avg_valid_loss = 0
    correct = [0] * 8
    total = [0] * 8
    val_confusion_matrices = [ConfusionMatrix(num_classes=class_num, labels=label_t) for _ in range(8)]
    pro_list = [[] for _ in range(8)]
    lab_list = [[] for _ in range(8)]

    label_names = ['diag', 'pn', 'bwv', 'vs', 'pig', 'str', 'dag', 'rs']
    task_keys = ['diag', 'pn', 'bwv', 'vs', 'pig', 'str', 'dag', 'rs']

    class_weights = valid_loader.dataset.class_weights

    with torch.no_grad():
        for der_data, cli_data, meta_data, meta_con, target in valid_loader:
            # Prepare target labels
            targets = [
                target[0].squeeze(1).cuda(),
                target[1].squeeze(1).cuda(),
                target[2].squeeze(1).cuda(),
                target[3].squeeze(1).cuda(),
                target[4].squeeze(1).cuda(),
                target[5].squeeze(1).cuda(),
                target[6].squeeze(1).cuda(),
                target[7].squeeze(1).cuda(),
            ]

            if options['cuda']:
                der_data, cli_data, meta_data = der_data.cuda(), cli_data.cuda(), meta_data.cuda().float()
                meta_data = meta_data.long()
                meta_con = meta_con.long()

            outputs = model(meta_data, meta_con, cli_data, der_data)
            
            # Use MMNestLoss criterion
            loss = criterion(outputs, targets)
            avg_valid_loss += loss.item()

            for i, (task_key, true) in enumerate(zip(task_keys, targets)):
                # Get ensemble prediction (last element)
                pred = outputs[task_key][-1]
                val_confusion_matrices[i].update(pred.cpu().numpy(), true.cpu().numpy())

                # Get ensemble probabilities (second to last element)
                probs = outputs[task_key][-2].cpu().numpy()

                # Check for NaNs or Infs in probabilities before extending pro_list
                if np.isnan(probs).any() or np.isinf(probs).any():
                    print(f"NaN or Inf detected in model output for {label_names[i]}")

                pro_list[i].extend(probs)
                lab_list[i].extend(true.cpu().numpy())

                correct[i] += pred.eq(true).sum().item()
                total[i] += true.size(0)

    avg_valid_loss /= len(valid_loader)
    accuracy = np.mean([c / t if t > 0 else 0 for c, t in zip(correct, total)])

    for i in range(8):
        pro_list[i] = np.vstack(pro_list[i])
        lab_list[i] = np.hstack(lab_list[i])

    for i, cm in enumerate(val_confusion_matrices):
        cm.summary(pro_list[i], lab_list[i], log_file)

    log_file.write(f"Validation Loss: {avg_valid_loss:.4f}, Accuracy: {accuracy:.4f}\n")
    return avg_valid_loss, accuracy


def test(model, criterion, test_loader, options, log_file):
    avg_test_loss = 0
    log_file.write(f"Model Test: \n")
    test_confusion_matrices = [ConfusionMatrix(num_classes=options['class_num'], labels=options['labels']) for _ in range(8)]
    pro_list = [[] for _ in range(8)]  # List to store predicted probabilities for each class
    lab_list = [[] for _ in range(8)]  # List to store true labels for each class
    total_correct = 0
    total_samples = 0

    # Define label names corresponding to your outputs
    label_names = ['diag', 'pn', 'bwv', 'vs', 'pig', 'str', 'dag', 'rs']
    task_keys = ['diag', 'pn', 'bwv', 'vs', 'pig', 'str', 'dag', 'rs']
    
    # Get class weights from the dataset
    class_weights = test_loader.dataset.class_weights

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
                meta_data = meta_data.long()
                meta_con = meta_con.long()

            # Forward pass
            outputs = model(meta_data, meta_con, cli_data, der_data)

            # Use MMNestLoss criterion
            loss = criterion(outputs, targets)
            avg_test_loss += loss.item()

            for i, (task_key, true) in enumerate(zip(task_keys, targets)):
                # Get ensemble prediction (last element)
                pred = outputs[task_key][-1]
                pred_np = pred.cpu().numpy()
                target_np = true.cpu().numpy()
                if len(target_np.shape) > 1:
                    target_np = target_np.squeeze()
                test_confusion_matrices[i].update(pred_np, target_np)

                # Get ensemble probabilities (second to last element)
                probs = outputs[task_key][-2]
                pro_list[i].extend(probs.cpu().numpy()) 
                lab_list[i].extend(true.cpu().numpy())

            # Calculate accuracy
            correct = sum([outputs[task_key][-1].eq(targets[i]).sum().item() for i, task_key in enumerate(task_keys)])
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
        # Now capture all returned values from summary, including fpr_dict and tpr_dict
        acc, f1_scores, auc_values, precision_list, sensitivity_list, specificity_list, fpr_dict, tpr_dict = cm.summary(pro_list[i], lab_list[i], log_file)
        
        # Append the fpr_dict and tpr_dict for ROC plotting
        fpr_dict_list.append(fpr_dict)
        tpr_dict_list.append(tpr_dict)

    # Plot ROC curves for each confusion matrix
    plot_roc_curves(options['log_path'], fpr_dict_list, tpr_dict_list, options['labels'], label_names)

def create_experiment_folders(base_dir):
    # Use a timestamp or unique identifier for each experiment
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    experiment_path = os.path.join(base_dir, timestamp)
    os.makedirs(experiment_path, exist_ok=True)

    # Create logs and models folders within the experiment directory
    log_path = os.path.join(experiment_path, "logs")
    model_path = os.path.join(experiment_path, "models")
    os.makedirs(log_path, exist_ok=True)
    os.makedirs(model_path, exist_ok=True)
    
    return log_path, model_path

def run_experiments(options, num_experiments):
    for i in range(num_experiments):
        print(f"Running experiment {i+1}/{num_experiments}")

        # Create unique directories for each experiment using create_experiment_folders
        options['log_path'], options['modal_path'] = create_experiment_folders(options['base_dir'])

        # Run the main function for this experiment
        main(options)

def main(options):
    # Create unique directories for each experiment
    options['log_path'], options['modal_path'] = create_experiment_folders(options['base_dir'])
    log_file = setup_logging(options['log_path'])
    log_params(options, log_file)

    derm_data_group, column_interest = load_dataset(dir_release=options['dir_release'])
    #derm_data_group = load_dataset(dir_release=options['dir_release'])
    #mata = derm_data_group.meta_train.values
    #print(mata)
    model, criterion = setup_model(options, options['class_num'])
    train_loader, valid_loader, test_loader = setup_data_loaders(options, derm_data_group)
    #print("Class weights tensor:", class_weights)
    #Current LR and optimiser
    optimizer = optim.AdamW(model.parameters(), lr=options['learning_rate'], betas=(0.9, 0.999), weight_decay=options['weight_decay'])
    lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=options['epochs'])
    best_accuracy = 0
    #best_valid_loss = float('inf')
    train_losses, valid_losses = [], []
    train_acc, valid_acc = [], []
    old_model_path = None
    start_time = time.time()
    patience_counter = 0  # Early stopping counter
    try:
        for epoch in range(options['epochs']):
            train_loss, train_accuracy = train_one_epoch(model, criterion, train_loader, optimizer, options, log_file, epoch)
            train_losses.append(train_loss)
            train_acc.append(train_accuracy)

            valid_loss, valid_accuracy = validate(model, criterion, valid_loader, options, options['labels'], options['class_num'], log_file)
            valid_losses.append(valid_loss)
            valid_acc.append(valid_accuracy)

            lr_scheduler.step()

            if valid_accuracy > best_accuracy:
                best_accuracy = valid_accuracy
                patience_counter = 0  # Reset patience counter
                new_model_path = os.path.join(options['modal_path'], f'best_model_{epoch}.pth')
                model_snapshot(model, new_model_path, old_model_path=old_model_path, only_best_model=True)
                old_model_path = new_model_path
                print("Found new best model, saving to disk...")
            else:
                patience_counter += 1  # Increment patience counter

            if patience_counter >= options['patience']:  # Early stopping condition
                print("Early stopping triggered. Stopping training...")
                break

            #if epoch % 10 == 0 or epoch == options['epochs'] - 1:
            #    model_snapshot(model, os.path.join(options['modal_path'], f'model-{epoch}.pth'))

            if np.isnan(train_loss):
                print("Training got into NaN values...\n\n")
                break

            end_time_epoch = time.time()
            total_training_time = end_time_epoch - start_time
            print(f"Total training time: {total_training_time:.4f}s")

        if old_model_path:
            print(old_model_path)
            model.load_state_dict(torch.load(old_model_path))
            model.eval()
            test(model, criterion, test_loader, options, log_file)

        plot_metrics(train_losses, valid_losses, train_acc, valid_acc, options['log_path'], options['epochs'])

    except Exception as e:
        import traceback
        traceback.print_exc()
    finally:
        log_file.close()

        # Free up memory by deleting objects
        del model, optimizer, train_loader, valid_loader, test_loader, lr_scheduler
        torch.cuda.empty_cache()  # Clear GPU memory
        gc.collect()  # Force garbage collection to release any residual memory

if __name__ == "__main__":
    OPTIONS = argparse.ArgumentParser()

    OPTIONS.add_argument('--epochs', dest='epochs', type=int, default=200)
    OPTIONS.add_argument('--base_dir', dest='base_dir', type=str, default="/home/matthewcockayne/Documents/PhD/experiments/results/")
    OPTIONS.add_argument('--dir_release', dest='dir_release', type=str, default="/home/matthewcockayne/datasets/Derm7pt/release_v0/release_v0")
    OPTIONS.add_argument('--labels', dest='labels', default=[0, 1, 2, 3, 4])
    OPTIONS.add_argument('--class_num', dest='class_num', type=int, default=5)
    OPTIONS.add_argument('--patience', dest='patience', type=int, default=50)
    OPTIONS.add_argument('--batch_size', dest='batch_size', type=int, default=16)
    OPTIONS.add_argument('--weight_decay', dest='weight_decay', type=float, default=1e-4)
    OPTIONS.add_argument('--learning_rate', dest='learning_rate', type=float, default=1e-4)
    OPTIONS.add_argument('--cuda', dest='cuda', type=bool, default=True)
    OPTIONS.add_argument('--pretrained', dest='pretrained', type=bool, default=True)
    OPTIONS.add_argument('--num_experiments', dest='num_experiments', type=int, default=1)
    
    #weight_decay_values = [1e-6]  # Add desired weight decay values
    #learning_rate_values = [1e-4]  # Add desired learning rate values

    args = OPTIONS.parse_args()
    PARAMS = vars(args)
    run_experiments(PARAMS, num_experiments=PARAMS['num_experiments'])