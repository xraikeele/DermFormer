"""
Hyperparameter optimisation:
- Code runs trials of hyperparameters to minimise validation loss

    Hyperparameters include:
    - epochs
    - batch size
    - patience
    - weight decay 
    - learning rate
    - optimiser
    - learning rate scheduler
    - cross attention layer depths
"""
import argparse
import os
import time
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.autograd import Variable
import optuna
from derm7pt.eval_metrics import ConfusionMatrix, plot_metrics, plot_roc_curves
from derm7pt.dataloader import load_dataset, dataset
from models.DermFormer import DermFormer
from tqdm import tqdm

def setup_logging(log_path, trial_number):
    trial_log_path = os.path.join(log_path, f'trial_{trial_number}')
    if not os.path.exists(trial_log_path):
        os.makedirs(trial_log_path)
    log_file = open(os.path.join(trial_log_path, 'trials_log.txt'), 'w')
    return log_file

def log_params(options, log_file, trial_number):
    log_file.write(f'Trial Number: {trial_number}\n')
    log_file.write('===========Training Params===============\n')
    for name, param in options.items():
        log_file.write(f'{name}: {param}\n')
    log_file.write('========================================\n')

def setup_model(options, class_num):
    model = DermFormer(class_num)
    if options['cuda']:
        model = model.cuda()
    return model

def model_snapshot(model, new_model_path, old_model_path=None, only_best_model=False):
    if only_best_model and old_model_path:
        os.remove(old_model_path)
    torch.save(model.state_dict(), new_model_path)

def setup_data_loaders(options, derm_data_group):
    train_loader = DataLoader(dataset(derm=derm_data_group, shape=(224, 224), mode='train'),
                              batch_size=options['batch_size'], shuffle=True, num_workers=4)
    valid_loader = DataLoader(dataset(derm=derm_data_group, shape=(224, 224), mode='valid'),
                              batch_size=1, shuffle=False, num_workers=2)
    test_loader = DataLoader(dataset(derm=derm_data_group, shape=(224, 224), mode='test'),
                             batch_size=1, shuffle=False, num_workers=2)
    return train_loader, valid_loader, test_loader

def train_one_epoch(model, train_loader, optimizer, options):
    model.train()
    train_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (der_data, cli_data, meta_data, meta_con, target) in enumerate(train_loader):
        diagnosis_label = target[0].squeeze(1).cuda()
        targets = [target[i].squeeze(1).cuda() for i in range(8)]

        if options['cuda']:
            der_data, cli_data, meta_data = der_data.cuda(), cli_data.cuda(), meta_data.cuda().float()
            der_data, cli_data, meta_data = Variable(der_data), Variable(cli_data), Variable(meta_data)
            meta_data = meta_data.long()
            meta_con = meta_con.long()

        optimizer.zero_grad()
        outputs = model(meta_data, meta_con, cli_data, der_data)

        class_weights = train_loader.dataset.class_weights
        weights = [class_weights[i].cuda() if options['cuda'] else class_weights[i] for i in range(8)]

        losses = [F.cross_entropy(output, target, weight=weight) for output, target, weight in zip(outputs, targets, weights)]
        loss = sum(losses) / len(losses)

        loss.backward()
        train_loss += loss.item()
        optimizer.step()

        predicted_results = outputs[0].data.max(1)[1]
        correct += predicted_results.eq(diagnosis_label).sum().item()
        total += diagnosis_label.size(0)

    accuracy = correct / total
    return train_loss / len(train_loader), accuracy

def validate(model, valid_loader, options):
    model.eval()
    valid_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for der_data, cli_data, meta_data, meta_con, target in valid_loader:
            diagnosis_label = target[0].squeeze(1).cuda()
            targets = [target[i].squeeze(1).cuda() for i in range(8)]

            if options['cuda']:
                der_data, cli_data, meta_data = der_data.cuda(), cli_data.cuda(), meta_data.cuda().float()
                der_data, cli_data, meta_data = Variable(der_data), Variable(cli_data), Variable(meta_data)
                meta_data = meta_data.long()
                meta_con = meta_con.long()

            outputs = model(meta_data, meta_con, cli_data, der_data)

            class_weights = valid_loader.dataset.class_weights
            weights = [class_weights[i].cuda() if options['cuda'] else class_weights[i] for i in range(8)]
            losses = [F.cross_entropy(output, target, weight=weight) for output, target, weight in zip(outputs, targets, weights)]
            loss = sum(losses) / len(losses)
            valid_loss += loss.item()

            predicted_results = outputs[0].data.max(1)[1]
            correct += predicted_results.eq(diagnosis_label).sum().item()
            total += diagnosis_label.size(0)

    valid_loss /= len(valid_loader)
    accuracy = correct / total

    return valid_loss, accuracy

def suggest_cross_attention_depths(trial):
    return [
        trial.suggest_int('cross_attention_depth_1', 1, 4),
        trial.suggest_int('cross_attention_depth_2', 1, 4),
        trial.suggest_int('cross_attention_depth_3', 1, 4),
        trial.suggest_int('cross_attention_depth_4', 1, 4),
        trial.suggest_int('cross_attention_depth_5', 1, 4),
    ]

def objective(trial):
    trial_number = trial.number
    options = {
        'epochs': trial.suggest_int('epochs', 10, 100),
        'dir_release': "/home/matthewcockayne/datasets/Derm7pt/release_v0/release_v0",
        'modal_path': "/home/matthewcockayne/Documents/PhD/MMCrossTransformer/hyperparameters/models/",
        'log_path': "/home/matthewcockayne/Documents/PhD/MMCrossTransformer/hyperparameters/logs/",
        'labels': [0, 1, 2, 3, 4],
        'class_num': 5,
        'patience': trial.suggest_int('patience', 10, 50),
        'batch_size': trial.suggest_categorical('batch_size', [16, 32, 64]),
        'weight_decay': trial.suggest_loguniform('weight_decay', 1e-5, 1e-2),
        'learning_rate': trial.suggest_loguniform('learning_rate', 1e-5, 1e-2),
        'optimizer': trial.suggest_categorical('optimizer', ['Adam', 'AdamW', 'SGD']),
        'scheduler': trial.suggest_categorical('scheduler', ['CosineAnnealingLR', 'StepLR', 'ExponentialLR']),
        'cuda': True,
        'pretrained': True,
        'cross_attention_depths': suggest_cross_attention_depths(trial)
    }
    
    log_file = setup_logging(options['log_path'], trial_number)
    log_params(options, log_file, trial_number)

    derm_data_group, _ = load_dataset(dir_release=options['dir_release'])
    model = setup_model(options, options['class_num'])
    
    if options['optimizer'] == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=options['learning_rate'], weight_decay=options['weight_decay'])
    elif options['optimizer'] == 'AdamW':
        optimizer = optim.AdamW(model.parameters(), lr=options['learning_rate'], weight_decay=options['weight_decay'])
    elif options['optimizer'] == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=options['learning_rate'], weight_decay=options['weight_decay'])

    if options['scheduler'] == 'CosineAnnealingLR':
        lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=options['epochs'])
    elif options['scheduler'] == 'StepLR':
        lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    elif options['scheduler'] == 'ExponentialLR':
        lr_scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)

    train_loader, valid_loader, test_loader = setup_data_loaders(options, derm_data_group)

    best_valid_loss = float('inf')
    start_time = time.time()
    patience_counter = 0
    old_model_path = None

    try:
        for epoch in range(options['epochs']):
            train_loss, train_accuracy = train_one_epoch(model, train_loader, optimizer, options)
            valid_loss, valid_accuracy = validate(model, valid_loader, options)

            lr_scheduler.step()

            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                patience_counter = 0
                new_model_path = os.path.join(options['modal_path'], f'trial_{trial_number}_best_model_epoch_{epoch}.pth')
                model_snapshot(model, new_model_path, old_model_path=old_model_path, only_best_model=True)
                old_model_path = new_model_path
                print("Found new best model, saving to disk...")
            else:
                patience_counter += 1

            if patience_counter >= options['patience']:
                break

            if np.isnan(train_loss):
                break

        end_time = time.time()
        total_training_time = end_time - start_time

        log_file.write(f'Trial Number: {trial_number}\n')
        log_file.write(f'Best Valid Loss: {best_valid_loss:.4f}\n')
        log_file.write(f'Total Training Time: {total_training_time:.4f}s\n')

    except Exception as e:
        import traceback
        traceback.print_exc()
    finally:
        log_file.close()
        torch.cuda.empty_cache()
        del train_loader, valid_loader, test_loader, model
        if 'derm_data_group' in locals():
            del derm_data_group

    return best_valid_loss

if __name__ == "__main__":
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=50)
    print('Best hyperparameters: ', study.best_params)