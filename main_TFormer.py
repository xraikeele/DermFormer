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
import argparse
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from torch.autograd import Variable
import os
import time
import torch.nn.functional as F
import matplotlib.pyplot as plt
from derm7pt.eval_metrics import ConfusionMatrix, plot_metrics, plot_roc_curves
from derm7pt.dataloader import load_dataset, dataset, train_data_transformation, test_data_transformation
from models.TFormer import TFormer

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
    model = TFormer(class_num)
    if options['cuda']:
        model = model.cuda()
    return model

def model_snapshot(model, new_model_path, old_model_path=None, only_best_model=False):
    if only_best_model and old_model_path:
        os.remove(old_model_path)
    torch.save(model.state_dict(), new_model_path)

def setup_data_loaders(options, derm_data_group):
    train_loader = DataLoader(dataset(derm=derm_data_group,shape=(224,224),mode='train'),
                                batch_size=options['batch_size'],shuffle=True,num_workers=4)
    valid_loader = DataLoader(dataset(derm= derm_data_group,shape=(224,224),mode='valid'),
                                batch_size=1,shuffle=False,num_workers=2)
    test_loader = DataLoader(dataset(derm=derm_data_group,shape=(224,224),mode='test'),
                               batch_size=1,shuffle=False,num_workers=2)
    return train_loader, valid_loader, test_loader

def train_one_epoch(model, train_loader, optimizer, options, log_file, epoch):
    model.train()
    train_loss = 0.0
    correct = [0] * 8
    total = [0] * 8
    print(f"Training epoch: {epoch}")
    log_file.write(f"Training epoch: {epoch}\n")
    #for batch_idx, (der_data, cli_data, meta_data, meta_con, target) in enumerate(train_loader):
    for batch_idx, (der_data, cli_data, meta_data, target) in enumerate(train_loader):
        diagnosis_label = target[0].squeeze(1).cuda()
        pn_label = target[1].squeeze(1).cuda()
        bmv_label = target[2].squeeze(1).cuda()
        vs_label = target[3].squeeze(1).cuda()
        pig_label = target[4].squeeze(1).cuda()
        str_label = target[5].squeeze(1).cuda()
        dag_label = target[6].squeeze(1).cuda()
        rs_label = target[7].squeeze(1).cuda()

        targets = [diagnosis_label, pn_label, bmv_label, vs_label, pig_label, str_label, dag_label, rs_label]

        if options['cuda']:
            der_data, cli_data, meta_data = der_data.cuda(), cli_data.cuda(), meta_data.cuda().float()
            der_data, cli_data, meta_data = Variable(der_data), Variable(cli_data), Variable(meta_data)
            #meta_data = meta_data.long()
            #meta_con = meta_con.long()

        optimizer.zero_grad()
        outputs = model(meta_data, cli_data, der_data)

        class_weights = train_loader.dataset.class_weights
        weights = [class_weights[i].cuda() if options['cuda'] else class_weights[i] for i in range(8)]

        losses = [F.cross_entropy(output, target, weight=weight) for output, target, weight in zip(outputs, targets, weights)]
        loss = sum(losses) / len(losses)

        loss.backward()
        avg_loss = loss.item()
        train_loss += avg_loss

        optimizer.step()

        predicted_results = [output.data.max(1)[1] for output in outputs]
        for i, (pred, true) in enumerate(zip(predicted_results, targets)):
            correct[i] += pred.eq(true).sum().item()
            total[i] += true.size(0)

        if batch_idx % 50 == 0 and batch_idx > 0:
            accuracy = np.mean([c / t if t > 0 else 0 for c, t in zip(correct, total)])
            print(f'Training epoch: {epoch} [{batch_idx * len(der_data)}/{len(train_loader.dataset)}], '
                  f'Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}, Learning rate: {optimizer.param_groups[0]["lr"]}')

    accuracy = np.mean([c / t if t > 0 else 0 for c, t in zip(correct, total)])
    log_file.write(f"Training Loss: {train_loss / len(train_loader):.4f}, Accuracy: {accuracy:.4f}\n")
    return train_loss / len(train_loader), accuracy

def validate(model, valid_loader, options, label_t, class_num, log_file):
    model.eval()
    avg_valid_loss = 0
    correct = [0] * 8
    total = [0] * 8
    val_confusion_matrices = [ConfusionMatrix(num_classes=class_num, labels=label_t) for _ in range(8)]
    pro_list = [[] for _ in range(8)]
    lab_list = [[] for _ in range(8)]

    with torch.no_grad():
        #for der_data, cli_data, meta_data, meta_con, target in valid_loader:
        for der_data, cli_data, meta_data, target in valid_loader:
            diagnosis_label = target[0].squeeze(1).cuda()
            pn_label = target[1].squeeze(1).cuda()
            bmv_label = target[2].squeeze(1).cuda()
            vs_label = target[3].squeeze(1).cuda()
            pig_label = target[4].squeeze(1).cuda()
            str_label = target[5].squeeze(1).cuda()
            dag_label = target[6].squeeze(1).cuda()
            rs_label = target[7].squeeze(1).cuda()

            targets = [diagnosis_label, pn_label, bmv_label, vs_label, pig_label, str_label, dag_label, rs_label]

            if options['cuda']:
                der_data, cli_data, meta_data = der_data.cuda(), cli_data.cuda(), meta_data.cuda().float()
                der_data, cli_data, meta_data = Variable(der_data), Variable(cli_data), Variable(meta_data)
                #meta_data = meta_data.long()
                #meta_con = meta_con.long()

            outputs = model(meta_data, cli_data, der_data)

            class_weights = valid_loader.dataset.class_weights
            weights = [class_weights[i].cuda() if options['cuda'] else class_weights[i] for i in range(8)]
            losses = [F.cross_entropy(output, target, weight=weight) for output, target, weight in zip(outputs, targets, weights)]
            loss = sum(losses) / len(losses)
            avg_valid_loss += loss.item()

            predicted_results = [output.data.max(1)[1] for output in outputs]
            for i, (pred, true) in enumerate(zip(predicted_results, targets)):
                val_confusion_matrices[i].update(pred.cpu().numpy(), true.cpu().numpy())
                pro_list[i].extend(F.softmax(outputs[i], dim=1).cpu().numpy())
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

def test(model, test_loader, options, log_file):
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
        for der_data, cli_data, meta_data, target in test_loader:
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
            outputs = model(meta_data, cli_data, der_data)

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
    plot_roc_curves(options['log_path'], fpr_dict_list, tpr_dict_list, options['labels'], names)

def main(options):
    log_file = setup_logging(options['log_path'])
    log_params(options, log_file)

    derm_data_group, column_interest = load_dataset(dir_release=options['dir_release'])
    #derm_data_group = load_dataset(dir_release=options['dir_release'])
    mata = derm_data_group.meta_train.values
    print(mata)
    model = setup_model(options, options['class_num'])
    #optimizer = optim.AdamW(model.parameters(), lr=options['learning_rate'], betas=(0.9, 0.999), weight_decay=options['weight_decay'])
    optimizer = optim.Adam(model.parameters(), lr=options['learning_rate'], betas=(0.9, 0.999), weight_decay=options['weight_decay'])
    lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=options['epochs'])
    #lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
    #lr_scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
    train_loader, valid_loader, test_loader = setup_data_loaders(options, derm_data_group)

    #best_accuracy = 0
    best_valid_loss = float('inf')
    train_losses, valid_losses = [], []
    train_acc, valid_acc = [], []
    old_model_path = None
    start_time = time.time()
    patience_counter = 0  # Early stopping counter
    try:
        for epoch in range(options['epochs']):
            train_loss, train_accuracy = train_one_epoch(model, train_loader, optimizer, options, log_file, epoch)
            train_losses.append(train_loss)
            train_acc.append(train_accuracy)

            valid_loss, valid_accuracy = validate(model, valid_loader, options, options['labels'], options['class_num'], log_file)
            valid_losses.append(valid_loss)
            valid_acc.append(valid_accuracy)

            lr_scheduler.step()

            #if valid_accuracy > best_accuracy:
            #    best_accuracy = valid_accuracy
            
            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
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

            if epoch % 10 == 0 or epoch == options['epochs'] - 1:
                model_snapshot(model, os.path.join(options['modal_path'], f'model-{epoch}.pth'))

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
            test(model, test_loader, options, log_file)

        plot_metrics(train_losses, valid_losses, train_acc, valid_acc, options['log_path'], options['epochs'])

    except Exception as e:
        import traceback
        traceback.print_exc()
    finally:
        log_file.close()

if __name__ == "__main__":
    OPTIONS = argparse.ArgumentParser()

    OPTIONS.add_argument('--epochs', dest='epochs', type=int, default=100)
    OPTIONS.add_argument('--dir_release', dest='dir_release', type=str, default="/home/matthewcockayne/datasets/Derm7pt/release_v0/release_v0")
    OPTIONS.add_argument('--modal_path', dest='modal_path', type=str, default="/home/matthewcockayne/Documents/PhD/experiments/test_TFormer/originaldataloader/models/")
    OPTIONS.add_argument('--log_path', dest='log_path', type=str, default="/home/matthewcockayne/Documents/PhD/experiments/test_TFormer/originaldataloader/logs/")
    OPTIONS.add_argument('--labels', dest='labels', default=[0, 1, 2, 3, 4])
    OPTIONS.add_argument('--class_num', dest='class_num', type=int, default=5)
    OPTIONS.add_argument('--patience', dest='patience', type=int, default=28)
    OPTIONS.add_argument('--cross_attention_depths', dest='cross_attention_depths', type=str, default='1,1,2,1,1')
    OPTIONS.add_argument('--batch_size', dest='batch_size', type=int, default=32)
    OPTIONS.add_argument('--weight_decay', dest='weight_decay', type=float, default=1e-4)
    OPTIONS.add_argument('--learning_rate', dest='learning_rate', type=float, default=1e-4)
    OPTIONS.add_argument('--cuda', dest='cuda', type=bool, default=True)
    OPTIONS.add_argument('--pretrained', dest='pretrained', type=bool, default=True)

    args = OPTIONS.parse_args()
    PARAMS = vars(args)
    PARAMS['cross_attention_depths'] = list(map(int, PARAMS['cross_attention_depths'].split(',')))
    main(PARAMS)