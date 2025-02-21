import numpy as np
import os
from prettytable import PrettyTable
import matplotlib.pyplot as plt
from sklearn.metrics import auc,roc_curve
import torch
"""
To DO:
- sort auc-roc plotting so that there are 8 plots with all subclasses plotted on one plot and then each is saved
instead of being overwritten
- Call plotmatrix in summary to plot and save confusion matrix for each class
- clean up code
"""
class ConfusionMatrix(object):
    def __init__(self, num_classes: int, labels: list):
        self.matrix = np.zeros((num_classes, num_classes))
        self.num_classes = num_classes
        self.labels = labels
        self.PrecisionofEachClass = [0.0 for _ in range(self.num_classes)]
        self.SensitivityofEachClass = [0.0 for _ in range(self.num_classes)]
        self.SpecificityofEachClass = [0.0 for _ in range(self.num_classes)]
        self.F1ScoreofEachClass = [0.0 for _ in range(self.num_classes)]
        self.AUCofEachClass = [0.0 for _ in range(self.num_classes)]
        self.acc = 0.0

    def update(self, pred, label):
        if isinstance(pred, int):
            pred = [pred]
        for p, t in zip(pred, label):
            self.matrix[int(p), int(t)] += 1
    
    def summary(self, pro_list, lab_list, File=None):
        sum_TP = 0
        for i in range(self.num_classes):
            sum_TP += self.matrix[i, i]
        self.acc = sum_TP / np.sum(self.matrix)
        if File is not None:
            File.write("The model accuracy is {}\n".format(self.acc))

        # Calculate AUC and capture the fpr_dict and tpr_dict
        auc_results = calculate_auc(pro_list, lab_list, self.num_classes, File)
        self.AUCofEachClass, fpr_dict, tpr_dict = auc_results

        table = PrettyTable()
        table.field_names = ["", "Precision", "Sensitivity", "Specificity", "F1 Score", "AUC"]

        precision_list = []
        sensitivity_list = []
        specificity_list = []

        for i in range(self.num_classes):
            TP = self.matrix[i, i]
            FP = np.sum(self.matrix[i, :]) - TP
            FN = np.sum(self.matrix[:, i]) - TP
            TN = np.sum(self.matrix) - TP - FP - FN

            Precision = round(TP / (TP + FP), 4) if TP + FP != 0 else 0.0
            Sensitivity = round(TP / (TP + FN), 4) if TP + FN != 0 else 0.0
            Specificity = round(TN / (TN + FP), 4) if TN + FP != 0 else 0.0
            F1Score = round(2 * Precision * Sensitivity / (Precision + Sensitivity), 4) if Precision + Sensitivity != 0 else 0.0

            self.PrecisionofEachClass[i] = Precision
            self.SensitivityofEachClass[i] = Sensitivity
            self.SpecificityofEachClass[i] = Specificity
            self.F1ScoreofEachClass[i] = F1Score

            precision_list.append(Precision)
            sensitivity_list.append(Sensitivity)
            specificity_list.append(Specificity)

            auc_value = self.AUCofEachClass[i] if i < len(self.AUCofEachClass) else "N/A"
            table.add_row([self.labels[i], Precision, Sensitivity, Specificity, F1Score, auc_value])

        if File is not None:
            File.write(str(table) + '\n')

        return self.acc, self.F1ScoreofEachClass, self.AUCofEachClass, precision_list, sensitivity_list, specificity_list, fpr_dict, tpr_dict
    """
    def summary(self, pro_list, lab_list, File=None):
        sum_TP = 0
        for i in range(self.num_classes):
            sum_TP += self.matrix[i, i]
        self.acc = sum_TP / np.sum(self.matrix)
        if File is not None:
            File.write("The model accuracy is {}\n".format(self.acc))

        # Calculate AUC
        auc_results = calculate_auc(pro_list, lab_list, self.num_classes, File)
        self.AUCofEachClass, _, _ = auc_results

        table = PrettyTable()
        table.field_names = ["", "Precision", "Sensitivity", "Specificity", "F1 Score", "AUC"]

        precision_list = []
        sensitivity_list = []
        specificity_list = []

        for i in range(self.num_classes):
            TP = self.matrix[i, i]
            FP = np.sum(self.matrix[i, :]) - TP
            FN = np.sum(self.matrix[:, i]) - TP
            TN = np.sum(self.matrix) - TP - FP - FN

            Precision = round(TP / (TP + FP), 4) if TP + FP != 0 else 0.0
            Sensitivity = round(TP / (TP + FN), 4) if TP + FN != 0 else 0.0
            Specificity = round(TN / (TN + FP), 4) if TN + FP != 0 else 0.0
            F1Score = round(2 * Precision * Sensitivity / (Precision + Sensitivity), 4) if Precision + Sensitivity != 0 else 0.0

            self.PrecisionofEachClass[i] = Precision
            self.SensitivityofEachClass[i] = Sensitivity
            self.SpecificityofEachClass[i] = Specificity
            self.F1ScoreofEachClass[i] = F1Score

            precision_list.append(Precision)
            sensitivity_list.append(Sensitivity)
            specificity_list.append(Specificity)

            auc_value = self.AUCofEachClass[i] if i < len(self.AUCofEachClass) else "N/A"
            table.add_row([self.labels[i], Precision, Sensitivity, Specificity, F1Score, auc_value])

        if File is not None:
            File.write(str(table) + '\n')

        return self.acc, self.F1ScoreofEachClass, self.AUCofEachClass, precision_list, sensitivity_list, specificity_list
    """
    def plotmatrix(self):
        plt.imshow(self.matrix, cmap=plt.cm.Blues)
        plt.xticks(range(self.num_classes), self.labels, rotation=45)
        plt.yticks(range(self.num_classes), self.labels)
        plt.colorbar()
        plt.xlabel('True Labels')
        plt.ylabel('Predicted Labels')

        thresh = self.matrix.max() / 2
        for x in range(self.num_classes):
            for y in range(self.num_classes):
                info = int(self.matrix[y, x])
                plt.text(x, y, info,
                         verticalalignment='center',
                         horizontalalignment='center',
                         color="white" if info > thresh else "black")

        plt.tight_layout()
        plt.show()

def calculate_auc(pro_list, lab_list, classnum, File):
    pro_array = np.array(pro_list)
    lab_array = np.array(lab_list)
    lab_tensor = torch.tensor(lab_array)
    lab_tensor = lab_tensor.reshape((lab_tensor.shape[0], 1))
    lab_onehot = torch.zeros(lab_tensor.shape[0], classnum)
    lab_onehot.scatter_(dim=1, index=lab_tensor, value=1)
    lab_onehot = np.array(lab_onehot)

    #print(f"pro_array shape: {pro_array.shape}")
    #print(f"lab_onehot shape: {lab_onehot.shape}")

    #table = PrettyTable()
    #table.field_names = ["Class", "AUC"]
    roc_auc = []
    fpr_dict = {}
    tpr_dict = {}

    min_classes = min(pro_array.shape[1], lab_onehot.shape[1])
    for i in range(min_classes):
        fpr, tpr, _ = roc_curve(lab_onehot[:, i], pro_array[:, i])
        auc_i = auc(fpr, tpr)
        roc_auc.append(auc_i)
        #table.add_row([i, auc_i])
        fpr_dict[i] = fpr
        tpr_dict[i] = tpr

    #File.write(str(table) + '\n')
    return roc_auc, fpr_dict, tpr_dict

def plot_roc_curves(log_path, fpr_dict_list, tpr_dict_list, class_labels, names):
    for i, name in enumerate(names):
        plt.figure(figsize=(10, 8))
        for j, label in enumerate(class_labels):
            if j in fpr_dict_list[i]:
                plt.plot(fpr_dict_list[i][j], tpr_dict_list[i][j], label=f'Class {label} (AUC = {auc(fpr_dict_list[i][j], tpr_dict_list[i][j]):.2f})')
        
        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'Receiver Operating Characteristic (ROC) Curves for {name} Class-')
        plt.legend(loc='lower right')
        plt.grid()
        plt.savefig(os.path.join(log_path, f'AUC_ROCplot_combined_{name}.png'))
        plt.close()

def plot_metrics(train_losses, valid_losses, train_acc, valid_acc, log_path, epochs):
    plt.figure(figsize=(8, 5))
    plt.plot(range(len(train_losses)), train_losses, label='Training Loss')
    plt.plot(range(len(valid_losses)), valid_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss Over Epochs')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(log_path, 'loss_plot.png'))
    plt.show()

    plt.figure(figsize=(8, 5))
    plt.plot(range(len(train_acc)), train_acc, label='Training Accuracy')
    plt.plot(range(len(valid_acc)), valid_acc, label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Accuracy Over Epochs')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(log_path, 'acc_plot.png'))
    plt.show()