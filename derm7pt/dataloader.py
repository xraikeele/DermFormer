# dataset_loader.py

import os
from pathlib import Path
import sys
sys.path.insert(0, '/home/matthewcockayne/Documents/PhD/Swin-Transformer/derm7pt')
import pandas as pd
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from PIL import Image
from torchvision import transforms
import torch
import cv2
import numpy as np
from sklearn.preprocessing import LabelEncoder
from derm7pt.dataset import Derm7PtDatasetGroupInfrequent

from albumentations import (
    Compose, VerticalFlip, HorizontalFlip, RandomRotate90, Rotate, ShiftScaleRotate, CLAHE, 
    RandomBrightnessContrast, ElasticTransform, GridDistortion, OpticalDistortion,
    GaussNoise, MotionBlur, RandomCrop, PadIfNeeded, HueSaturationValue, RGBShift, CoarseDropout, GridDropout, ChannelDropout
)

aug = Compose([
    VerticalFlip(p=0.5),
    HorizontalFlip(p=0.5),
    Rotate(limit=45, p=0.5),
    #RandomRotate90(p=0.5),
    #CLAHE(p=0.5),
    CoarseDropout(max_holes=20, max_height=1, max_width=1, min_holes=20, min_height=1, min_width=1, fill_value=0, p=0.5)
    #CoarseDropout(max_holes=8, max_height=8, max_width=8, min_holes=None, min_height=None, min_width=None, 
    #              fill_value=0, mask_fill_value=None, always_apply=None, p=0.5)
], p=1.0)

def load_dataset(dir_release):
    dir_meta = os.path.join(dir_release, 'meta')
    dir_images = os.path.join(dir_release, 'images')

    meta_df = pd.read_csv(os.path.join(dir_meta, 'meta.csv'))
    
    columns_of_interest = [
        'level_of_diagnostic_difficulty', 'elevation', 'location', 'sex', 'management'
    ]

    train_indexes = list(pd.read_csv(os.path.join(dir_meta, 'train_indexes.csv'))['indexes'])
    valid_indexes = list(pd.read_csv(os.path.join(dir_meta, 'valid_indexes.csv'))['indexes'])
    test_indexes = list(pd.read_csv(os.path.join(dir_meta, 'test_indexes.csv'))['indexes'])

    derm_data_group = Derm7PtDatasetGroupInfrequent(
        dir_images=dir_images,
        metadata_df=meta_df.copy(),
        train_indexes=train_indexes,
        valid_indexes=valid_indexes,
        test_indexes=test_indexes
    )

    derm_data_group.dataset_stats()

    derm_data_group.meta_train = derm_data_group.meta_train[columns_of_interest]
    derm_data_group.meta_valid = derm_data_group.meta_valid[columns_of_interest]
    derm_data_group.meta_test = derm_data_group.meta_test[columns_of_interest]
    return derm_data_group, columns_of_interest

def load_image(path, shape):
    img = cv2.imread(path)
    img = cv2.resize(img, (shape[0], shape[1]))
    return img

class dataset(Dataset):
    def __init__(self, derm, shape, mode='train'):
        self.shape = shape
        self.mode = mode
        self.derm_paths = derm.get_img_paths(data_type=mode, img_type='derm')
        self.clinic_paths = derm.get_img_paths(data_type=mode, img_type='clinic')
        self.labels = derm.get_labels(data_type=mode, one_hot=False)

        if not self.derm_paths or not self.clinic_paths:
            raise ValueError(f"No image paths found for mode '{mode}'")
        
        if self.mode == 'train':
            self.meta = self.preprocess_metadata(derm.meta_train)
        elif self.mode == 'valid':
            self.meta = self.preprocess_metadata(derm.meta_valid)
        else:
            self.meta = self.preprocess_metadata(derm.meta_test)

        self.class_weights = self.calculate_class_weights()

    def preprocess_metadata(self, meta_df):
        encoder = LabelEncoder()
        for col in meta_df.columns:
            if meta_df[col].dtype == 'object':
                meta_df[col] = encoder.fit_transform(meta_df[col])
        return meta_df.values.astype(np.float32)
    """
    def calculate_class_weights(self):
        class_weights = []

        for label_name, label_data in self.labels.items():
            class_counts = np.bincount(label_data)
            total_samples = len(label_data)
            class_frequency = class_counts / total_samples
            class_weight = 1 / (class_frequency + 1e-3)
            class_weight /= class_weight.sum()
            class_weights.append(torch.Tensor(class_weight))

        return class_weights
    """
    def calculate_class_weights(self):
        class_weights = []

        for label_name, label_data in self.labels.items():
            class_counts = np.bincount(label_data)
            total_samples = len(label_data)
            class_frequency = class_counts / total_samples
            class_weight = 1 / (class_frequency + 1e-3)  # To avoid division by zero
            class_weight /= class_weight.sum()

            # Optional: Clipping weights to avoid large imbalances
            class_weight = np.clip(class_weight, 0.5, 2.0)  # Adjust the range based on your data
            class_weights.append(torch.Tensor(class_weight))

        return class_weights
    def __getitem__(self, index):
        dermoscopy_img_path = self.derm_paths[index]
        clinic_img_path = self.clinic_paths[index]
        dermoscopy_img = load_image(dermoscopy_img_path, self.shape)
        clinic_img = load_image(clinic_img_path, self.shape)

        if self.mode == 'train':
            augmented = aug(image=clinic_img, mask=dermoscopy_img)
            clinic_img = augmented['image']
            dermoscopy_img = augmented['mask']

        clinic_img = torch.from_numpy(np.transpose(clinic_img, (2, 0, 1)).astype('float32') / 255)
        dermoscopy_img = torch.from_numpy(np.transpose(dermoscopy_img, (2, 0, 1)).astype('float32') / 255)

        DIAG = torch.LongTensor([self.labels['DIAG'][index]])
        PN = torch.LongTensor([self.labels['PN'][index]])
        BWV = torch.LongTensor([self.labels['BWV'][index]])
        VS = torch.LongTensor([self.labels['VS'][index]])
        PIG = torch.LongTensor([self.labels['PIG'][index]])
        STR = torch.LongTensor([self.labels['STR'][index]])
        DaG = torch.LongTensor([self.labels['DaG'][index]])
        RS = torch.LongTensor([self.labels['RS'][index]])

        metadata = torch.from_numpy(self.meta[index])
        meta_con = torch.randn(0, 0)

        return dermoscopy_img, clinic_img, metadata, meta_con, [DIAG, PN, BWV, VS, PIG, STR, DaG, RS]

    def __len__(self):
        return len(self.clinic_paths)

if __name__ == '__main__':
    dir_release = "/home/matthewcockayne/Documents/PhD/data/Derm7pt/release_v0/release_v0"
    derm_data_group, columns_of_interest = load_dataset(dir_release)
    
    dataset_train = dataset(derm_data_group, (224, 224), mode='train')
    dataset_valid = dataset(derm_data_group, (224, 224), mode='valid')
    dataset_test = dataset(derm_data_group, (224, 224), mode='test')

    # Creating data loaders
    train_loader = DataLoader(dataset_train, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)
    valid_loader = DataLoader(dataset_valid, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(dataset_test, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)

    print(f"Number of samples in training dataset: {len(dataset_train)}")
    print(f"Number of samples in validation dataset: {len(dataset_valid)}")
    print(f"Number of samples in test dataset: {len(dataset_test)}")