import os
from pathlib import Path
import sys
sys.path.insert(0, '/home/matthewcockayne/Documents/PhD/MMCrossTransformer/derm7pt')
import pandas as pd
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from PIL import Image
from torchvision import transforms
import torch
import cv2
import numpy as np
import random
from sklearn.preprocessing import LabelEncoder
#from derm7pt.dataset import Derm7PtDatasetGroupInfrequent
from dataset import Derm7PtDatasetGroupInfrequent
import matplotlib.pyplot as plt
import random

from albumentations import (
    Compose, VerticalFlip, HorizontalFlip, RandomRotate90, Rotate, ShiftScaleRotate, CLAHE, 
    RandomBrightnessContrast, ElasticTransform, GridDistortion, OpticalDistortion,
    GaussNoise, MotionBlur, RandomCrop, PadIfNeeded, HueSaturationValue, RGBShift, CoarseDropout, GridDropout, ChannelDropout
)

aug = Compose([
    VerticalFlip(p=0.5),
    #HorizontalFlip(p=0.5),
    #Rotate(limit=45, p=0.5),
    #RandomRotate90(p=0.5),
    #CLAHE(p=0.5),
    #CoarseDropout(max_holes=8, max_height=8, max_width=8, min_holes=None, min_height=None, min_width=None, 
    #              fill_value=0, mask_fill_value=None, always_apply=None, p=0.5)
], p=1.0)

def load_noise(dir_release):
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
"""
def apply_noise(image, corruption_percentage):
    
    Apply random noise to a given percentage of an image's pixels.
    
    Args:
        image (numpy.ndarray): The input image in shape (H, W, C).
        corruption_percentage (float): The percentage of pixels to corrupt (0 to 1).
    
    Returns:
        numpy.ndarray: The corrupted image.
   
    height, width, channels = image.shape
    num_pixels = height * width
    num_corrupted = int(num_pixels * corruption_percentage)
    
    # Generate random pixel indices
    corrupted_indices = random.sample(range(num_pixels), num_corrupted)
    corrupted_coords = np.unravel_index(corrupted_indices, (height, width))
    
    # Add random noise to these pixel positions
    noise = np.random.randint(0, 256, size=(num_corrupted, channels), dtype=np.uint8)
    image[corrupted_coords[0], corrupted_coords[1], :] = noise
    
    return image
"""
def apply_noise(image, corruption_percentage):
    """
    Apply random noise to a given percentage of an image's pixels.
    
    Args:
        image (numpy.ndarray): The input image in shape (H, W, C).
        corruption_percentage (float): The percentage of pixels to corrupt (0 to 1).
    
    Returns:
        numpy.ndarray: The corrupted image.
    """
    height, width, channels = image.shape
    num_pixels = height * width
    num_corrupted = int(num_pixels * corruption_percentage)
    
    if num_corrupted == 0:
        return image  # No corruption needed

    # Generate random pixel indices
    corrupted_indices = np.random.choice(num_pixels, num_corrupted, replace=False)
    corrupted_coords = np.unravel_index(corrupted_indices, (height, width))

    # Add random noise to these pixel positions
    noise = np.random.randint(0, 256, size=(num_corrupted, channels), dtype=np.uint8)
    image[corrupted_coords[0], corrupted_coords[1], :] = noise

    return image

class dataset_noise(Dataset):
    def __init__(self, derm, shape, mode='train', derm_noise=0.0, clinical_noise=0.0):
        self.shape = shape
        self.mode = mode
        self.derm_paths = derm.get_img_paths(data_type=mode, img_type='derm')
        self.clinic_paths = derm.get_img_paths(data_type=mode, img_type='clinic')
        self.labels = derm.get_labels(data_type=mode, one_hot=False)
        self.derm_noise = derm_noise
        self.clinical_noise = clinical_noise

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
    
    def __getitem__(self, index):
        dermoscopy_img_path = self.derm_paths[index]
        clinic_img_path = self.clinic_paths[index]
        dermoscopy_img = load_image(dermoscopy_img_path, self.shape)
        clinic_img = load_image(clinic_img_path, self.shape)

        # Apply noise to dermoscopy and clinical images
        if self.derm_noise > 0:
            dermoscopy_img = apply_noise(dermoscopy_img, self.derm_noise)
        if self.clinical_noise > 0:
            clinic_img = apply_noise(clinic_img, self.clinical_noise)

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
    
def visualize_noise_effect(dataset, steps=5, save_path="noise_levels"):
    """
    Display a grid of images showing the effect of applying derm_noise
    from 0.0 to 1.0 in equal steps.
    
    Args:
        dataset (dataset_noise): The dataset to sample an image from.
        steps (int): Number of steps in the noise range.
        save_path (str, optional): Path to save the figure.
    """
    # Select a random image from the dataset
    random_index = random.randint(0, len(dataset) - 1)
    dermoscopy_img, _, _, _, _ = dataset[random_index]
    
    # Convert tensor image to numpy array
    dermoscopy_img = dermoscopy_img.permute(1, 2, 0).numpy()
    
    # Define noise levels
    noise_levels = np.round(np.linspace(0.0, 1.0, steps), 1)
    rows = 2
    cols = steps // 2 if steps % 2 == 0 else (steps // 2) + 1
    fig, axes = plt.subplots(rows, cols, figsize=(15, 6), constrained_layout=True)
    axes = axes.flatten()
    for i, noise in enumerate(noise_levels):
            noisy_img = apply_noise(dermoscopy_img.copy(), noise)
            
            axes[i].imshow(noisy_img)
            axes[i].set_title(f"Noise: {noise:.2f}")
            axes[i].axis("off")
        
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        
        plt.show()
    

if __name__ == '__main__':
    dir_release = "/home/matthewcockayne/Documents/PhD/data/Derm7pt/release_v0/release_v0"
    derm_data_group, columns_of_interest = load_noise(dir_release)
    
    dataset_train = dataset_noise(derm_data_group, (224, 224), mode='train')
    dataset_valid = dataset_noise(derm_data_group, (224, 224), mode='valid')
    dataset_test = dataset_noise(derm_data_group, (224, 224), mode='test')

    # Creating data loaders
    train_loader = DataLoader(dataset_train, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)
    valid_loader = DataLoader(dataset_valid, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(dataset_test, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)

    print(f"Number of samples in training dataset: {len(dataset_train)}")
    print(f"Number of samples in validation dataset: {len(dataset_valid)}")
    print(f"Number of samples in test dataset: {len(dataset_test)}")    

    visualize_noise_effect(dataset_train, steps=10)