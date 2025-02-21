import os
from pathlib import Path
import sys
import random
sys.path.insert(0, '/home/matthewcockayne/Documents/PhD/Swin-Transformer/derm7pt')
import pandas as pd
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from PIL import Image
import torchvision
from torchvision import transforms
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
#from derm7pt.dataset import Derm7PtDatasetGroupInfrequent
from dataset import Derm7PtDatasetGroupInfrequent
from albumentations import (
    Compose, VerticalFlip, HorizontalFlip, RandomRotate90, Rotate, ShiftScaleRotate, CLAHE, 
    RandomBrightnessContrast, ElasticTransform, GridDistortion, OpticalDistortion,
    GaussNoise, MotionBlur, RandomCrop, PadIfNeeded, HueSaturationValue, RGBShift, CoarseDropout, GridDropout, ChannelDropout
)

aug = Compose([
    VerticalFlip(p=0.5),
    #HorizontalFlip(p=0.5),
    #Rotate(limit=45, p=0.5),
    #CLAHE(p=0.5),
    #CoarseDropout(max_holes=8, max_height=8, max_width=8, min_holes=None, min_height=None, min_width=None, 
    #              fill_value=0, mask_fill_value=None, always_apply=None, p=0.5)
], p=1.0)

# /////////////// Distortion Helpers ///////////////

import skimage as sk
from skimage.filters import gaussian
from io import BytesIO
from wand.image import Image as WandImage
from wand.api import library as wandlibrary
import wand.color as WandColor
import ctypes
from PIL import Image as PILImage
import cv2
from scipy.ndimage import zoom as scizoom
from scipy.ndimage.interpolation import map_coordinates
import warnings

warnings.simplefilter("ignore", UserWarning)


def auc(errs):  # area under the alteration error curve
    area = 0
    for i in range(1, len(errs)):
        area += (errs[i] + errs[i - 1]) / 2
    area /= len(errs) - 1
    return area


def disk(radius, alias_blur=0.1, dtype=np.float32):
    if radius <= 8:
        L = np.arange(-8, 8 + 1)
        ksize = (3, 3)
    else:
        L = np.arange(-radius, radius + 1)
        ksize = (5, 5)
    X, Y = np.meshgrid(L, L)
    aliased_disk = np.array((X ** 2 + Y ** 2) <= radius ** 2, dtype=dtype)
    aliased_disk /= np.sum(aliased_disk)

    # supersample disk to antialias
    return cv2.GaussianBlur(aliased_disk, ksize=ksize, sigmaX=alias_blur)


# Tell Python about the C method
wandlibrary.MagickMotionBlurImage.argtypes = (ctypes.c_void_p,  # wand
                                              ctypes.c_double,  # radius
                                              ctypes.c_double,  # sigma
                                              ctypes.c_double)  # angle


# Extend wand.image.Image class to include method signature
class MotionImage(WandImage):
    def motion_blur(self, radius=0.0, sigma=0.0, angle=0.0):
        wandlibrary.MagickMotionBlurImage(self.wand, radius, sigma, angle)


# modification of https://github.com/FLHerne/mapgen/blob/master/diamondsquare.py
def plasma_fractal(mapsize=224, wibbledecay=3):
    """
    Generate a heightmap using diamond-square algorithm.
    Return square 2d array, side length 'mapsize', of floats in range 0-255.
    'mapsize' must be a power of two.
    """
    assert (mapsize & (mapsize - 1) == 0)
    maparray = np.empty((mapsize, mapsize), dtype=np.float_)
    maparray[0, 0] = 0
    stepsize = mapsize
    wibble = 100

    def wibbledmean(array):
        return array / 4 + wibble * np.random.uniform(-wibble, wibble, array.shape)

    def fillsquares():
        """For each square of points stepsize apart,
           calculate middle value as mean of points + wibble"""
        cornerref = maparray[0:mapsize:stepsize, 0:mapsize:stepsize]
        squareaccum = cornerref + np.roll(cornerref, shift=-1, axis=0)
        squareaccum += np.roll(squareaccum, shift=-1, axis=1)
        maparray[stepsize // 2:mapsize:stepsize,
        stepsize // 2:mapsize:stepsize] = wibbledmean(squareaccum)

    def filldiamonds():
        """For each diamond of points stepsize apart,
           calculate middle value as mean of points + wibble"""
        mapsize = maparray.shape[0]
        drgrid = maparray[stepsize // 2:mapsize:stepsize, stepsize // 2:mapsize:stepsize]
        ulgrid = maparray[0:mapsize:stepsize, 0:mapsize:stepsize]
        ldrsum = drgrid + np.roll(drgrid, 1, axis=0)
        lulsum = ulgrid + np.roll(ulgrid, -1, axis=1)
        ltsum = ldrsum + lulsum
        maparray[0:mapsize:stepsize, stepsize // 2:mapsize:stepsize] = wibbledmean(ltsum)
        tdrsum = drgrid + np.roll(drgrid, 1, axis=1)
        tulsum = ulgrid + np.roll(ulgrid, -1, axis=0)
        ttsum = tdrsum + tulsum
        maparray[stepsize // 2:mapsize:stepsize, 0:mapsize:stepsize] = wibbledmean(ttsum)

    while stepsize >= 2:
        fillsquares()
        filldiamonds()
        stepsize //= 2
        wibble /= wibbledecay

    maparray -= maparray.min()
    return maparray / maparray.max()


def clipped_zoom(img, zoom_factor):
    h = img.shape[0]
    # ceil crop height(= crop width)
    ch = int(np.ceil(h / zoom_factor))

    top = (h - ch) // 2
    img = scizoom(img[top:top + ch, top:top + ch], (zoom_factor, zoom_factor, 1), order=1)
    # trim off any extra pixels
    trim_top = (img.shape[0] - h) // 2

    return img[trim_top:trim_top + h, trim_top:trim_top + h]


def imshow(img, title=None):
    # Unnormalize and display the image
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    if title:
        plt.title(title)
    plt.show()
# /////////////// End Distortion Helpers ///////////////


# /////////////// Distortions ///////////////

def gaussian_noise(x, severity=1):
    c = [0.04, 0.08, .12, .15, .18][severity - 1]

    x = np.array(x) / 255.
    return np.clip(x + np.random.normal(size=x.shape, scale=c), 0, 1) * 255


def shot_noise(x, severity=1):
    c = [250, 100, 50, 30, 15][severity - 1]

    x = np.array(x) / 255.
    return np.clip(np.random.poisson(x * c) / c, 0, 1) * 255


def impulse_noise(x, severity=1):
    c = [.01, .02, .05, .08, .14][severity - 1]

    x = sk.util.random_noise(np.array(x) / 255., mode='s&p', amount=c)
    return np.clip(x, 0, 1) * 255


def speckle_noise(x, severity=1):
    c = [.15, .2, 0.25, 0.3, 0.35][severity - 1]

    x = np.array(x) / 255.
    return np.clip(x + x * np.random.normal(size=x.shape, scale=c), 0, 1) * 255


def gaussian_blur(x, severity=1):
    c = [.5, .75, 1, 1.25, 1.5][severity - 1]

    x = gaussian(np.array(x) / 255., sigma=c, channel_axis=True)
    return np.clip(x, 0, 1) * 255


def glass_blur(x, severity=1):
    # sigma, max_delta, iterations
    c = [(0.1, 1, 1), (0.5, 1, 1), (0.6, 1, 2), (0.7, 2, 1), (0.9, 2, 2)][severity - 1]

    # Apply gaussian blur
    x = np.uint8(gaussian(np.array(x) / 255., sigma=c[0], channel_axis=True) * 255)

    # Get image height and width
    h, w = x.shape[:2]

    # Locally shuffle pixels
    for i in range(c[2]):  # Shuffle for specified number of iterations
        for h_idx in range(h):  # Loop through height
            for w_idx in range(w):  # Loop through width
                dx, dy = np.random.randint(-c[1], c[1], size=(2,))  # Random offset
                h_prime, w_prime = h_idx + dy, w_idx + dx

                # Ensure h_prime and w_prime are within valid bounds
                if 0 <= h_prime < h and 0 <= w_prime < w:
                    # Swap pixels at (h, w) and (h_prime, w_prime)
                    x[h_idx, w_idx], x[h_prime, w_prime] = x[h_prime, w_prime], x[h_idx, w_idx]

    # Apply gaussian blur again after pixel shuffling
    return np.clip(gaussian(x / 255., sigma=c[0], channel_axis=True), 0, 1) * 255


def defocus_blur(x, severity=1):
    c = [(0.5, 0.6), (1, 0.1), (1.5, 0.1), (2.5, 0.01), (3, 0.1)][severity - 1]

    x = np.array(x) / 255.
    kernel = disk(radius=c[0], alias_blur=c[1])

    channels = []
    for d in range(3):
        channels.append(cv2.filter2D(x[:, :, d], -1, kernel))
    channels = np.array(channels).transpose((1, 2, 0))  # 3x64x64 -> 64x64x3

    return np.clip(channels, 0, 1) * 255

def motion_blur(x, severity=1):
    # Define motion blur parameters based on severity
    c = [(10, 1), (10, 1.5), (10, 2), (10, 2.5), (12, 3)][severity - 1]

    # Convert NumPy array to PIL Image if it's not already and enforce RGB
    if isinstance(x, np.ndarray):
        x = PILImage.fromarray(np.uint8(x)).convert("RGB")

    # Save the image to a BytesIO buffer to pass it to wand
    output = BytesIO()
    x.save(output, format='PNG')
    
    # Process the image with motion blur using the wand library
    x = MotionImage(blob=output.getvalue())
    x.motion_blur(radius=c[0], sigma=c[1], angle=np.random.uniform(-45, 45))

    # Convert the result back to a NumPy array
    x = cv2.imdecode(np.frombuffer(x.make_blob(), np.uint8), cv2.IMREAD_UNCHANGED)

    # If the result is grayscale, convert to 3 channels
    if len(x.shape) == 2 or x.shape[-1] == 1:
        x = cv2.cvtColor(x, cv2.COLOR_GRAY2BGR)
    
    # Convert BGR to RGB
    if x.shape[-1] == 3:
        x = x[..., [2, 1, 0]]
    
    return np.clip(x, 0, 255)

def zoom_blur(x, severity=1):
    c = [np.arange(1, 1.06, 0.01), np.arange(1, 1.11, 0.01), np.arange(1, 1.16, 0.01),
         np.arange(1, 1.21, 0.01), np.arange(1, 1.26, 0.01)][severity - 1]

    x = (np.array(x) / 255.).astype(np.float32)
    out = np.zeros_like(x)
    for zoom_factor in c:
        out += clipped_zoom(x, zoom_factor)

    x = (x + out) / (len(c) + 1)
    return np.clip(x, 0, 1) * 255

"""
def fog(x, severity=1):
    c = [(.4,3), (.7,3), (1,2.5), (1.5,2), (2,1.75)][severity - 1]

    x = np.array(x) / 255.
    max_val = x.max()
    x += c[0] * plasma_fractal(wibbledecay=c[1])[:224, :224][..., np.newaxis]
    return np.clip(x * max_val / (max_val + c[0]), 0, 1) * 255

def frost(x, severity=1):
    c = [(1, 0.3), (0.9, 0.4), (0.8, 0.45), (0.75, 0.5), (0.7, 0.55)][severity - 1]
    idx = np.random.randint(5)
    filename = ['./frost1.png', './frost2.png', './frost3.png', './frost4.jpg', './frost5.jpg', './frost6.jpg'][idx]
    frost = cv2.imread(filename)
    frost = cv2.resize(frost, (0, 0), fx=0.3, fy=0.3)
    # randomly crop and convert to rgb
    x_start, y_start = np.random.randint(0, frost.shape[0] - 64), np.random.randint(0, frost.shape[1] - 64)
    frost = frost[x_start:x_start + 64, y_start:y_start + 64][..., [2, 1, 0]]

    return np.clip(c[0] * np.array(x) + c[1] * frost, 0, 255)
"""

def snow(x, severity=1):
    c = [(0.1, 0.2, 1, 0.6, 8, 3, 0.8),
         (0.1, 0.2, 1, 0.5, 10, 4, 0.8),
         (0.15, 0.3, 1.75, 0.55, 10, 4, 0.7),
         (0.25, 0.3, 2.25, 0.6, 12, 6, 0.65),
         (0.3, 0.3, 1.25, 0.65, 14, 12, 0.6)][severity - 1]

    # Normalize the image to [0, 1]
    x = np.array(x, dtype=np.float32) / 255.0

    # Generate snow layer with random normal distribution
    snow_layer = np.random.normal(size=x.shape[:2], loc=c[0], scale=c[1])
    # Apply zoom and clipping
    snow_layer = clipped_zoom(snow_layer[..., np.newaxis], c[2])
    snow_layer[snow_layer < c[3]] = 0

    # Convert snow_layer to a PIL Image and apply motion blur
    snow_layer_pil = PILImage.fromarray((np.clip(snow_layer.squeeze(), 0, 1) * 255).astype(np.uint8), mode='L')
    output = BytesIO()
    snow_layer_pil.save(output, format='PNG')
    snow_layer_wand = MotionImage(blob=output.getvalue())
    snow_layer_wand.motion_blur(radius=c[4], sigma=c[5], angle=np.random.uniform(-135, -45))
    snow_layer = cv2.imdecode(np.frombuffer(snow_layer_wand.make_blob(), np.uint8), cv2.IMREAD_UNCHANGED) / 255.0
    snow_layer = snow_layer[..., np.newaxis]

    # Instead of converting x to grayscale with cvtColor, compute the average channel value
    gray = np.mean(x, axis=2, keepdims=True)

    # Combine the snow and the original image in a color-preserving way
    x = c[6] * x + (1 - c[6]) * np.maximum(x, gray * 1.5 + 0.5)

    # Add snow layer and a rotated version of it
    result = np.clip(x + snow_layer + np.rot90(snow_layer, k=2), 0, 1) * 255
    return result


def spatter(x, severity=1):
    c = [(0.62,0.1,0.7,0.7,0.6,0),
         (0.65,0.1,0.8,0.7,0.6,0),
         (0.65,0.3,1,0.69,0.6,0),
         (0.65,0.1,0.7,0.68,0.6,1),
         (0.65,0.1,0.5,0.67,0.6,1)][severity - 1]
    x = np.array(x, dtype=np.float32) / 255.

    liquid_layer = np.random.normal(size=x.shape[:2], loc=c[0], scale=c[1])

    liquid_layer = gaussian(liquid_layer, sigma=c[2])
    liquid_layer[liquid_layer < c[3]] = 0
    if c[5] == 0:
        liquid_layer = (liquid_layer * 255).astype(np.uint8)
        dist = 255 - cv2.Canny(liquid_layer, 50, 150)
        dist = cv2.distanceTransform(dist, cv2.DIST_L2, 5)
        _, dist = cv2.threshold(dist, 20, 20, cv2.THRESH_TRUNC)
        dist = cv2.blur(dist, (3, 3)).astype(np.uint8)
        dist = cv2.equalizeHist(dist)
        #     ker = np.array([[-1,-2,-3],[-2,0,0],[-3,0,1]], dtype=np.float32)
        #     ker -= np.mean(ker)
        ker = np.array([[-2, -1, 0], [-1, 1, 1], [0, 1, 2]])
        dist = cv2.filter2D(dist, cv2.CV_8U, ker)
        dist = cv2.blur(dist, (3, 3)).astype(np.float32)

        m = cv2.cvtColor(liquid_layer * dist, cv2.COLOR_GRAY2BGRA)
        m /= np.max(m, axis=(0, 1))
        m *= c[4]

        # water is pale turqouise
        color = np.concatenate((175 / 255. * np.ones_like(m[..., :1]),
                                238 / 255. * np.ones_like(m[..., :1]),
                                238 / 255. * np.ones_like(m[..., :1])), axis=2)

        color = cv2.cvtColor(color, cv2.COLOR_BGR2BGRA)
        x = cv2.cvtColor(x, cv2.COLOR_BGR2BGRA)

        return cv2.cvtColor(np.clip(x + m * color, 0, 1), cv2.COLOR_BGRA2BGR) * 255
    else:
        m = np.where(liquid_layer > c[3], 1, 0)
        m = gaussian(m.astype(np.float32), sigma=c[4])
        m[m < 0.8] = 0
        #         m = np.abs(m) ** (1/c[4])

        # mud brown
        color = np.concatenate((63 / 255. * np.ones_like(x[..., :1]),
                                42 / 255. * np.ones_like(x[..., :1]),
                                20 / 255. * np.ones_like(x[..., :1])), axis=2)

        color *= m[..., np.newaxis]
        x *= (1 - m[..., np.newaxis])

        return np.clip(x + color, 0, 1) * 255


def contrast(x, severity=1):
    c = [.4, .3, .2, .1, 0.05][severity - 1]

    x = np.array(x) / 255.
    means = np.mean(x, axis=(0, 1), keepdims=True)
    return np.clip((x - means) * c + means, 0, 1) * 255


def brightness(x, severity=1):
    c = [.1, .2, .3, .4, .5][severity - 1]
    
    # Ensure image is RGB (3 channels)
    if isinstance(x, np.ndarray):
        if len(x.shape) == 2 or x.shape[-1] == 1:
            x = np.repeat(np.expand_dims(x, axis=-1), 3, axis=-1)
    
    x = np.array(x) / 255.
    x = sk.color.rgb2hsv(x)
    x[:, :, 2] = np.clip(x[:, :, 2] + c, 0, 1)
    x = sk.color.hsv2rgb(x)
    
    return np.clip(x, 0, 1) * 255


def saturate(x, severity=1):
    c = [(0.3, 0), (0.1, 0), (2, 0), (5, 0.1), (30, 0.2)][severity - 1]

    x = np.array(x) / 255.
    x = sk.color.rgb2hsv(x)
    x[:, :, 1] = np.clip(x[:, :, 1] * c[0] + c[1], 0, 1)
    x = sk.color.hsv2rgb(x)

    return np.clip(x, 0, 1) * 255


def jpeg_compression(x, severity=1):
    c = [65, 58, 50, 40, 25][severity - 1]

    # Convert NumPy array to PIL Image if necessary and ensure RGB
    if isinstance(x, np.ndarray):
        x = PILImage.fromarray(np.uint8(x)).convert("RGB")

    # Perform JPEG compression in memory
    output = BytesIO()
    x.save(output, 'JPEG', quality=c)
    
    # Reload the compressed image from memory as a PIL Image and convert to RGB
    output.seek(0)
    x = PILImage.open(output).convert("RGB")
    
    # Convert back to a NumPy array for further processing
    return np.array(x)


def pixelate(x, severity=1):
    c = [0.9, 0.8, 0.7, 0.6, 0.5][severity - 1]
    
    # Ensure the input is a PIL Image and in RGB
    if isinstance(x, np.ndarray):
        x = PILImage.fromarray(np.uint8(x)).convert("RGB")
    
    # Calculate the new dimensions
    small_size = (int(x.size[0] * c), int(x.size[1] * c))
    
    # Resize to smaller, then back to original
    x = x.resize(small_size, PILImage.Resampling.BOX)
    x = x.resize(x.size, PILImage.Resampling.BOX)
    
    return np.array(x)


# mod of https://gist.github.com/erniejunior/601cdf56d2b424757de5
def elastic_transform(image, severity=1):
    IMSIZE = 224
    c = [(IMSIZE*0, IMSIZE*0, IMSIZE*0.08),
         (IMSIZE*0.05, IMSIZE*0.3, IMSIZE*0.06),
         (IMSIZE*0.1, IMSIZE*0.08, IMSIZE*0.06),
         (IMSIZE*0.1, IMSIZE*0.03, IMSIZE*0.03),
         (IMSIZE*0.16, IMSIZE*0.03, IMSIZE*0.02)][severity - 1]

    image = np.array(image, dtype=np.float32) / 255.
    shape = image.shape
    shape_size = shape[:2]

    # random affine
    center_square = np.float32(shape_size) // 2
    square_size = min(shape_size) // 3
    pts1 = np.float32([center_square + square_size,
                       [center_square[0] + square_size, center_square[1] - square_size],
                       center_square - square_size])
    pts2 = pts1 + np.random.uniform(-c[2], c[2], size=pts1.shape).astype(np.float32)
    M = cv2.getAffineTransform(pts1, pts2)
    image = cv2.warpAffine(image, M, shape_size[::-1], borderMode=cv2.BORDER_REFLECT_101)

    dx = (gaussian(np.random.uniform(-1, 1, size=shape[:2]),
                   c[1], mode='reflect', truncate=3) * c[0]).astype(np.float32)
    dy = (gaussian(np.random.uniform(-1, 1, size=shape[:2]),
                   c[1], mode='reflect', truncate=3) * c[0]).astype(np.float32)
    dx, dy = dx[..., np.newaxis], dy[..., np.newaxis]

    x, y, z = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]), np.arange(shape[2]))
    indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1)), np.reshape(z, (-1, 1))
    return np.clip(map_coordinates(image, indices, order=1, mode='reflect').reshape(shape), 0, 1) * 255


# /////////////// End Distortions ///////////////

def load_corrupt(dir_release):
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
    # Force loading in color as had greyscale issue
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    img = cv2.resize(img, (shape[0], shape[1]))
    return img

class dataset_corrupt(Dataset):
    def __init__(self, derm, shape, mode='train', chosen_distortion=None, distort_target='both', severity=1):
        """
        Initialize the dataset_corrupt class.
        
        Parameters:
        - derm: Derm7PtDatasetGroupInfrequent object
        - shape: Tuple specifying the target image shape (height, width)
        - mode: 'train', 'valid', or 'test'
        - chosen_distortion: Name of the distortion to apply during testing
        - distort_target: Specify which images to distort ('derm', 'clinic', or 'both')
        - severity: Distortion severity level (1-5, if want more severe expand sigma values)
        """
        self.shape = shape
        self.mode = mode
        self.chosen_distortion = chosen_distortion
        self.distort_target = distort_target  # Which modality to distort
        self.severity = severity  # severity of corruptions
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

        self.distortions = {
            'gaussian_noise': gaussian_noise,
            'shot_noise': shot_noise,
            'impulse_noise': impulse_noise,
            'speckle_noise': speckle_noise,
            'gaussian_blur': gaussian_blur,
            'glass_blur': glass_blur,
            'defocus_blur': defocus_blur,
            'motion_blur': motion_blur,
            'zoom_blur': zoom_blur,
            'snow': snow,
            'spatter': spatter,
            'contrast': contrast,
            'brightness': brightness,
            'saturate': saturate,
            'jpeg_compression': jpeg_compression,
            'pixelate': pixelate,
            'elastic_transform': elastic_transform
        }

        if self.chosen_distortion and self.chosen_distortion not in self.distortions:
            raise ValueError(f"Invalid distortion: {self.chosen_distortion}. Available options: {list(self.distortions.keys())}")

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

        if self.mode == 'train':
            augmented = aug(image=clinic_img, mask=dermoscopy_img)
            clinic_img = augmented['image']
            dermoscopy_img = augmented['mask']

        if self.mode == 'test' and self.chosen_distortion:
            distortion = self.distortions[self.chosen_distortion]
            # Pass the severity parameter
            if self.distort_target == 'derm':
                dermoscopy_img = distortion(dermoscopy_img, severity=self.severity)
            elif self.distort_target == 'cli':
                clinic_img = distortion(clinic_img, severity=self.severity)
            elif self.distort_target == 'both':
                dermoscopy_img = distortion(dermoscopy_img, severity=self.severity)
                clinic_img = distortion(clinic_img, severity=self.severity)

            dermoscopy_img = cv2.resize(dermoscopy_img, (self.shape[1], self.shape[0]))
            clinic_img = cv2.resize(clinic_img, (self.shape[1], self.shape[0]))

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
    
def save_image(img_tensor, filename):
    """Convert a tensor to an image and save it."""
    img = img_tensor / 2 + 0.5  
    npimg = img_tensor.numpy()
    img = np.transpose(npimg, (1, 2, 0))
    pil_img = Image.fromarray((img * 255).astype(np.uint8))  # Convert to 8-bit pixel values
    pil_img.save(filename)

def plot_corruption_types_single_image(dataset, distortions, image_index=0, severity=1, save_path=None):
    fig, axes = plt.subplots(4, 5, figsize=(27, 18))  
    axes = axes.flatten()
    
    # Get the image from the dataset 
    dermoscopy_imgs, clinic_imgs, metadata, meta_con, labels = dataset[image_index]
    
    print(f"image shape: {dermoscopy_imgs.shape}")  # [3,224,224]
    image = dermoscopy_imgs.numpy()
    print(f"Original image shape: {image.shape}")  # [3,224,224]
    
    # Convert from [C, H, W] to [H, W, C]
    if image.shape[0] == 3:
        image = np.transpose(image, (1, 2, 0))
    
    # Convert from [0, 1] to [0, 255] 
    if image.max() <= 1:
        image = (image * 255).astype(np.uint8)
    
    # Convert BGR to RGB (if necessary)
    if image.shape[-1] == 3:
        image = image[..., ::-1]
    
    print("Final image shape (should be [H, W, 3]):", image.shape)
    
    # Plot the original image
    axes[0].imshow(image)
    axes[0].axis('off')
    axes[0].set_title("Original Image")
    
    # Loop through the distortions and apply each one with the specified severity
    for i, distortion_name in enumerate(distortions):
        distortion = dataset.distortions.get(distortion_name, None)
        if distortion is None:
            print(f"Distortion '{distortion_name}' not found!")
            continue

        corrupted_img = distortion(image, severity=severity)
        if corrupted_img.dtype != np.uint8:
            corrupted_img = np.clip(corrupted_img, 0, 255).astype(np.uint8)
        
        # Convert BGR to RGB if distortion reverses dimensions
        if corrupted_img.shape[-1] == 3:
            corrupted_img = corrupted_img[..., ::-1]

        axes[i + 1].imshow(corrupted_img)
        axes[i + 1].axis('off')
        axes[i + 1].set_title(distortion_name)
    
    # Hide last two unused subplots
    for j in range(len(distortions) + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, format='png')
        print(f"Figure saved at {save_path}")
    plt.show()

if __name__ == '__main__':
    dir_release = "/home/matthewcockayne/Documents/PhD/data/Derm7pt/release_v0/release_v0"
    derm_data_group, columns_of_interest = load_corrupt(dir_release)
    
    # Load datasets
    dataset_train = dataset_corrupt(derm_data_group, (224, 224), mode='train')
    dataset_valid = dataset_corrupt(derm_data_group, (224, 224), mode='valid')
    dataset_test = dataset_corrupt(derm_data_group, (224, 224), mode='test', severity=5)

    # Creating data loaders
    train_loader = DataLoader(dataset_train, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)
    valid_loader = DataLoader(dataset_valid, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(dataset_test, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)

    print(f"Number of samples in training dataset: {len(dataset_train)}")
    print(f"Number of samples in validation dataset: {len(dataset_valid)}")
    print(f"Number of samples in test dataset: {len(dataset_test)}")

    dataiter = iter(test_loader) 
    dermoscopy_imgs, clinic_imgs, metadata, meta_con, labels = next(dataiter)
    image_index = 0
    image = dermoscopy_imgs[image_index].numpy()
    print(f"Image shape after loading: {image.shape}")

    distortions_to_plot = [
            'gaussian_noise', 'shot_noise', 'impulse_noise', 'speckle_noise',
            'gaussian_blur', 'glass_blur', 'defocus_blur', 'motion_blur',
            'zoom_blur', 'snow', 'spatter', 'contrast', 'brightness',
            'saturate', 'jpeg_compression', 'pixelate', 'elastic_transform'
    ]
    plot_corruption_types_single_image(dataset_test, distortions_to_plot, severity=5, save_path="corruptions_image_exampletest5.png")

