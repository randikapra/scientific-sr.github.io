import os
import json
import numpy as np
import tensorflow as tf
from torch.utils.data import Dataset, DataLoader
import cv2
import lmdb

# Load configuration
config_path = '../config/train.json'  # Update this path accordingly
with open(config_path, 'r') as f:
    config = json.load(f)

# Define the LMDBDataset class for loading data from LMDB files
class LMDBDataset(Dataset):
    def __init__(self, lmdb_path, target_shape):
        """
        Initializes the LMDBDataset instance.

        Args:
        - lmdb_path (str): Path to the LMDB file.
        - target_shape (tuple): Shape of the target images.
        """
        self.env = lmdb.open(lmdb_path, readonly=True, lock=False)
        with self.env.begin() as txn:
            self.length = txn.stat()['entries']
        self.target_shape = target_shape

    def __len__(self):
        """
        Returns the number of entries in the LMDB file.

        Returns:
        - int: Number of entries.
        """
        return self.length

    def __getitem__(self, idx):
        """
        Retrieves an image from the LMDB file.

        Args:
        - idx (int): Index of the image to retrieve.

        Returns:
        - np.ndarray: Retrieved and preprocessed image.
        """
        with self.env.begin() as txn:
            key = f'{idx:08}'.encode()
            image_bytes = txn.get(key)
            image = np.frombuffer(image_bytes, dtype=np.float32)  # Use float32 to match encoding
            image = image.reshape(self.target_shape)
            image = np.transpose(image, (1, 2, 0))  # Channels last

            # Normalize image values to the range [0, 1]
            image = np.clip(image, 0.0, 1.0)
            
            # Convert RGB to BGR (Uncomment the following line if needed)
            # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            return image

# Update the paths and target shapes
train_hr_path = config['train_hr_path']
train_lr_path = config['train_lr_path']
valid_hr_path = config['valid_hr_path']
valid_lr_path = config['valid_lr_path']
target_shape_hr = tuple(config['target_shape_hr'])
target_shape_lr = tuple(config['target_shape_lr'])
batch_size = config['batch_size']

# Initialize the datasets
train_hr_dataset = LMDBDataset(train_hr_path, target_shape_hr)  # Ensure target_shape matches LMDB preprocessing
train_lr_dataset = LMDBDataset(train_lr_path, target_shape_lr)  # Ensure target_shape matches LMDB preprocessing
valid_hr_dataset = LMDBDataset(valid_hr_path, target_shape_hr)  # Ensure target_shape matches LMDB preprocessing
valid_lr_dataset = LMDBDataset(valid_lr_path, target_shape_lr)  # Ensure target_shape matches LMDB preprocessing

# Create DataLoaders
train_hr_dataloader = DataLoader(train_hr_dataset, batch_size=batch_size, shuffle=True)
train_lr_dataloader = DataLoader(train_lr_dataset, batch_size=batch_size, shuffle=True)
valid_hr_dataloader = DataLoader(valid_hr_dataset, batch_size=batch_size, shuffle=False)
valid_lr_dataloader = DataLoader(valid_lr_dataset, batch_size=batch_size, shuffle=False)
