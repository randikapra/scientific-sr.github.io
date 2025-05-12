import os
import cv2
from torch.utils.data import Dataset
from PIL import Image  # Import the Image module from Pillow
import json

def get_image_dimensions(image_path):
    with Image.open(image_path) as img:
        width, height = img.size
    return width, height

def update_json_file(config_path, config):
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)

class PairedImageDataset(Dataset):
    def __init__(self, lr_image_dir, hr_image_dir, lr_target_size, hr_target_size, train_config):
        self.lr_image_paths = sorted([os.path.join(lr_image_dir, fname) for fname in os.listdir(lr_image_dir)])
        self.hr_image_paths = sorted([os.path.join(hr_image_dir, fname) for fname in os.listdir(hr_image_dir)])
        self.train_config = train_config
        self.config_path = "/home/oshadi/SISR-Final_Year_Project/envs/SISR-Project/Final-Model/SINSR/code/config/train.json"

    def __len__(self):
        return len(self.lr_image_paths)

    def __getitem__(self, idx):
        lr_image_path = self.lr_image_paths[idx]
        hr_image_path = self.hr_image_paths[idx]
        
        lr_image = cv2.imread(lr_image_path)
        hr_image = cv2.imread(hr_image_path)
        
        if lr_image is None or hr_image is None:
            raise ValueError(f"Error reading image {lr_image_path} or {hr_image_path}")

        lr_image = cv2.cvtColor(lr_image, cv2.COLOR_BGR2RGB)
        hr_image = cv2.cvtColor(hr_image, cv2.COLOR_BGR2RGB)
        
        # Get the dimensions of the current HR image
        hr_width, hr_height = get_image_dimensions(hr_image_path)
        
        # Update the configuration dynamically
        self.train_config['vgg_input_shape'] = [hr_height, hr_width, 3]
        
        # Write the updated configuration back to the JSON file
        update_json_file(self.config_path, self.train_config)
        
        lr_image = lr_image / 127.5 - 1
        hr_image = hr_image / 255
        
        return lr_image, hr_image
