import tensorflow as tf
from tensorflow.keras.models import load_model
import cv2
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.models import load_model
import cv2
import numpy as np
from PIL import Image
import os
import tensorflow as tf
from tensorflow.keras.models import load_model
import cv2
import numpy as np
# Preprocess the LR image
def preprocess_lr_image(image_path):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"The file {image_path} does not exist.")
    lr_image = cv2.imread(image_path)
    lr_image = cv2.cvtColor(lr_image, cv2.COLOR_BGR2RGB)
    lr_image = lr_image / 127.5 - 1  # Normalize LR image to [-1, 1]
    lr_image = np.expand_dims(lr_image, axis=0)
    return lr_image

# Preprocess the HR image
def preprocess_hr_image(image_path):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"The file {image_path} does not exist.")
    hr_image = cv2.imread(image_path)
    hr_image = cv2.cvtColor(hr_image, cv2.COLOR_BGR2RGB)
    hr_image = hr_image / 255  # Normalize HR image to [0, 1]
    hr_image = np.expand_dims(hr_image, axis=0)
    return hr_image

# Extract HR features using EfficientNet
def extract_hr_features(hr_image):
    hr_features = efficientnet(hr_image, training=False)
    return hr_features

# # Generate SR image
# def generate_sr_image(lr_image_path, hr_image_path, output_path):
#     lr_image = preprocess_lr_image(lr_image_path)
#     hr_image = preprocess_hr_image(hr_image_path)
#     hr_features = extract_hr_features(hr_image)
    
#     # Ensure both inputs are tensors
#     lr_image = tf.convert_to_tensor(lr_image, dtype=tf.float32)
#     hr_features = tf.convert_to_tensor(hr_features, dtype=tf.float32)
    
#     sr_image = generator([lr_image, hr_features], training=False)
#     sr_image = (sr_image + 1) / 2.0  # Convert to [0, 1]
#     sr_image = np.squeeze(sr_image, axis=0)  # Remove batch dimension
#     sr_image = (sr_image * 255).astype(np.uint8)
#     sr_image = cv2.cvtColor(sr_image, cv2.COLOR_RGB2BGR)
    
#     # Save as JPG using OpenCV
#     cv2.imwrite(output_path, sr_image, [int(cv2.IMWRITE_JPEG_QUALITY), 95])  # Save with quality 95
#############

# # Generate SR image
def generate_sr_image(lr_image_path, hr_image_path, output_path):
    lr_image = preprocess_lr_image(lr_image_path)
    hr_image = preprocess_hr_image(hr_image_path)
    hr_features = extract_hr_features(hr_image)
    
    # Ensure both inputs are tensors
    lr_image = tf.convert_to_tensor(lr_image, dtype=tf.float32)
    hr_features = tf.convert_to_tensor(hr_features, dtype=tf.float32)
    
    sr_image = generator([lr_image, hr_features], training=False)
    sr_image = (sr_image + 1) / 2.0  # Convert to [0, 1]
    sr_image = np.squeeze(sr_image, axis=0)  # Remove batch dimension
    sr_image = (sr_image * 255).astype(np.uint8)
    sr_image = cv2.cvtColor(sr_image, cv2.COLOR_RGB2BGR)
    # cv2.imwrite(output_path, sr_image)
    # Save as JPG
    sr_image_pil = Image.fromarray(sr_image)
    sr_image_pil.save(output_path, format='JPEG')

    print('Completed...')

# #################
# Load the pre-trained generator model
generator = load_model("../models/generator/best_model.keras", safe_mode=False)

# Load the EfficientNet model
efficientnet = tf.keras.applications.EfficientNetB7(include_top=False, weights='imagenet')
efficientnet.trainable = False

# Directories
lr_dir = "../data/train_LR"
hr_dir = "../data/train_HR"
output_dir = "/home/oshadi/SISR-Final_Year_Project/envs/SISR-Project/Final-Model/SINSR/experiment/conference/gen7_t1_14d"

# Ensure the output directory exists
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Loop through each file in the LR directory
for lr_file in os.listdir(lr_dir):
    if lr_file.endswith(".jpg"):  # Process only JPG files
        lr_image_path = os.path.join(lr_dir, lr_file)
        hr_image_path = os.path.join(hr_dir, lr_file)  # Corresponding HR image
        output_path = os.path.join(output_dir, f"1_{lr_file}")  # Save SR image with modified name
        
        try:
            generate_sr_image(lr_image_path, hr_image_path, output_path)
            print(f"Generated SR image for {lr_file} and saved to {output_path}")
        except Exception as e:
            print(f"Error processing {lr_file}: {e}")
