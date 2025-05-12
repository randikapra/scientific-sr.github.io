import os
import csv
import tensorflow as tf
import numpy as np
import torch
import lpips

# Initialize LPIPS model
lpips_model = lpips.LPIPS(net='vgg')
# Function to load image
def load_image(image_path):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.convert_image_dtype(image, tf.float32)  # Convert to float32 in [0, 1]
    return image

# Function to crop a 64x64 region from the center
def crop_center(image, crop_height=384, crop_width=384):
    shape = tf.shape(image)
    height, width = shape[0], shape[1]
    start_height = (height - crop_height) // 2
    start_width = (width - crop_width) // 2
    return tf.image.crop_to_bounding_box(image, start_height, start_width, crop_height, crop_width)

# Compute PSNR
def compute_psnr(y_true, y_pred):
    psnr_value = tf.image.psnr(y_true, y_pred, max_val=1.0)
    return psnr_value.numpy()

# Compute SSIM
def compute_ssim(y_true, y_pred):
    ssim_value = tf.image.ssim(y_true, y_pred, max_val=1.0)
    return ssim_value.numpy()

# Compute LPIPS
def compute_lpips(y_true, y_pred):
    y_true = y_true.numpy()
    y_pred = y_pred.numpy()
    
    # Ensure input shapes are [batch_size, height, width, channels]
    if len(y_true.shape) == 3:
        y_true = np.expand_dims(y_true, axis=0)
    if len(y_pred.shape) == 3:
        y_pred = np.expand_dims(y_pred, axis=0)
    
    # Convert images to tensors and normalize to [-1, 1]
    y_true_torch = torch.tensor(y_true * 2 - 1).permute(0, 3, 1, 2).float()
    y_pred_torch = torch.tensor(y_pred * 2 - 1).permute(0, 3, 1, 2).float()
    
    # Compute LPIPS score
    lpips_value = lpips_model(y_true_torch, y_pred_torch)
    return lpips_value.mean().item()

# Paths to directories
sr_image_dir = "../experiment/sr
true_image_path = "../data/train_HR"

# Save metrics to CSV
output_csv = "../experiment/sr/eval_metrics.csv"

with open(output_csv, mode="w", newline="") as file:
    writer = csv.writer(file)
    # Write the header
    writer.writerow(["Image Name", "PSNR", "SSIM", "LPIPS"])
    
    # Loop through each SR image in the folder
    for sr_image_name in os.listdir(sr_image_dir):
        sr_image_path = os.path.join(sr_image_dir, sr_image_name)
        
        # Construct the corresponding true image path
        # Extract the base name (remove prefix like "1_") from the SR image
        true_image_name = "_".join(sr_image_name.split("_")[1:])  # Adjust based on naming convention
        true_image_full_path = os.path.join(true_image_path, true_image_name)
        
        # Skip if the true image does not exist
        if not os.path.exists(true_image_full_path):
            print(f"True image not found for {sr_image_name}. Skipping.")
            continue
        
        # Load the images
        sr_image = load_image(sr_image_path)
        true_image = load_image(true_image_full_path)
        
        # # Crop 64x64 regions from the center
        # sr_image_cropped = crop_center(sr_image)
        # true_image_cropped = crop_center(true_image)
        
        # Compute evaluation metrics on cropped regions
        psnr_value = compute_psnr(true_image, sr_image)
        ssim_value = compute_ssim(true_image, sr_image)
        lpips_value = compute_lpips(true_image, sr_image)
        
        # Write metrics to the CSV file
        writer.writerow([sr_image_name, psnr_value, ssim_value, lpips_value])
        print(f"Metrics for {sr_image_name} - PSNR: {psnr_value}, SSIM: {ssim_value}, LPIPS: {lpips_value}")

print(f"Metrics saved to {output_csv}")
