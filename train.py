import os
import tensorflow as tf
from utils.config_loader import load_config, load_loss_weights
from utils.utilities import interpolate_weights, update_loss_weights, get_lr, clear_gpu_memory
from utils.model_setup import setup_model_and_optimizers
from utils.training_functions import pretrain_encoder, train, joint_train
from torch.utils.data import DataLoader
import importlib
from utils.data_loader import PairedImageDataset

policy = tf.keras.mixed_precision.Policy('mixed_float16')
tf.keras.mixed_precision.set_global_policy(policy)

model_config = load_config('../config/model_config.json')
train_config = load_config('../config/train.json')
loss_weights_config = load_loss_weights('../config/loss_config.json')

architecture_module = importlib.import_module(f'architectures.{model_config["model_architecture"]}')
strategy = tf.distribute.MirroredStrategy()

train_hr_path = train_config['train_hr_path']
train_lr_path = train_config['train_lr_path']
valid_hr_path = train_config['valid_hr_path']
valid_lr_path = train_config['valid_lr_path']
target_size_hr = tuple(train_config['target_size_hr'])
target_size_lr = tuple(train_config['target_size_lr'])
batch_size = train_config['batch_size']

train_dataset = PairedImageDataset(train_lr_path, train_hr_path, target_size_lr, target_size_hr, train_config)
valid_dataset = PairedImageDataset(valid_lr_path, valid_hr_path, target_size_lr, target_size_hr, train_config)

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=0)


'''
encoder_save_path = train_config["encoder_save_path"]
encoder_optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)

# Pretrain Encoder
print("Starting pre-training for the encoder...")
encoder = architecture_module.encoder()
pretrain_encoder(train_dataloader, encoder, encoder_optimizer, strategy, epochs=10)
encoder.save_weights(os.path.join(encoder_save_path, 'pretrained_encoder.weights.h5'))
print("Encoder pre-training completed and weights saved!")
'''

clear_gpu_memory()
train(train_config, model_config, loss_weights_config, architecture_module, strategy, train_dataloader, valid_dataloader)

'''
print("Starting joint training for GAN and encoder...")
joint_train(train_config, model_config, loss_weights_config, architecture_module, strategy, train_dataloader, valid_dataloader)
print("Joint training completed successfully!")
'''
