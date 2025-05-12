from tensorflow.keras.callbacks import ModelCheckpoint  # type: ignore
import tensorflow as tf
import os
import numpy as np
import cv2
from PIL import Image
from matplotlib import pyplot as plt
from utils.utilities import interpolate_weights, update_loss_weights, get_lr, clear_gpu_memory, get_dynamic_weights
from utils.train_util import distributed_train_step, distributed_validation_step, distributed_pretrain_step, distributed_pretrain_encoder_step
from utils.callbacks import get_callbacks
from torch.utils.data import DataLoader
from utils.model_setup import setup_model_and_optimizers#, update_vgg_input_shape
import tensorflow as tf
from tensorflow.keras import layers, Model # type: ignore
import pandas as pd # type: ignore
# Initialize lists to store loss values
train_total_loss = []
train_adv_loss = []
train_perc_loss = []
train_grad_loss = []
train_second_grad_loss = []
train_struct_loss = []
train_aux_loss = []

val_total_loss = []
val_adv_loss = []
val_perc_loss = []
val_grad_loss = []
val_second_grad_loss = []
val_struct_loss = []
val_aux_loss = []


def pretrain_generator(train_dataloader, generator, generator_optimizer, strategy, epochs=10):
    for epoch in range(epochs):
        for LR_batch, HR_batch in train_dataloader:
            HR_batch = tf.convert_to_tensor(HR_batch.numpy().astype('float32'))
            LR_batch = tf.convert_to_tensor(LR_batch.numpy().astype('float32'))
            
            pretrain_loss = distributed_pretrain_step(LR_batch, HR_batch, generator, generator_optimizer, strategy)
            print(f"Pre-Training Epoch {epoch + 1}, Loss: {pretrain_loss.numpy()}")
        
    generator.save_weights('/path/to/pretrained_generator.weights.h5')

def pretrain_encoder(train_dataloader, encoder, encoder_optimizer, strategy, epochs=10):
    """Pre-trains the encoder to mimic HR feature maps."""
    for epoch in range(epochs):
        for LR_batch, HR_features_batch in train_dataloader:
            HR_features_batch = tf.convert_to_tensor(HR_features_batch.numpy().astype('float32'))
            LR_batch = tf.convert_to_tensor(LR_batch.numpy().astype('float32'))

            pretrain_loss = distributed_pretrain_encoder_step(LR_batch, HR_features_batch, encoder, encoder_optimizer, strategy)
            print(f"Encoder Pre-Training Epoch {epoch + 1}, Loss: {pretrain_loss.numpy()}")

    encoder.save_weights('/home/oshadi/SISR-Final_Year_Project/envs/SISR-Project/Final-Model/SINSR/models/pretrained_encoder.weights.h5')

def save_individual_losses(train_losses, val_losses, loss_name, log_path):
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label=f'Train {loss_name.capitalize()} Loss')
    plt.plot(val_losses, label=f'Validation {loss_name.capitalize()} Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title(f'{loss_name.capitalize()} Loss Curves')
    plt.legend()
    plt.grid(True)
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    plt.savefig(os.path.join(log_path, f'{loss_name}_loss_curves.png'))
    plt.close()

def save_losses_to_csv(epoch, train_total_loss, val_total_loss, train_adv_loss, val_adv_loss, train_perc_loss, val_perc_loss, train_grad_loss, val_grad_loss, train_second_grad_loss, val_second_grad_loss, train_struct_loss, val_struct_loss, train_aux_loss, val_aux_loss, log_path):
    losses_dict = {
        'epoch': list(range(1, epoch + 2)),
        'train_total_loss': train_total_loss,
        'val_total_loss': val_total_loss,
        'train_adv_loss': train_adv_loss,
        'val_adv_loss': val_adv_loss,
        'train_perc_loss': train_perc_loss,
        'val_perc_loss': val_perc_loss,
        'train_grad_loss': train_grad_loss,
        'val_grad_loss': val_grad_loss,
        'train_second_grad_loss': train_second_grad_loss,
        'val_second_grad_loss': val_second_grad_loss,
        'train_struct_loss': train_struct_loss,
        'val_struct_loss': val_struct_loss,
        'train_aux_loss': train_aux_loss,
        'val_aux_loss': val_aux_loss
    }

    losses_df = pd.DataFrame(losses_dict)
    losses_df.to_csv(f'{log_path}/losses.csv', index=False)



def train(train_config, model_config, loss_weights_config, architecture_module, strategy, train_dataloader, valid_dataloader):
    generator, discriminator, efficientnet, vgg, generator_optimizer, discriminator_optimizer = setup_model_and_optimizers(strategy, architecture_module, model_config, train_config)
    
    initial_weights = loss_weights_config['initial_weights']
    final_weights = loss_weights_config['final_weights']
    current_weights = initial_weights.copy()
    # Create the callback for saving the best model
    checkpoint_path = os.path.join(train_config['model_save_path'], 'best_model.keras')
    checkpoint = ModelCheckpoint(filepath=checkpoint_path, monitor='val_total_loss', save_best_only=True, mode='min', verbose=5)

    callbacks = get_callbacks(train_config)
    callbacks.append(checkpoint)

    for callback in callbacks:
        callback.set_model(generator)
        callback.model.optimizer = generator_optimizer

    for epoch in range(train_config['epochs']):
        train_loss = {'total': 0, 'adv': 0, 'perc': 0, 'grad': 0, 'second_grad': 0, 'struct': 0, 'aux': 0}
        step = 0
        
        for lr_images, hr_images in train_dataloader:
            hr_images = tf.convert_to_tensor(hr_images.numpy().astype('float32'))
            lr_images = tf.convert_to_tensor(lr_images.numpy().astype('float32'))

            gen_loss, disc_loss, generated_images, individual_losses = distributed_train_step(
                lr_images, hr_images, generator, discriminator, efficientnet, vgg,
                generator_optimizer, discriminator_optimizer, strategy,
                lambda_adv=current_weights['adv'],
                lambda_perceptual=current_weights['perc'],
                lambda_gradient=current_weights['grad'],
                lambda_second=current_weights['second_grad'],
                lambda_struct=current_weights['struct'],
                lambda_aux=current_weights['aux']
            )

            gathered_generated_images = strategy.gather(generated_images, axis=0)
            print(f"Epoch {epoch + 1}, Step {step}, Generator Loss: {gen_loss.numpy()}, Discriminator Loss: {disc_loss.numpy()}")

            train_loss['total'] += gen_loss.numpy()
            for key in individual_losses:
                train_loss[key] += individual_losses[key].numpy()
            step += 1

        num_train_batches = len(train_dataloader)
        train_total_loss.append(float(train_loss['total'] / num_train_batches))
        train_adv_loss.append(float(train_loss['adv'] / num_train_batches))
        train_perc_loss.append(float(train_loss['perc'] / num_train_batches))
        train_grad_loss.append(float(train_loss['grad'] / num_train_batches))
        train_second_grad_loss.append(float(train_loss['second_grad'] / num_train_batches))
        train_struct_loss.append(float(train_loss['struct'] / num_train_batches))
        train_aux_loss.append(float(train_loss['aux'] / num_train_batches))

        val_loss = {'total': 0, 'adv': 0, 'perc': 0, 'grad': 0, 'second_grad': 0, 'struct': 0, 'aux': 0}

        for lr_val_images, hr_val_images in valid_dataloader:
            hr_val_images = tf.convert_to_tensor(hr_val_images.numpy().astype('float32'))
            lr_val_images = tf.convert_to_tensor(lr_val_images.numpy().astype('float32'))

            val_t_loss, val_individual_losses = distributed_validation_step(
                lr_val_images, hr_val_images, generator, discriminator, efficientnet, vgg, strategy,
                lambda_adv=current_weights['adv'],
                lambda_perceptual=current_weights['perc'],
                lambda_gradient=current_weights['grad'],
                lambda_second=current_weights['second_grad'],
                lambda_struct=current_weights['struct'],
                lambda_aux=current_weights['aux']
            )

            val_loss['total'] += val_t_loss.numpy()
            for key in val_individual_losses:
                val_loss[key] += val_individual_losses[key].numpy()

        num_val_batches = len(valid_dataloader)
        val_avg_loss = {key: val_loss[key] / num_val_batches for key in val_loss}
        val_total_loss.append(float(val_avg_loss['total']))
        val_adv_loss.append(float(val_avg_loss['adv']))
        val_perc_loss.append(float(val_avg_loss['perc']))
        val_grad_loss.append(float(val_avg_loss['grad']))
        val_second_grad_loss.append(float(val_avg_loss['second_grad']))
        val_struct_loss.append(float(val_avg_loss['struct']))
        val_aux_loss.append(float(val_avg_loss['aux']))

        # Update the ModelCheckpoint at the end of each epoch
        checkpoint.on_epoch_end(epoch, logs={'val_total_loss': val_avg_loss['total']})

        if (epoch + 1) % 5 == 0:
            current_weights = get_dynamic_weights(epoch, train_config['epochs'], initial_weights, final_weights)

        print(f"Epoch {epoch + 1}/{train_config['epochs']}, Train Total Loss: {train_total_loss[-1]}")
        print(f" Adv Loss: {train_adv_loss[-1]}, Perc Loss: {train_perc_loss[-1]}, Grad Loss: {train_grad_loss[-1]}, Second Grad Loss: {train_second_grad_loss[-1]}, Struct Loss: {train_struct_loss[-1]}, Aux Loss: {train_aux_loss[-1]}")
        print(f" Validation Total Loss: {val_total_loss[-1]}")
        print(f" Adv Loss: {val_adv_loss[-1]}, Perc Loss: {val_perc_loss[-1]}, Grad Loss: {val_grad_loss[-1]}, Second Grad Loss: {val_second_grad_loss[-1]}, Struct Loss: {val_struct_loss[-1]}, Aux Loss: {val_aux_loss[-1]}")
        
        # for i in range(gathered_generated_images.shape[0]):
        #     generated_image_np = (gathered_generated_images[i].numpy() * 255).astype(np.uint8)
        #     generated_image_bgr = cv2.cvtColor(generated_image_np, cv2.COLOR_RGB2BGR)
        #     if not os.path.exists(train_config['save_path']): os.makedirs(train_config['save_path'])
        #     cv2.imwrite(os.path.join(train_config['save_path'], f'epoch_{epoch}_step_{step}_replica_{i}.jpg'), generated_image_bgr)

        for i in range(gathered_generated_images.shape[0]):
            # Convert Tensor to numpy array and scale pixel values
            generated_image_np = (gathered_generated_images[i].numpy() * 255).astype(np.uint8)
            # Convert to PIL Image
            generated_image_pil = Image.fromarray(generated_image_np)
            # Ensure directory exists
            if not os.path.exists(train_config['save_path']):
                os.makedirs(train_config['save_path'])
            # Set the save path for the image
            save_path = os.path.join(train_config['save_path'], f'epoch_{epoch}_step_{step}_replica_{i}.png')
            # Save the image in PNG format (lossless compression)
            generated_image_pil.save(save_path, format='PNG')

        save_losses_to_csv(epoch, train_total_loss, val_total_loss, train_adv_loss, val_adv_loss, train_perc_loss, val_perc_loss, train_grad_loss, val_grad_loss, train_second_grad_loss, val_second_grad_loss, train_struct_loss, val_struct_loss, train_aux_loss, val_aux_loss, train_config["log_path"])
    
    print("Training completed successfully!")
    for loss_name in ['total', 'adv', 'perc', 'grad', 'second_grad', 'struct', 'aux']:
        save_individual_losses(globals()[f"train_{loss_name}_loss"], globals()[f"val_{loss_name}_loss"], loss_name, train_config['log_path'])


def joint_train(train_config, model_config, loss_weights_config, architecture_module, strategy, train_dataloader, valid_dataloader):
    """Fine-tunes the encoder and GAN together."""
    # Setup models and optimizers
    generator, discriminator, encoder, efficientnet, vgg, generator_optimizer, discriminator_optimizer, encoder_optimizer = setup_model_and_optimizers(
        strategy, architecture_module, model_config, train_config
    )

    initial_weights = loss_weights_config['initial_weights']
    final_weights = loss_weights_config['final_weights']
    current_weights = initial_weights.copy()

    # Define the checkpoint for saving the best model
    checkpoint_path = os.path.join(train_config['model_save_path'], 'best_joint_model.keras')
    checkpoint = ModelCheckpoint(filepath=checkpoint_path, monitor='val_total_loss', save_best_only=True, mode='min', verbose=5)

    callbacks = get_callbacks(train_config)
    callbacks.append(checkpoint)

    for callback in callbacks:
        callback.set_model(generator)
        callback.model.optimizer = generator_optimizer

    for epoch in range(train_config['epochs']):
        train_loss = {'total': 0, 'adv': 0, 'perc': 0, 'grad': 0, 'second_grad': 0, 'struct': 0, 'aux': 0}
        step = 0

        for lr_images, hr_images in train_dataloader:
            hr_images = tf.convert_to_tensor(hr_images.numpy().astype('float32'))
            lr_images = tf.convert_to_tensor(lr_images.numpy().astype('float32'))

            hr_features = encoder(lr_images, training=True)  # Encoder generates features

            # Train the GAN and encoder jointly
            gen_loss, disc_loss, generated_images, individual_losses = distributed_train_step(
                lr_images, hr_images, generator, discriminator, encoder, efficientnet,
                generator_optimizer, discriminator_optimizer, strategy,
                lambda_adv=current_weights['adv'],
                lambda_perceptual=current_weights['perc'],
                lambda_gradient=current_weights['grad'],
                lambda_second=current_weights['second_grad'],
                lambda_struct=current_weights['struct'],
                lambda_aux=current_weights['aux']
            )

            print(f"Epoch {epoch + 1}, Step {step}, Generator Loss: {gen_loss.numpy()}, Discriminator Loss: {disc_loss.numpy()}")

            train_loss['total'] += gen_loss.numpy()
            for key in individual_losses:
                train_loss[key] += individual_losses[key].numpy()
            step += 1

        num_train_batches = len(train_dataloader)
        train_total_loss.append(float(train_loss['total'] / num_train_batches))
        train_adv_loss.append(float(train_loss['adv'] / num_train_batches))
        train_perc_loss.append(float(train_loss['perc'] / num_train_batches))
        train_grad_loss.append(float(train_loss['grad'] / num_train_batches))
        train_second_grad_loss.append(float(train_loss['second_grad'] / num_train_batches))
        train_struct_loss.append(float(train_loss['struct'] / num_train_batches))
        train_aux_loss.append(float(train_loss['aux'] / num_train_batches))

        # Validation
        val_loss = {'total': 0, 'adv': 0, 'perc': 0, 'grad': 0, 'second_grad': 0, 'struct': 0, 'aux': 0}

        for lr_val_images, hr_val_images in valid_dataloader:
            hr_val_images = tf.convert_to_tensor(hr_val_images.numpy().astype('float32'))
            lr_val_images = tf.convert_to_tensor(lr_val_images.numpy().astype('float32'))

            val_t_loss, val_individual_losses = distributed_validation_step(
                lr_val_images, hr_val_images, generator, discriminator, encoder, efficientnet, vgg, strategy,
                lambda_adv=current_weights['adv'],
                lambda_perceptual=current_weights['perc'],
                lambda_gradient=current_weights['grad'],
                lambda_second=current_weights['second_grad'],
                lambda_struct=current_weights['struct'],
                lambda_aux=current_weights['aux']
            )

            val_loss['total'] += val_t_loss.numpy()
            for key in val_individual_losses:
                val_loss[key] += val_individual_losses[key].numpy()

        num_val_batches = len(valid_dataloader)
        val_avg_loss = {key: val_loss[key] / num_val_batches for key in val_loss}
        val_total_loss.append(float(val_avg_loss['total']))
        val_adv_loss.append(float(val_avg_loss['adv']))
        val_perc_loss.append(float(val_avg_loss['perc']))
        val_grad_loss.append(float(val_avg_loss['grad']))
        val_second_grad_loss.append(float(val_avg_loss['second_grad']))
        val_struct_loss.append(float(val_avg_loss['struct']))
        val_aux_loss.append(float(val_avg_loss['aux']))

        # Save the best joint model
        checkpoint.on_epoch_end(epoch, logs={'val_total_loss': val_avg_loss['total']})

        if (epoch + 1) % 5 == 0:
            current_weights = get_dynamic_weights(epoch, train_config['epochs'], initial_weights, final_weights)

        print(f"Epoch {epoch + 1}/{train_config['epochs']}, Train Total Loss: {train_total_loss[-1]}")
        print(f" Adv Loss: {train_adv_loss[-1]}, Perc Loss: {train_perc_loss[-1]}, Grad Loss: {train_grad_loss[-1]}, Second Grad Loss: {train_second_grad_loss[-1]}, Struct Loss: {train_struct_loss[-1]}, Aux Loss: {train_aux_loss[-1]}")
        print(f" Validation Total Loss: {val_total_loss[-1]}")
        print(f" Adv Loss: {val_adv_loss[-1]}, Perc Loss: {val_perc_loss[-1]}, Grad Loss: {val_grad_loss[-1]}, Second Grad Loss: {val_second_grad_loss[-1]}, Struct Loss: {val_struct_loss[-1]}, Aux Loss: {val_aux_loss[-1]}")

        save_losses_to_csv(epoch, train_total_loss, val_total_loss, train_adv_loss, val_adv_loss, train_perc_loss, val_perc_loss, train_grad_loss, val_grad_loss, train_second_grad_loss, val_second_grad_loss, train_struct_loss, val_struct_loss, train_aux_loss, val_aux_loss, train_config["log_path"])

    print("Joint training completed successfully!")
    for loss_name in ['total', 'adv', 'perc', 'grad', 'second_grad', 'struct', 'aux']:
        save_individual_losses(globals()[f"train_{loss_name}_loss"], globals()[f"val_{loss_name}_loss"], loss_name, train_config['log_path'])

    # Save final model weights
    generator.save_weights(os.path.join(train_config['model_save_path'], 'final_generator.weights.h5'))
    discriminator.save_weights(os.path.join(train_config['model_save_path'], 'final_discriminator.weights.h5'))
    encoder.save_weights(os.path.join(train_config['encoder_save_path'], 'final_encoder.weights.h5'))
















