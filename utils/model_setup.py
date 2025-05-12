from tensorflow.keras.applications import VGG19, EfficientNetB7 # type: ignore
from tensorflow.keras.models import Model, load_model   # type: ignore
import tensorflow as tf
from tensorflow.keras import layers # type: ignore
import os

def create_flexible_vgg():
    vgg_base = VGG19(include_top=False, weights='imagenet')
    vgg_base.trainable = False
    output_layers = [vgg_base.get_layer(name).output for name in ['block1_conv2', 'block2_conv2', 'block3_conv3', 'block4_conv3', 'block5_conv3']]
    model = Model(inputs=vgg_base.input, outputs=output_layers)
    return model

def setup_model_and_optimizers(strategy, architecture_module, model_config, train_config):
    initial_lr_gen = train_config['learning_rates']['generator']
    initial_lr_disc = train_config['learning_rates']['discriminator']
    initial_lr_enc = 0.0001
  
    lr_schedule_gen = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=initial_lr_gen,
        decay_steps=100000,
        decay_rate=0.96,
        staircase=True)

    lr_schedule_disc = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=initial_lr_disc,
        decay_steps=100000,
        decay_rate=0.96,
        staircase=True)

    lr_schedule_enc = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=initial_lr_enc,
        decay_steps=100000,
        decay_rate=0.96,
        staircase=True)
     
    with strategy.scope():
        # generator = architecture_module.generator()
        # generator_path = os.path.join(train_config['model_save_path'], 'best_model.keras')
        generator_path ='/home/oshadi/SISR-Final_Year_Project/envs/SISR-Project/Final-Model/SINSR/models/generator_5pl/best_model.keras'
        generator = load_model(generator_path, safe_mode=False) # Load the model with safe mode disabled
        discriminator = architecture_module.discriminator()
        # encoder = architecture_module.encoder()  # Added encoder initialization

        efficientnet = EfficientNetB7(include_top=False, weights='imagenet')
        efficientnet.trainable = False

        vgg = create_flexible_vgg()

        generator_optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule_gen)
        discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule_disc)
        # encoder_optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule_enc)  # Added encoder optimizer

    return generator, discriminator, efficientnet, vgg,  generator_optimizer, discriminator_optimizer
    # return generator, discriminator, encoder, efficientnet, vgg, generator_optimizer, discriminator_optimizer, encoder_optimizer

