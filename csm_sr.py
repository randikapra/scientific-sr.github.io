from tensorflow.keras.applications import VGG19, EfficientNetB7 # type: ignore
from tensorflow.keras.models import Model # type: ignore
import tensorflow as tf # type: ignore
from tensorflow.keras import layers # type: ignore

# Custom PixelShuffle Layer
class PixelShuffle(layers.Layer):
    def __init__(self, scale, **kwargs):
        super(PixelShuffle, self).__init__(**kwargs)
        self.scale = scale

    def call(self, inputs):
        return tf.nn.depth_to_space(inputs, block_size=self.scale)

# Custom Resize Layer
class ResizeLayer(layers.Layer):
    def call(self, inputs):
        lr_inputs, hr_features_resized = inputs
        lr_shape = tf.shape(lr_inputs)[1:3]
        return tf.image.resize(hr_features_resized, lr_shape, method='bilinear')

# Define Residual Dense Block (RDB) with Mixed Convolution Types
def residual_dense_block(x, filters, growth_rate=32, layers_in_block=5):
    concat_features = [x]
    for _ in range(layers_in_block):
        x = layers.Conv2D(growth_rate, (3, 3), padding='same')(x)  # Standard convolution
        x = layers.Activation('relu')(x)
        
        # Dilated convolution
        x = layers.Conv2D(growth_rate, (3, 3), padding='same', dilation_rate=2)(x)
        x = layers.Activation('relu')(x)
        
        # Depthwise separable convolution
        x = layers.SeparableConv2D(growth_rate, (3, 3), padding='same')(x)
        x = layers.Activation('relu')(x)
        
        concat_features.append(x)
        x = layers.Concatenate()(concat_features)
    
    x = layers.Conv2D(filters, (3, 3), padding='same')(x)
    return x

# Define Residual-in-Residual Dense Block (RRDB)
def rrdb(x, filters, growth_rate=32, res_block=5):
    res = layers.Conv2D(filters, (3, 3), padding='same')(x)
    for _ in range(res_block):
        x = residual_dense_block(x, filters, growth_rate)
    return layers.Add()([x, layers.Lambda(lambda x: x * 0.2)(res)])

# Define Multi-Scale Convolutional Block
def multi_scale_conv_block(x, filters):
    conv_1x1 = layers.Conv2D(filters, (1, 1), padding='same')(x)
    conv_3x3 = layers.Conv2D(filters, (3, 3), padding='same')(x)
    conv_5x5 = layers.Conv2D(filters, (5, 5), padding='same')(x)
    return layers.Concatenate()([conv_1x1, conv_3x3, conv_5x5])


# Encoder to extract features from the LR image
def encoder(input_shape=(None, None, 3)):
    lr_input = layers.Input(shape=input_shape)
    
    # Initial convolutional layers
    x = layers.Conv2D(64, (3, 3), padding='same', activation='relu')(lr_input)
    x = layers.Conv2D(128, (3, 3), padding='same', activation='relu')(x)
    
    # Multi-scale feature extraction
    scale1 = layers.Conv2D(256, (1, 1), padding='same', activation='relu')(x)
    scale2 = layers.Conv2D(256, (3, 3), padding='same', activation='relu')(x)
    scale3 = layers.Conv2D(256, (5, 5), padding='same', activation='relu')(x)
    multi_scale_features = layers.Concatenate()([scale1, scale2, scale3])
    
    # Global and local attention (optional)
    attention = layers.Conv2D(256, (3, 3), padding='same', activation='sigmoid')(multi_scale_features)
    attended_features = layers.Multiply()([multi_scale_features, attention])
    
    # Output layer
    encoder_output = layers.Conv2D(256, (3, 3), padding='same', activation='relu')(attended_features)
    
    return Model(inputs=lr_input, outputs=encoder_output, name="Encoder")


def generator(input_shape=(None, None, 3), feature_shape=(None, None, 2560)):
    lr_inputs = layers.Input(shape=input_shape)
    hr_features = layers.Input(shape=feature_shape)

    # Adjust number of channels to match the expected shape
    # hr_features_resized = PixelShuffle(scale=8)(hr_features)
    hr_features_resized = PixelShuffle(scale=8)(hr_features)
    # Resize hr_features_resized to match lr_inputs dimensions using the custom ResizeLayer
    hr_features_resized = ResizeLayer()([lr_inputs, hr_features_resized])
    

    # Original scale
    x = layers.Conv2D(128, (3, 3), padding='same')(lr_inputs)
    x = layers.Activation('relu')(x)
    
    # Concatenate HR features with LR features
    x = layers.Concatenate()([x, hr_features_resized])
    
    # Apply multi-scale convolutional block
    # x = multi_scale_conv_block(x, 128)
    
    # Original scale processing (scale1)
    scale1 = x
    for _ in range(4):
        scale1 = rrdb(scale1, 128)

    # Downscale by 2 (scale2)
    scale2 = layers.AveragePooling2D(pool_size=(2, 2))(x)
    for _ in range(4):
        scale2 = rrdb(scale2, 128)
    # Upscale by 2
    scale2 = PixelShuffle(scale=2)(scale2)
    
    # Upscale by 2 (scale4)
    scale4 = PixelShuffle(scale=2)(x)
    for _ in range(4):
        scale4 = rrdb(scale4, 128)
    # Downscale by 2
    scale4 = layers.AveragePooling2D(pool_size=(2, 2))(scale4)
    
    # Concatenate multi-scale features
    multi_scale = layers.Concatenate()([scale1, scale2, scale4])
    
    # Additional convolutional layers
    multi_scale = layers.Conv2D(128, (3, 3), padding='same')(multi_scale)
    multi_scale = layers.Activation('relu')(multi_scale)
    
    # # Upscale by 2
    multi_scale = PixelShuffle(scale=2)(multi_scale)
    multi_scale = layers.Conv2D(128, (3, 3), padding='same')(multi_scale)
    multi_scale = layers.Activation('relu')(multi_scale)
    
    # Upscale by 2
    multi_scale = PixelShuffle(scale=2)(multi_scale)
    multi_scale = layers.Conv2D(3, (3, 3), padding='same')(multi_scale)
    
    # Final output
    outputs = layers.Conv2D(3, (3, 3), padding='same', activation='tanh')(multi_scale)
    
    return Model(inputs=[lr_inputs, hr_features], outputs=outputs)

# Residual Block for Discriminator
def res_block(x, filters):
    res = x
    x = layers.Conv2D(filters, (3, 3), padding='same')(x)
    x = layers.LeakyReLU(alpha=0.2)(x)
    x = layers.Conv2D(filters, (3, 3), padding='same')(x)
    x = layers.Add()([x, res])
    return x

def discriminator(input_shape=(None, None, 3)):
    inputs = layers.Input(shape=input_shape)

    # Initial convolution block
    x = layers.Conv2D(32, (3, 3), padding='same')(inputs)
    x = layers.LeakyReLU(alpha=0.2)(x)

    # Multi-scale processing branches
    scale1 = res_block(x, 32)
    scale2 = layers.AveragePooling2D(pool_size=(2, 2))(x)
    scale2 = res_block(scale2, 32)
    scale2 = PixelShuffle(scale=2)(scale2)
    scale3 = layers.AveragePooling2D(pool_size=(4, 4))(x)
    scale3 = res_block(scale3, 32)
    scale3 = PixelShuffle(scale=4)(scale3)

    multi_scale = layers.Concatenate()([scale1, scale2, scale3])

    # Additional convolutional layers after concatenation
    x = layers.Conv2D(32, (3, 3), padding='same')(multi_scale)
    x = layers.LeakyReLU(alpha=0.2)(x)
    
    for filters in [64, 128, 256, 512]:
        x = layers.Conv2D(filters, (4, 4), strides=(2, 2), padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU(negative_slope=0.2)(x)
    
    x = layers.Conv2D(1, (4, 4), padding='same')(x)
    return Model(inputs, x)


