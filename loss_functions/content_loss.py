def perceptual_loss(vgg, y_true, y_pred):
    y_true = tf.keras.applications.vgg19.preprocess_input(y_true * 255.0)
    y_pred = tf.keras.applications.vgg19.preprocess_input(y_pred * 255.0)

    layers = ['block1_conv2', 'block2_conv2', 'block3_conv3', 'block4_conv3', 'block5_conv3']

    feature_extractor = tf.keras.Model(inputs=vgg.input, outputs=[vgg.get_layer(name).output for name in layers])
    # weights = [2.0, 2.0, 1.0, 1.0, 1.5]  # Emphasize higher layers
    vgg_features_true = feature_extractor(y_true)
    vgg_features_pred = feature_extractor(y_pred)
    
    perc_loss = tf.reduce_mean([tf.reduce_mean(tf.square(f_true - f_pred)) for f_true, f_pred in zip(vgg_features_true, vgg_features_pred)]) ## added weight for this..
    # perc_loss = tf.add_n([w * tf.reduce_mean(tf.square(f_true - f_pred))
    #                       for w, f_true, f_pred in zip(weights, vgg_features_true, vgg_features_pred)])
    return perc_loss

def contextual_loss(y_true, y_pred, h=0.5):
    def cosine_similarity(x1, x2):
        x1 = tf.nn.l2_normalize(x1, axis=-1)
        x2 = tf.nn.l2_normalize(x2, axis=-1)
        return tf.reduce_sum(x1 * x2, axis=-1)
    
    def contextual_similarity(y_true_patches, y_pred_patches):
        cs = cosine_similarity(y_true_patches, y_pred_patches)
        return cs / (tf.reduce_sum(cs, axis=-1, keepdims=True) + 1e-5)
    
    y_true_patches = tf.image.extract_patches(y_true, sizes=[1, 3, 3, 1], strides=[1, 1, 1, 1], rates=[1, 1, 1, 1], padding='SAME')
    y_pred_patches = tf.image.extract_patches(y_pred, sizes=[1, 3, 3, 1], strides=[1, 1, 1, 1], rates=[1, 1, 1, 1], padding='SAME')
    
    y_true_patches = tf.reshape(y_true_patches, (tf.shape(y_true)[0], -1, 3, 3, tf.shape(y_true)[-1]))
    y_pred_patches = tf.reshape(y_pred_patches, (tf.shape(y_pred)[0], -1, 3, 3, tf.shape(y_pred)[-1]))
    
    cs = contextual_similarity(y_true_patches, y_pred_patches)
    cs = tf.clip_by_value(cs, 1e-10, 1.0)  # Clip values to avoid log(0)
    # cont_loss = -tf.reduce_sum(tf.math.log(cs))
    cont_loss = -tf.reduce_sum(tf.math.log(cs)) / tf.cast(tf.size(cs), tf.float32) # Normalize loss return cont_loss
    return cont_loss

def content_loss(y_true, y_pred, lambda_pl=0.25,lambda_cl=0.25):
    return lambda_cl * contextual_loss(y_true, y_pred) + lambda_pl *perceptual_loss(vgg, y_true, y_pred)
