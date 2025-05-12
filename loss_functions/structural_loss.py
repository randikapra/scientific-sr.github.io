def gradient_loss(y_true, y_pred):
    grad_true = tf.image.sobel_edges(y_true)
    grad_pred = tf.image.sobel_edges(y_pred)
    return tf.reduce_mean(tf.square(grad_true - grad_pred))

def second_order_gradient_loss(y_true, y_pred):
    grad_true = tf.image.sobel_edges(y_true)
    grad_pred = tf.image.sobel_edges(y_pred)
    grad2_true_x = tf.image.sobel_edges(grad_true[:, :, :, 0, :])
    grad2_true_y = tf.image.sobel_edges(grad_true[:, :, :, 1, :])
    grad2_pred_x = tf.image.sobel_edges(grad_pred[:, :, :, 0, :])
    grad2_pred_y = tf.image.sobel_edges(grad_pred[:, :, :, 1, :])
    grad2_true = tf.concat([grad2_true_x, grad2_true_y], axis=-1)
    grad2_pred = tf.concat([grad2_pred_x, grad2_pred_y], axis=-1)
    return tf.reduce_mean(tf.square(grad2_true - grad2_pred))

def total_variation_loss(y_pred):
    return tf.reduce_mean(tf.image.total_variation(y_pred))

def structure_similarity_loss(y_true, y_pred):
    return 1 - tf.reduce_mean(tf.image.ssim(y_true, y_pred, max_val=1.0))

def structural_loss(y_true, y_pred, lambda_gl=0.25,lambda_sgl=0.25, lambda_sm=1.0):
    return lambda_gl * gradient_loss(y_true, y_pred) + lambda_sgl * second_order_gradient_loss(y_true, y_pred) + lambda_sm * structure_similarity_loss(y_true, y_pred)
    # return lambda_tv * total_variation_loss(y_pred) + lambda_sm * structure_similarity_loss(y_true, y_pred)
