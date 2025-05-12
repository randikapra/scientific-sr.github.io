def color_consistency_loss(y_true, y_pred):
    return tf.reduce_mean(tf.abs(y_true - y_pred))

def texture_matching_loss(y_true, y_pred):
    grad_true = tf.image.sobel_edges(y_true)
    grad_pred = tf.image.sobel_edges(y_pred)
    return tf.reduce_mean(tf.abs(grad_true - grad_pred))
  
def texture_loss(y_true, y_pred, lambda_ccl=0.25,lambda_tl=0.25):
    return lambda_cl * color_consistency_loss(y_true, y_pred) + lambda_tl * texture_matching_loss(y_true, y_pred)
