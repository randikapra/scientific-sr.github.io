import tensorflow as tf
import numpy as np

def interpolate_weights(epoch, total_epochs, initial_weights, final_weights):
    alpha = epoch / total_epochs
    interpolated_weights = {key: (1 - alpha) * initial_weights[key] + alpha * final_weights[key] for key in initial_weights}
    return interpolated_weights

def update_loss_weights(losses, current_weights, smoothing_factor=0.1):
    total_loss = sum(losses.values())
    updated_weights = {}
    
    for key, value in losses.items():
        ratio = value / total_loss
        smoothed_ratio = (1 - smoothing_factor) * ratio + smoothing_factor * 1/len(losses)
        if key in current_weights:
            updated_weights[key] = current_weights[key] * (1 + (smoothed_ratio - 0.5) * 0.1)
        else:
            print(f"Warning: Key {key} not found in current_weights. Using default weight of 1.0.")
            updated_weights[key] = 1.0 * (1 + (smoothed_ratio - 0.5) * 0.1)
    
    total_current_weight = sum(current_weights.values())
    total_updated_weight = sum(updated_weights.values())
    for key in updated_weights:
        updated_weights[key] = updated_weights[key] * (total_current_weight / total_updated_weight)
    
    print(updated_weights)
    return updated_weights


def get_dynamic_weights(epoch, total_epochs, initial_weights, final_weights):
    dynamic_weights = {}
    for key in initial_weights.keys():
        dynamic_weights[key] = initial_weights[key] + (final_weights[key] - initial_weights[key]) * (epoch / total_epochs)
    return dynamic_weights

# def get_lr(epoch, base_lr, decay_rate=0.1, decay_epochs=5):
#     if epoch < decay_epochs:
#         return base_lr
#     else:
#         return base_lr * np.exp(-decay_rate * (epoch - decay_epochs))

def get_lr(base_lr, decay_steps=100000, decay_rate=0.96):
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=base_lr,
        decay_steps=decay_steps,
        decay_rate=decay_rate,
        staircase=True
    )
    return lr_schedule


def clear_gpu_memory():
    tf.keras.backend.clear_session()
    print("GPU memory cleared")
