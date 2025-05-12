import json

def load_config(config_path):
    with open(config_path, 'r') as f:
        return json.load(f)

def load_loss_weights(config_path):
    with open(config_path, 'r') as f:
        return json.load(f)
