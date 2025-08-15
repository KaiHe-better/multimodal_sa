import os
import torch

# DEVICE = torch.device("cuda")

# # MOSI SETTING
# ACOUSTIC_DIM = 74
# VISUAL_DIM = 47
# TEXT_DIM = 768

# # MOSEI SETTING
# ACOUSTIC_DIM = 74
# VISUAL_DIM = 35
# TEXT_DIM = 768
    
def get_config(dataset):
    if dataset == 'mosi':
        return {
            'ACOUSTIC_DIM': 74,
            'VISUAL_DIM': 47,
            'TEXT_DIM': 768
        }
    elif dataset == 'mosei':
        return {
            'ACOUSTIC_DIM': 74,
            'VISUAL_DIM': 35,
            'TEXT_DIM': 768
        }
    elif dataset == 'chsims':
        return {
            'ACOUSTIC_DIM': 33,   
            'VISUAL_DIM': 709,    
            'TEXT_DIM': 768       
        }
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")
