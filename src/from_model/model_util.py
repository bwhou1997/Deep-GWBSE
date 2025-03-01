import torch
import torch.nn as nn
import torch.nn.functional as F
from e2cnn import gspaces, nn as e2nn
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import os
from tqdm import tqdm
from functools import wraps
import logging

def print_model_size(model, model_name="Model"):
    param_size = 0
    param_number = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
        param_number += param.numel()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    size_all_mb = (param_size + buffer_size) / 1024**2
    print(f'{model_name} parameters: {param_number} with size of {size_all_mb:.3f} MB')
    return param_number

def timeCudaWatch(func):
    """Decorator to measure execution time of a function."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        
        start.record()
        result = func(*args, **kwargs)
        end.record()
        torch.cuda.synchronize()
        
        logging.debug(f"{func.__name__} execution time: {start.elapsed_time(end) * 1e-3:.3f} s")
        return result
    return wrapper

