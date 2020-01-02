import numpy as np


'''Reverse transform array'''
def reverse_transform(arr):
    arr = arr.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    arr = std * arr + mean
    arr = np.clip(arr, 0, 1)
    arr = (arr * 255).astype(np.uint8)
    return arr
