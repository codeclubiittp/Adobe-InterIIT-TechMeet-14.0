import numpy as np
import torch
from PIL import Image

def decode_normals(pred):
    if isinstance(pred, torch.Tensor):
        arr = pred.detach().cpu().float().numpy()
    elif isinstance(pred, Image.Image):
        arr = np.asarray(pred).astype(np.float32) / 255.0
    else:
        arr = np.array(pred).astype(np.float32)

    while arr.ndim > 3:
        arr = arr.squeeze(0)

    if arr.max() > 1.5:
        arr /= 255.0

    normals = arr * 2.0 - 1.0
    norm = np.linalg.norm(normals, axis=-1, keepdims=True) + 1e-8
    return (normals / norm).astype(np.float32)

def visualize_normals(normals):
    return (normals * 0.5 + 0.5).clip(0, 1)

