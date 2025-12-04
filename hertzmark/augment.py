# augment.py
import cv2
import numpy as np
import torch

def jpeg_compress(img, quality):
    img_np = (img.permute(1,2,0).cpu().numpy() * 255).astype(np.uint8)
    enc = cv2.imencode(".jpg", img_np,
                       [cv2.IMWRITE_JPEG_QUALITY, int(quality)])[1]
    dec = cv2.imdecode(enc, 1)
    dec = torch.tensor(dec).float().permute(2,0,1) / 255.0
    return dec

def strong_augment(img):
    # Random JPEG
    if np.random.rand() < 0.5:
        img = jpeg_compress(img, np.random.randint(40, 95))

    # Blur
    if np.random.rand() < 0.3:
        k = np.random.choice([3, 5])
        arr = (img.permute(1,2,0).numpy() * 255).astype(np.uint8)
        arr = cv2.GaussianBlur(arr, (k,k), 1)
        img = torch.tensor(arr).float().permute(2,0,1) / 255.0

    # Resize down and up
    if np.random.rand() < 0.3:
        H, W = img.size(1), img.size(2)
        scale = np.random.uniform(0.6, 1.0)
        newH, newW = int(H*scale), int(W*scale)
        arr = cv2.resize((img.permute(1,2,0).numpy()*255).astype(np.uint8),
                         (newW,newH))
        arr = cv2.resize(arr, (W,H))
        img = torch.tensor(arr).float().permute(2,0,1)/255.0

    return img
