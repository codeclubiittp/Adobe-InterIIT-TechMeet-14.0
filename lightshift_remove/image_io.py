import numpy as np
from PIL import Image
import base64
from io import BytesIO

def load_image_from_bytes(img_bytes):
    pil = Image.open(BytesIO(img_bytes)).convert("RGB")
    arr = np.asarray(pil).astype(np.float32) / 255.0
    return arr, pil

def to_uint8_image(arr):
    arr8 = (np.clip(arr, 0, 1) * 255).astype(np.uint8)
    return Image.fromarray(arr8)

def encode_base64(pil_img):
    buffer = BytesIO()
    pil_img.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")

