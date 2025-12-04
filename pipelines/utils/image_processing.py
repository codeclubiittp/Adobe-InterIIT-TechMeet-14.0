import cv2
import numpy as np
from PIL import Image

def letterbox(im, new_shape=(640, 640), color=(114, 114, 114)):
    shape = im.shape[:2]  
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])

    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  

    dw /= 2  
    dh /= 2

    if shape[::-1] != new_unpad:  
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  
    return im, r, (dw, dh)

def upscale_simple_lanczos(image: Image.Image, scale_factor: int = 2) -> Image.Image:
    new_size = (image.width * scale_factor, image.height * scale_factor)
    return image.resize(new_size, Image.LANCZOS)

def upscale_opencv_super_res(image: Image.Image, scale_factor: int = 2) -> Image.Image:
    img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    new_size = (image.width * scale_factor, image.height * scale_factor)
    upscaled = cv2.resize(img_cv, new_size, interpolation=cv2.INTER_CUBIC)
    return Image.fromarray(cv2.cvtColor(upscaled, cv2.COLOR_BGR2RGB))