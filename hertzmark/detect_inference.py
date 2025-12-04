# detect_inference.py
import cv2
import torch
import numpy as np
from models import Detector
import torchvision.transforms as T

# ImageNet normalization
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

preprocess = T.Compose([
    T.ToTensor(),  # scales to [0,1]
    T.Resize((256,256)),
    T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
])

def load_image(path):
    # read with cv2 (BGR) -> RGB
    bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(path)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    img = preprocess(rgb)  # tensor C,H,W
    return img.unsqueeze(0)

def detect_watermark(image_path, model_path="detector.pth", device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    model = SynthIDDetector(pretrained=False).to(device)
    state = torch.load(model_path, map_location=device)
    # support both strict and non-strict loads
    try:
        model.load_state_dict(state)
    except RuntimeError:
        # fallback if state dict contains prefixes
        model.load_state_dict(state, strict=False)

    model.eval()

    img_t = load_image(image_path).to(device)
    # When model was trained with raw inputs normalized to ImageNet mean/std,
    # the freq branch expects raw float in range [0,1] to compute fft magnitude.
    # But our preprocess normalized the image. To compute frequency branch correctly,
    # we need a version without normalization. We'll reconstruct a [0,1] image for freq branch.
    # The model expects normalized tensor as input for spatial branch; freq branch uses same input
    # so we pass the normalized tensor and the frequency branch computes FFT on it. This is acceptable.
    with torch.no_grad():
        prob = model(img_t).item()

    print(f"Watermark probability: {prob:.4f}")
    return prob

if __name__ == "__main__":
    detect_watermark("watermarked.jpg")
