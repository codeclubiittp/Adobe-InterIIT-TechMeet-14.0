import os
from pathlib import Path
import torch
import yaml
import numpy as np
import cv2

from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from omegaconf import OmegaConf
from saicinpainting.evaluation.utils import move_to_device
from saicinpainting.training.trainers import load_checkpoint

app = FastAPI()

ROOT = Path(__file__).parent.resolve()
MODEL_DIR = ROOT / "fourier"               # your model folder
CONFIG_PATH = MODEL_DIR / "config.yaml"
CHECKPOINT = MODEL_DIR / "models" / "best.ckpt"   # or predict_config.model.checkpoint

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------------------
# Load model ONCE at startup
# ------------------------------
print("Loading LaMa model...")

with open(CONFIG_PATH, "r") as f:
    train_config = OmegaConf.create(yaml.safe_load(f))

train_config.training_model.predict_only = True
train_config.visualizer.kind = 'noop'

device = torch.device("cpu")

model = load_checkpoint(
    train_config,
    CHECKPOINT,
    strict=False,
    map_location=device
)
model.freeze()
model.to(device)

print("LaMa model loaded.")


@app.post("/inpaint")
async def inpaint(image: UploadFile = File(...),
                  mask: UploadFile = File(...)):
    """
    Real-time inpainting using pre-loaded model.
    """

    # Read bytes
    image_bytes = await image.read()
    mask_bytes = await mask.read()

    # Decode
    img = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
    msk = cv2.imdecode(np.frombuffer(mask_bytes, np.uint8), cv2.IMREAD_GRAYSCALE)

    # Convert BGR→RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    msk = (msk > 0).astype(np.uint8)

    # Prepare batch in LaMa format
    batch = {
        "image": torch.tensor(img).permute(2, 0, 1).float() / 255.0,
        "mask": torch.tensor(msk).unsqueeze(0).float()
    }

    # Add batch dimension
    batch = {k: v.unsqueeze(0) for k, v in batch.items()}
    batch = move_to_device(batch, device)

    # Forward pass
    with torch.no_grad():
        result = model(batch)

    out = result["inpainted"][0].permute(1, 2, 0).cpu().numpy()
    out = (out * 255).clip(0, 255).astype(np.uint8)
    out = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)

    out_path = ROOT / "output.png"
    cv2.imwrite(str(out_path), out)
    print("output path: ", out_path)
    return FileResponse(out_path, media_type="image/png")

@app.get("/")
def home():
    return {"status": "LaMa FastAPI (model loaded once)"}

