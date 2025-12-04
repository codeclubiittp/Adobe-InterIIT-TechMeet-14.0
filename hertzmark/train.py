# train.py (excerpt with changes)
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim

from models import Embedder, Detector
from dataset import ImageOnlyDataset  # new dataset
import random
# from models_detector import SDetector

# settings
DATA_FOLDER = "data/filmset"
IMAGE_SIZE = 256
BATCH = 32
EPOCHS = 10
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# models
embedder = Embedder().to(DEVICE)
detector = Detector().to(DEVICE)


# key vector on GPU
torch.manual_seed(1234)
key_vec = torch.randn(1, 128).to(DEVICE)

dataset = ImageOnlyDataset(DATA_FOLDER, IMAGE_SIZE)
loader = DataLoader(dataset, batch_size=BATCH, shuffle=True, num_workers=4, pin_memory=True)

opt = optim.Adam(list(embedder.parameters()) + list(detector.parameters()), lr=1e-4)
bce = nn.BCELoss()

for epoch in range(EPOCHS):
    for imgs in loader:
        # imgs: (B,3,H,W) on CPU (because DataLoader moves them here)
        imgs = imgs.to(DEVICE, non_blocking=True)

        # Decide per-sample labels randomly (50% watermark)
        B = imgs.size(0)
        is_wm = torch.rand(B) < 0.5  # boolean mask

        # Create inputs for detector:
        inputs = imgs.clone()
        if is_wm.any():
            # prepare a batch key_vec repeated
            key_batch = key_vec.repeat(int(is_wm.sum()), 1)
            # select indices to watermark
            idxs = is_wm.nonzero(as_tuple=False).squeeze(1)
            # embed those examples
            with torch.no_grad():  # remove this if you want gradients to flow into embedder too
                # remove with no_grad -> embedder trains; keep it if you want joint training
                pass

            # If you want joint training for embedder too, DON'T use no_grad:
            wm_imgs = embedder(inputs[idxs], key_batch)   # outputs on DEVICE
            inputs[idxs] = wm_imgs

        # Compose labels for detector
        labels = is_wm.float().to(DEVICE).unsqueeze(1)

        # Forward pass through detector
        preds = detector(inputs)
        loss = bce(preds, labels)

        opt.zero_grad()
        loss.backward()
        opt.step()

    print(f"Epoch {epoch+1}/{EPOCHS} | loss: {loss.item():.4f}")

torch.save(embedder.state_dict(), "embedder.pth")
torch.save(detector.state_dict(), "detector.pth")
print("Training complete. Models saved.")
