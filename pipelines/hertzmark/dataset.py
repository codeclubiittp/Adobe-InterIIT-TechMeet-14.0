# dataset.py
import cv2
import torch
from torch.utils.data import Dataset
import glob
import numpy as np
from augment import strong_augment

class ImageOnlyDataset(Dataset):
    def __init__(self, folder, image_size=256):
        self.paths = sorted(glob.glob(folder + "/*.jpg") +
                            glob.glob(folder + "/*.jpeg") +
                            glob.glob(folder + "/*.png"))
        assert len(self.paths) > 0, "No images found in dataset!"
        self.size = image_size

    def __len__(self):
        return len(self.paths)

    def load_image(self, path):
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.size, self.size))
        img = torch.tensor(img).float().permute(2,0,1) / 255.0
        return img

    def __getitem__(self, idx):
        path = self.paths[idx]
        img = self.load_image(path)
        img = strong_augment(img)  # keep augmentations here (optional)
        return img
