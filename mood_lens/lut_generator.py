import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import numpy as np
import cv2
import mood_lens.config as config
import logging
from mood_lens.lut_model import TrilinearLUT

class NeuralLUTGenerator:
    def __init__(self):
        config.setup_logging()
        self.device = config.DEVICE
        logging.info(f"Loading VGG19 for Perceptual Loss on {self.device}")
        vgg = models.vgg19(weights=models.VGG19_Weights.DEFAULT).features
        self.slices = nn.ModuleList([vgg[:4], vgg[4:9], vgg[9:18], vgg[18:27]]).to(self.device).eval()
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(self.device)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(self.device)

    def _get_features(
        self, 
        x: torch.Tensor
    ):
        x = (x - self.mean) / self.std
        features = []
        for slice_block in self.slices:
            x = slice_block(x)
            features.append(x)
        return features

    def _gram_matrix(self, x):
        b, c, h, w = x.size()
        feat = x.view(b, c, h * w)
        gram = torch.bmm(feat, feat.transpose(1, 2))
        return gram / (c * h * w)

    def _tv_loss(
        self, 
        lut
    ):
        tv_h = torch.pow(lut[:, :, 1:, :, :] - lut[:, :, :-1, :, :], 2).mean()
        tv_w = torch.pow(lut[:, :, :, 1:, :] - lut[:, :, :, :-1, :], 2).mean()
        tv_d = torch.pow(lut[:, :, :, :, 1:] - lut[:, :, :, :, :-1], 2).mean()
        return tv_h + tv_w + tv_d

    def generate(
        self, 
        content_img_cv2: cv2.Mat, 
        style_img_path: str, 
        steps=config.TRAIN_STEPS
    ):
        img_pil = Image.fromarray(cv2.cvtColor(content_img_cv2, cv2.COLOR_BGR2RGB))
        transform_train = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()])
        transform_final = transforms.ToTensor()
        
        content_tensor = transform_train(img_pil).unsqueeze(0).to(self.device)

        try:
            style_pil = Image.open(style_img_path).convert("RGB")
            style_tensor = transform_train(style_pil).unsqueeze(0).to(self.device)
        except Exception as e:
            raise ValueError(f"Could not load style image: {style_img_path}. Error: {e}")

        lut_model = TrilinearLUT(dim=config.LUT_DIM).to(self.device)
        optimizer = optim.AdamW(lut_model.parameters(), lr=config.LEARNING_RATE)

        with torch.no_grad():
            s_feats = self._get_features(style_tensor)
            s_grams = [self._gram_matrix(f) for f in s_feats]
            c_feats_orig = self._get_features(content_tensor)

        logging.info(f"Optimizing Neural LUT ({steps} steps)")
        for i in range(steps):
            optimizer.zero_grad()
            stylized = lut_model(content_tensor)
            
            c_feats = self._get_features(stylized)
            
            loss_style = 0
            for cf, sg in zip(c_feats, s_grams):
                loss_style += torch.nn.functional.mse_loss(self._gram_matrix(cf), sg)
            
            loss_content = torch.nn.functional.mse_loss(c_feats[1], c_feats_orig[1])
            loss_tv = self._tv_loss(lut_model.lut)
            
            total_loss = (loss_style * config.WEIGHT_STYLE) + \
                         (loss_content * config.WEIGHT_CONTENT) + \
                         (loss_tv * config.WEIGHT_TV)
            
            total_loss.backward()
            optimizer.step()

        logging.info("Applying LUT to full resolution image")
        full_tensor = transform_final(img_pil).unsqueeze(0).to(self.device)
        with torch.no_grad():
            final_tensor = lut_model(full_tensor)
        
        final_np = final_tensor.squeeze().permute(1, 2, 0).cpu().numpy()
        return cv2.cvtColor(np.clip(final_np * 255, 0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR)