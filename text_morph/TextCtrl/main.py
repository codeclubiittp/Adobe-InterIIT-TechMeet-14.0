import torch
import os
from src.trainer.CtrlBase import ControlBase
from PIL import Image
import numpy as np
import torchvision.transforms as T
from tqdm import tqdm
from src.MuSA.GaMuSA_app import text_editing
from src.MuSA.GaMuSA import GaMuSA
from argparse import ArgumentParser
from pytorch_lightning import seed_everything
from utils import create_model, load_state_dict


# Add this new class
class GaMuSAModelManager:
    """Singleton to manage model loading and reuse"""
    _instance = None
    _model = None
    _pipeline = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def get_pipeline(self, ckpt_path="weights/model.pth"):
        """Get or create the pipeline"""
        if self._pipeline is None:
            print("[INFO] Loading model for the first time...")
            cfg_path = 'configs/inference.yaml'
            self._model = create_model(cfg_path).cuda()
            self._model.load_state_dict(load_state_dict(ckpt_path), strict=False)
            self._model.eval()
            
            monitor_cfg = {
                "max_length": 25,
                "loss_weight": 1.,
                "attention": 'position',
                "backbone": 'transformer',
                "backbone_ln": 3,
                "checkpoint": "weights/vision_model.pth",
                "charset_path": "src/module/abinet/data/charset_36.txt"
            }
            self._pipeline = GaMuSA(self._model, monitor_cfg)
            print("[INFO] Model loaded successfully!")
        return self._pipeline


# Initialize global instance
model_manager = GaMuSAModelManager()


def load_image(image_path, image_height=256, image_width=256):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    img = Image.open(image_path)
    image = T.ToTensor()(T.Resize((image_height, image_width))(img.convert("RGB")))
    image = image.to(device)
    return image.unsqueeze(0)


def create_parser():
    parser = ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--target_height", default=256)
    parser.add_argument("--teaget_width", default=256)
    parser.add_argument("--style_height", default=256)
    parser.add_argument("--style_width", default=256)
    parser.add_argument("--ckpt_path", type=str, default="weights/model.pth")
    parser.add_argument("--dataset_dir", type=str, default="example/")
    parser.add_argument("--output_dir", type=str, default="example_result/")
    parser.add_argument("--starting_layer", default=10, type=int)
    parser.add_argument("--num_inference_steps", default=50)
    parser.add_argument("--num_sample_per_image", default=1, type=int)
    parser.add_argument("--guidance_scale", default=2, type=float)
    parser.add_argument("--benchmark", default=True)
    return parser


def process_single_image(image_path, style_text, target_text, 
                         ckpt_path="weights/model.pth",
                         starting_layer=10, 
                         num_inference_steps=50,
                         guidance_scale=2,
                         seed=42):
    """Process a single image using the shared model pipeline."""
    
    # Use the singleton model manager instead of loading model each time
    pipeline = model_manager.get_pipeline(ckpt_path)
    
    seed_everything(seed)
    
    w, h = Image.open(image_path).size
    source_image = load_image(image_path)
    style_image = load_image(image_path)
    
    result = text_editing(pipeline, source_image, style_image, style_text, target_text,
                         starting_layer=starting_layer,
                         ddim_steps=num_inference_steps,
                         scale=guidance_scale)
    
    reconstruction_image, GaMuSA_image = result[:]
    reconstruction_image = Image.fromarray((reconstruction_image * 255).astype(np.uint8)).resize((w, h))
    GaMuSA_image = Image.fromarray((GaMuSA_image * 255).astype(np.uint8)).resize((w, h))
    
    return reconstruction_image, GaMuSA_image


def process_multiple_images(image_data_list, 
                           ckpt_path="weights/model.pth",
                           starting_layer=10, 
                           num_inference_steps=50,
                           guidance_scale=2,
                           seed=42):
    """Process multiple images with a single model load."""
    
    # Use the singleton model manager
    pipeline = model_manager.get_pipeline(ckpt_path)
    
    seed_everything(seed)
    
    results = []
    for image_path, style_text, target_text in image_data_list:
        w, h = Image.open(image_path).size
        source_image = load_image(image_path)
        style_image = load_image(image_path)
        
        result = text_editing(pipeline, source_image, style_image, style_text, target_text,
                             starting_layer=starting_layer,
                             ddim_steps=num_inference_steps,
                             scale=guidance_scale)
        
        reconstruction_image, GaMuSA_image = result[:]
        reconstruction_image = Image.fromarray((reconstruction_image * 255).astype(np.uint8)).resize((w, h))
        GaMuSA_image = Image.fromarray((GaMuSA_image * 255).astype(np.uint8)).resize((w, h))
        
        results.append((reconstruction_image, GaMuSA_image))
    
    return results


# Keep your main() function unchanged...