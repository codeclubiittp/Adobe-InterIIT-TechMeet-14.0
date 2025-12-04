import faiss
import pandas as pd
import torch
import os
import pickle
import numpy as np
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from torchvision import transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import logging
import mood_lens.config as config
from mood_lens.dataset import ImageDataset

class VSearchEngine:
    def __init__(
        self,
        clip_model_name="openai/clip-vit-base-patch32"
    ):
        config.setup_logging()
        self.device = config.DEVICE
        
        logging.info("Initializing Semantic Search Engine")
        self.model = CLIPModel.from_pretrained(
            clip_model_name, 
            cache_dir=config.CACHE_DIR
        ).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(
            clip_model_name, 
            cache_dir=config.CACHE_DIR,
            use_fast=True
        )
        self.model.eval()

        if os.path.exists(config.INDEX_PATH) and os.path.exists(config.METADATA_PATH):
            logging.info("Existing index found. Loading from disk.")
            self.load_index()
        else:
            logging.info("No existing index found. Building new index.")
            self.build_index()

    def get_clip_transform(self):
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), 
                                 (0.26862954, 0.26130258, 0.27577711))
        ])

    def build_index(self):
        logging.info(f"Building New Vector Database from {config.CSV_PATH}")
        df = pd.read_csv(config.CSV_PATH)
        
        valid_paths = []
        valid_emotions = []
        
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Checking Files"):
            if os.path.exists(row['image_path']):
                valid_paths.append(row['image_path'])
                valid_emotions.append(row['emotion'])
        
        dataset = ImageDataset(valid_paths, self.get_clip_transform())
        loader = DataLoader(dataset, batch_size=config.BATCH_SIZE, num_workers=4, shuffle=False)
        
        all_embeddings = []
        self.metadata_map = {} 
        
        current_id = 0
        with torch.no_grad():
            for batch_imgs, batch_paths in tqdm(loader, desc="Embedding Batches"):
                batch_imgs = batch_imgs.to(self.device)
                features = self.model.get_image_features(pixel_values=batch_imgs)
                features = features / features.norm(p=2, dim=-1, keepdim=True)
                features_np = features.cpu().numpy().astype('float32')
                
                for i, path in enumerate(batch_paths):
                    if path != "ERROR":
                        all_embeddings.append(features_np[i])
                        self.metadata_map[current_id] = {
                            "path": path,
                            "emotion": valid_emotions[current_id]
                        }
                        current_id += 1

        if not all_embeddings:
            logging.error("No valid images found to index.")
            return

        embedding_matrix = np.vstack(all_embeddings)
        d = embedding_matrix.shape[1] 
        self.index = faiss.IndexFlatIP(d) 
        self.index.add(embedding_matrix)
        
        logging.info(f"Index built with {self.index.ntotal} vectors.")
        faiss.write_index(self.index, config.INDEX_PATH)
        with open(config.METADATA_PATH, 'wb') as f:
            pickle.dump(self.metadata_map, f)
        logging.info("Index and Metadata Saved to Disk.")

    def load_index(self):
        logging.info("Loading existing FAISS index from disk")
        self.index = faiss.read_index(config.INDEX_PATH)
        with open(config.METADATA_PATH, 'rb') as f:
            self.metadata_map = pickle.load(f)
        logging.info(f"Loaded {self.index.ntotal} vectors.")

    def search(self, input_path, target_emotion, k=3):
        try:
            img = Image.open(input_path).convert("RGB")
            t = self.get_clip_transform()
            img_t = t(img).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                feat = self.model.get_image_features(pixel_values=img_t)
                feat = feat / feat.norm(p=2, dim=-1, keepdim=True)
                query_vector = feat.cpu().numpy().astype('float32')
        except Exception as e:
            logging.error(f"Error embedding input: {e}")
            return []

        D, I = self.index.search(query_vector, 300) 
        
        final_candidates = []
        for score, idx in zip(D[0], I[0]):
            if idx in self.metadata_map:
                meta = self.metadata_map[idx]
                if meta['emotion'] == target_emotion:
                    final_candidates.append((meta['path'], score))
                    if len(final_candidates) >= k:
                        break
        return final_candidates