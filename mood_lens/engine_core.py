import torch
import torch.optim as optim
import cv2
import numpy as np
import os
import random
import gc
import logging
from PIL import Image
from tqdm import tqdm
from torchvision import transforms

import mood_lens.config as config
from mood_lens.dataset import ImageDataset
from mood_lens.lut_model import TrilinearLUT 

def check_image_path(path):
    """Helper to verify image path and debug if missing."""
    if os.path.exists(path):
        return True
    
    # Try to fix common relative path issues
    abs_path = os.path.abspath(path)
    logging.error(f"Image not found at: {path}")
    logging.error(f"Resolved absolute path: {abs_path}")
    
    # Check if we are already inside the root directory and the path duplicates it
    # e.g., CWD is /app/pipeline and path is pipeline/dataset/...
    if not os.path.exists(path):
        cwd = os.getcwd()
        logging.info(f"Current Working Directory: {cwd}")
        
    return False

def colour_correct(
    input_path: str, 
    target_emotion: str, 
    matcher, 
    generator, 
    profile: bool = False
):
    filename = os.path.basename(input_path)
    logging.info(f"Processing: {filename} -> {target_emotion}")

    logging.info(f"Searching FAISS for top {config.TOP_K_MATCHES} matches...")
    top_matches = matcher.search(input_path, target_emotion, k=config.TOP_K_MATCHES)
    
    if not top_matches:
        logging.warning(f"No matches found for emotion '{target_emotion}'")
        return

    ref_paths = [m[0] for m in top_matches]
    for p, s in top_matches:
        logging.info(f"Found: {p} (Score: {s:.3f})")

    logging.info("Optimizing One-Shot LUT")
    
    lut_model = TrilinearLUT(dim=config.LUT_DIM).to(config.DEVICE)
    optimizer = optim.AdamW(lut_model.parameters(), lr=config.LEARNING_RATE)
    
    train_transform = transforms.Compose([
        transforms.Resize(config.TRAIN_SIZE),
        transforms.ToTensor()
    ])
    
    try:
        content_tensor = ImageDataset([input_path], train_transform)[0][0].unsqueeze(0).to(config.DEVICE)
        target_grams_list = []
        c_feats_orig = None

        # preprocessing
        def preprocess():
            target_grams_list.clear()
            with torch.no_grad():
                c_feats_orig = generator._get_features(content_tensor)
                
                for r_path in ref_paths:
                    # VALIDATION ADDED
                    if not check_image_path(r_path):
                        continue

                    s_tensor = ImageDataset([r_path], train_transform)[0][0].unsqueeze(0).to(config.DEVICE)
                    
                    if s_tensor.sum() > 0: 
                        s_feats = generator._get_features(s_tensor)
                        s_grams = [generator._gram_matrix(f) for f in s_feats]
                        target_grams_list.append(s_grams)
                    else:
                        logging.warning(f"Image loaded as empty (sum=0): {r_path}")

            return c_feats_orig
            
        if profile:
            with torch.profiler.record_function("LUT_Preprocess"):
                c_feats_orig = preprocess()
        else:
            c_feats_orig = preprocess()

        if not target_grams_list: 
            logging.error("Target grams list is empty. Check image paths or loading logic.")
            return
        if c_feats_orig is None:
            logging.error("Content features could not be extracted.")
            return

        # training
        def train():
            lut_model.train()
            pbar = tqdm(range(config.TRAIN_STEPS), desc=f"Tuning ({target_emotion})", leave=False)

            def forward_pass():
                        current_target_grams = random.choice(target_grams_list)
                        stylized = lut_model(content_tensor)
                        c_feats = generator._get_features(stylized)
                        
                        loss_style = sum([torch.nn.functional.mse_loss(generator._gram_matrix(cf), sg) 
                                        for cf, sg in zip(c_feats, current_target_grams)])
                        loss_content = torch.nn.functional.mse_loss(c_feats[1], c_feats_orig[1])
                        loss_tv = generator._tv_loss(lut_model.lut)
                        
                        loss = (loss_style * config.WEIGHT_STYLE) + (loss_content * config.WEIGHT_CONTENT) + (loss_tv * config.WEIGHT_TV)
                        return loss
            
            vram_forward_pass_avg = 0.0
            vram_backward_pass_avg = 0.0
            with torch.autocast(device_type=config.DEVICE, dtype=torch.bfloat16):
                for i in pbar:
                    optimizer.zero_grad()
                    loss = None
                    
                    if profile:
                        with torch.profiler.record_function("LUT_ForwardPass"):
                            loss = forward_pass()
                            vram_forward_pass_avg += torch.cuda.memory_allocated() / (1024 * 1024)
                    else:
                        loss = forward_pass()
                    
                    if profile:
                        with torch.profiler.record_function("LUT_BackwardPass"):
                            loss.backward()
                            optimizer.step()
                            pbar.set_postfix({'loss': f"{loss.item():.2f}"})
                            vram_backward_pass_avg += torch.cuda.memory_allocated() / (1024 * 1024)
                    else:
                        loss.backward()
                        optimizer.step()
                        pbar.set_postfix({'loss': f"{loss.item():.2f}"})

            return vram_forward_pass_avg / config.TRAIN_STEPS, vram_backward_pass_avg / config.TRAIN_STEPS

        vram_forward_pass = 0.0
        vram_backward_pass = 0.0

        if profile:
            with torch.profiler.record_function("LUT_Training"):
                vram_forward_pass, vram_backward_pass = train()
        else:
            train()

        logging.info("Generating Output")
        lut_model.eval()
        img_pil = Image.open(input_path).convert("RGB")
        full_tensor = transforms.ToTensor()(img_pil).unsqueeze(0).to(config.DEVICE)

        # inference
        with torch.no_grad():
            output_tensor = lut_model(full_tensor)
        
        # post processing
        success = False
        out_name = ""
        def postprocess():
            out_np = output_tensor.squeeze().permute(1, 2, 0).cpu().numpy()
            out_bgr = cv2.cvtColor(np.clip(out_np * 255, 0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
            
            if not os.path.exists(config.OUTPUT_ROOT_DIR):
                os.makedirs(config.OUTPUT_ROOT_DIR)

            out_name = os.path.join(config.OUTPUT_ROOT_DIR, f"result_{target_emotion}_{filename}")
            success = cv2.imwrite(out_name, out_bgr)
            return success, out_name 

        if profile:
            with torch.profiler.record_function("LUT_Postprocess"):
                success, out_name = postprocess()
        else:
            success, out_name = postprocess()
        
        if success:
            logging.info(f"Success: Saved to {out_name}")
        else:
            logging.error(f"OpenCV failed to write image to {out_name}")

        return vram_forward_pass, vram_backward_pass
            
    except Exception as e:
        logging.critical(f"Error during training: {e}", exc_info=True)
    finally:
        del lut_model
        del optimizer
        if 'content_tensor' in locals(): del content_tensor
        if 'target_grams_list' in locals(): del target_grams_list
        torch.cuda.empty_cache()
        gc.collect()

def colour_correct_multivariant(
    input_path: str, 
    target_emotion: str, 
    matcher, 
    generator,
    n_variants: int = 3,
    profile: bool = False
):
    filename = os.path.basename(input_path)
    logging.info(f"Searching FAISS for top {n_variants} matches for: {filename} -> {target_emotion}")
    top_matches = matcher.search(input_path, target_emotion, k=n_variants)
    
    if not top_matches:
        logging.warning(f"No matches found for emotion '{target_emotion}'")
        return 0.0, 0.0

    num_found = len(top_matches)
    logging.info(f"Found {num_found} matches.")

    logging.info("Pre-calculating content features")
    train_transform = transforms.Compose([
        transforms.Resize(config.TRAIN_SIZE),
        transforms.ToTensor()
    ])
    
    try:
        content_dataset = ImageDataset([input_path], train_transform)
        content_tensor = content_dataset[0][0].unsqueeze(0).to(config.DEVICE)
        
        with torch.no_grad():
            c_feats_orig = generator._get_features(content_tensor)
    except Exception as e:
        logging.error(f"Failed to load content image: {e}")
        return 0.0, 0.0

    total_vram_fwd = 0.0
    total_vram_bwd = 0.0

    for idx, (ref_path, score) in enumerate(top_matches):
        variant_id = idx + 1
        logging.info(f"--- Processing Variant {variant_id}/{num_found} ---")
        logging.info(f"Reference: {ref_path} (Score: {score:.3f})")

        # --- DIAGNOSTIC CHECK ---
        if not check_image_path(ref_path):
            logging.warning(f"Skipping variant {variant_id} due to missing file.")
            continue
        # ------------------------

        lut_model = None
        optimizer = None
        
        try:
            lut_model = TrilinearLUT(dim=config.LUT_DIM).to(config.DEVICE)
            optimizer = optim.AdamW(lut_model.parameters(), lr=config.LEARNING_RATE)
            target_grams = []
            
            with torch.no_grad():
                s_tensor = ImageDataset([ref_path], train_transform)[0][0].unsqueeze(0).to(config.DEVICE)
                
                # --- TENSOR CHECK ---
                if s_tensor.sum() <= 0:
                    logging.warning(f"Skipping variant {variant_id}: Image tensor is empty or black.")
                    continue
                # --------------------

                s_feats = generator._get_features(s_tensor)
                target_grams = [generator._gram_matrix(f) for f in s_feats]
            
            if not target_grams:
                logging.warning(f"Skipping variant {variant_id}: Could not extract style features.")
                continue

            def train_step():
                lut_model.train()
                pbar_desc = f"Tuning V{variant_id}"
                pbar = tqdm(range(config.TRAIN_STEPS), desc=pbar_desc, leave=False)

                vram_f = 0.0
                vram_b = 0.0

                with torch.autocast(device_type=config.DEVICE, dtype=torch.bfloat16):
                    for i in pbar:
                        optimizer.zero_grad()
                        
                        def forward_op():
                            stylized = lut_model(content_tensor)
                            c_feats = generator._get_features(stylized)
                            
                            loss_style = sum([torch.nn.functional.mse_loss(generator._gram_matrix(cf), sg) 
                                            for cf, sg in zip(c_feats, target_grams)])
                            loss_content = torch.nn.functional.mse_loss(c_feats[1], c_feats_orig[1])
                            loss_tv = generator._tv_loss(lut_model.lut)
                            
                            loss = (loss_style * config.WEIGHT_STYLE) + \
                                   (loss_content * config.WEIGHT_CONTENT) + \
                                   (loss_tv * config.WEIGHT_TV)
                            return loss

                        loss = None
                        if profile:
                            with torch.profiler.record_function(f"LUT_Fwd_V{variant_id}"):
                                loss = forward_op()
                                vram_f += torch.cuda.memory_allocated() / (1024 * 1024)
                        else:
                            loss = forward_op()

                        if profile:
                            with torch.profiler.record_function(f"LUT_Bwd_V{variant_id}"):
                                loss.backward()
                                optimizer.step()
                                pbar.set_postfix({'loss': f"{loss.item():.2f}"})
                                vram_b += torch.cuda.memory_allocated() / (1024 * 1024)
                        else:
                            loss.backward()
                            optimizer.step()
                            pbar.set_postfix({'loss': f"{loss.item():.2f}"})
                
                return vram_f / config.TRAIN_STEPS, vram_b / config.TRAIN_STEPS

            v_f, v_b = 0.0, 0.0
            if profile:
                with torch.profiler.record_function(f"LUT_Train_V{variant_id}"):
                    v_f, v_b = train_step()
            else:
                train_step()
            
            total_vram_fwd += v_f
            total_vram_bwd += v_b

            logging.info(f"Generating Output for Variant {variant_id}")
            lut_model.eval()
            img_pil = Image.open(input_path).convert("RGB")
            full_tensor = transforms.ToTensor()(img_pil).unsqueeze(0).to(config.DEVICE)

            with torch.no_grad():
                output_tensor = lut_model(full_tensor)

            out_np = output_tensor.squeeze().permute(1, 2, 0).cpu().numpy()
            out_bgr = cv2.cvtColor(np.clip(out_np * 255, 0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
            
            if not os.path.exists(config.OUTPUT_ROOT_DIR):
                os.makedirs(config.OUTPUT_ROOT_DIR)

            out_name = os.path.join(config.OUTPUT_ROOT_DIR, f"result_{target_emotion}_{filename}_v{variant_id}.png")
            cv2.imwrite(out_name, out_bgr)
            logging.info(f"Saved: {out_name}")

        except Exception as e:
            logging.critical(f"Error processing variant {variant_id}: {e}", exc_info=True)
        
        finally:
            if lut_model: del lut_model
            if optimizer: del optimizer
            if 'target_grams' in locals(): del target_grams
            torch.cuda.empty_cache()
            gc.collect()

    if 'content_tensor' in locals(): del content_tensor
    torch.cuda.empty_cache()
    gc.collect()
    
    avg_fwd = total_vram_fwd / num_found if num_found > 0 else 0.0
    avg_bwd = total_vram_bwd / num_found if num_found > 0 else 0.0
    
    return avg_fwd, avg_bwd