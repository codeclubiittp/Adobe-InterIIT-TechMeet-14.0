import os
import time
import gc
import torch
import requests
import numpy as np
import onnxruntime as ort
from PIL import Image
from diffusers import DiffusionPipeline, EulerAncestralDiscreteScheduler, ControlNetModel
from torch.nn.attention import sdpa_kernel, SDPBackend

# ==========================================
# 1. Depth Estimation Module (ONNX)
# ==========================================
def run_depth_estimation(
    model_path: str,
    image_path: str,
) -> Image.Image:
    """
    Runs ONNX depth estimation and returns a PIL Image suitable for ControlNet.
    Includes explicit memory cleanup.
    """
    print(f"[Depth] Loading model from {model_path}...")
    
    # Run in a block so we can delete the session explicitly
    try:
        onnx_session = ort.InferenceSession(
            path_or_bytes=model_path,
            providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
        )

        # Preprocessing
        image = Image.open(image_path).convert("RGB")
        original_size = image.size
        
        # Resize to model input requirement (Depth Anything usually uses 518)
        image_input = image.resize(size=(518, 518))
        image_np = np.array(image_input, dtype=np.float32).transpose((2, 0, 1)) / 255.0
        image_np = np.expand_dims(image_np, axis=0)

        output_names = [o.name for o in onnx_session.get_outputs()]
        input_names = [i.name for i in onnx_session.get_inputs()]

        onnx_input_feed = {input_names[0]: image_np}

        print("[Depth] Running inference...")
        start_time = time.time()
        depth_estimate = onnx_session.run(
            output_names=output_names,
            input_feed=onnx_input_feed
        )
        inference_time = time.time() - start_time
        print(f"[Depth] Inference finished in {inference_time:.4f}s")

        # Post-processing
        depth_map = depth_estimate[0].squeeze() 
        depth_min = depth_map.min()
        depth_max = depth_map.max()
        normalized_map = (depth_map - depth_min) / (depth_max - depth_min)
        
        # Convert to PIL Image (Uint8)
        depth_pil = Image.fromarray((normalized_map * 255).astype(np.uint8)).convert("RGB")
        depth_pil = depth_pil.resize(original_size)
    
    finally:
        # CRITICAL: Clean up ONNX Runtime memory
        if 'onnx_session' in locals():
            del onnx_session
        gc.collect()
        torch.cuda.empty_cache()
        print("[Depth] Memory cleaned up.")

    return depth_pil

# ==========================================
# 2. View Synthesis Module (Diffusers)
# ==========================================
def run_view_synthesis(
    image_path: str,
    depth_image: Image.Image,
    output_path: str = "output_grid.png"
) -> Image.Image:
    """
    Runs Zero123++ with Depth ControlNet to generate novel views.
    Uses enable_model_cpu_offload for 8GB VRAM support.
    """
    print("[ViewSyn] Loading Diffusion Pipeline...")
    
    # Load ControlNet
    controlnet = ControlNetModel.from_pretrained(
        "sudo-ai/controlnet-zp11-depth-v1", 
        torch_dtype=torch.float16  # Changed to float16 for better compatibility
    )

    # Load Pipeline
    pipeline = DiffusionPipeline.from_pretrained(
        "sudo-ai/zero123plus-v1.1", 
        custom_pipeline="sudo-ai/zero123plus-pipeline",
        torch_dtype=torch.float16
    )
    
    pipeline.add_controlnet(controlnet, conditioning_scale=0.75)
    
    # Configure Scheduler
    pipeline.scheduler = EulerAncestralDiscreteScheduler.from_config(
        pipeline.scheduler.config, timestep_spacing='trailing'
    )
    
    # CRITICAL FIX: Do not use pipeline.to('cuda:0')
    # Use enable_model_cpu_offload() to fit in 8GB VRAM
    #print("[ViewSyn] Enabling model CPU offload...")
    #pipeline.enable_model_cpu_offload() 
    
    # Optional: Enable VAE tiling if you still OOM
    pipeline.enable_vae_tiling()

    cond_image = Image.open(image_path).convert("RGB")
    pipeline.to("cuda")

    print("[ViewSyn] Generating views...")
    
    # Run the pipeline
    # Note: enable_model_cpu_offload handles device placement automatically
    with torch.autocast(device_type="cuda", dtype=torch.float16):
         # Removed explicit SDPA context as Diffusers handles this by default now,
         # but you can keep it if you specifically need EFFICIENT_ATTENTION.
         # Ideally, rely on Diffusers default attention processors.
        result = pipeline(
            cond_image, 
            depth_image=depth_image, 
            num_inference_steps=36
        ).images[0]

    result.save(output_path)
    print(f"[ViewSyn] Grid saved to {output_path}")
    return result

# ==========================================
# 3. Grid Splitting Logic
# ==========================================
def split_grid_image(grid_image: Image.Image, save_dir: str):
    """
    Splits the Zero123++ 3x2 grid into 6 separate images.
    """
    os.makedirs(save_dir, exist_ok=True)
    width, height = grid_image.size
    
    # Zero123++ outputs 3 columns and 2 rows
    single_w = width // 3
    single_h = height // 2
    
    count = 0
    # Row major iteration
    for row in range(2):
        for col in range(3):
            left = col * single_w
            upper = row * single_h
            right = left + single_w
            lower = upper + single_h
            
            crop = grid_image.crop((left, upper, right, lower))
            save_path = os.path.join(save_dir, f"view_{count}.png")
            crop.save(save_path)
            print(f"Saved view {count} to {save_path}")
            count += 1

# ==========================================
# Main Execution
# ==========================================
if __name__ == "__main__":
    # CONFIGURATION
    INPUT_IMAGE = "assets/i4.jpg"
    ONNX_MODEL = "models/qualcomm-depth-anything-v2/model.onnx"
    OUTPUT_DIR = "outputs"
    
    # Optional: Set allocator config to reduce fragmentation
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. Generate Depth (and clean up immediately)
    depth_pil = run_depth_estimation(ONNX_MODEL, INPUT_IMAGE)
    depth_pil.save(f"{OUTPUT_DIR}/debug_depth.png")

    # 2. Generate View Grid
    grid_result = run_view_synthesis(
        INPUT_IMAGE, 
        depth_pil, 
        output_path=f"{OUTPUT_DIR}/combined_grid.png"
    )

    # 3. Split Views
    print("[Post-Process] Splitting grid into individual views...")
    split_grid_image(grid_result, save_dir=f"{OUTPUT_DIR}/views")