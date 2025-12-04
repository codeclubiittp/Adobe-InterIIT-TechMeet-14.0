import time
import cv2
import numpy as np
import onnxruntime as ort
import torch
from PIL import Image
from typing import List, Dict, Any, Tuple
import matplotlib.pyplot as plt

# --- TFLite/ONNX Imports ---
import tflite_runtime.interpreter as interpreter
from pprint import pprint

# --- Diffusers Imports ---
from diffusers import StableDiffusionControlNetInpaintPipeline, ControlNetModel, UniPCMultistepScheduler
from diffusers.utils import load_image

# --- Profiler (Assuming utils.profiler and utils.onnx exist) ---
class profiler:
    def __init__(self):
        self.start_time = None
    def begin(self):
        self.start_time = time.perf_counter()
    def end(self, name=""):
        if self.start_time is None:
            print(f"Profiler error: 'begin()' was not called for '{name}'")
            return
        elapsed = time.perf_counter() - self.start_time
        print(f"[{name}] Elapsed Time: {elapsed:.4f}s")
        self.start_time = None

def log_session_details(session, other={}):
    print(f"--- ONNX Session Details ({other.get('SessionType', 'N/A')}) ---")
    print(f"Providers: {session.get_providers()}")

pf = profiler()
CACHE_DIR = ".hf/"


# --- Function 1: Grounding DINO Object Detection (Unchanged) ---
def object_detection_gdino_tiny(
    image_path: str,
    text: str,
    box_threshold: float = 0.4
) -> Tuple[List[Dict[str, Any]], Tuple[int, int]]:
    
    from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
    
    model_id = "IDEA-Research/grounding-dino-tiny"
    device = "cpu"
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)
    image = Image.open(image_path).convert("RGB")
    original_size = image.size # (width, height)

    print(f"[GDINO] Processing image and text: '{text}'")
    pf.begin()
    inputs = processor(images=image, text=text, return_tensors="pt").to(device)
    pf.end("GDINO Pre-process")
    pf.begin()
    with torch.no_grad():
        outputs = model(**inputs)
    pf.end("GDINO Inference")

    target_sizes = torch.tensor([image.size[::-1]], dtype=torch.float32)
    outputs.logits = outputs.logits.cpu()
    outputs.pred_boxes = outputs.pred_boxes.cpu()
    pf.begin()
    results = processor.post_process_grounded_object_detection(
        outputs, input_ids=inputs["input_ids"].cpu(), threshold=0.0, target_sizes=target_sizes
    )
    pf.end("GDINO Post-process")

    filtered_results = []
    for res in results:
        scores, boxes, labels = res["scores"], res["boxes"], res["text_labels"]
        keep = scores > box_threshold
        filtered_results.append({
            "boxes": boxes[keep],
            "scores": scores[keep],
            "labels": [labels[i] for i in torch.nonzero(keep).flatten()],
        })
    for res in filtered_results:
        for label, score, box in zip(res["labels"], res["scores"], res["boxes"]):
            print(f"[GDINO] Found: {label}: {score:.2f} -> {box.tolist()}")

    return filtered_results, original_size


# --- Function 2: MobileSAM Segmentation (Unchanged) ---
def run_mobilesam_segmentation(
    encoder_path: str,
    decoder_path: str,
    image_path: str,
    original_w: int,
    original_h: int,
    point_coords_scaled: list,
    point_labels: list,
) -> np.ndarray:
    
    tfi_sam_encoder = interpreter.Interpreter(model_path=encoder_path)
    tfi_sam_decoder = interpreter.Interpreter(model_path=decoder_path)
    tfi_sam_encoder.allocate_tensors()
    tfi_sam_decoder.allocate_tensors()

    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_resized = cv2.resize(image_rgb, (1024, 1024))
    image_np = np.array(image_resized, dtype=np.float32) / 255.0
    image_np = np.expand_dims(image_np, axis=0) 

    coords = np.array(point_coords_scaled, dtype=np.float32).reshape(1, 2, 2)
    labels = np.array(point_labels, dtype=np.float32).reshape(1, 2)

    print(f"[MobileSAM] Coords: {coords} (Shape: {coords.shape})")
    print(f"[MobileSAM] Labels: {labels} (Shape: {labels.shape})")

    encoder_inputs = tfi_sam_encoder.get_input_details()
    tfi_sam_encoder.set_tensor(encoder_inputs[0]['index'], image_np)
    tfi_sam_encoder.invoke()
    encoder_outputs = tfi_sam_encoder.get_output_details()
    image_embeddings = tfi_sam_encoder.get_tensor(encoder_outputs[0]['index'])

    decoder_inputs = tfi_sam_decoder.get_input_details()
    tfi_sam_decoder.set_tensor(decoder_inputs[0]['index'], image_embeddings)
    tfi_sam_decoder.set_tensor(decoder_inputs[1]['index'], coords)
    tfi_sam_decoder.set_tensor(decoder_inputs[2]['index'], labels)
    tfi_sam_decoder.invoke()

    decoder_outputs = tfi_sam_decoder.get_output_details()
    final_mask = tfi_sam_decoder.get_tensor(decoder_outputs[0]['index'])

    mask_256 = final_mask[0, :, :, 0]
    mask_1024 = cv2.resize(mask_256, (1024, 1024), interpolation=cv2.INTER_LINEAR)
    mask_original_size = cv2.resize(mask_1024, (original_w, original_h), interpolation=cv2.INTER_LINEAR)
    
    binary_mask = (mask_original_size > 0.50)
    binary_mask_uint8 = binary_mask.astype(np.uint8) * 255
    
    print("[MobileSAM] Segmentation complete.")
    return binary_mask_uint8


# --- Function 3: Compositing (Unchanged) ---
def paste_object_with_matte(
    background_rgb: np.ndarray, 
    cutout_rgb: np.ndarray, 
    matte_8bit: np.ndarray, 
    position: Tuple[int, int]
) -> np.ndarray:
    
    x_paste, y_paste = position
    h_cut, w_cut, _ = cutout_rgb.shape

    # Handle clipping
    x_cut_start, y_cut_start = 0, 0
    if x_paste < 0: x_cut_start = -x_paste; w_cut -= x_cut_start; x_paste = 0
    if y_paste < 0: y_cut_start = -y_paste; h_cut -= y_cut_start; y_paste = 0
    if x_paste + w_cut > background_rgb.shape[1]: w_cut = background_rgb.shape[1] - x_paste
    if y_paste + h_cut > background_rgb.shape[0]: h_cut = background_rgb.shape[0] - y_paste

    cutout_rgb = cutout_rgb[y_cut_start:y_cut_start+h_cut, x_cut_start:x_cut_start+w_cut]
    matte_8bit = matte_8bit[y_cut_start:y_cut_start+h_cut, x_cut_start:x_cut_start+w_cut]

    if h_cut <= 0 or w_cut <= 0: return background_rgb

    bg_slice = background_rgb[y_paste : y_paste + h_cut, x_paste : x_paste + w_cut]
    matte_norm = np.expand_dims(matte_8bit.astype(np.float32) / 255.0, axis=-1)
    
    composited_slice = (cutout_rgb * matte_norm) + (bg_slice * (1 - matte_norm))
    composited_slice = composited_slice.astype(np.uint8)
    
    final_image = background_rgb.copy()
    final_image[y_paste : y_paste + h_cut, x_paste : x_paste + w_cut] = composited_slice
    
    print(f"[Paste] Pasted object at {position}.")
    return final_image

# --- Function 4: Depth Estimation (Unchanged) ---
def depth_estimate(
    model_path: str,
    source_image_pil: Image.Image
) -> np.ndarray:

    onnx_session = ort.InferenceSession(path_or_bytes=model_path)
    image = source_image_pil.resize(size=(518, 518))
    image_np = np.array(image, dtype=np.float32).transpose((2, 0, 1)) / 255.0
    image_np = np.expand_dims(image_np, axis=0)

    output_names = [o.name for o in onnx_session.get_outputs()]
    input_names = [i.name for i in onnx_session.get_inputs()]
    onnx_input_feed = {input_names[0]: image_np}

    pf.begin()
    depth_estimate_output = onnx_session.run(
        output_names=output_names, input_feed=onnx_input_feed
    )
    pf.end("Depth Estimation Time")

    depth_map = depth_estimate_output[0].squeeze()
    depth_min = depth_map.min()
    depth_max = depth_map.max()
    normalized_map = (depth_map - depth_min) / (depth_max - depth_min)
    
    return normalized_map

# --- Function 5: Upscaling (Unchanged) ---
def upscale_simple_lanczos(image: Image.Image, scale_factor: int = 2) -> Image.Image:
    print(f"Upscaling with Lanczos by {scale_factor}x...")
    new_size = (image.width * scale_factor, image.height * scale_factor)
    return image.resize(new_size, Image.LANCZOS)


# --- *** NEW/MODIFIED DIFFUSION FUNCTIONS *** ---

def load_controlnet_pipeline(
    controlnet_depth_path: str, 
    controlnet_canny_path: str
) -> StableDiffusionControlNetInpaintPipeline:
    """
    Loads and returns the ControlNet Inpainting pipeline.
    """
    print("Loading ControlNet & Inpainting Pipeline...")
    pf.begin()
    controlnet_depth = ControlNetModel.from_pretrained(
        controlnet_depth_path, torch_dtype=torch.float16, cache_dir=CACHE_DIR
    )
    controlnet_canny = ControlNetModel.from_pretrained(
        controlnet_canny_path, torch_dtype=torch.float16, cache_dir=CACHE_DIR
    )

    pipe = StableDiffusionControlNetInpaintPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5", 
        controlnet=[controlnet_depth, controlnet_canny], 
        safety_checker=None, 
        torch_dtype=torch.float16,
        cache_dir=CACHE_DIR
    )
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.to("cuda")
    pf.end("Diffusion Pipeline Load Time")
    return pipe

def run_diffusion_inpainting(
    pipe: StableDiffusionControlNetInpaintPipeline,
    original_image: np.ndarray, # Original image (RGB)
    object_mask: np.ndarray,  # 8-bit full-size mask of object to remove
    depth_model_path: str,
    prompt: str,
    negative_prompt: str
) -> np.ndarray:
    """
    FIRST diffusion pass: Inpaints the background where the object was.
    """
    print("\n--- Starting Diffusion Inpainting (Pass 1) ---")
    
    # --- A. Prepare Input Images ---
    base_pil = Image.fromarray(original_image).resize((512, 512))
    
    # 1. ControlNet: Depth Map (from original image)
    depth_map_array = depth_estimate(depth_model_path, base_pil)
    image_scaled = (depth_map_array * 255).astype(np.uint8)
    control_image_depth = Image.fromarray(np.stack([image_scaled]*3, axis=-1))
    control_image_depth.save("outputs/rearrange_v5_02_inpainting_depth.jpg")
    print("Generated Inpainting Depth map.")

    # 2. ControlNet: Canny Map (from original image)
    image_cv = cv2.cvtColor(np.array(base_pil), cv2.COLOR_RGB2BGR)
    canny_edges = cv2.Canny(image_cv, 100, 200)
    control_image_canny = Image.fromarray(cv2.cvtColor(canny_edges, cv2.COLOR_GRAY2BGR))
    control_image_canny.save("outputs/rearrange_v5_03_inpainting_canny.jpg")
    print("Generated Inpainting Canny map.")
    
    # 3. Inpainting Mask (from original SAM mask)
    # Resize mask to 512x512
    mask_512 = cv2.resize(object_mask, (512, 512), interpolation=cv2.INTER_NEAREST)
    # Dilate mask to give model blending room
    kernel = np.ones((25, 25), np.uint8)
    dilated_mask = cv2.dilate(mask_512, kernel, iterations=1)
    
    mask_image_pil = Image.fromarray(dilated_mask)
    mask_image_pil.save("outputs/rearrange_v5_04_inpainting_mask.jpg")
    print("Generated Inpainting mask.")
    
    # --- B. Run Inpainting ---
    print("Running diffusion inpainting...")
    controlnet_scales = [0.8, 0.5]
    generator = torch.Generator(device="cuda").manual_seed(1234)

    pf.begin()
    with torch.inference_mode():
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            inpainted_image_pil = pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                image=base_pil,
                mask_image=mask_image_pil, 
                control_image=[control_image_depth, control_image_canny],
                num_inference_steps=30,
                guidance_scale=7.5,
                controlnet_conditioning_scale=controlnet_scales,
                generator=generator,
                strength=1.0 # Strength 1.0 = fully replace masked area
            ).images[0]
    pf.end("Diffusion Inpainting Time (Pass 1)")
    
    # Resize back to original dimensions
    inpainted_image_pil = inpainted_image_pil.resize((original_image.shape[1], original_image.shape[0]), Image.LANCZOS)
    return np.array(inpainted_image_pil)


def run_diffusion_harmonization(
    pipe: StableDiffusionControlNetInpaintPipeline,
    base_composite_image: np.ndarray, # The "fake" pasted image (RGB)
    object_mask_cropped: np.ndarray,  # The 8-bit cropped mask of the object
    paste_position: Tuple[int, int],
    depth_model_path: str,
    prompt: str,
    negative_prompt: str
) -> Image.Image:
    """
    SECOND diffusion pass: Harmonizes the newly pasted object.
    """
    print("\n--- Starting Diffusion Harmonization (Pass 2) ---")
    
    # --- A. Prepare Input Images ---
    base_pil = Image.fromarray(base_composite_image).resize((512, 512))
    
    # 1. ControlNet: Depth Map (from composite image)
    depth_map_array = depth_estimate(depth_model_path, base_pil)
    image_scaled = (depth_map_array * 255).astype(np.uint8)
    control_image_depth = Image.fromarray(np.stack([image_scaled]*3, axis=-1))
    control_image_depth.save("outputs/rearrange_v5_06_harmonize_depth.jpg")
    print("Generated Harmonize Depth map.")

    # 2. ControlNet: Canny Map (from composite image)
    image_cv = cv2.cvtColor(np.array(base_pil), cv2.COLOR_RGB2BGR)
    canny_edges = cv2.Canny(image_cv, 100, 200)
    control_image_canny = Image.fromarray(cv2.cvtColor(canny_edges, cv2.COLOR_GRAY2BGR))
    control_image_canny.save("outputs/rearrange_v5_07_harmonize_canny.jpg")
    print("Generated Harmonize Canny map.")
    
    # 3. Harmonization Mask (from pasted object)
    x_paste, y_paste = paste_position
    x_paste_512 = int(x_paste * (512 / base_composite_image.shape[1]))
    y_paste_512 = int(y_paste * (512 / base_composite_image.shape[0]))
    h_orig, w_orig = object_mask_cropped.shape[:2]
    h_512 = int(h_orig * (512 / base_composite_image.shape[0]))
    w_512 = int(w_orig * (512 / base_composite_image.shape[1]))
    
    mask_512_cropped = cv2.resize(object_mask_cropped, (w_512, h_512), interpolation=cv2.INTER_NEAREST)
    inpainting_mask = np.zeros((512, 512), dtype=np.uint8)
    
    y1, y2 = y_paste_512, y_paste_512 + h_512
    x1, x2 = x_paste_512, x_paste_512 + w_512
    y1c, y2c = max(0, y1), min(512, y2)
    x1c, x2c = max(0, x1), min(512, x2)
    my1, my2 = max(0, -y1), h_512 - max(0, y2 - 512)
    mx1, mx2 = max(0, -x1), w_512 - max(0, x2 - 512)
    inpainting_mask[y1c:y2c, x1c:x2c] = mask_512_cropped[my1:my2, mx1:mx2]
    
    kernel = np.ones((25, 25), np.uint8)
    dilated_mask = cv2.dilate(inpainting_mask, kernel, iterations=1)
    
    mask_image_pil = Image.fromarray(dilated_mask)
    mask_image_pil.save("outputs/rearrange_v5_08_harmonize_mask.jpg")
    print("Generated Harmonize mask.")
    
    # --- B. Run Harmonization ---
    print("Running diffusion harmonization...")
    controlnet_scales = [0.8, 0.5]
    generator = torch.Generator(device="cuda").manual_seed(1234)

    pf.begin()
    with torch.inference_mode():
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            harmonized_image = pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                image=base_pil,
                mask_image=mask_image_pil, 
                control_image=[control_image_depth, control_image_canny],
                num_inference_steps=30,
                guidance_scale=7.5,
                controlnet_conditioning_scale=controlnet_scales,
                generator=generator,
                strength=0.9 # Strength 0.9 = Harmonize, respect base image
            ).images[0]
    pf.end("Diffusion Harmonization Time (Pass 2)")
    
    return harmonized_image


# --- Main Execution Pipeline ---

if __name__ == "__main__":
    # --- CONFIGURATION ---
    IMAGE_PATH = "assets/i3.jpg"
    SAM_ENCODER_PATH = "models/MobileSam_MobileSAMEncoder_float.tflite"
    SAM_DECODER_PATH = "models/MobileSam_MobileSAMDecoder_float.tflite"
    DEPTH_MODEL_PATH = "models/qualcomm-depth-anything-v2/model.onnx"
    
    CONTROLNET_DEPTH_PATH = "lllyasviel/sd-controlnet-depth"
    CONTROLNET_CANNY_PATH = "lllyasviel/sd-controlnet-canny"
    
    OBJECT_TO_MOVE = "pillow."
    NEW_POSITION = (50, 4000)
    UPSCALE_FACTOR = 2
    
    # *** Prompts for TWO diffusion passes ***
    BACKGROUND_INPAINT_PROMPT = "a living room with a couch and windows, cowhide rug, photorealistic, 4k"
    HARMONIZATION_PROMPT = "make the image look good and blend everything with it"
    NEGATIVE_PROMPT = "blurry, low quality, unrealistic, cartoon, disfigured, watermark, text"
    
    import os
    os.makedirs("outputs", exist_ok=True)
    
    # --- Load original image ---
    original_image_cv = cv2.imread(IMAGE_PATH)
    original_image_rgb = cv2.cvtColor(original_image_cv, cv2.COLOR_BGR2RGB)
    (orig_h, orig_w) = original_image_rgb.shape[:2]

    # --- STEP 1: Detect
    print("--- Step 1: Detecting Object ---")
    detection_results, _ = object_detection_gdino_tiny(
        image_path=IMAGE_PATH, text=OBJECT_TO_MOVE
    )
    if not detection_results or not detection_results[0]["boxes"].numel():
        print(f"Error: Could not find '{OBJECT_TO_MOVE}'. Exiting."); exit()
    bbox = detection_results[0]["boxes"][0].tolist()
    print(f"Object '{OBJECT_TO_MOVE}' found at {[int(v) for v in bbox]}")

    x0, y0, x1, y1 = bbox
    center_x_scaled = ((x0 + x1) / 2 / orig_w) * 1024
    center_y_scaled = ((y0 + y1) / 2 / orig_h) * 1024
    sam_point_coords = [[center_x_scaled, center_y_scaled], [0.0, 0.0]]
    sam_point_labels = [1, -1]

    # --- STEP 2: Segment
    print("\n--- Step 2: Segmenting Object (MobileSAM) ---")
    full_mask_8bit = run_mobilesam_segmentation(
        encoder_path=SAM_ENCODER_PATH, decoder_path=SAM_DECODER_PATH,
        image_path=IMAGE_PATH, original_w=orig_w, original_h=orig_h,
        point_coords_scaled=sam_point_coords, point_labels=sam_point_labels
    )
    cv2.imwrite("outputs/rearrange_v5_01_sam_mask.png", full_mask_8bit)

    # --- STEP 3: Load Diffusion Pipeline (one time) ---
    controlnet_pipe = load_controlnet_pipeline(CONTROLNET_DEPTH_PATH, CONTROLNET_CANNY_PATH)

    # --- STEP 4: Diffusion Inpaint Background (Pass 1) ---
    inpainted_background_rgb = run_diffusion_inpainting(
        pipe=controlnet_pipe,
        original_image=original_image_rgb,
        object_mask=full_mask_8bit,
        depth_model_path=DEPTH_MODEL_PATH,
        prompt=BACKGROUND_INPAINT_PROMPT,
        negative_prompt=NEGATIVE_PROMPT
    )
    cv2.imwrite("outputs/rearrange_v5_05_diffused_bg.jpg", cv2.cvtColor(inpainted_background_rgb, cv2.COLOR_RGB2BGR))

    # --- STEP 5: Crop Cutout & Paste (Initial Composite) ---
    print("\n--- Step 5: Creating Initial 'Fake' Composite ---")
    contours, _ = cv2.findContours(full_mask_8bit, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours: print("Error: No contours in SAM mask."); exit()
    mx, my, mw, mh = cv2.boundingRect(contours[0])
    
    cutout_rgb = original_image_rgb[my : my + mh, mx : mx + mw]
    cutout_matte = full_mask_8bit[my : my + mh, mx : mx + mw]
    
    initial_composite_rgb = paste_object_with_matte(
        background_rgb=inpainted_background_rgb,
        cutout_rgb=cutout_rgb,
        matte_8bit=cutout_matte,
        position=NEW_POSITION
    )
    cv2.imwrite("outputs/rearrange_v5_09_initial_composite.jpg", cv2.cvtColor(initial_composite_rgb, cv2.COLOR_RGB2BGR))
    
    # --- STEP 6: Harmonize with Diffusion (Pass 2) ---
    harmonized_image_pil = run_diffusion_harmonization(
        pipe=controlnet_pipe,
        base_composite_image=initial_composite_rgb,
        object_mask_cropped=cutout_matte,
        paste_position=NEW_POSITION,
        depth_model_path=DEPTH_MODEL_PATH,
        prompt=HARMONIZATION_PROMPT,
        negative_prompt=NEGATIVE_PROMPT
    )
    harmonized_image_pil.save("outputs/rearrange_v5_10_harmonized_base.jpg")

    # --- STEP 7: Upscale
    print(f"\n--- Step 7: Upscaling Final Image ---")
    pf.begin()
    final_upscaled_image = upscale_simple_lanczos(harmonized_image_pil, scale_factor=UPSCALE_FACTOR)
    pf.end(f"Lanczos Upscaling x{UPSCALE_FACTOR}")
    
    final_upscaled_image.save('outputs/rearrange_v5_11_final_upscaled.jpg')

    # --- STEP 8: Show Final Comparison ---
    print("\n--- V5 Pipeline Complete ---")
    plt.figure(figsize=(24, 8))
    
    plt.subplot(1, 4, 1)
    plt.imshow(original_image_rgb)
    plt.title("1. Original")
    plt.axis("off")

    plt.subplot(1, 4, 2)
    plt.imshow(inpainted_background_rgb)
    plt.title("2. Diffused BG (Pass 1)")
    plt.axis("off")

    plt.subplot(1, 4, 3)
    plt.imshow(initial_composite_rgb)
    plt.title("3. 'Fake' Composite")
    plt.axis("off")

    plt.subplot(1, 4, 4)
    plt.imshow(final_upscaled_image)
    plt.title("4. Harmonized (Pass 2)")
    plt.axis("off")

    plt.tight_layout()
    plt.savefig("outputs/rearrange_v5_12_comparison.png")
    print("Saved final comparison plot.")