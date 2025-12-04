import time
import cv2
import numpy as np
import onnxruntime as ort
import torch
from PIL import Image, ImageDraw, ImageFont
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
import matplotlib.pyplot as plt
from typing import List, Dict, Any, Tuple

# --- NEW IMPORTS ---
import tflite_runtime.interpreter as interpreter
from pprint import pprint

# --- Function 1: Object Detection (Unchanged) ---
def object_detection_gdino_tiny(
    image_path: str,
    text: str,
    box_threshold: float = 0.4
) -> Tuple[List[Dict[str, Any]], Tuple[int, int]]:
    """
    Detects objects in an image based on a text prompt.
    Returns a list of filtered results and the original image size.
    """
    model_id = "IDEA-Research/grounding-dino-tiny"
    device = "cpu"

    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)

    image = Image.open(image_path).convert("RGB")
    original_size = image.size # (width, height)

    print(f"[GDINO] Processing image and text: '{text}'")
    begin = time.perf_counter()
    inputs = processor(images=image, text=text, return_tensors="pt").to(device)
    end = time.perf_counter()
    print(f"[GDINO] Image Processing Time: {(end - begin):.4f}s")

    begin = time.perf_counter()
    with torch.no_grad():
        outputs = model(**inputs)
    end = time.perf_counter()
    print(f"[GDINO] Inference Time: {(end - begin):.4f}s")

    target_sizes = torch.tensor([image.size[::-1]], dtype=torch.float32)
    outputs.logits = outputs.logits.cpu()
    outputs.pred_boxes = outputs.pred_boxes.cpu()

    begin = time.perf_counter()
    results = processor.post_process_grounded_object_detection(
        outputs,
        input_ids=inputs["input_ids"].cpu(),
        threshold=0.0,
        target_sizes=target_sizes
    )
    end = time.perf_counter()
    print(f"[GDINO] Post Processing Time: {(end - begin):.4f}s")

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


# --- MODIFIED Function 2: MobileSAM Segmentation ---
# (Integrated your function and modified to return the mask)
def run_mobilesam_segmentation(
    encoder_path: str,
    decoder_path: str,
    image_path: str,
    original_w: int,
    original_h: int,
    point_coords_scaled: list,
    point_labels: list,
) -> np.ndarray:
    """
    Runs MobileSAM segmentation using a scaled point and returns
    the final 8-bit mask resized to the original image dimensions.
    """
    tfi_sam_encoder = interpreter.Interpreter(model_path=encoder_path)
    tfi_sam_decoder = interpreter.Interpreter(model_path=decoder_path)

    #(Optional: uncomment to debug shapes)
    # print("--- SAM Encoder ---")
    # pprint(tfi_sam_encoder.get_input_details())
    # print("--- SAM Decoder ---")
    # pprint(tfi_sam_decoder.get_input_details())

    tfi_sam_encoder.allocate_tensors()
    tfi_sam_decoder.allocate_tensors()

    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_resized = cv2.resize(image_rgb, (1024, 1024))
    image_np = np.array(image_resized, dtype=np.float32) / 255.0
    image_np = np.expand_dims(image_np, axis=0) 

    # Prepare points and labels
    # Expects point_coords_scaled = [[x, y]]
    # Expects point_labels = [1]
    coords = np.array(point_coords_scaled, dtype=np.float32).reshape(1, 2, 2)
    
    # point_labels is now a list of 2 labels, shape (2,)
    # We reshape it to (1, 2)
    labels = np.array(point_labels, dtype=np.float32).reshape(1, 2)

    print(f"[MobileSAM] Coords: {coords} (Shape: {coords.shape})")
    print(f"[MobileSAM] Labels: {labels} (Shape: {labels.shape})")

    # Run Encoder
    encoder_inputs = tfi_sam_encoder.get_input_details()
    tfi_sam_encoder.set_tensor(encoder_inputs[0]['index'], image_np)
    tfi_sam_encoder.invoke()
    encoder_outputs = tfi_sam_encoder.get_output_details()
    image_embeddings = tfi_sam_encoder.get_tensor(encoder_outputs[0]['index'])

    # Run Decoder
    decoder_inputs = tfi_sam_decoder.get_input_details()
    tfi_sam_decoder.set_tensor(decoder_inputs[0]['index'], image_embeddings)
    tfi_sam_decoder.set_tensor(decoder_inputs[1]['index'], coords)
    tfi_sam_decoder.set_tensor(decoder_inputs[2]['index'], labels)
    tfi_sam_decoder.invoke()

    decoder_outputs = tfi_sam_decoder.get_output_details()
    final_mask = tfi_sam_decoder.get_tensor(decoder_outputs[0]['index'])
    # final_score = tfi_sam_decoder.get_tensor(decoder_outputs[1]['index'])
    # print(f"[MobileSAM] Score: {final_score[0][0]}")

    # Upscale mask from 256x256 to 1024x1024
    mask_256 = final_mask[0, :, :, 0]
    mask_1024 = cv2.resize(
        mask_256,
        (1024, 1024),  
        interpolation=cv2.INTER_LINEAR
    )
    
    # Resize mask from 1024x1024 to original image size
    mask_original_size = cv2.resize(
        mask_1024,
        (original_w, original_h),
        interpolation=cv2.INTER_LINEAR
    )

    # Binarize and convert to 8-bit
    binary_mask = (mask_original_size > 0.50) # Use 0.5 as threshold
    binary_mask_uint8 = binary_mask.astype(np.uint8) * 255
    
    print("[MobileSAM] Segmentation complete.")
    
    return binary_mask_uint8


# --- MODIFIED Function 3: Inpainting (Unchanged) ---
def run_lama_inpainting(
    model_path: str,
    image_path: str,
    mask_array: np.ndarray  # <-- CHANGED
) -> np.ndarray:
    """
    Inpaints an image using a provided mask array.
    Returns the final inpainted image as an RGB numpy array.
    """
    onnx_session = ort.InferenceSession(
        model_path,
        providers=["CPUExecutionProvider"]
    )
    
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    orig_h, orig_w = image.shape[:2]

    # --- Use provided mask_array ---
    if mask_array.ndim == 3:
        mask = cv2.cvtColor(mask_array, cv2.COLOR_BGR2GRAY)
    else:
        mask = mask_array
    _, mask = cv2.threshold(mask, 127, 1, cv2.THRESH_BINARY)
    # ---

    image_resized = cv2.resize(image, (512, 512))
    mask_resized = cv2.resize(mask, (512, 512)) 
    
    image_norm = image_resized.astype(np.float32) / 255.0
    mask_broadcastable = np.expand_dims(mask_resized, axis=-1)
    image_masked = image_norm * (1 - mask_broadcastable)
    image_onnx = np.transpose(image_masked, (2, 0, 1))
    image_onnx = np.expand_dims(image_onnx, axis=0).astype(np.float32)

    mask_onnx = np.expand_dims(mask_resized, axis=0)
    mask_onnx = np.expand_dims(mask_onnx, axis=0).astype(np.float32)

    input_names = [input.name for input in onnx_session.get_inputs()]
    output_names = [output.name for output in onnx_session.get_outputs()]

    inputs = {
        input_names[0]: image_onnx,
        input_names[1]: mask_onnx
    }
    
    print("[LaMa] Running inpainting inference...")
    begin = time.perf_counter()
    painted_image_onnx = onnx_session.run(output_names, inputs)[0]
    end = time.perf_counter()
    print(f"[LaMa] Inference Time: {(end - begin):.4f}s")

    painted_image_norm = np.squeeze(painted_image_onnx, axis=0)
    painted_image_norm = np.transpose(painted_image_norm, (1, 2, 0))
    painted_image_norm = np.clip(painted_image_norm, 0, 1)
    painted_image_denorm = (painted_image_norm * 255.0).astype(np.uint8) 
    
    final_image_resized = cv2.resize(painted_image_denorm, (orig_w, orig_h), interpolation=cv2.INTER_CUBIC)
    
    print("[LaMa] Inpainting complete.")
    return final_image_resized


# --- Function 4: Compositing (Unchanged) ---
def paste_object_with_matte(
    background_rgb: np.ndarray, 
    cutout_rgb: np.ndarray, 
    matte_8bit: np.ndarray, 
    position: Tuple[int, int]
) -> np.ndarray:
    """
    Pastes a matted cutout onto a background image at a new position.
    """
    x_paste, y_paste = position
    h_cut, w_cut, _ = cutout_rgb.shape

    # --- Handle Edge Cases ---
    # Store original paste coords for later
    x_paste_orig, y_paste_orig = x_paste, y_paste
    
    # Store original cutout shape
    h_cut_orig, w_cut_orig = cutout_rgb.shape[:2]

    # Check if paste is completely off-screen
    if x_paste >= background_rgb.shape[1] or y_paste >= background_rgb.shape[0]:
        print("[Paste] Error: Paste position is completely off-screen.")
        return background_rgb
    if x_paste + w_cut <= 0 or y_paste + h_cut <= 0:
        print("[Paste] Error: Paste position is completely off-screen.")
        return background_rgb

    # Clip cutout if it starts off-screen
    x_cut_start, y_cut_start = 0, 0
    if x_paste < 0:
        x_cut_start = -x_paste
        w_cut -= x_cut_start
        x_paste = 0
    if y_paste < 0:
        y_cut_start = -y_paste
        h_cut -= y_cut_start
        y_paste = 0

    cutout_rgb = cutout_rgb[y_cut_start:y_cut_start+h_cut, x_cut_start:x_cut_start+w_cut]
    matte_8bit = matte_8bit[y_cut_start:y_cut_start+h_cut, x_cut_start:x_cut_start+w_cut]

    # Clip cutout if it ends off-screen
    if x_paste + w_cut > background_rgb.shape[1]:
        w_cut = background_rgb.shape[1] - x_paste
        cutout_rgb = cutout_rgb[:, :w_cut]
        matte_8bit = matte_8bit[:, :w_cut]
    if y_paste + h_cut > background_rgb.shape[0]:
        h_cut = background_rgb.shape[0] - y_paste
        cutout_rgb = cutout_rgb[:h_cut, :]
        matte_8bit = matte_8bit[:h_cut, :]
    # --- End Edge Cases ---

    if h_cut <= 0 or w_cut <= 0:
        print("[Paste] Error: Paste resulted in zero-sized object.")
        return background_rgb

    bg_slice = background_rgb[y_paste : y_paste + h_cut, x_paste : x_paste + w_cut]
    
    matte_norm = np.expand_dims(matte_8bit.astype(np.float32) / 255.0, axis=-1)
    
    composited_slice = (cutout_rgb * matte_norm) + (bg_slice * (1 - matte_norm))
    composited_slice = composited_slice.astype(np.uint8)
    
    final_image = background_rgb.copy()
    final_image[y_paste : y_paste + h_cut, x_paste : x_paste + w_cut] = composited_slice
    
    print(f"[Paste] Pasted object at {position} (final coords: {(x_paste, y_paste)}).")
    return final_image


# --- NEW Function 5: Fast Shadow Creation ---
def create_and_blend_shadow(
    background_rgb: np.ndarray,
    cutout_matte: np.ndarray,
    cutout_rgb_shape: Tuple[int, int], # (h_cut, w_cut)
    paste_position: Tuple[int, int]
) -> np.ndarray:
    """
    Creates a fake shadow and blends it onto the background.
    """
    print("[ShadowFX] Creating fast shadow...")
    x_paste, y_paste = paste_position
    h_cut, w_cut = cutout_rgb_shape
    bg_h, bg_w = background_rgb.shape[:2]

    # 1. Create the shadow shape from the matte
    # Use a heavy blur. Kernel size must be odd.
    shadow_matte = cv2.GaussianBlur(cutout_matte, (41, 41), 0)

    # 2. Define a simple shear/skew transformation
    # This will "flatten" the shadow
    shear_factor = 0.5
    M = np.array([
        [1, shear_factor, 0],
        [0, 1, 0]
    ], dtype=np.float32)

    # 3. Apply the transformation
    # We need a wider canvas for the sheared shadow
    shadow_transformed = cv2.warpAffine(shadow_matte, M, (int(w_cut + h_cut * shear_factor), h_cut))
    
    # 4. Create the shadow "color" (just a grayscale intensity)
    # Convert to float [0, 1] and make it dark (e.g., max 0.6 opacity)
    shadow_strength = 0.65 # How dark the shadow is
    shadow_float = (shadow_transformed.astype(np.float32) / 255.0) * shadow_strength
    
    # Invert it: 1.0 = no shadow, (1.0 - shadow_strength) = full shadow
    # This is our multiplier
    shadow_multiplier = 1.0 - shadow_float
    # Add a 3rd dimension for broadcasting
    shadow_multiplier = np.expand_dims(shadow_multiplier, axis=-1)
    
    shadow_h, shadow_w = shadow_multiplier.shape[:2]

    # 5. Calculate shadow paste position
    # We want to align the *bottom-left* of the shadow with the *bottom-left* of the object paste
    y_shadow_paste = y_paste + h_cut - shadow_h 
    x_shadow_paste = x_paste # Align left edges

    # 6. Handle Edge Cases (Clipping)
    x_shadow_start_in_bg = max(0, x_shadow_paste)
    y_shadow_start_in_bg = max(0, y_shadow_paste)
    
    x_shadow_end_in_bg = min(bg_w, x_shadow_paste + shadow_w)
    y_shadow_end_in_bg = min(bg_h, y_shadow_paste + shadow_h)
    
    # Calculate shadow slice (what part of shadow to use)
    x_shadow_start_in_shadow = max(0, -x_shadow_paste)
    y_shadow_start_in_shadow = max(0, -y_shadow_paste)
    
    x_shadow_end_in_shadow = x_shadow_start_in_shadow + (x_shadow_end_in_bg - x_shadow_start_in_bg)
    y_shadow_end_in_shadow = y_shadow_start_in_shadow + (y_shadow_end_in_bg - y_shadow_start_in_bg)

    # Get the slice of the shadow
    shadow_multiplier_clipped = shadow_multiplier[y_shadow_start_in_shadow : y_shadow_end_in_shadow, x_shadow_start_in_shadow : x_shadow_end_in_shadow]

    if shadow_multiplier_clipped.size == 0:
        print("[ShadowFX] Warning: Shadow is completely off-screen.")
        return background_rgb

    # 7. Apply the shadow (Multiply blend)
    bg_slice = background_rgb[y_shadow_start_in_bg : y_shadow_end_in_bg, x_shadow_start_in_bg : x_shadow_end_in_bg]

    # Convert bg to float, apply multiply, convert back
    bg_slice_float = bg_slice.astype(np.float32)
    blended_slice = bg_slice_float * shadow_multiplier_clipped
    blended_slice = blended_slice.astype(np.uint8)

    # 8. Place the blended slice back onto a copy of the background
    background_with_shadow = background_rgb.copy()
    background_with_shadow[y_shadow_start_in_bg : y_shadow_end_in_bg, x_shadow_start_in_bg : x_shadow_end_in_bg] = blended_slice
    
    print("[ShadowFX] Fast shadow applied.")
    return background_with_shadow


# --- Main Execution Pipeline ---

if __name__ == "__main__":
    # --- CONFIGURATION ---
    IMAGE_PATH = "assets/i3.jpg"        # <-- Your input image
    LAMA_MODEL_PATH = "models/qualcomm-lama-dilated/model.onnx" # <-- Your LaMa ONNX model
    SAM_ENCODER_PATH = "models/MobileSam_MobileSAMEncoder_float.tflite" # <-- Your MobileSAM Encoder
    SAM_DECODER_PATH = "models/MobileSam_MobileSAMDecoder_float.tflite" # <-- Your MobileSAM Decoder
    
    # Try a more specific prompt!
    OBJECT_TO_MOVE = "box."          # <-- Prompt for Grounding DINO
    NEW_POSITION = (50, 3800)            # <-- (x, y) coords to paste the object
    
    import os
    os.makedirs("outputs", exist_ok=True)
    
    # --- STEP 1: Detect the object and get its center point ---
    print("--- Step 1: Detecting Object ---")
    detection_results, (orig_w, orig_h) = object_detection_gdino_tiny(
        image_path=IMAGE_PATH,
        text=OBJECT_TO_MOVE
    )
    
    if not detection_results or not detection_results[0]["boxes"].numel():
        print(f"Error: Could not find '{OBJECT_TO_MOVE}' in the image. Exiting.")
        exit()
        
    bbox = detection_results[0]["boxes"][0].tolist()
    bbox_int = [int(v) for v in bbox]
    print(f"Object '{OBJECT_TO_MOVE}' found at {bbox_int}")

    # Calculate center point and SCALE it for SAM's 1024x1024 input
    x0, y0, x1, y1 = bbox
    center_x_orig = (x0 + x1) / 2
    center_y_orig = (y0 + y1) / 2
    
    center_x_scaled = (center_x_orig / orig_w) * 1024
    center_y_scaled = (center_y_orig / orig_h) * 1024
    
    sam_point_coords = [
        [center_x_scaled, center_y_scaled], # Real point
        [0.0, 0.0]                          # Dummy padding point
    ]
    sam_point_labels = [1, -1]

    # --- STEP 2: Get a precise mask for the object ---
    print("\n--- Step 2: Segmenting Object (MobileSAM) ---")
    full_mask_8bit = run_mobilesam_segmentation(
        encoder_path=SAM_ENCODER_PATH,
        decoder_path=SAM_DECODER_PATH,
        image_path=IMAGE_PATH,
        original_w=orig_w,
        original_h=orig_h,
        point_coords_scaled=sam_point_coords,
        point_labels=sam_point_labels
    )
    cv2.imwrite("outputs/rearrange_v3_01_sam_mask.png", full_mask_8bit)
    print("Saved 'outputs/rearrange_v3_01_sam_mask.png'")

    # --- STEP 3: Remove the object from the background (Inpainting) ---
    print("\n--- Step 3: Inpainting Background (LaMa) ---")
    inpainted_background_rgb = run_lama_inpainting(
        model_path=LAMA_MODEL_PATH,
        image_path=IMAGE_PATH,
        mask_array=full_mask_8bit  # <-- Pass the precise mask
    )
    cv2.imwrite("outputs/rearrange_v3_02_inpainted_bg.jpg", cv2.cvtColor(inpainted_background_rgb, cv2.COLOR_RGB2BGR))
    print("Saved 'outputs/rearrange_v3_02_inpainted_bg.jpg'")

    # --- STEP 4: Get the cropped cutout and its matte ---
    print("\n--- Step 4: Cropping Cutout & Matte ---")
    original_image_cv = cv2.imread(IMAGE_PATH)
    original_image_rgb = cv2.cvtColor(original_image_cv, cv2.COLOR_BGR2RGB)
    
    # Find bounding box of the *mask* to crop tightly
    contours, _ = cv2.findContours(full_mask_8bit, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        print("Error: No contours found in SAM mask. Cannot proceed.")
        exit()
        
    mx, my, mw, mh = cv2.boundingRect(contours[0]) # Get box of largest contour
    
    # Crop the original image
    cutout_rgb = original_image_rgb[my : my + mh, mx : mx + mw]
    # Crop the mask
    cutout_matte = full_mask_8bit[my : my + mh, mx : mx + mw]
    
    cv2.imwrite("outputs/rearrange_v3_03_cutout.png", cv2.cvtColor(cutout_rgb, cv2.COLOR_RGB2BGR))
    cv2.imwrite("outputs/rearrange_v3_04_cutout_matte.png", cutout_matte)

    # --- *** NEW STEP 4.5: Create and blend shadow *** ---
    print("\n--- Step 4.5: Creating Fast Shadow ---")
    background_with_shadow_rgb = create_and_blend_shadow(
        background_rgb=inpainted_background_rgb,
        cutout_matte=cutout_matte,
        cutout_rgb_shape=cutout_rgb.shape[:2], # (h, w)
        paste_position=NEW_POSITION
    )
    cv2.imwrite("outputs/rearrange_v3_04.5_bg_with_shadow.jpg", cv2.cvtColor(background_with_shadow_rgb, cv2.COLOR_RGB2BGR))
    print("Saved 'outputs/rearrange_v3_04.5_bg_with_shadow.jpg'")


    # --- STEP 5: Paste the matted object onto the new background ---
    print("\n--- Step 5: Pasting Object to New Location ---")
    final_image_rgb = paste_object_with_matte(
        background_rgb=background_with_shadow_rgb, # <-- CHANGED
        cutout_rgb=cutout_rgb,
        matte_8bit=cutout_matte,
        position=NEW_POSITION
    )
    
    # --- STEP 6: Save and show the final result ---
    print("\n--- Step 6: Saving Final Image ---")
    final_image_bgr = cv2.cvtColor(final_image_rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite("outputs/rearrange_v3_05_final_result.jpg", final_image_bgr)
    print("Saved 'outputs/rearrange_v3_05_final_result.jpg'")

    # Show a comparison
    plt.figure(figsize=(24, 8)) # Made figure wider
    plt.subplot(1, 4, 1)
    plt.imshow(original_image_rgb)
    plt.title("1. Original Image")
    plt.axis("off")

    plt.subplot(1, 4, 2)
    plt.imshow(inpainted_background_rgb)
    plt.title("2. Inpainted Background")
    plt.axis("off")

    plt.subplot(1, 4, 3)
    plt.imshow(background_with_shadow_rgb) # <-- Added new plot
    plt.title("3. Background + Fake Shadow")
    plt.axis("off")

    plt.subplot(1, 4, 4)
    plt.imshow(final_image_rgb)
    plt.title(f"4. Final Result (Pasted at {NEW_POSITION})")
    plt.axis("off")

    plt.tight_layout()
    plt.savefig("outputs/rearrange_v3_06_comparison.png")
    print("Saved 'outputs/rearrange_v3_06_comparison.png'")
    print("\n--- V3 Pipeline Complete ---")