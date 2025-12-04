import time
import cv2
import numpy as np
import onnxruntime as ort
import torch
from PIL import Image, ImageDraw, ImageFont
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
import matplotlib.pyplot as plt
from typing import List, Dict, Any, Tuple

# --- MODIFIED Function 1: Object Detection ---
# (Modified to return all filtered results)

def object_detection_gdino_tiny(
    image_path: str,
    text: str,
    box_threshold: float = 0.4
) -> List[Dict[str, Any]]:
    """
    Detects objects in an image based on a text prompt.
    Returns a list of filtered results.
    """
    model_id = "IDEA-Research/grounding-dino-tiny"
    device = "cpu"

    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)

    image = Image.open(image_path).convert("RGB")

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

    # (Removed image drawing/showing from here, will do at the end)
    for res in filtered_results:
        for label, score, box in zip(res["labels"], res["scores"], res["boxes"]):
            print(f"[GDINO] Found: {label}: {score:.2f} -> {box.tolist()}")

    return filtered_results


# --- MODIFIED Function 2: Inpainting ---
# (Modified to accept a bounding box for the mask and return the inpainted image)

def run_lama_inpainting(
    model_path: str,
    image_path: str,
    mask_bbox: List[float]
) -> np.ndarray:
    """
    Inpaints an image using a mask generated from a bounding box.
    Returns the final inpainted image as an RGB numpy array.
    """
    onnx_session = ort.InferenceSession(
        model_path,
        providers=["CPUExecutionProvider"]
    )
    # log_session_details(onnx_session) # Optional: uncomment for debugging

    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    orig_h, orig_w = image.shape[:2]

    # --- Create mask from bounding box ---
    mask = np.zeros((orig_h, orig_w), dtype=np.uint8)
    x0, y0, x1, y1 = [int(v) for v in mask_bbox]
    # Add a small buffer/padding to the mask to ensure full removal
    pad = 10 
    y0, y1 = max(0, y0 - pad), min(orig_h, y1 + pad)
    x0, x1 = max(0, x0 - pad), min(orig_w, x1 + pad)
    mask[y0:y1, x0:x1] = 1
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
    
    # Resize back to original dimensions
    final_image_resized = cv2.resize(painted_image_denorm, (orig_w, orig_h), interpolation=cv2.INTER_CUBIC)
    
    print("[LaMa] Inpainting complete.")
    return final_image_resized # Return the RGB numpy array


# --- MODIFIED Function 3: Matting ---
# (Modified to accept an in-memory image (numpy array) instead of a file path)

def modnet_matting(
    model_path: str,
    image_bgr: np.ndarray
) -> np.ndarray:
    """
    Performs portrait matting on a BGR numpy image array.
    Returns an 8-bit (0-255) single-channel alpha matte.
    """
    onnx_session = ort.InferenceSession(
        model_path,
        providers=["CPUExecutionProvider"]
    )
    # log_session_details(onnx_session) # Optional: uncomment for debugging

    input_name = onnx_session.get_inputs()[0].name
    output_name = onnx_session.get_outputs()[0].name

    original_h, original_w, _ = image_bgr.shape
    img_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (512,512), interpolation=cv2.INTER_LINEAR)
    img_normalized = img_resized.astype(np.float32) / 255.0
    img_chw = img_normalized.transpose(2, 0, 1)
    input_tensor = np.expand_dims(img_chw, axis=0)
    
    ort_inputs = {input_name: input_tensor}
    
    # print("[MODNet] Running matting inference...")
    ort_outputs = onnx_session.run([output_name], ort_inputs)

    alpha_matte = ort_outputs[0].squeeze() 
    alpha_matte = (alpha_matte * 255).astype(np.uint8)

    # Resize matte back to the input image's original size
    alpha_matte_original_size = cv2.resize(
        alpha_matte, 
        (original_w, original_h), 
        interpolation=cv2.INTER_LINEAR
    )
    
    # print("[MODNet] Matting complete.")
    return alpha_matte_original_size


# --- NEW Function 4: Compositing ---
# (New function to paste the cutout onto the background)

def paste_object_with_matte(
    background_rgb: np.ndarray, 
    cutout_rgb: np.ndarray, 
    matte_8bit: np.ndarray, 
    position: Tuple[int, int]
) -> np.ndarray:
    """
    Pastes a matted cutout onto a background image at a new position.
    
    Args:
        background_rgb: Full-size background (RGB numpy array).
        cutout_rgb: The cropped object to paste (RGB numpy array).
        matte_8bit: The alpha matte for the cutout (8-bit numpy array).
        position: (x, y) tuple for the top-left corner to paste.

    Returns:
        The final composited image (RGB numpy array).
    """
    x_paste, y_paste = position
    h_cut, w_cut, _ = cutout_rgb.shape

    # Ensure paste coordinates are within bounds
    if x_paste + w_cut > background_rgb.shape[1] or y_paste + h_cut > background_rgb.shape[0]:
        print(f"[Paste] Warning: Paste position {position} with size {w_cut}x{h_cut} is partially out of bounds.")
        # Adjust dimensions if out of bounds (simple crop)
        w_cut = min(w_cut, background_rgb.shape[1] - x_paste)
        h_cut = min(h_cut, background_rgb.shape[0] - y_paste)
        cutout_rgb = cutout_rgb[:h_cut, :w_cut]
        matte_8bit = matte_8bit[:h_cut, :w_cut]

    # Get the slice of the background where we will paste
    bg_slice = background_rgb[y_paste : y_paste + h_cut, x_paste : x_paste + w_cut]
    
    # Normalize matte to [0, 1] and add channel dimension
    matte_norm = np.expand_dims(matte_8bit.astype(np.float32) / 255.0, axis=-1)
    
    # Alpha compositing formula: (FG * Alpha) + (BG * (1 - Alpha))
    composited_slice = (cutout_rgb * matte_norm) + (bg_slice * (1 - matte_norm))
    composited_slice = composited_slice.astype(np.uint8)
    
    # Create a copy of the background to place the composite
    final_image = background_rgb.copy()
    final_image[y_paste : y_paste + h_cut, x_paste : x_paste + w_cut] = composited_slice
    
    print(f"[Paste] Pasted object at {position}.")
    return final_image


# --- Main Execution Pipeline ---

if __name__ == "__main__":
    # --- CONFIGURATION ---
    IMAGE_PATH = "assets/i1.jpg"        # <-- Your input image
    LAMA_MODEL_PATH = "models/qualcomm-lama-dilated/model.onnx" # <-- Your LaMa ONNX model
    MODNET_MODEL_PATH = "models/modnet_photographic_portrait_matting.onnx" # <-- Your MODNet ONNX model
    
    OBJECT_TO_MOVE = "a human."         # <-- Prompt for Grounding DINO
    NEW_POSITION = (50, 100)            # <-- (x, y) coords to paste the object
    
    # Create outputs directory if it doesn't exist
    
    # --- STEP 1: Detect the object to move ---
    print("--- Step 1: Detecting Object ---")
    detection_results = object_detection_gdino_tiny(
        image_path=IMAGE_PATH,
        text=OBJECT_TO_MOVE
    )
    
    if not detection_results or not detection_results[0]["boxes"].numel():
        print(f"Error: Could not find '{OBJECT_TO_MOVE}' in the image. Exiting.")
        exit()
        
    # Get the bounding box of the first detected object
    bbox = detection_results[0]["boxes"][0].tolist()
    bbox_int = [int(v) for v in bbox]
    print(f"Object '{OBJECT_TO_MOVE}' found at {bbox_int}")

    # --- STEP 2: Get the original image and the cutout ---
    original_image_cv = cv2.imread(IMAGE_PATH)
    x0, y0, x1, y1 = bbox_int
    # Crop the original image to get the object
    person_crop_bgr = original_image_cv[y0:y1, x0:x1]
    person_crop_rgb = cv2.cvtColor(person_crop_bgr, cv2.COLOR_BGR2RGB)

    # --- STEP 3: Create a precise matte for the cutout ---
    print("\n--- Step 2: Creating Precise Matte ---")
    alpha_matte_crop = modnet_matting(
        model_path=MODNET_MODEL_PATH,
        image_bgr=person_crop_bgr
    )
    cv2.imwrite("outputs/rearrange_01_cropped_matte.png", alpha_matte_crop)
    print("Saved 'outputs/rearrange_01_cropped_matte.png'")

    # --- STEP 4: Remove the object from the background (Inpainting) ---
    print("\n--- Step 3: Inpainting Background ---")
    inpainted_background_rgb = run_lama_inpainting(
        model_path=LAMA_MODEL_PATH,
        image_path=IMAGE_PATH,
        mask_bbox=bbox
    )
    cv2.imwrite("outputs/rearrange_02_inpainted_bg.jpg", cv2.cvtColor(inpainted_background_rgb, cv2.COLOR_RGB2BGR))
    print("Saved 'outputs/rearrange_02_inpainted_bg.jpg'")

    # --- STEP 5: Paste the matted object onto the new background ---
    print("\n--- Step 4: Pasting Object to New Location ---")
    final_image_rgb = paste_object_with_matte(
        background_rgb=inpainted_background_rgb,
        cutout_rgb=person_crop_rgb,
        matte_8bit=alpha_matte_crop,
        position=NEW_POSITION
    )
    
    # --- STEP 6: Save and show the final result ---
    print("\n--- Step 5: Saving Final Image ---")
    final_image_bgr = cv2.cvtColor(final_image_rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite("outputs/rearrange_03_final_result.jpg", final_image_bgr)
    print("Saved 'outputs/rearrange_03_final_result.jpg'")

    # Show a comparison
    plt.figure(figsize=(18, 6))
    plt.subplot(1, 3, 1)
    plt.imshow(cv2.cvtColor(original_image_cv, cv2.COLOR_BGR2RGB))
    plt.title("1. Original Image")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(inpainted_background_rgb)
    plt.title("2. Inpainted Background")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(final_image_rgb)
    plt.title(f"3. Final Result (Pasted at {NEW_POSITION})")
    plt.axis("off")

    plt.tight_layout()
    plt.savefig("outputs/rearrange_04_comparison.png")
    print("Saved 'outputs/rearrange_04_comparison.png'")
    print("\n--- Pipeline Complete ---")