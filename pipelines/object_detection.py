import onnxruntime as ort
import cv2
import numpy as np
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import time
import torch
from PIL import Image, ImageDraw, ImageFont
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

from utils import log_session_details

# def _visualize_object_detection(
#     image: np.ndarray,
#     boxes: np.ndarray,      
#     scores: np.ndarray,     
#     class_idx: np.ndarray,  
#     savefile: str = "images/estimated_boxes.png",
#     score_thresh: float = 0.4, 
#     nms_thresh: float = 0.45   # Standard NMS threshold
# ):
#     print(f"Raw scores min: {np.min(scores):.4f}, max: {np.max(scores):.4f}, mean: {np.mean(scores):.4f}")

#     scores = 1/(1 + np.exp(-scores))  # Convert logits to [0, 1]

#     # V DEBUGGING: Check scores after sigmoid
#     print(f"Sigmoid scores min: {np.min(scores):.4f}, max: {np.max(scores):.4f}, mean: {np.mean(scores):.4f}")
    
#     mask = scores > score_thresh
#     boxes_masked = boxes[mask]
#     scores_masked = scores[mask]
#     class_idx_masked = class_idx[mask]

#     if len(boxes_masked) == 0:
#         print(f"No objects found above threshold {score_thresh}")
#         return

#     # --- 2. Convert Box Format for NMS ---
#     # cv2.dnn.NMSBoxes expects (x_min, y_min, w, h)
#     boxes_for_nms = boxes_masked.copy()
#     boxes_for_nms[:, 0] = boxes_masked[:, 0] - boxes_masked[:, 2] / 2  # cx -> x_min
#     boxes_for_nms[:, 1] = boxes_masked[:, 1] - boxes_masked[:, 3] / 2  # cy -> y_min

#     # --- 3. Apply Non-Maximum Suppression (NMS) ---
#     indices = cv2.dnn.NMSBoxes(
#         boxes_for_nms.tolist(),
#         scores_masked.tolist(),
#         score_thresh,
#         nms_thresh
#     )

#     # NMSBoxes returns a 2D array, flatten it
#     if len(indices.shape) > 1:
#         indices = indices.flatten()

#     # --- 4. Filter Detections by NMS ---
#     final_boxes = boxes_masked[indices]
#     final_scores = scores_masked[indices]
#     final_classes = class_idx_masked[indices]

#     print(f"Visualizing {len(final_boxes)} final objects after NMS.")

#     # --- 5. Scale and Draw ---
#     h, w, _ = image.shape
#     plt.figure(figsize=(10, 10))
#     plt.imshow(image)
#     ax = plt.gca()

#     for i, box in enumerate(final_boxes):
#         score = final_scores[i]
#         class_id = final_classes[i]

#         # Convert normalized (cx, cy, w, h) to scaled (x_min, y_min)
#         cx, cy, bw, bh = box
#         x_min = (cx - bw / 2) * w
#         y_min = (cy - bh / 2) * h
#         box_width = bw * w
#         box_height = bh * h

#         # Create a Rectangle patch
#         rect = patches.Rectangle(
#             (x_min, y_min), box_width, box_height,
#             linewidth=2, edgecolor='r', facecolor='none'
#         )
#         ax.add_patch(rect)
        
#         # Add text label (Class ID + Score)
#         label = f"Class {class_id}: {score:.2f}"
#         plt.text(
#             x_min, y_min - 5, label,
#             color='white', fontsize=9,
#             bbox=dict(facecolor='red', alpha=0.5, pad=1)
#         )
            
#     plt.axis("off")
#     plt.savefig(savefile)
    

def object_detection_qualcomm_yolox(
    image_path: str = "images/i1.jpg",
    savefig: bool = True
):
    """
    Returns
    -------
    [{'name': 'boxes', 'shape': [1, 8400, 4], 'type': 'tensor(float)'},
    {'name': 'scores', 'shape': [1, 8400], 'type': 'tensor(float)'},
    {'name': 'class_idx', 'shape': [1, 8400], 'type': 'tensor(uint8)'}]
    """
    onnx_session = ort.InferenceSession(
        "models/qualcomm-yolo-x/model.onnx",
        providers=["CPUExecutionProvider"]
    )

    log_session_details(onnx_session)

    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_resized = cv2.resize(image, (640, 640))

    image_np = np.array(image_resized, dtype=np.float32)
    # [-1, 1] normalization
    image_np = (image_np / 127.5) - 1.0
    # HWC->CHW
    image_np = np.transpose(image_np, (2, 0, 1))
    # add batch dimension
    image_np = np.expand_dims(image_np, axis=0)

    print(f"image (shape) : {image_np.shape}")
    print(f"image (dtype) : {image_np.dtype}")

    input_names = [input.name for input in onnx_session.get_inputs()]
    output_names = [output.name for output in onnx_session.get_outputs()]

    input_feed = {
        input_names[0] : image_np
    } 

    boxes, scores, class_idx = onnx_session.run(
        output_names=output_names,
        input_feed=input_feed
    )

    return boxes, scores, class_idx
    # _visualize_object_detection(image, boxes[0], scores[0], class_idx[0])

def object_detection_gdino_tiny(
    image_path: str = "images/i1.jpg",
    box_threshold: float = 0.4
):
    model_id = "IDEA-Research/grounding-dino-tiny"
    device =  "cpu"

    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)

    image = Image.open(image_path).convert("RGB")
    text = "a human. an alien."

    begin = time.perf_counter()
    inputs = processor(images=image, text=text, return_tensors="pt").to(device)
    end = time.perf_counter()

    print(f"device:{device} | Image Processing Time: {(end - begin):.4f}s")

    being = time.perf_counter()
    with torch.no_grad():
        outputs = model(**inputs)
    end = time.perf_counter()

    print(f"device:{device} | Inference Time: {(end - begin):.4f}s")

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

    print(f"device:{device} | Post Processing Time: {(end - begin):.4f}s")

    filtered_results = []
    for res in results:
        scores, boxes, labels = res["scores"], res["boxes"], res["text_labels"]
        keep = scores > box_threshold
        filtered_results.append({
            "boxes": boxes[keep],
            "scores": scores[keep],
            "labels": [labels[i] for i in torch.nonzero(keep).flatten()],
        })

    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()

    for res in filtered_results:
        boxes, scores, labels = res["boxes"], res["scores"], res["labels"]
        for box, score, label in zip(boxes, scores, labels):
            x0, y0, x1, y1 = box.tolist()
            draw.rectangle([x0, y0, x1, y1], outline="red", width=3)
            text_label = f"{label} ({score:.2f})"
            text_bbox = draw.textbbox((0, 0), text_label, font=font)
            draw.rectangle([x0, y0 - text_bbox[3], x0 + text_bbox[2], y0], fill="red")
            draw.text((x0, y0 - text_bbox[3]), text_label, fill="white", font=font)

    image.show()

    for res in filtered_results:
        for label, score, box in zip(res["labels"], res["scores"], res["boxes"]):
            print(f"{label}: {score:.2f} -> {box.tolist()}")

    return box, score, label

if __name__ == "__main__":
    object_detection_gdino_tiny()