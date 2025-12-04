import onnxruntime as ort
import cv2
import numpy as np
import time
import torch
from PIL import Image, ImageDraw, ImageFont
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

from pipelines.utils.onnx_helpers import log_session_details
from pipelines.utils.visualization import visualize_object_detection

def run_yolox_detection(
    model_path: str,
    image_path: str = "images/i1.jpg",
    savefig: bool = True,
    save_path: str = "outputs/yolox_estimated.png",
    score_thresh: float = 0.4,
    nms_thresh: float = 0.45
):
    onnx_session = ort.InferenceSession(
        model_path,
        providers=["CPUExecutionProvider"]
    )

    log_session_details(onnx_session)

    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_resized = cv2.resize(image_rgb, (640, 640))

    image_np = np.array(image_resized, dtype=np.float32)
    image_np = (image_np / 127.5) - 1.0
    image_np = np.transpose(image_np, (2, 0, 1))
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
    
    if savefig:
        visualize_object_detection(
            image_rgb, 
            boxes[0], 
            scores[0], 
            class_idx[0], 
            savefile=save_path,
            score_thresh=score_thresh,
            nms_thresh=nms_thresh
        )

    return boxes, scores, class_idx

def run_gdino_detection(
    image_path: str = "images/i1.jpg",
    text_prompt: str = "a human. an alien.",
    box_threshold: float = 0.4,
    model_id: str = "IDEA-Research/grounding-dino-tiny",
    device: str = "cpu",
    save_path: str = "outputs/gdino_detected.png"
):
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)

    image = Image.open(image_path).convert("RGB")

    begin = time.perf_counter()
    inputs = processor(images=image, text=text_prompt, return_tensors="pt").to(device)
    end = time.perf_counter()

    print(f"device:{device} | image processing time: {(end - begin):.4f}s")

    begin = time.perf_counter()
    with torch.no_grad():
        outputs = model(**inputs)
    end = time.perf_counter()

    print(f"device:{device} | inference time: {(end - begin):.4f}s")

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

    print(f"device:{device} | post processing time: {(end - begin):.4f}s")

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

    image.save(save_path)
    print(f"saved gdino detection to {save_path}")

    for res in filtered_results:
        for label, score, box in zip(res["labels"], res["scores"], res["boxes"]):
            print(f"{label}: {score:.2f} -> {box.tolist()}")

    return filtered_results