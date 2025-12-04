import cv2
import torch
import numpy as np
from PIL import Image
from facenet_pytorch import MTCNN
from emotiefflib.facial_analysis import EmotiEffLibRecognizer
import torch.profiler as tp

device = 'cuda' if torch.cuda.is_available() else 'cpu'
with tp.record_function("emotiefflib-mtcnn-loading"):
    mtcnn = MTCNN(keep_all=False, device=device)
    model_name = 'enet_b0_8_best_vgaf' 
    print(f"Loading EmotiEffLib model: {model_name}...")
    fer = EmotiEffLibRecognizer(engine="onnx", model_name=model_name, device=device)

def process_image(image_path):
    frame_bgr = cv2.imread(image_path)
    if frame_bgr is None:
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(frame_rgb)

    print("Detecting faces...")
    with tp.record_function("emotiefflib-mtcnn-inference"):
        boxes, probs = mtcnn.detect(pil_image)

    if boxes is None:
        print("No faces detected.")
        return

    # B. Process Each Face
    for i, box in enumerate(boxes):
        # Crop the face (with some margin if desired, MTCNN is usually tight)
        x1, y1, x2, y2 = [int(b) for b in box]
        
        # Clamp coordinates to image dimensions
        h, w, _ = frame_rgb.shape
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        
        face_img = frame_rgb[y1:y2, x1:x2]
        
        if face_img.size == 0:
            continue

        # C. Predict Emotions
        # EmotiEffLib expects the cropped face array (RGB)
        with tp.record_function("emotiefflib-emotion-inference"):
            emotion, scores = fer.predict_emotions(face_img, logits=False)
        
        print(f"\n[Face {i}]")
        print(f"  > Bbox: {x1}, {y1}, {x2}, {y2}")
        print(f"  > Emotion: {emotion}")
        print(f"  > Scores: {scores}")

        # Optional: Draw on image
        cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame_bgr, f"{emotion}", (x1, y1 - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Save or Show result
    output_path = "output_emotion.jpg"
    cv2.imwrite(output_path, frame_bgr)
    print(f"\nResult saved to {output_path}")

with tp.profile(
activities=[
    tp.ProfilerActivity.CUDA,
    tp.ProfilerActivity.CPU
],
record_shapes=True,
profile_memory=True,
with_stack=True,
with_flops=True,
with_modules=True,
) as profiler:
    process_image('assets/i7.jpg')

profiler.export_chrome_trace("emotion_trace.json")