import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import json
import torch
from doctr.io import DocumentFile
from doctr.models import ocr_predictor

def initialize_model():
    print(f"CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU Device: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nInitializing doctr OCR model on {device.upper()}...")
    model = ocr_predictor(pretrained=True).to(device)
    print("Model loaded successfully!")
    return model

def process_doctr_results(model, image_path, min_conf=0.0):
    doc = DocumentFile.from_images(image_path)
    
    print("\nRunning OCR with doctr...")
    result = model(doc)
    
    img = cv2.imread(image_path)
    h, w = img.shape[:2]
    
    group_data = []
    group_idx = 0
    
    for page_idx, page in enumerate(result.pages):
        print(f"\nProcessing Page {page_idx}...")
        
        for block in page.blocks:
            for line in block.lines:
                line_words = []
                line_text = " ".join([word.value for word in line.words])
                
                line_conf = np.mean([word.confidence for word in line.words]) if line.words else 0
                if line_conf < min_conf or not line_text.strip():
                    continue
                
                line_geometry = line.geometry
                line_box_norm = [
                    [line_geometry[0][0], line_geometry[0][1]],
                    [line_geometry[1][0], line_geometry[0][1]],
                    [line_geometry[1][0], line_geometry[1][1]],
                    [line_geometry[0][0], line_geometry[1][1]]
                ]
                
                line_box = [[int(p[0] * w), int(p[1] * h)] for p in line_box_norm]
                
                print(f"\n  Group {group_idx}: '{line_text}' (conf: {line_conf:.2f})")
                
                group_info = {
                    'group_index': group_idx,
                    'full_text': line_text,
                    'confidence': float(line_conf),
                    'group_box': line_box,
                    'words': []
                }
                
                for word_idx, word in enumerate(line.words):
                    word_geom = word.geometry
                    word_box_norm = [
                        [word_geom[0][0], word_geom[0][1]],
                        [word_geom[1][0], word_geom[0][1]],
                        [word_geom[1][0], word_geom[1][1]],
                        [word_geom[0][0], word_geom[1][1]]
                    ]
                    
                    word_box = [[int(p[0] * w), int(p[1] * h)] for p in word_box_norm]
                    
                    x_coords = [p[0] for p in word_box]
                    y_coords = [p[1] for p in word_box]
                    x_min, x_max = min(x_coords), max(x_coords)
                    y_min, y_max = min(y_coords), max(y_coords)
                    
                    word_info = {
                        'word_index': word_idx,
                        'word': word.value,
                        'box': word_box,
                        'bounding_box': {
                            'x': x_min,
                            'y': y_min,
                            'width': x_max - x_min,
                            'height': y_max - y_min
                        },
                        'confidence': float(word.confidence)
                    }
                    
                    group_info['words'].append(word_info)
                    
                    print(f"    Word {word_idx}: '{word.value}' @ [{x_min},{y_min},{x_max},{y_max}] (conf: {word.confidence:.2f})")
                
                group_data.append(group_info)
                group_idx += 1
    
    print(f"\n{'='*50}")
    print(f"Processed {len(group_data)} text groups")
    print(f"{'='*50}")
    
    return group_data

def draw_results(image_path, group_data, save_path=None, draw_words=True, show_conf=True):
    img = cv2.imread(image_path)
    out = img.copy()
    h, w = out.shape[:2]
    
    for group in group_data:
        group_pts = np.array([[int(p[0]), int(p[1])] for p in group['group_box']], 
                            dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(out, [group_pts], True, (0, 255, 0), 2)
        
        label = group['full_text']
        if show_conf:
            label = f"{label} {group['confidence']:.2f}"
        
        x0 = min([p[0] for p in group['group_box']])
        y0 = min([p[1] for p in group['group_box']])
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        fs = max(0.5, min(1.0, w/1000))
        (tw, th), base = cv2.getTextSize(label, font, fs, 1)
        
        rx1, ry1 = x0, max(0, y0 - th - base - 2)
        rx2, ry2 = rx1 + tw + 4, ry1 + th + base + 4
        cv2.rectangle(out, (rx1, ry1), (rx2, ry2), (0, 255, 0), -1)
        cv2.putText(out, label, (rx1 + 2, ry2 - base - 2), font, fs, (0, 0, 0), 1, cv2.LINE_AA)
        
        if draw_words:
            for word_info in group['words']:
                word_pts = np.array([[int(p[0]), int(p[1])] for p in word_info['box']], 
                                   dtype=np.int32).reshape((-1, 1, 2))
                cv2.polylines(out, [word_pts], True, (255, 0, 0), 1)
                
                if show_conf and len(group['words']) <= 5:
                    word_x = min([p[0] for p in word_info['box']])
                    word_y = max([p[1] for p in word_info['box']])
                    word_label = f"{word_info['word']}"
                    cv2.putText(out, word_label, (word_x, word_y + 15), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 0, 0), 1, cv2.LINE_AA)
    
    if save_path:
        cv2.imwrite(save_path, out)
        print(f"\nVisualization saved to: {save_path}")

def detect_text(image_path, output_json="ocr_results.json", 
                visualization_path="out_doctr.jpg", min_conf=0.3, draw_visualization=True):
    model = initialize_model()
    
    group_data = process_doctr_results(model, image_path, min_conf=min_conf)
    
    ocr_results = {
        'total_groups': len(group_data),
        'groups': []
    }
    
    for group in group_data:
        group_entry = {
            'group_index': group['group_index'],
            'full_text': group['full_text'],
            'confidence': group['confidence'],
            'group_box': group['group_box'],
            'words': []
        }
        
        for word in group['words']:
            word_entry = {
                'word_index': word['word_index'],
                'word': word['word'],
                'box': word['box'],
                'bounding_box': word['bounding_box'],
                'confidence': word['confidence']
            }
            group_entry['words'].append(word_entry)
        
        ocr_results['groups'].append(group_entry)
    
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(ocr_results, f, indent=2, ensure_ascii=False)
    
    print(f"\nAll OCR data exported to: {output_json}")
    
    if draw_visualization:
        draw_results(image_path, group_data, save_path=visualization_path, 
                    draw_words=True, show_conf=True)
    
    return ocr_results


image_path = "sample_images/3.jpg"
results = detect_text(image_path, output_json="ocr.json", visualization_path="out.jpg", min_conf=0.3)