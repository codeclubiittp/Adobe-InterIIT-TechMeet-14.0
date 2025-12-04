import cv2
import numpy as np
import json
import easyocr
import torch
from doctr.io import DocumentFile
from doctr.models import ocr_predictor

def initialize_models():
    """Initialize both doctr (for bbox detection) and EasyOCR (for text recognition)"""
    print(f"CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU Device: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"\nInitializing doctr OCR model on {device.upper()} (for bounding boxes)...")
    doctr_model = ocr_predictor(pretrained=True).to(device)
    
    print(f"Initializing EasyOCR model (for text recognition)...")
    easyocr_reader = easyocr.Reader(['en'], gpu=torch.cuda.is_available())
    
    print("Models loaded successfully!")
    return doctr_model, easyocr_reader

def crop_image_region(img, box_points):
    """Crop image region based on bounding box points"""
    x_coords = [p[0] for p in box_points]
    y_coords = [p[1] for p in box_points]
    x_min, x_max = max(0, min(x_coords)), max(x_coords)
    y_min, y_max = max(0, min(y_coords)), max(y_coords)
    
    return img[y_min:y_max, x_min:x_max], (x_min, y_min)

def recognize_text_with_easyocr(easyocr_reader, img_crop):
    """Use EasyOCR to recognize text in a cropped region"""
    if img_crop.size == 0:
        return "", 0.0
    
    results = easyocr_reader.readtext(img_crop, detail=1)
    
    if not results:
        return "", 0.0
    
    # Combine all detected text and take average confidence
    texts = [text for (_, text, _) in results]
    confidences = [conf for (_, _, conf) in results]
    
    combined_text = " ".join(texts)
    avg_conf = np.mean(confidences) if confidences else 0.0
    
    return combined_text, avg_conf

def process_hybrid_results(doctr_model, easyocr_reader, image_path, min_conf=0.0):
    """Use doctr for bbox detection and EasyOCR for text recognition"""
    doc = DocumentFile.from_images(image_path)
    
    print("\nDetecting bounding boxes with doctr...")
    result = doctr_model(doc)
    
    img = cv2.imread(image_path)
    h, w = img.shape[:2]
    
    group_data = []
    group_idx = 0
    
    for page_idx, page in enumerate(result.pages):
        print(f"\nProcessing Page {page_idx}...")
        
        for block in page.blocks:
            for line in block.lines:
                # Get line geometry from doctr
                line_geometry = line.geometry
                line_box_norm = [
                    [line_geometry[0][0], line_geometry[0][1]],
                    [line_geometry[1][0], line_geometry[0][1]],
                    [line_geometry[1][0], line_geometry[1][1]],
                    [line_geometry[0][0], line_geometry[1][1]]
                ]
                
                line_box = [[int(p[0] * w), int(p[1] * h)] for p in line_box_norm]
                
                # Crop the line region
                line_crop, (offset_x, offset_y) = crop_image_region(img, line_box)
                
                # Use EasyOCR to recognize text in this region
                line_text, line_conf = recognize_text_with_easyocr(easyocr_reader, line_crop)
                
                if line_conf < min_conf or not line_text.strip():
                    continue
                
                print(f"\nGroup {group_idx}: '{line_text}' (conf: {line_conf:.2f})")
                
                group_info = {
                    'group_index': group_idx,
                    'full_text': line_text,
                    'confidence': float(line_conf),
                    'group_box': line_box,
                    'words': []
                }
                
                # Process individual words using doctr's word-level detection
                for word_idx, word in enumerate(line.words):
                    word_geom = word.geometry
                    word_box_norm = [
                        [word_geom[0][0], word_geom[0][1]],
                        [word_geom[1][0], word_geom[0][1]],
                        [word_geom[1][0], word_geom[1][1]],
                        [word_geom[0][0], word_geom[1][1]]
                    ]
                    
                    word_box = [[int(p[0] * w), int(p[1] * h)] for p in word_box_norm]
                    
                    # Crop word region
                    word_crop, (word_offset_x, word_offset_y) = crop_image_region(img, word_box)
                    
                    # Use EasyOCR for word recognition
                    word_text, word_conf = recognize_text_with_easyocr(easyocr_reader, word_crop)
                    
                    # Fallback to doctr text if EasyOCR fails
                    if not word_text.strip():
                        word_text = word.value
                        word_conf = word.confidence
                    
                    x_coords = [p[0] for p in word_box]
                    y_coords = [p[1] for p in word_box]
                    x_min, x_max = min(x_coords), max(x_coords)
                    y_min, y_max = min(y_coords), max(y_coords)
                    
                    word_info = {
                        'word_index': word_idx,
                        'word': word_text,
                        'box': word_box,
                        'bounding_box': {
                            'x': x_min,
                            'y': y_min,
                            'width': x_max - x_min,
                            'height': y_max - y_min
                        },
                        'confidence': float(word_conf)
                    }
                    
                    group_info['words'].append(word_info)
                    
                    print(f"  Word {word_idx}: '{word_text}' @ [{x_min},{y_min},{x_max},{y_max}] (conf: {word_conf:.2f})")
                
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
                visualization_path="out_hybrid.jpg", min_conf=0.3, draw_visualization=True):
    """
    Hybrid OCR: doctr for bounding boxes + EasyOCR for text recognition
    """
    doctr_model, easyocr_reader = initialize_models()
    
    group_data = process_hybrid_results(doctr_model, easyocr_reader, image_path, min_conf=min_conf)
    
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