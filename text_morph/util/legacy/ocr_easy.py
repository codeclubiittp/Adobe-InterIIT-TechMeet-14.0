import cv2
import numpy as np
import json
import easyocr
import torch

def initialize_model(languages=['en'], gpu=True):
    """
    Initialize EasyOCR model
    languages: list of language codes (e.g., ['en'], ['en', 'hi'], ['en', 'ch_sim'])
    gpu: whether to use GPU if available
    """
    print(f"CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU Device: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    
    use_gpu = gpu and torch.cuda.is_available()
    device = 'GPU' if use_gpu else 'CPU'
    
    print(f"\nInitializing EasyOCR model on {device}...")
    print(f"Languages: {languages}")
    
    reader = easyocr.Reader(languages, gpu=use_gpu)
    print("Model loaded successfully!")
    return reader

def process_easyocr_results(reader, image_path, min_conf=0.0):
    """
    Process image with EasyOCR and extract text groups
    """
    print("\nRunning OCR with EasyOCR...")
    
    # Read image to get dimensions
    img = cv2.imread(image_path)
    h, w = img.shape[:2]
    
    # Run EasyOCR
    # Returns list of (bbox, text, confidence)
    results = reader.readtext(image_path)
    
    group_data = []
    
    for group_idx, (bbox, text, conf) in enumerate(results):
        # Skip low confidence results
        if conf < min_conf or not text.strip():
            continue
        
        # Convert bbox to integer coordinates
        # EasyOCR returns [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
        group_box = [[int(p[0]), int(p[1])] for p in bbox]
        
        print(f"\nGroup {group_idx}: '{text}' (conf: {conf:.2f})")
        
        # Calculate bounding box
        x_coords = [p[0] for p in group_box]
        y_coords = [p[1] for p in group_box]
        x_min, x_max = min(x_coords), max(x_coords)
        y_min, y_max = min(y_coords), max(y_coords)
        
        # Split text into words for compatibility with original structure
        words = text.split()
        word_list = []
        
        if words:
            # Estimate word positions (approximate, since EasyOCR gives line-level detection)
            total_width = x_max - x_min
            word_width = total_width / len(words) if words else total_width
            
            for word_idx, word in enumerate(words):
                # Approximate word position
                word_x_start = x_min + (word_idx * word_width)
                word_x_end = word_x_start + word_width
                
                word_box = [
                    [int(word_x_start), y_min],
                    [int(word_x_end), y_min],
                    [int(word_x_end), y_max],
                    [int(word_x_start), y_max]
                ]
                
                word_info = {
                    'word_index': word_idx,
                    'word': word,
                    'box': word_box,
                    'bounding_box': {
                        'x': int(word_x_start),
                        'y': y_min,
                        'width': int(word_x_end - word_x_start),
                        'height': y_max - y_min
                    },
                    'confidence': float(conf)  # Same confidence for all words in the line
                }
                
                word_list.append(word_info)
                print(f"  Word {word_idx}: '{word}' @ [{int(word_x_start)},{y_min},{int(word_x_end)},{y_max}] (conf: {conf:.2f})")
        
        group_info = {
            'group_index': group_idx,
            'full_text': text,
            'confidence': float(conf),
            'group_box': group_box,
            'words': word_list
        }
        
        group_data.append(group_info)
    
    print(f"\n{'='*50}")
    print(f"Processed {len(group_data)} text groups")
    print(f"{'='*50}")
    
    return group_data

def draw_results(image_path, group_data, save_path=None, draw_words=True, show_conf=True):
    """
    Draw bounding boxes and text on the image
    """
    img = cv2.imread(image_path)
    out = img.copy()
    h, w = out.shape[:2]
    
    for group in group_data:
        # Draw group bounding box (green)
        group_pts = np.array([[int(p[0]), int(p[1])] for p in group['group_box']], 
                            dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(out, [group_pts], True, (0, 255, 0), 2)
        
        # Prepare label
        label = group['full_text']
        if show_conf:
            label = f"{label} {group['confidence']:.2f}"
        
        # Position for label
        x0 = min([p[0] for p in group['group_box']])
        y0 = min([p[1] for p in group['group_box']])
        
        # Draw label background and text
        font = cv2.FONT_HERSHEY_SIMPLEX
        fs = max(0.5, min(1.0, w/1000))
        (tw, th), base = cv2.getTextSize(label, font, fs, 1)
        
        rx1, ry1 = x0, max(0, y0 - th - base - 2)
        rx2, ry2 = rx1 + tw + 4, ry1 + th + base + 4
        cv2.rectangle(out, (rx1, ry1), (rx2, ry2), (0, 255, 0), -1)
        cv2.putText(out, label, (rx1 + 2, ry2 - base - 2), font, fs, (0, 0, 0), 1, cv2.LINE_AA)
        
        # Draw individual word boxes (blue) if requested
        if draw_words and group['words']:
            for word_info in group['words']:
                word_pts = np.array([[int(p[0]), int(p[1])] for p in word_info['box']], 
                                   dtype=np.int32).reshape((-1, 1, 2))
                cv2.polylines(out, [word_pts], True, (255, 0, 0), 1)
                
                # Draw word labels for short groups
                if show_conf and len(group['words']) <= 5:
                    word_x = min([p[0] for p in word_info['box']])
                    word_y = max([p[1] for p in word_info['box']])
                    word_label = f"{word_info['word']}"
                    cv2.putText(out, word_label, (word_x, word_y + 15), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 0, 0), 1, cv2.LINE_AA)
    
    if save_path:
        cv2.imwrite(save_path, out)
        print(f"\nVisualization saved to: {save_path}")
    
    return out

def detect_text(image_path, output_json="ocr_results.json", 
                visualization_path="out_easyocr.jpg", min_conf=0.3, 
                draw_visualization=True, languages=['en'], gpu=True):
    """
    Main function to detect text in an image using EasyOCR
    
    Args:
        image_path: Path to input image
        output_json: Path to save JSON results
        visualization_path: Path to save visualization
        min_conf: Minimum confidence threshold (0-1)
        draw_visualization: Whether to draw and save visualization
        languages: List of language codes for EasyOCR
        gpu: Whether to use GPU if available
    """
    # Initialize model
    reader = initialize_model(languages=languages, gpu=gpu)
    
    # Process image
    group_data = process_easyocr_results(reader, image_path, min_conf=min_conf)
    
    # Prepare OCR results
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
    
    # Save JSON results
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(ocr_results, f, indent=2, ensure_ascii=False)
    
    print(f"\nAll OCR data exported to: {output_json}")
    
    # Draw and save visualization
    if draw_visualization:
        draw_results(image_path, group_data, save_path=visualization_path, 
                    draw_words=True, show_conf=True)
    
    return ocr_results


# Example usage
if __name__ == "__main__":
    image_path = "sample_images/3.jpg"
    
    # For English only
    results = detect_text(
        image_path, 
        output_json="ocr.json", 
        visualization_path="out.jpg", 
        min_conf=0.3,
        languages=['en'],  # Add more languages as needed: ['en', 'hi', 'ch_sim', etc.]
        gpu=True
    )
    
    # Print summary
    print(f"\n{'='*50}")
    print(f"Total text groups detected: {results['total_groups']}")
    print(f"{'='*50}")
    
    # Print all detected text
    for group in results['groups']:
        print(f"Group {group['group_index']}: {group['full_text']} (conf: {group['confidence']:.2f})")