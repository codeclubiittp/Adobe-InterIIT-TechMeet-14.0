import json
import os
from PIL import Image

def crop_words_from_ocr(json_path, image_path, output_dir="ocr_output"):
    with open(json_path, 'r') as f:
        ocr_data = json.load(f)
    
    img = Image.open(image_path)
    os.makedirs(output_dir, exist_ok=True)
    
    for group in ocr_data['groups']:
        group_idx = group['group_index']
        group_folder = os.path.join(output_dir, str(group_idx))
        os.makedirs(group_folder, exist_ok=True)
        
        text_entries = []
        
        for word_data in group['words']:
            word_idx = word_data['word_index']
            word_text = word_data['word']
            bbox = word_data['bounding_box']
            
            x, y, width, height = bbox['x'], bbox['y'], bbox['width'], bbox['height']
            cropped = img.crop((x, y, x + width, y + height))
            
            image_filename = f"{group_idx}_{word_idx}.png"
            image_path_out = os.path.join(group_folder, image_filename)
            cropped.save(image_path_out)
            
            text_entries.append(f"{image_filename} {word_text}")
        
        text_file_path = os.path.join(group_folder, "words.txt")
        with open(text_file_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(text_entries))
        
        print(f"Processed group {group_idx}: {len(text_entries)} words")
    
    print(f"\nCompleted! Output saved to: {output_dir}")
    print(f"Total groups processed: {ocr_data['total_groups']}")

crop_words_from_ocr("ocr.json", "sample_images/3.jpg")