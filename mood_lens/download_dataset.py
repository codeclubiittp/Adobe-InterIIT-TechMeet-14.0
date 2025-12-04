import os
import pandas as pd
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm  
import logging
import config

INPUT_FILE = 'photos.csv000'
OUTPUT_CSV = 'dataset_manifest.csv'
IMAGE_DIR = 'mood_lens/dataset/images' 
MAX_WORKERS = 8
SAVE_INTERVAL = 250  

def setup_directories():
    if not os.path.exists(IMAGE_DIR):
        os.makedirs(IMAGE_DIR)
        logging.info(f"[INFO] Created directory: {IMAGE_DIR}")

def append_to_csv(data_list, file_path, write_header=False):
    if not data_list:
        return

    df = pd.DataFrame(data_list)
    cols = ['photo_id', 'photo_description', 'image_path']
    for col in cols:
        if col not in df.columns:
            df[col] = ""
            
    df = df[cols]
    df.to_csv(file_path, mode='a', index=False, header=write_header)

def process_row(row):
    photo_id = row.get('photo_id')
    description = row.get('ai_description')
    # temporary
    if pd.isna(description):
        description = ""
    else:
        description = str(description).replace('\n', ' ').strip()

    url = row.get('photo_image_url')
    if pd.isna(url):
        url = row.get('photo_url')

    if not photo_id or not url:
        return None

    save_path = os.path.join(IMAGE_DIR, f"{photo_id}.jpg")
    download_success = False
    
    if os.path.exists(save_path):
        download_success = True
    else:
        try:
            headers = {'User-Agent': 'Mozilla/5.0'}
            response = requests.get(url, stream=True, timeout=15, headers=headers)
            if response.status_code == 200:
                with open(save_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                download_success = True
        except Exception:
            download_success = False

    if download_success:
        return {
            "photo_id": photo_id,
            "photo_description": description,
            "image_path": save_path
        }
    return None

def main():
    config.setup_logging()
    setup_directories()
    if os.path.exists(OUTPUT_CSV):
        os.remove(OUTPUT_CSV)
        logging.info(f"Removed old {OUTPUT_CSV} to start fresh.")

    logging.info(f"Loading {INPUT_FILE}")
    try:
        df = pd.read_csv(INPUT_FILE, sep='\t', on_bad_lines='skip')
        if 'photo_id' not in df.columns:
            logging.info("Tabs failed, trying comma separation")
            df = pd.read_csv(INPUT_FILE, sep=',')
    except Exception as e:
        logging.info(f"Could not read file: {e}")
        return

    logging.info(f"Found {len(df)} rows. Starting parallel download")
    
    results_buffer = []
    is_first_batch = True
 
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_row = {executor.submit(process_row, row): row for _, row in df.iterrows()}
        for future in tqdm(as_completed(future_to_row), total=len(future_to_row), unit="img"):
            res = future.result()
            
            if res:
                results_buffer.append(res)
            
            if len(results_buffer) >= SAVE_INTERVAL:
                append_to_csv(results_buffer, OUTPUT_CSV, write_header=is_first_batch)
                results_buffer = []  
                is_first_batch = False 
    if results_buffer:
        append_to_csv(results_buffer, OUTPUT_CSV, write_header=is_first_batch)

    logging.info(f"Processing complete. Manifest saved to {OUTPUT_CSV}")

if __name__ == "__main__":
    main()