import torch
import datetime
import os
import logging
from logging.handlers import RotatingFileHandler

from dotenv import load_dotenv

load_dotenv()

DATASET_ROOT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset")
#DATASET_ROOT_DIR = os.getenv("dataset_path")
print(DATASET_ROOT_DIR)
OUTPUT_ROOT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
MISC_ROOT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "misc")

INDEX_PATH = os.path.join(DATASET_ROOT_DIR, "vector_db.index")
METADATA_PATH = os.path.join(DATASET_ROOT_DIR, "metadata_map.pkl")
CSV_PATH = os.path.join(DATASET_ROOT_DIR, "labeled_dataset.csv")
PROFILE_CHART_PATH = os.path.join(MISC_ROOT_DIR, "profiling_charts")
CACHE_DIR = os.path.join(MISC_ROOT_DIR, ".mood_lens_hf_cache")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

LUT_DIM = 33
TRAIN_STEPS = 125
LEARNING_RATE = 1e-2
BATCH_SIZE = 32
TRAIN_SIZE = (400, 400)
TOP_K_MATCHES = 3

WEIGHT_STYLE = 1e5
WEIGHT_CONTENT = 12.0
WEIGHT_TV = 12.0

def setup_logging(log_file=None):
    if log_file is None:
        log_file = os.path.join(MISC_ROOT_DIR, f"mood_lens_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M')}.log")
    
    os.makedirs(DATASET_ROOT_DIR, exist_ok=True)
    os.makedirs(OUTPUT_ROOT_DIR, exist_ok=True)
    os.makedirs(MISC_ROOT_DIR, exist_ok=True)
    os.makedirs(PROFILE_CHART_PATH, exist_ok=True)
    
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    if logger.handlers:
        return

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        "%H:%M:%S"
    ))
    
    file_handler = RotatingFileHandler(
        log_file, maxBytes=5 * 1024 * 1024, backupCount=3
    )
    file_handler.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    ))

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)