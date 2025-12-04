import os
import pandas as pd
import logging
import config
from pipeline_colour_correction.vsearch import VSearchEngine
from pipeline_colour_correction.lut_generator import NeuralLUTGenerator
from pipeline_colour_correction.engine_core import colour_correct

def dump_info():
    logging.info("Configuration Settings:")
    logging.info(f"Device: {config.DEVICE}")
    logging.info(f"lut_Dim: {config.LUT_DIM}")
    logging.info(f"lr: {config.LEARNING_RATE}")
    logging.info(f"Train Size: {config.TRAIN_SIZE}")    
    logging.info(f"Metadata Dir: {config.METADATA_PATH}")

if __name__ == "__main__":
    config.setup_logging()
    
    matcher = VSearchEngine()
    
    logging.info("Loading Shared VGG19 Model for LUT Generation")
    shared_generator = NeuralLUTGenerator()
    
    test_image = "pipeline_colour_correction/inputs/1.jpg"
    
    if not os.path.exists(test_image):
        logging.error(f"Input image not found at {test_image}")
        exit()

    df = pd.read_csv(config.CSV_PATH)
    all_emotions = sorted(df['emotion'].unique().tolist())
    
    logging.info(f"Found {len(all_emotions)} unique emotions.")
    
    for i, emotion in enumerate(all_emotions):
        logging.info(f"[{i+1}/{len(all_emotions)}] | Starting job for: {emotion.upper()}")
        colour_correct(test_image, emotion, matcher, shared_generator)