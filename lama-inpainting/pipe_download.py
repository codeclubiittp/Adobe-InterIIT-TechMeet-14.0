import numpy as np
import cv2
from datasets import load_dataset
from PIL import Image
import os
import shutil
import random
import argparse
from ultralytics import SAM


def init_sam_model(model_name="sam2.1_l.pt", device="cuda:0"):
    """Initialize SAM model"""
    model = SAM(model_name)
    model.to(device)
    return model


def show_difference_mask_sam(source_image, target_image, model, threshold_ratio=0.15):
    """Generate SAM mask from image differences"""
    source_gray = source_image.convert("L")
    target_gray = target_image.convert("L")

    src_arr = np.array(source_gray, dtype=np.int16)
    tgt_arr = np.array(target_gray, dtype=np.int16)

    # Absolute difference
    diff = np.abs(src_arr - tgt_arr)

    # Raw mask thresholded
    threshold_value = threshold_ratio * diff.max()
    raw_mask = (diff > threshold_value).astype(np.uint8) * 255

    # Smooth mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    tight_mask = cv2.morphologyEx(raw_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    tight_mask = cv2.morphologyEx(tight_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    tight_mask = cv2.GaussianBlur(tight_mask, (3, 3), 0)
    tight_mask = (tight_mask > 127).astype(np.uint8) * 255

    # Find contours and fit ellipse
    contours, _ = cv2.findContours(tight_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        all_points = np.vstack(contours)
        if len(all_points) >= 5:
            ellipse = cv2.fitEllipse(all_points)
            mask_ellipse = np.zeros_like(tight_mask)
            cv2.ellipse(mask_ellipse, ellipse, 255, -1)
        else:
            mask_ellipse = tight_mask.copy()

        # Construct bounding box from the ellipse
        ys, xs = np.where(mask_ellipse > 0)
        x_min, x_max = xs.min(), xs.max()
        y_min, y_max = ys.min(), ys.max()
        padding = 10
        x_min = max(0, x_min - padding)
        y_min = max(0, y_min - padding)
        x_max = min(mask_ellipse.shape[1], x_max + padding)
        y_max = min(mask_ellipse.shape[0], y_max + padding)
        bbox = [x_min, y_min, x_max, y_max]

        # Pass bounding box to SAM for segmentation
        results = model(target_image, bboxes=[bbox])
        sam_mask = results[0].masks.data[0].cpu().numpy() if results[0].masks is not None else None

        return sam_mask
    else:
        return None


def download_and_process_dataset(output_folder, max_size_gb=50, start_idx=1, 
                                 sam_model_name="sam2.1_l.pt", device="cuda:0"):
    """Download PIPE dataset and process with SAM"""
    
    max_size_bytes = max_size_gb * 1024 * 1024 * 1024
    os.makedirs(output_folder, exist_ok=True)
    
    # Initialize SAM model
    print("Initializing SAM model...")
    model = init_sam_model(sam_model_name, device)
    
    # Load dataset
    print("Loading PIPE dataset...")
    dataset = load_dataset("paint-by-inpaint/PIPE", streaming=True)
    train_iter = dataset["train"]
    
    total_size = 0
    idx = 0
    
    for row in train_iter:
        idx += 1
        if idx < start_idx:
            continue
        
        # Check size limit
        if total_size >= max_size_bytes:
            print(f"Reached size limit of {max_size_gb} GB")
            break
        
        # Check if images exist
        if "source_img" not in row or "target_img" not in row or row["source_img"] is None or row["target_img"] is None:
            print(f"Missing source or target image at index {idx}, skipping.")
            continue
        
        source_image = row["source_img"]
        target_image = row["target_img"]
        
        print(f"Processing image pair {idx}")
        
        try:
            sam_mask = show_difference_mask_sam(source_image, target_image, model)
            
            if sam_mask is None:
                print(f"No SAM mask for index {idx}, skipping.")
                continue
            
            # Create folder for this row
            row_folder = os.path.join(output_folder, str(idx))
            os.makedirs(row_folder, exist_ok=True)
            
            # Save images with swapped names (as per original code)
            source_path = os.path.join(row_folder, "source.jpg")
            target_path = os.path.join(row_folder, "target.jpg")
            mask_path = os.path.join(row_folder, "mask.png")
            
            target_image.save(source_path)
            source_image.save(target_path)
            
            mask_image = Image.fromarray((sam_mask * 255).astype(np.uint8))
            mask_image.save(mask_path)
            
            row_size = os.path.getsize(source_path) + os.path.getsize(target_path) + os.path.getsize(mask_path)
            total_size += row_size
            
            print(f"Saved to {row_folder}")
            print(f"Total size so far: {total_size / (1024**3):.2f} GB")
            print("-" * 50)
            
        except Exception as e:
            print(f"Error processing index {idx}: {e}")
            continue
    
    print(f"Processing complete. Total attempted: {idx} samples, {total_size / (1024**3):.2f} GB")


def split_dataset(parent_dir, output_dir, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
    """Split processed dataset into train/val/test splits"""
    
    # Make split dirs
    for split in ["train", "val", "test"]:
        os.makedirs(os.path.join(output_dir, split), exist_ok=True)
    
    # Numbered folders only
    folders = sorted([f for f in os.listdir(parent_dir) if f.isdigit()])
    random.shuffle(folders)
    
    n = len(folders)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)
    
    split_map = {
        "train": folders[:train_end],
        "val": folders[train_end:val_end],
        "test": folders[val_end:]
    }
    
    for split, dirs in split_map.items():
        for folder in dirs:
            fpath = os.path.join(parent_dir, folder)
            
            source = os.path.join(fpath, "source.jpg")
            mask = os.path.join(fpath, "mask.png")
            
            if not (os.path.exists(source) and os.path.exists(mask)):
                print(f"Skipping {folder}")
                continue
            
            # Copy into train/val/test directly
            out_img = os.path.join(output_dir, split, f"{folder}.jpg")
            out_mask = os.path.join(output_dir, split, f"{folder}_mask.png")
            
            shutil.copy(source, out_img)
            shutil.copy(mask, out_mask)
    
    print("Dataset split complete!")


def main():
    parser = argparse.ArgumentParser(description="Download and process PIPE dataset with SAM")
    
    # Download arguments
    parser.add_argument("--output_folder", type=str, default="pipe_dataset",
                       help="Output folder for downloaded dataset")
    parser.add_argument("--max_size_gb", type=float, default=50,
                       help="Maximum dataset size in GB")
    parser.add_argument("--start_idx", type=int, default=1,
                       help="Starting index for processing")
    parser.add_argument("--sam_model", type=str, default="sam2.1_l.pt",
                       help="SAM model name")
    parser.add_argument("--device", type=str, default="cuda:0",
                       help="Device to run SAM on")
    
    # Split arguments
    parser.add_argument("--split", action="store_true",
                       help="Split dataset into train/val/test")
    parser.add_argument("--split_output", type=str, default="pipe_split",
                       help="Output folder for split dataset")
    parser.add_argument("--train_ratio", type=float, default=0.8,
                       help="Training set ratio")
    parser.add_argument("--val_ratio", type=float, default=0.1,
                       help="Validation set ratio")
    parser.add_argument("--test_ratio", type=float, default=0.1,
                       help="Test set ratio")
    
    args = parser.parse_args()
    
    # Download and process dataset
    print("Starting dataset download and processing...")
    download_and_process_dataset(
        output_folder=args.output_folder,
        max_size_gb=args.max_size_gb,
        start_idx=args.start_idx,
        sam_model_name=args.sam_model,
        device=args.device
    )
    
    # Optionally split dataset
    if args.split:
        print("\nSplitting dataset...")
        split_dataset(
            parent_dir=args.output_folder,
            output_dir=args.split_output,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio
        )


if __name__ == "__main__":
    main()



# # Basic usage - download and process dataset
# python script.py --output_folder pipe_dataset --max_size_gb 50

# # Start from specific index
# python script.py --output_folder pipe_dataset --start_idx 94

# # Download and split in one go
# python script.py --output_folder pipe_dataset --split --split_output pipe_split

# # Custom split ratios
# python script.py --output_folder pipe_dataset --split --train_ratio 0.7 --val_ratio 0.15 --test_ratio 0.15

# # Use different SAM model or device
# python script.py --sam_model sam2.1_b.pt --device cuda:1