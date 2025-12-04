import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.colors as mcolors
from PIL import Image

def _get_colors(num_colors):
    cmap = plt.cm.get_cmap('hsv', num_colors)
    return [mcolors.to_hex(cmap(i)) for i in range(num_colors)]

def visualize_pose_predictions(
    image: np.ndarray, 
    coords1: np.ndarray, 
    coords2: np.ndarray, 
    scores1: np.ndarray, 
    scores2: np.ndarray, 
    savefile: str = "images/estimated_pose.png",
    score_thresh: float = 0.5
):
    coords = np.concatenate([coords1[0], coords2[0]], axis=0)  
    scores = np.concatenate([scores1[0], scores2[0]], axis=0)  
    
    h, w, _ = image.shape
    coords_scaled = coords.copy()
    coords_scaled[..., 0::2] *= w   
    coords_scaled[..., 1::2] *= h   
    
    scores = 1/(1 + np.exp(-scores))
    
    mask = scores[:, 0] > score_thresh
    coords_scaled = coords_scaled[mask]
    scores = scores[mask]
    
    num_poses = len(coords_scaled)
    print(f"visualizing {num_poses} poses above threshold {score_thresh}")

    if num_poses == 0:
        return

    colors = _get_colors(num_poses)
    plt.figure(figsize=(10, 10)) 
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    
    for i, c in enumerate(coords_scaled):
        color = colors[i]
        for j in range(0, len(c), 2):
            x, y = c[j], c[j+1]
            plt.scatter(x, y, s=20, c=color, alpha=0.9)
        plt.text(c[0], c[1] - 5, f"{scores[i][0]:.2f}", color='white', fontsize=9, 
                 bbox=dict(facecolor=color, alpha=0.8, pad=1)) 
    plt.axis("off")
    plt.savefig(savefile)
    plt.close()

def visualize_object_detection(
    image: np.ndarray,
    boxes: np.ndarray,      
    scores: np.ndarray,     
    class_idx: np.ndarray,  
    savefile: str = "images/estimated_boxes.png",
    score_thresh: float = 0.4, 
    nms_thresh: float = 0.45   
):
    print(f"raw scores min: {np.min(scores):.4f}, max: {np.max(scores):.4f}, mean: {np.mean(scores):.4f}")

    scores = 1/(1 + np.exp(-scores)) 

    print(f"sigmoid scores min: {np.min(scores):.4f}, max: {np.max(scores):.4f}, mean: {np.mean(scores):.4f}")
    
    mask = scores > score_thresh
    boxes_masked = boxes[mask]
    scores_masked = scores[mask]
    class_idx_masked = class_idx[mask]

    if len(boxes_masked) == 0:
        print(f"no objects found above threshold {score_thresh}")
        return

    boxes_for_nms = boxes_masked.copy()
    boxes_for_nms[:, 0] = boxes_masked[:, 0] - boxes_masked[:, 2] / 2  
    boxes_for_nms[:, 1] = boxes_masked[:, 1] - boxes_masked[:, 3] / 2  

    indices = cv2.dnn.NMSBoxes(
        boxes_for_nms.tolist(),
        scores_masked.tolist(),
        score_thresh,
        nms_thresh
    )

    if isinstance(indices, tuple) and len(indices) == 0:
         print("no objects found after nms")
         return
    
    if len(indices.shape) > 1:
        indices = indices.flatten()

    final_boxes = boxes_masked[indices]
    final_scores = scores_masked[indices]
    final_classes = class_idx_masked[indices]

    print(f"visualizing {len(final_boxes)} final objects after nms.")

    h, w, _ = image.shape
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    ax = plt.gca()

    for i, box in enumerate(final_boxes):
        score = final_scores[i]
        class_id = final_classes[i]

        cx, cy, bw, bh = box
        x_min = (cx - bw / 2) * w
        y_min = (cy - bh / 2) * h
        box_width = bw * w
        box_height = bh * h

        rect = patches.Rectangle(
            (x_min, y_min), box_width, box_height,
            linewidth=2, edgecolor='r', facecolor='none'
        )
        ax.add_patch(rect)
        
        label = f"class {class_id}: {score:.2f}"
        plt.text(
            x_min, y_min - 5, label,
            color='white', fontsize=9,
            bbox=dict(facecolor='red', alpha=0.5, pad=1)
        )
            
    plt.axis("off")
    plt.savefig(savefile)
    plt.close()

def visualize_depth_map(normalized_map: np.ndarray, save_path: str):
    cmap = plt.get_cmap('magma') 
    colored_map_rgba = cmap(normalized_map)
    colored_map_rgb_uint8 = (colored_map_rgba[:, :, :3] * 255).astype(np.uint8)
    Image.fromarray(colored_map_rgb_uint8).save(save_path)
    print(f"saved depth map to {save_path}")