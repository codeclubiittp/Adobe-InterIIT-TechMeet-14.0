import numpy as np
from typing import Optional, Tuple, List
from ultralytics import SAM

# Configuration
SAM_WEIGHTS = "util/sam2.1_b.pt"
EXPAND_PX = 6
IOU_THRESHOLD = 0.1

# Initialize SAM model
sam_model = SAM(SAM_WEIGHTS)


def bbox_from_mask(mask: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    """Extract bounding box from binary mask."""
    ys, xs = np.where(mask)
    if ys.size == 0:
        return None
    x1, y1, x2, y2 = int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())
    return x1, y1, x2, y2


def expand_box(box: Tuple[int, int, int, int], expand_px: int, img_w: int, img_h: int) -> Tuple[int, int, int, int]:
    """Expand bounding box by expand_px pixels."""
    x1, y1, x2, y2 = box
    x1 = max(0, x1 - expand_px)
    y1 = max(0, y1 - expand_px)
    x2 = min(img_w - 1, x2 + expand_px)
    y2 = min(img_h - 1, y2 + expand_px)
    return x1, y1, x2, y2


def compute_iou(box1: Tuple[int, int, int, int], box2: Tuple[int, int, int, int]) -> float:
    """Compute Intersection over Union (IoU) between two boxes."""
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2
    
    xi1 = max(x1_1, x1_2)
    yi1 = max(y1_1, y1_2)
    xi2 = min(x2_1, x2_2)
    yi2 = min(y2_1, y2_2)
    
    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    
    box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
    box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
    
    union_area = box1_area + box2_area - inter_area
    
    if union_area == 0:
        return 0
    
    return inter_area / union_area


def merge_overlapping_boxes(bboxes: List[Tuple[int, int, int, int]], threshold: float = 0.1) -> List[Tuple[int, int, int, int]]:
    """Merge overlapping bounding boxes based on IoU threshold."""
    if len(bboxes) <= 1:
        return bboxes
    
    merged = []
    used = [False] * len(bboxes)
    
    for i in range(len(bboxes)):
        if used[i]:
            continue
        
        current_box = list(bboxes[i])
        used[i] = True
        merged_any = True
        
        while merged_any:
            merged_any = False
            for j in range(len(bboxes)):
                if used[j]:
                    continue
                
                iou = compute_iou(current_box, bboxes[j])
                
                if iou > threshold:
                    x1 = min(current_box[0], bboxes[j][0])
                    y1 = min(current_box[1], bboxes[j][1])
                    x2 = max(current_box[2], bboxes[j][2])
                    y2 = max(current_box[3], bboxes[j][3])
                    current_box = [x1, y1, x2, y2]
                    used[j] = True
                    merged_any = True
        
        merged.append(tuple(current_box))
    
    return merged


def run_point_segmentation(img_rgb: np.ndarray, px: int, py: int) -> Tuple[Optional[np.ndarray], Optional[Tuple[int, int, int, int]]]:
    """Run SAM segmentation on a single point and return mask and bbox."""
    try:
        H, W = img_rgb.shape[:2]
        
        results = sam_model(
            img_rgb,
            points=[[int(px), int(py)]],
            labels=[1]
        )
        
        r = results[0]
        
        if not hasattr(r, 'masks') or r.masks is None:
            return None, None
        
        masks_arr = r.masks.data
        
        if masks_arr is None or len(masks_arr) == 0:
            return None, None
        
        mask = masks_arr[0].cpu().numpy() if hasattr(masks_arr[0], 'cpu') else masks_arr[0]
        mask = (mask > 0.5).astype(bool)
        
        bbox = bbox_from_mask(mask)
        if bbox is not None:
            bbox = expand_box(bbox, EXPAND_PX, W, H)
        
        return mask, bbox
        
    except Exception as e:
        print(f"SAM inference error: {e}")
        return None, None



from pydantic import BaseModel
from typing import List


class Point(BaseModel):
    x: int
    y: int


class AddPointRequest(BaseModel):
    session_id: str
    point: Point


class BoundingBox(BaseModel):
    x1: int
    y1: int
    x2: int
    y2: int


class SessionState(BaseModel):
    session_id: str
    points: List[Point]
    bboxes: List[BoundingBox]
    merged_bboxes: List[BoundingBox]