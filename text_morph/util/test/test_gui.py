import cv2
import numpy as np
from ultralytics import SAM, YOLO
import easyocr
import os

SAM_WEIGHTS = "sam2.1_b.pt"
IMG_PATH = "samples/3.jpg"
EXPAND_PX = 6
IOU_THRESHOLD = 0.1

# Initialize YOLO and EasyOCR
pwd = os.getcwd()
yolo_path = os.path.join(pwd, "yolo_mscoco", "weights", "best.pt")
yolo_model = YOLO(yolo_path)
reader = easyocr.Reader(['en'], gpu=True)

def text_ocr(img, cd):
    results = []
    if isinstance(img, str):
        img = cv2.imread(img)
        if img is None:
            raise FileNotFoundError(img)
    h, w = img.shape[:2]

    for i in cd:
        x1, y1, x2, y2 = i[0]
        crop = img[y1:y2, x1:x2]
        if crop.size == 0:
            results.append("")
            continue
        ocr = reader.readtext(crop, detail=1, paragraph=True)
        text = ""
        if ocr:
            text = " ".join([t[1] for t in ocr if len(t) > 1 and t[1]])
        results.append(text)
    return results

def get_boxes(img_path, scale=0.6, pad=200):
    if isinstance(img_path, str):
        orig = cv2.imread(img_path)
        if orig is None:
            raise FileNotFoundError(img_path)
    else:
        orig = img_path
    h_orig, w_orig = orig.shape[:2]

    img = cv2.resize(orig, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    h, w = img.shape[:2]
    white = np.ones((h + 2*pad, w + 2*pad, 3), dtype=np.uint8) * 255
    white[pad:pad+h, pad:pad+w] = img
    padded = white

    results = yolo_model.predict(padded)
    r = results[0]

    boxes = r.boxes.xyxy
    confs = r.boxes.conf

    try:
        boxes = boxes.cpu().numpy()
        confs = confs.cpu().numpy()
    except:
        boxes = np.array(boxes)
        confs = np.array(confs)

    boxes[:, [0, 2]] -= pad
    boxes[:, [1, 3]] -= pad
    boxes = boxes / scale

    boxes[:, 0] = np.clip(boxes[:, 0], 0, w_orig - 1)
    boxes[:, 2] = np.clip(boxes[:, 2], 0, w_orig - 1)
    boxes[:, 1] = np.clip(boxes[:, 1], 0, h_orig - 1)
    boxes[:, 3] = np.clip(boxes[:, 3], 0, h_orig - 1)

    out = []
    for i in range(len(boxes)):
        coords = [int(round(x)) for x in boxes[i].tolist()]
        out.append([coords, float(confs[i])])

    return out

def bbox_from_mask(mask):
    ys, xs = np.where(mask)
    if ys.size == 0:
        return None
    x1, y1, x2, y2 = int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())
    return x1, y1, x2, y2

def expand_box(box, expand_px, img_w, img_h):
    x1, y1, x2, y2 = box
    x1 = max(0, x1 - expand_px)
    y1 = max(0, y1 - expand_px)
    x2 = min(img_w - 1, x2 + expand_px)
    y2 = min(img_h - 1, y2 + expand_px)
    return x1, y1, x2, y2

def compute_iou(box1, box2):
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

def merge_overlapping_boxes(bboxes, threshold=0.1):
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

def run():
    model = SAM(SAM_WEIGHTS)
    img_bgr = cv2.imread(IMG_PATH)
    if img_bgr is None:
        raise SystemExit(f"Image not found: {IMG_PATH}")
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    H, W = img_rgb.shape[:2]

    window = "SAM Live Click - Left: add | Right: remove last | c: clear | e: export | q: quit"
    points = []
    masks_list = []
    bboxes_list = []

    def update_display():
        disp = img_bgr.copy()
        
        for mask in masks_list:
            mask_vis = (mask.astype(np.uint8) * 120)[:, :, None]
            overlay = cv2.cvtColor(mask_vis, cv2.COLOR_GRAY2BGR)
            disp = cv2.addWeighted(disp, 1.0, overlay, 0.6, 0)
        
        merged_bboxes = merge_overlapping_boxes(bboxes_list, IOU_THRESHOLD)
        
        for bbox in merged_bboxes:
            x1, y1, x2, y2 = bbox
            cv2.rectangle(disp, (x1, y1), (x2, y2), (0, 0, 255), 2)
        
        for (px, py) in points:
            cv2.circle(disp, (px, py), 4, (0, 255, 0), -1)
        
        cv2.imshow(window, disp)

    def run_point_seg(px, py):
        try:
            results = model(
                img_rgb,
                points=[[int(px), int(py)]],
                labels=[1]
            )
            
            r = results[0]
            
            if not hasattr(r, 'masks') or r.masks is None:
                print(f"No mask returned for point ({px}, {py})")
                return None, None
            
            masks_arr = r.masks.data
            
            if masks_arr is None or len(masks_arr) == 0:
                print(f"Empty mask for point ({px}, {py})")
                return None, None
            
            mask = masks_arr[0].cpu().numpy() if hasattr(masks_arr[0], 'cpu') else masks_arr[0]
            mask = (mask > 0.5).astype(bool)
            
            bbox = bbox_from_mask(mask)
            if bbox is not None:
                bbox = expand_box(bbox, EXPAND_PX, W, H)
            
            return mask, bbox
            
        except Exception as e:
            print(f"SAM inference error: {e}")
            import traceback
            traceback.print_exc()
            return None, None

    def mouse_cb(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append((x, y))
            mask, bbox = run_point_seg(x, y)
            
            if mask is not None:
                masks_list.append(mask)
            if bbox is not None:
                bboxes_list.append(bbox)
            
            update_display()
            
        elif event == cv2.EVENT_RBUTTONDOWN:
            if points:
                points.pop()
                if masks_list:
                    masks_list.pop()
                if bboxes_list:
                    bboxes_list.pop()
                update_display()

    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window, mouse_cb)
    cv2.imshow(window, img_bgr)

    while True:
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord("q"):
            break
        elif key == ord("c"):
            points.clear()
            masks_list.clear()
            bboxes_list.clear()
            cv2.imshow(window, img_bgr)
        elif key == ord("e"):
            merged_bboxes = merge_overlapping_boxes(bboxes_list, IOU_THRESHOLD)
            word_count = 0
            
            for i, bbox in enumerate(merged_bboxes):
                x1, y1, x2, y2 = bbox
                crop = img_bgr[y1:y2+1, x1:x2+1]
                
                # Get YOLO boxes for this crop
                yolo_boxes = get_boxes(crop)
                
                # Get OCR text for this crop
                ocr_results = text_ocr(crop, yolo_boxes)
                
                # Save each word in its own folder
                for j, (box_data, text) in enumerate(zip(yolo_boxes, ocr_results)):
                    bx1, by1, bx2, by2 = box_data[0]
                    word_crop = crop[by1:by2+1, bx1:bx2+1]
                    
                    if word_crop.size == 0:
                        continue
                    
                    # Create folder for this word
                    word_folder = f"word_{word_count}"
                    os.makedirs(word_folder, exist_ok=True)
                    
                    # Save cropped word image
                    img_filename = os.path.join(word_folder, "image.png")
                    cv2.imwrite(img_filename, word_crop)
                    
                    # Save OCR text
                    txt_filename = os.path.join(word_folder, "text.txt")
                    with open(txt_filename, 'w', encoding='utf-8') as f:
                        f.write(text)
                    
                    print(f"Saved {word_folder}/ with image and text")
                    word_count += 1
                
        elif key == ord("r"):
            if points:
                points.pop()
                if masks_list:
                    masks_list.pop()
                if bboxes_list:
                    bboxes_list.pop()
                update_display()

    cv2.destroyAllWindows()

if __name__ == "__main__":
    run()