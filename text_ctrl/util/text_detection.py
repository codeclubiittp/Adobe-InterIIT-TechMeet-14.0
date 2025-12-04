from ultralytics import YOLO
import cv2
import os
import numpy as np

# loading the model weights
pwd = os.getcwd()
path = os.path.join(pwd,"yolo_mscoco","weights", "best.pt")
model = YOLO(path)

# getting bounding boxes coordinates for individual words present in the cropped image
def get_boxes(img_path, scale=0.6, pad=200):
    # load original image (so we can map back)
    if isinstance(img_path, str):
        orig = cv2.imread(img_path)
        if orig is None:
            raise FileNotFoundError(img_path)
    else:
        orig = img_path
    h_orig, w_orig = orig.shape[:2]

    # resize and pad (detection happens here)
    img = cv2.resize(orig, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    h, w = img.shape[:2]
    white = np.ones((h + 2*pad, w + 2*pad, 3), dtype=np.uint8) * 255
    white[pad:pad+h, pad:pad+w] = img
    padded = white

    results = model.predict(padded)
    r = results[0]

    boxes = r.boxes.xyxy
    confs = r.boxes.conf

    try:
        boxes = boxes.cpu().numpy()
        confs = confs.cpu().numpy()
    except:
        boxes = np.array(boxes)
        confs = np.array(confs)

    # remove padding (still in resized coords)
    boxes[:, [0, 2]] -= pad
    boxes[:, [1, 3]] -= pad

    # map resized coords back to original image coords
    boxes = boxes / scale

    # clip to original dims
    boxes[:, 0] = np.clip(boxes[:, 0], 0, w_orig - 1)
    boxes[:, 2] = np.clip(boxes[:, 2], 0, w_orig - 1)
    boxes[:, 1] = np.clip(boxes[:, 1], 0, h_orig - 1)
    boxes[:, 3] = np.clip(boxes[:, 3], 0, h_orig - 1)

    out = []
    for i in range(len(boxes)):
        coords = [int(round(x)) for x in boxes[i].tolist()]
        out.append([coords, float(confs[i])])

    return out



# Example Usage ##
# result = get_boxes("crop_0.png")
# >> [
#  [ [x1, y1, x2, y2], confidence ],
#  [ [x1, y1, x2, y2], confidence ],
#  ]
