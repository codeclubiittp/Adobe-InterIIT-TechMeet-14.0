import cv2
import numpy as np
import easyocr

reader = easyocr.Reader(['en'], gpu=True)


def text_ocr(img,cd):
    results = []
    if isinstance(img, str):
        img = cv2.imread(img)
        if img is None:
            raise FileNotFoundError(img)
    h,w = img.shape[:2]

    for i in cd:
        x1,y1,x2,y2 = i[0]
        crop = img[y1:y2, x1:x2]
        if crop.size==0:
            results.append("")
            continue
        ocr = reader.readtext(crop,detail=1,paragraph=True)
        text=""
        if ocr:
            text = " ".join([t[1] for t in ocr if len(t) > 1 and t[1]])
        results.append(text)
    return results



