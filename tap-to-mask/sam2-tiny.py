import os
import numpy as np
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import Response, JSONResponse
from PIL import Image
import cv2
import torch
from transformers import pipeline
import io
import base64
from typing import List, Dict, Any, Union, Optional
from fastapi.middleware.cors import CORSMiddleware
try:
    from pycocotools import mask as coco_mask
    _have_pycocotools = True
except Exception:
    _have_pycocotools = False


def tensor_to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return x


def decode_possible_rle(rle_obj):
    if not _have_pycocotools:
        return None
    try:
        if isinstance(rle_obj, dict) and 'counts' in rle_obj and 'size' in rle_obj:
            decoded = coco_mask.decode(rle_obj)
            if decoded.ndim == 3:
                decoded = np.any(decoded, axis=2).astype(np.uint8)
            return (decoded > 0).astype(bool)
    except Exception:
        return None
    return None


def normalize_mask_obj(m):
    if isinstance(m, dict):
        for key in ("mask", "mask_image", "segmentation", "segmentation_mask", "seg", "rle"):
            if key in m and m[key] is not None:
                val = m[key]
                rle_decoded = decode_possible_rle(val)
                if rle_decoded is not None:
                    return rle_decoded.astype(bool)

                if isinstance(val, (np.ndarray, list)):
                    arr = np.asarray(val)
                    if arr.ndim == 3:
                        arr = arr[..., 0]
                    return (arr > 0).astype(bool)

                if isinstance(val, torch.Tensor):
                    arr = tensor_to_numpy(val)
                    if arr.ndim == 3:
                        arr = arr[..., 0]
                    return (arr > 0).astype(bool)

                if isinstance(val, Image.Image):
                    arr = np.asarray(val.convert("L"))
                    return (arr > 0).astype(bool)

        for v in m.values():
            try:
                if isinstance(v, (np.ndarray, torch.Tensor, Image.Image)):
                    arr = v
                    if isinstance(arr, torch.Tensor):
                        arr = tensor_to_numpy(arr)
                    if isinstance(arr, Image.Image):
                        arr = np.asarray(arr.convert("L"))
                    arr = np.asarray(arr)
                    if arr.ndim == 3:
                        arr = arr[..., 0]
                    return (arr > 0).astype(bool)
            except Exception:
                continue

        return None

    if isinstance(m, torch.Tensor):
        arr = tensor_to_numpy(m)
        if arr.ndim == 3:
            arr = arr[..., 0]
        return (arr > 0).astype(bool)

    if isinstance(m, np.ndarray):
        arr = m
        if arr.ndim == 3:
            arr = arr[..., 0]
        return (arr > 0).astype(bool)

    if isinstance(m, Image.Image):
        arr = np.asarray(m.convert("L"))
        return (arr > 0).astype(bool)

    rle_decoded = decode_possible_rle(m)
    if rle_decoded is not None:
        return rle_decoded.astype(bool)

    return None


def pick_mask_for_point(masks, x, y):
    normalized_list = []
    debug_log = []

    for i, m in enumerate(masks):
        mask_bool = normalize_mask_obj(m)
        if mask_bool is None:
            debug_log.append(f"Mask {i}: could not normalize, type={type(m)}")
        else:
            normalized_list.append((i, mask_bool))

    if not normalized_list:
        return None, debug_log

    for i, mask_bool in normalized_list:
        h, w = mask_bool.shape[:2]
        if 0 <= y < h and 0 <= x < w and mask_bool[y, x]:
            debug_log.append(f"Mask {i} selected because tap is inside.")
            return mask_bool, debug_log

    best = None
    best_dist = float("inf")
    for i, mask_bool in normalized_list:
        ys, xs = np.where(mask_bool)
        if len(xs) == 0:
            continue
        cx = xs.mean()
        cy = ys.mean()
        d = (cx - x) ** 2 + (cy - y) ** 2
        if d < best_dist:
            best_dist = d
            best = (i, mask_bool)

    if best:
        debug_log.append(f"Mask {best[0]} selected by centroid, dist={best_dist}")
        return best[1], debug_log

    return None, debug_log


def create_rgba(img_arr, mask_bool):
    mask = (mask_bool.astype("uint8") * 255)

    if img_arr.shape[:2] != mask.shape[:2]:
        mask = cv2.resize(mask, (img_arr.shape[1], img_arr.shape[0]), cv2.INTER_NEAREST)

    rgba = np.zeros((img_arr.shape[0], img_arr.shape[1], 4), dtype=np.uint8)
    rgba[..., :3] = img_arr
    rgba[..., 3] = mask
    return rgba


app = FastAPI(title="SAM2 Tiny Point Segment API")

MODEL = "facebook/sam2-hiera-tiny"
DEVICE = 0 if torch.cuda.is_available() else -1
GEN = None

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins (React is usually localhost:5173 or 3000)
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

@app.on_event("startup")
def load_pipeline():
    global GEN
    GEN = pipeline("mask-generation", model=MODEL, device=DEVICE)


@app.post("/segment")
async def segment(
    input_image: UploadFile = File(...),
    x: int = Form(...),
    y: int = Form(...)
):
    if GEN is None:
        raise RuntimeError("Pipeline not loaded yet.")

    raw = await input_image.read()
    img = Image.open(io.BytesIO(raw)).convert("RGB")
    img_arr = np.asarray(img)

    # Run segmentation
    outputs = GEN(img)

    if isinstance(outputs, dict):
        if "masks" in outputs:
            masks = outputs["masks"]
        elif "predictions" in outputs:
            masks = outputs["predictions"]
        else:
            masks = [outputs]
    elif isinstance(outputs, (list, tuple)):
        masks = list(outputs)
    else:
        masks = [outputs]

    chosen, debug_log = pick_mask_for_point(masks, x, y)

    if chosen is None:
        return JSONResponse({"error": "no mask found", "debug": debug_log})

    # Convert mask into PNG bytes
    mask_uint8 = (chosen.astype("uint8") * 255)
    mask_img = Image.fromarray(mask_uint8)
    mask_bytes = io.BytesIO()
    mask_img.save(mask_bytes, format="PNG")
    mask_bytes.seek(0)

    # Build RGBA PNG
    rgba_arr = create_rgba(img_arr, chosen)
    rgba_img = Image.fromarray(rgba_arr)
    rgba_bytes = io.BytesIO()
    rgba_img.save(rgba_bytes, format="PNG")
    rgba_bytes.seek(0)

    return {
        # .decode('utf-8') turns the base64 bytes into a standard string for JSON
        "mask_png": base64.b64encode(mask_bytes.getvalue()).decode("utf-8"),
        "rgba_png": base64.b64encode(rgba_bytes.getvalue()).decode("utf-8"),
        "debug": debug_log,
    }
