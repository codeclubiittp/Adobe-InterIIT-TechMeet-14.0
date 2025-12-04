#!/usr/bin/env python3
"""
run_lama_onnx.py

Usage:
    python run_lama_onnx.py --onnx_zip path/to/onnx.zip \
        --image path/to/input.jpg --mask path/to/mask.png \
        --out out.png

If your mask is embedded as alpha in the input image, you can omit --mask;
the script will use the alpha channel as mask.

Preprocessing modes (--norm):
  - 0_1       : scale pixels to [0,1]             (default)
  - neg1_1    : scale to [-1,1]
  - imagenet  : normalize with ImageNet mean/std

You can override size (--size). Many LaMa exports expect 512 or 1024; default is 512.
"""
import argparse
import os
import zipfile
import tempfile
import glob
from pathlib import Path
import numpy as np
from PIL import Image
import onnxruntime as ort
import cv2
from tqdm import tqdm

def find_onnx_in_zip(zip_path):
    with zipfile.ZipFile(zip_path, 'r') as z:
        candidates = [n for n in z.namelist() if n.lower().endswith('.onnx')]
        if not candidates:
            raise RuntimeError(f"No .onnx file found inside {zip_path}")
        # prefer top-level or first candidate
        return candidates[0]

def load_image(path):
    img = Image.open(path).convert('RGBA')  # read alpha if present
    return img

def prepare_image_and_mask(img_pil, mask_pil, target_size):
    # img_pil is RGBA PIL.Image
    # mask_pil is single-channel PIL.Image or None
    w, h = img_pil.size
    # If mask not provided, try alpha channel
    if mask_pil is None:
        if img_pil.mode == 'RGBA':
            alpha = np.array(img_pil.split()[-1])
            mask = (alpha > 0).astype(np.uint8) * 0  # alpha>0 means content => keep (0)
            # Need mask=255 for holes; invert: where alpha==0 -> hole
            mask = np.where(alpha == 0, 255, 0).astype(np.uint8)
        else:
            raise RuntimeError("No mask provided and no alpha channel found in the input image.")
    else:
        mask = mask_pil.convert('L')
        mask = np.array(mask)
        # Binarize: treat >127 as hole (255)
        mask = np.where(mask > 127, 255, 0).astype(np.uint8)

    # Convert input rgb
    rgb = img_pil.convert('RGB')
    # Resize both to target_size keeping aspect by center-cropping/padding? We'll resize directly:
    target_w, target_h = target_size, target_size
    rgb = rgb.resize((target_w, target_h), Image.LANCZOS)
    mask = Image.fromarray(mask).resize((target_w, target_h), Image.NEAREST)
    rgb_arr = np.array(rgb)  # H W C (0..255)
    mask_arr = np.array(mask)  # H W (0/255)

    return rgb_arr, mask_arr

def preprocess(rgb_arr, mask_arr, norm_mode):
    # rgb_arr: H W 3, uint8
    # mask_arr: H W, uint8 (0 or 255)
    H, W = rgb_arr.shape[:2]

    img_f = rgb_arr.astype(np.float32)
    if norm_mode == '0_1':
        img_f = img_f / 255.0
    elif norm_mode == 'neg1_1':
        img_f = (img_f / 127.5) - 1.0
    elif norm_mode == 'imagenet':
        img_f = img_f / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img_f = (img_f - mean) / std
    else:
        raise ValueError("Unknown normalization mode: " + norm_mode)

    # model usually expects NCHW
    img_nchw = img_f.transpose(2, 0, 1)[np.newaxis, :]

    # mask to single channel with 1 for hole, 0 for keep
    mask_bin = (mask_arr > 127).astype(np.float32)[np.newaxis, np.newaxis, :, :]  # NCHW
    return img_nchw.astype(np.float32), mask_bin.astype(np.float32)

def run_onnx(session, input_name, mask_name, img_tensor, mask_tensor, additional_inputs={}):
    """
    session.run expects a dict mapping input names to numpy arrays.
    Some ONNX exports may have different input names; we try to detect them.
    """
    feed = {}
    # Provide detected names
    if input_name is not None:
        feed[input_name] = img_tensor
    if mask_name is not None:
        feed[mask_name] = mask_tensor
    # Add other optional inputs if present
    for k,v in additional_inputs.items():
        feed[k] = v
    outputs = session.run(None, feed)
    return outputs

def postprocess_and_save(output_arr, orig_size, out_path, norm_mode):
    # output_arr expected NCHW or NHWC; convert to HWC uint8
    out = output_arr
    # Try common shapes
    if out.ndim == 4:
        out = out[0]
    # if NCHW
    if out.shape[0] == 3 or out.shape[0] == 4:
        out = out.transpose(1,2,0)  # H W C

    # If normalized, map back to 0-255 depending on norm_mode
    if norm_mode == '0_1':
        out = np.clip(out * 255.0, 0, 255).astype(np.uint8)
    elif norm_mode == 'neg1_1':
        out = ((np.clip(out, -1, 1) + 1.0) * 127.5).astype(np.uint8)
    elif norm_mode == 'imagenet':
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        out = (out * std + mean) * 255.0
        out = np.clip(out, 0, 255).astype(np.uint8)
    else:
        out = np.clip(out, 0, 255).astype(np.uint8)

    # Resize back to original image size
    if (orig_size[0], orig_size[1]) != (out.shape[1], out.shape[0]):
        out = cv2.resize(out, (orig_size[0], orig_size[1]), interpolation=cv2.INTER_LINEAR)

    Image.fromarray(out).save(out_path)
    print(f"Saved result to {out_path}")

def guess_input_names(session):
    # Try to identify likely input names for image and mask
    input_meta = session.get_inputs()
    names = [inp.name for inp in input_meta]
    # heuristics:
    img_name = None
    mask_name = None
    for n in names:
        ln = n.lower()
        if 'image' in ln or 'input' in ln or 'rgb' in ln or 'img' in ln:
            img_name = n
        if 'mask' in ln or 'hole' in ln or 'mask_input' in ln:
            mask_name = n
    # fallback: first input as image, second as mask (if exists)
    if img_name is None and len(names) > 0:
        img_name = names[0]
    if mask_name is None and len(names) > 1:
        mask_name = names[1]
    return img_name, mask_name

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--onnx_zip', required=True, help='Path to onnx.zip downloaded from HuggingFace (or path to .onnx directly)')
    p.add_argument('--image', required=True, help='Path to input image (RGB or RGBA if using alpha as mask)')
    p.add_argument('--mask', required=False, default=None, help='Path to mask image (white=hole, black=keep). If omitted and image has alpha, alpha used.')
    p.add_argument('--out', default='out.png', help='Output image path')
    p.add_argument('--size', type=int, default=512, help='Model input size (square). Many LaMa variants use 512 or 1024')
    p.add_argument('--norm', choices=['0_1','neg1_1','imagenet'], default='0_1', help='Normalization used by the model')
    p.add_argument('--device', choices=['cpu','cuda'], default='cpu', help='onnxruntime provider target')
    args = p.parse_args()

        # Resolve ONNX file
    if args.onnx_zip.lower().endswith('.onnx'):
        onnx_path = args.onnx_zip
        if not os.path.exists(onnx_path):
            raise FileNotFoundError(onnx_path)
        tmpdir = None
    else:
        if not os.path.exists(args.onnx_zip):
            raise FileNotFoundError(args.onnx_zip)

        tmpdir = tempfile.TemporaryDirectory()
        print("Extracting entire ONNX package...")
        with zipfile.ZipFile(args.onnx_zip, 'r') as z:
            z.extractall(tmpdir.name)

        # search for .onnx file
        onnx_files = list(Path(tmpdir.name).rglob("*.onnx"))
        if len(onnx_files) == 0:
            raise RuntimeError("No .onnx file found inside zip")

        onnx_path = str(onnx_files[0])
        print("Using ONNX file:", onnx_path)


    print("Loading ONNX model:", onnx_path)
    # Setup providers
    providers = ['CPUExecutionProvider']
    if args.device == 'cuda':
        providers = ['CUDAExecutionProvider','CPUExecutionProvider']
    sess = ort.InferenceSession(onnx_path, providers=providers)

    # detect input names
    img_name, mask_name = guess_input_names(sess)
    print("Detected input names - image:", img_name, " mask:", mask_name)

    # Load image and mask
    img_pil = load_image(args.image)
    mask_pil = None
    if args.mask:
        mask_pil = Image.open(args.mask)

    orig_size = img_pil.size  # width, height
    # Make sure size divisible by 8
    target_size = args.size
    if target_size % 8 != 0:
        raise ValueError("Target size must be divisible by 8 for LaMa-like models.")
    rgb_arr, mask_arr = prepare_image_and_mask(img_pil, mask_pil, target_size)

    img_tensor, mask_tensor = preprocess(rgb_arr, mask_arr, args.norm)

    # attempt to map to model input names; if model expects different extras, they won't be provided
    # Some exported models expect also a 'ones' or 'coord' inputs; we try simple run first.
    try:
        outputs = run_onnx(sess, img_name, mask_name, img_tensor, mask_tensor)
    except Exception as e:
        # Try feeding by raw input order instead of names
        print("First run failed, retrying by feeding inputs in order. Error:", e)
        inp_meta = sess.get_inputs()
        feed = {}
        # assign image to first float input and mask to second if found
        if len(inp_meta) >= 1:
            feed[inp_meta[0].name] = img_tensor
        if len(inp_meta) >= 2:
            feed[inp_meta[1].name] = mask_tensor
        outputs = sess.run(None, feed)

    # Assume first output is the inpainted image
    out_arr = outputs[0]
    postprocess_and_save(out_arr, orig_size, args.out, args.norm)

    if tmpdir:
        tmpdir.cleanup()

if __name__ == '__main__':
    main()
