#!/usr/bin/env python3
"""
Relighting using MiDaS depth → normals + your shading, spotlight, guided upsample.
Now includes PyTorch profiling for the MiDaS inference part only.

Usage:
    python relight_midas.py --input input.jpg --out relight_out \
        --az 45 --el 30 --intensity 1.2 --ambient 0.25 --spot_mode directional
"""

import argparse
import math
from pathlib import Path
import numpy as np
from PIL import Image
import cv2
import torch
import torch.nn.functional as F
import timm

# ================================================================
# Utility Functions
# ================================================================

def load_image(path):
    img = Image.open(path).convert("RGB")
    arr = np.asarray(img).astype(np.float32) / 255.0
    return arr, img

def save_image_uint8(path, arr):
    arr8 = np.clip(arr * 255.0 + 0.5, 0, 255).astype(np.uint8)
    Image.fromarray(arr8).save(path)

def srgb_to_linear(img):
    a = 0.055
    mask = img <= 0.04045
    out = np.empty_like(img)
    out[mask] = img[mask] / 12.92
    out[~mask] = ((img[~mask] + a)/(1.0 + a))**2.4
    return out

def linear_to_srgb(img):
    a = 0.055
    mask = img <= 0.0031308
    out = np.empty_like(img)
    out[mask] = img[mask] * 12.92
    out[~mask] = (1.0 + a)*(img[~mask]**(1/2.4)) - a
    return np.clip(out, 0, 1)

# ================================================================
# Guided Upsampling
# ================================================================

def guided_upsample_shading(sh_low, guide_full, radius=8, eps=1e-4):
    Hf, Wf = guide_full.shape[:2]
    sh = cv2.resize(sh_low, (Wf, Hf), interpolation=cv2.INTER_LINEAR)

    if guide_full.ndim == 3:
        guide_gray = 0.299*guide_full[...,0] + 0.587*guide_full[...,1] + 0.114*guide_full[...,2]
    else:
        guide_gray = guide_full
    
    I = guide_gray.astype(np.float32)
    p = sh.astype(np.float32)
    r = radius

    mean_I  = cv2.boxFilter(I, -1, (r,r))
    mean_p  = cv2.boxFilter(p, -1, (r,r))
    mean_Ip = cv2.boxFilter(I*p, -1, (r,r))
    cov_Ip  = mean_Ip - mean_I*mean_p

    mean_II = cv2.boxFilter(I*I, -1, (r,r))
    var_I   = mean_II - mean_I*mean_I

    a = cov_Ip / (var_I + eps)
    b = mean_p - a*mean_I

    mean_a = cv2.boxFilter(a, -1, (r,r))
    mean_b = cv2.boxFilter(b, -1, (r,r))
    out = mean_a * I + mean_b
    return np.clip(out, 0, 1)

# ================================================================
# MiDaS Depth → Normals
# ================================================================

@torch.no_grad()
def load_midas(device="cuda"):
    model = torch.hub.load("intel-isl/MiDaS", "DPT_Hybrid")
    model.to(device).eval()
    transform = torch.hub.load("intel-isl/MiDaS", "transforms").dpt_transform
    return model, transform

def depth_to_normals(depth):
    depth = depth.astype(np.float32)
    H, W = depth.shape

    dzdx = cv2.Sobel(depth, cv2.CV_32F, 1, 0, ksize=3)
    dzdy = cv2.Sobel(depth, cv2.CV_32F, 0, 1, ksize=3)

    nx = -dzdx
    ny = -dzdy
    nz = np.ones_like(depth)

    normals = np.stack([nx, ny, nz], axis=-1)
    n = np.linalg.norm(normals, axis=-1, keepdims=True) + 1e-8
    return (normals / n).astype(np.float32)

def normals_visual(normals):
    return (normals * 0.5 + 0.5).clip(0, 1)

# ================================================================
# Shading
# ================================================================

def compute_shading_from_normals(
    normals,
    light_azimuth_deg=45.0,
    light_elevation_deg=30.0,
    spot_mode="directional",
    cone_angle_deg=10.0,
    spot_exponent=60.0,
    screen_center=(0.5, 0.5),
    screen_falloff_radius=0.25
):
    H, W = normals.shape[:2]

    az = math.radians(light_azimuth_deg)
    el = math.radians(light_elevation_deg)
    L = np.array([
        math.cos(el)*math.cos(az),
        math.cos(el)*math.sin(az),
        math.sin(el)
    ], dtype=np.float32).reshape(1,1,3)

    cos_theta = np.sum(normals * L, axis=-1)
    cos_theta = np.clip(cos_theta, 0, 1)

    if spot_mode == "directional":
        ang = np.arccos(np.clip(cos_theta, -1, 1))
        cone = math.radians(cone_angle_deg)
        spot_mask = np.where(ang <= cone,
                             ((cone - ang) / cone)**spot_exponent,
                             0)
        return np.clip(cos_theta * spot_mask, 0, 1)

    elif spot_mode == "screen":
        ys = (np.arange(H) + 0.5)/H
        xs = (np.arange(W) + 0.5)/W
        xv, yv = np.meshgrid(xs, ys)

        dx = xv - screen_center[0]
        dy = yv - screen_center[1]
        dist = np.sqrt(dx*dx + dy*dy)

        spot_mask = np.clip(1 - dist/screen_falloff_radius, 0, 1)
        spot_mask = spot_mask**(spot_exponent/10)
        return np.clip(cos_theta * spot_mask, 0, 1)

    return cos_theta

# ================================================================
# Relighting
# ================================================================

def apply_relighting(img_srgb, shading, intensity=1.0, ambient=0.25):
    img_lin = srgb_to_linear(img_srgb)
    lighting = ambient + intensity * shading[...,None]
    out_lin = img_lin * lighting
    return np.clip(linear_to_srgb(out_lin), 0, 1)

# ================================================================
# Main Pipeline (Profiling added)
# ================================================================

def relight_pipeline(
    image_path,
    out_dir,
    device="cuda",
    downsize=768,
    az=45.0, el=30.0,
    intensity=1.2, ambient=0.25,
    guided_radius=8, guided_eps=1e-4,
    spot_mode="directional",
    spot_cone=10.0, spot_exponent=60.0,
    screen_center=(0.5,0.5),
    screen_spot_radius=0.25
):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    img_rgb, pil = load_image(image_path)
    H, W = img_rgb.shape[:2]

    # -------------------------
    # Load MiDaS
    # -------------------------
    model, transform = load_midas(device)

    # Resize for MiDaS
    if max(H,W) > downsize:
        scale = downsize / max(H,W)
        new_w, new_h = int(W*scale), int(H*scale)
        img_small = cv2.resize(img_rgb, (new_w,new_h), interpolation=cv2.INTER_AREA)
    else:
        img_small = img_rgb.copy()

    # Prepare tensor
    input_tensor = transform((img_small*255).astype(np.uint8)).to(device)

    # ============================================================
    # ⭐ PROFILING: MiDaS Inference Only ⭐
    # ============================================================
    print("[INFO] Profiling MiDaS inference...")

    with torch.autograd.profiler.profile(
        use_cuda=(device=="cuda"),
        record_shapes=True,
        profile_memory=True
    ) as prof:

        depth = model(input_tensor)

    print(prof.key_averages().table(sort_by="self_cpu_time_total"))
    # ============================================================

    depth = depth.cpu().detach().numpy().squeeze()
    depth_norm = (depth - depth.min()) / (depth.max() - depth.min())

    # Convert depth → normals
    normals_low = depth_to_normals(depth_norm)

    # Compute shading
    shading_low = compute_shading_from_normals(
        normals_low,
        light_azimuth_deg=az,
        light_elevation_deg=el,
        spot_mode=spot_mode,
        cone_angle_deg=spot_cone,
        spot_exponent=spot_exponent,
        screen_center=screen_center,
        screen_falloff_radius=screen_spot_radius
    )

    # Guided upsampling
    shading_full = guided_upsample_shading(
        shading_low, srgb_to_linear(img_rgb),
        radius=guided_radius, eps=guided_eps
    )

    # Relighting
    relit = apply_relighting(img_rgb, shading_full, intensity, ambient)

    # Save outputs
    save_image_uint8(out_dir/"normals.png", normals_visual(normals_low))
    save_image_uint8(out_dir/"shading.png", shading_full)
    save_image_uint8(out_dir/"relit.png", relit)

    print(f"[DONE] Results in: {out_dir}")

# ================================================================
# CLI
# ================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--out", default="relight_out")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--downsize", type=int, default=768)

    parser.add_argument("--az", type=float, default=45.0)
    parser.add_argument("--el", type=float, default=30.0)
    parser.add_argument("--intensity", type=float, default=1.2)
    parser.add_argument("--ambient", type=float, default=0.25)

    parser.add_argument("--guided_radius", type=int, default=8)
    parser.add_argument("--guided_eps", type=float, default=1e-4)

    parser.add_argument("--spot_mode", choices=["directional","screen"], default="directional")
    parser.add_argument("--spot_cone", type=float, default=10.0)
    parser.add_argument("--spot_exponent", type=float, default=60.0)
    parser.add_argument("--spot_center_x", type=float, default=0.5)
    parser.add_argument("--spot_center_y", type=float, default=0.5)
    parser.add_argument("--screen_spot_radius", type=float, default=0.25)

    args = parser.parse_args()

    relight_pipeline(
        image_path=args.input,
        out_dir=args.out,
        device=args.device,
        downsize=args.downsize,
        az=args.az,
        el=args.el,
        intensity=args.intensity,
        ambient=args.ambient,
        guided_radius=args.guided_radius,
        guided_eps=args.guided_eps,
        spot_mode=args.spot_mode,
        spot_cone=args.spot_cone,
        spot_exponent=args.spot_exponent,
        screen_center=(args.spot_center_x, args.spot_center_y),
        screen_spot_radius=args.screen_spot_radius,
    )
