#!/usr/bin/env python3
"""
relight_with_marigold_spotlight.py

Directional + Spotlight relighting using Marigold normals LCM.
Adds focused light control (directional or screen-space spotlight).

Usage:
    python relight_with_marigold_spotlight.py \
        --input input.jpg \
        --out relight_out \
        --az 45 --el 30 \
        --intensity 1.2 \
        --ambient 0.2 \
        --spot_mode directional \
        --spot_cone 8 \
        --spot_exponent 80
"""

import argparse
import math
from pathlib import Path
import numpy as np
from PIL import Image
import cv2
import torch
from transformers import DPTFeatureExtractor, DPTForDepthEstimation

from huggingface_hub import login
login(token="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")


# ---------------------- Guided Filter ----------------------
try:
    from guided_filter.guided_filter import guided_filter as gf_native
    HAVE_GUIDED_NATIVE = True
except Exception:
    HAVE_GUIDED_NATIVE = False

import onnxruntime as ort

def depth_anything_predict_normals(pil_img, onnx_path, save_depth_path="depth.png", device="cuda",
                                 smoothness=0.1, normal_strength=1):
    """
    Improved normal calculation from depth
    
    Args:
        smoothness: Higher values = more smoothing (0.5 to 2.0)
        normal_strength: Controls gradient scale (0.05 to 0.3)
    """
    # Convert PIL → RGB float32 [0,1]
    img = np.asarray(pil_img).astype(np.float32) / 255.0
    H, W = img.shape[:2]

    # Depth Anything expects 518×518
    size = 518
    img_resized = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    inp = img_resized.transpose(2, 0, 1)[None, ...]  # BCHW
    inp = inp.astype(np.float32)

    # ONNX inference
    sess = ort.InferenceSession(onnx_path)
    depth_pred = sess.run(None, {sess.get_inputs()[0].name: inp})[0]

    depth_small = depth_pred[0].squeeze()          # 518×518
    depth_full = cv2.resize(depth_small, (W, H))   # back to original resolution
    
    # ---- ADAPTIVE DEPTH PROCESSING ----
    # Normalize depth to 0-1 range for consistent processing
    depth_normalized = (depth_full - depth_full.min()) / (depth_full.max() - depth_full.min() + 1e-8)
    
    # Apply bilateral filter with adaptive parameters based on smoothness
    sigma_color = 0.03 * smoothness
    sigma_space = 5 * smoothness
    depth_filtered = cv2.bilateralFilter(depth_normalized, d=9, 
                                       sigmaColor=sigma_color, 
                                       sigmaSpace=sigma_space)
    
    # Additional Gaussian blur based on smoothness
    blur_size = max(3, int(3 * smoothness))
    if blur_size % 2 == 0:  # Ensure odd kernel size
        blur_size += 1
    depth_filtered = cv2.GaussianBlur(depth_filtered, (blur_size, blur_size), 0.7 * smoothness)
    
    # ---- ROBUST NORMAL CALCULATION ----
    # Use Scharr filter for better rotation invariance and smoother gradients
    scale = normal_strength
    dx = cv2.Scharr(depth_filtered, cv2.CV_32F, 1, 0, scale=scale)
    dy = cv2.Scharr(depth_filtered, cv2.CV_32F, 0, 1, scale=scale)
    
    # Apply gradient smoothing
    gradient_blur = max(1, int(2 * smoothness))
    if gradient_blur % 2 == 0:
        gradient_blur += 1
    dx = cv2.GaussianBlur(dx, (gradient_blur, gradient_blur), 0.5 * smoothness)
    dy = cv2.GaussianBlur(dy, (gradient_blur, gradient_blur), 0.5 * smoothness)
    
    # Calculate adaptive z-component based on local variance
    gradient_magnitude = np.sqrt(dx**2 + dy**2)
    local_variance = cv2.GaussianBlur(gradient_magnitude**2, (5, 5), 1.0)
    adaptive_z = 1.0 / (1.0 + local_variance * 3.0)
    
    normals = np.dstack([-dx, -dy, adaptive_z])
    
    # Normalize with careful numerical handling
    norm = np.linalg.norm(normals, axis=2, keepdims=True)
    valid_mask = norm > 1e-10
    normals = np.where(valid_mask, normals / norm, np.array([0, 0, 1], dtype=np.float32))
    
    # Post-processing: remove outliers and smooth
    for i in range(3):
        normals[:, :, i] = cv2.medianBlur(normals[:, :, i], 3)
    
    # Final normalization
    norm = np.linalg.norm(normals, axis=2, keepdims=True)
    normals = np.where(norm > 1e-10, normals / norm, np.array([0, 0, 1], dtype=np.float32))
    
    # Ensure consistent orientation (mostly pointing up)
    dot_with_up = np.sum(normals * np.array([0, 0, 1], dtype=np.float32), axis=2)
    flip_mask = dot_with_up < 0
    normals[flip_mask] *= -1
    
    # ---- SAVE FOR DEBUGGING ----
    if save_depth_path is not None:
        out_dir = Path(save_depth_path).parent
        # Save original depth
        cv2.imwrite(save_depth_path, (depth_normalized * 255).astype(np.uint8))
        # Save filtered depth
        cv2.imwrite(str(out_dir / "depth_filtered.png"), 
                   (depth_filtered * 255).astype(np.uint8))
        # Save gradient magnitude visualization
        grad_viz = (gradient_magnitude / gradient_magnitude.max() * 255).astype(np.uint8)
        cv2.imwrite(str(out_dir / "gradient_magnitude.png"), grad_viz)

    return normals.astype(np.float32)


# ---------------------- Utilities ----------------------
def load_image(path_or_url):
    if str(path_or_url).lower().startswith(("http://", "https://")):
        from io import BytesIO
        import requests
        r = requests.get(path_or_url, timeout=30)
        r.raise_for_status()
        pil = Image.open(BytesIO(r.content)).convert("RGB")
    else:
        pil = Image.open(str(path_or_url)).convert("RGB")
    arr = np.asarray(pil).astype(np.float32) / 255.0
    return arr, pil

def save_image_uint8(path, arr):
    arr8 = np.clip(arr * 255.0 + 0.5, 0, 255).astype(np.uint8)
    Image.fromarray(arr8).save(path)

def srgb_to_linear(img):
    a = 0.055
    mask = img <= 0.04045
    out = np.empty_like(img)
    out[mask] = img[mask] / 12.92
    out[~mask] = ((img[~mask] + a) / (1.0 + a)) ** 2.4
    return out

def linear_to_srgb(img):
    a = 0.055
    mask = img <= 0.0031308
    out = np.empty_like(img)
    out[mask] = img[mask] * 12.92
    out[~mask] = (1.0 + a) * (img[~mask] ** (1.0 / 2.4)) - a
    return np.clip(out, 0.0, 1.0)

# ---------------------- Guided upsample ----------------------
def guided_upsample_shading(sh_low, guide_full, radius=8, eps=1e-4):
    Hf, Wf = guide_full.shape[:2]
    sh_init = cv2.resize(sh_low, (Wf, Hf), interpolation=cv2.INTER_LINEAR).astype(np.float32)
    if guide_full.ndim == 3:
        guide_gray = 0.299 * guide_full[..., 0] + 0.587 * guide_full[..., 1] + 0.114 * guide_full[..., 2]
    else:
        guide_gray = guide_full.astype(np.float32)

    if HAVE_GUIDED_NATIVE:
        out = gf_native(guide_gray.astype(np.float32), sh_init.astype(np.float32), r=radius, eps=eps)
    else:
        I = guide_gray.astype(np.float32)
        p = sh_init.astype(np.float32)
        r = radius
        eps_local = eps
        mean_I = cv2.boxFilter(I, -1, (r, r))
        mean_p = cv2.boxFilter(p, -1, (r, r))
        mean_Ip = cv2.boxFilter(I * p, -1, (r, r))
        cov_Ip = mean_Ip - mean_I * mean_p
        mean_II = cv2.boxFilter(I * I, -1, (r, r))
        var_I = mean_II - mean_I * mean_I
        a = cov_Ip / (var_I + eps_local)
        b = mean_p - a * mean_I
        mean_a = cv2.boxFilter(a, -1, (r, r))
        mean_b = cv2.boxFilter(b, -1, (r, r))
        out = mean_a * I + mean_b
    return np.clip(out, 0.0, 1.0)

# ---------------------- Normals ----------------------
def decode_normals_from_tensor_or_image(pred):
    if isinstance(pred, torch.Tensor):
        arr = pred.detach().cpu().float().numpy()
    elif isinstance(pred, Image.Image):
        arr = np.asarray(pred).astype(np.float32) / 255.0
    else:
        arr = np.array(pred).astype(np.float32)

    while arr.ndim > 3:
        arr = np.squeeze(arr, axis=0)

    if arr.max() > 1.5:
        arr = arr / 255.0

    normals = arr * 2.0 - 1.0
    norm = np.linalg.norm(normals, axis=-1, keepdims=True) + 1e-8
    normals = normals / norm
    return normals.astype(np.float32)

def visualize_normals(normals):
    return (normals * 0.5 + 0.5).clip(0.0, 1.0)

# ---------------------- Shading & spotlight ----------------------
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
    lx = math.cos(el) * math.cos(az)
    ly = math.cos(el) * math.sin(az)
    lz = math.sin(el)
    L_dir = np.array([lx, ly, lz], dtype=np.float32).reshape(1, 1, 3)
    
    # Calculate dot product between normals and light direction
    cos_theta = np.sum(normals * L_dir, axis=-1)
    cos_theta_clamped = np.clip(cos_theta, 0.0, 1.0)  # Only positive for lighting
    
    if spot_mode == "directional":
        # For directional spotlight, use angular falloff
        cos_cone = math.cos(math.radians(cone_angle_deg))
        # Calculate angular difference from light direction
        angular_diff = np.arccos(np.clip(cos_theta, -1.0, 1.0))  # Actual angle in radians
        cone_rad = math.radians(cone_angle_deg)
        
        # Smooth falloff based on angular difference
        spot_mask = np.where(angular_diff <= cone_rad,
                            ((cone_rad - angular_diff) / cone_rad) ** spot_exponent,
                            0.0)
        shading = cos_theta_clamped * spot_mask

    elif spot_mode == "screen":
        # Screen-space spotlight
        ys = (np.arange(H) + 0.5) / H
        xs = (np.arange(W) + 0.5) / W
        xv, yv = np.meshgrid(xs, ys)
        cx, cy = screen_center
        dx = (xv - cx)
        dy = (yv - cy)
        dist = np.sqrt(dx * dx + dy * dy)
        max_r = screen_falloff_radius
        
        # Smooth circular falloff
        spot_mask = np.clip(1.0 - (dist / max_r), 0.0, 1.0)
        spot_mask = spot_mask ** (spot_exponent / 10.0)  # More reasonable exponent scaling
        shading = cos_theta_clamped * spot_mask
        
    else:  # Simple directional
        shading = cos_theta_clamped

    return np.clip(shading, 0.0, 1.0)

def apply_relighting(img_srgb, shading_full, intensity=1.0, ambient=0.25, light_color=(1.0, 1.0, 1.0)):
    img_lin = srgb_to_linear(img_srgb)
    light_color_arr = np.array(light_color, dtype=np.float32).reshape(1, 1, 3)
    lighting_factor = (ambient + intensity * shading_full[..., None]) * light_color_arr
    relit_lin = img_lin * lighting_factor

    return np.clip(linear_to_srgb(relit_lin), 0.0, 1.0)

# ---------------------- Main pipeline ----------------------
def relight_pipeline(
    image_path,
    out_dir,
    device="cuda",
    normal_model_name="prs-eth/marigold-normals-lcm-v0-1",
    downsize_long_side=768,
    light_az=45.0,
    light_el=30.0,
    intensity=1.2,
    ambient=0.25,
    guided_radius=8,
    guided_eps=1e-4,
    brush_mask_path=None,
    force_cpu=False,
    spot_mode="directional",
    spot_cone_deg=10.0,
    spot_exponent=60.0,
    spot_center=(0.5, 0.5),
    screen_spot_radius=0.25
):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    img_rgb, _ = load_image(image_path)
    H, W = img_rgb.shape[:2]
    print(f"[INFO] Loaded image {image_path} → {W}x{H}")

    if force_cpu:
        device = "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        print("[WARN] CUDA not available, falling back to CPU")
        device = "cpu"

    torch_device = torch.device(device)
    dtype = torch.float16 if torch_device.type == "cuda" else torch.float32

    # Resize input
    if max(H, W) > downsize_long_side:
        scale = downsize_long_side / float(max(H, W))
        new_w, new_h = int(W * scale), int(H * scale)
        img_for_model = cv2.resize(img_rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)
    else:
        img_for_model = img_rgb.copy()
    pil_for_model = Image.fromarray((img_for_model * 255.0).astype(np.uint8))

    # print(f"[INFO] Loading normals pipeline `{normal_model_name}` ...")
    print("[INFO] Predicting normals using Depth-Anything-V2 ONNX…")
    normals_low = depth_anything_predict_normals(
        pil_for_model,
        onnx_path="./models/job_jpre24w05_optimized_onnx/model.onnx",   # unpack your zip
        device=device,
    )

    # normals_low_smooth = cv2.medianBlur(normals_low, 3) # Or bilateral filter
    # normals_low = normals_low_smooth
    save_image_uint8(out_dir / "predicted_normals.png", visualize_normals(normals_low))

    shading_low = compute_shading_from_normals(
        normals_low,
        light_azimuth_deg=light_az,
        light_elevation_deg=light_el,
        spot_mode=spot_mode,
        cone_angle_deg=spot_cone_deg,
        spot_exponent=spot_exponent,
        screen_center=spot_center,
        screen_falloff_radius=screen_spot_radius,
    )

    shading_full = guided_upsample_shading(shading_low, srgb_to_linear(img_rgb), radius=guided_radius, eps=guided_eps)
    # After computing shading_full, add:
    print(f"[DEBUG] Shading stats - min: {shading_full.min():.3f}, max: {shading_full.max():.3f}, mean: {shading_full.mean():.3f}")

    # Save intermediate results for inspection
    save_image_uint8(out_dir / "shading_visual.png", shading_full)
    save_image_uint8(out_dir / "normals_visual.png", visualize_normals(normals_low))
    relit = apply_relighting(img_rgb, shading_full, intensity=intensity, ambient=ambient)

    if brush_mask_path:
        mask = cv2.imread(str(brush_mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is not None:
            mask_resized = cv2.resize(mask, (W, H), interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255.0
            mask3 = np.stack([mask_resized] * 3, axis=-1)
            relit = relit * mask3 + img_rgb * (1.0 - mask3)

    save_image_uint8(out_dir / "relit_output.png", relit)
    cv2.imwrite(str(out_dir / "shading_gray.png"), (shading_full * 255).astype(np.uint8))
    cv2.imwrite(str(out_dir / "shading_color.png"), cv2.applyColorMap((shading_full * 255).astype(np.uint8), cv2.COLORMAP_JET))
    print(f"[DONE] Results saved to {out_dir}")

    return {"relit": relit, "normals": normals_low, "shading": shading_full}

# ---------------------- CLI ----------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Directional/Spotlight relighting using Marigold normals LCM")
    parser.add_argument("--input", required=True)
    parser.add_argument("--out", default="relight_out")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--downsize", type=int, default=768)
    parser.add_argument("--az", type=float, default=45.0)
    parser.add_argument("--el", type=float, default=30.0)
    parser.add_argument("--intensity", type=float, default=1.2) #intensity of hte new light
    parser.add_argument("--ambient", type=float, default=0.25) #how much the old light shoul dmake a difference, for spot light, make it small
    parser.add_argument("--guided_radius", type=int, default=8)
    parser.add_argument("--guided_eps", type=float, default=1e-4)
    parser.add_argument("--brush_mask", default=None)
    parser.add_argument("--force_cpu", action="store_true")
    parser.add_argument("--spot_mode", choices=["directional","screen"], default="directional")
    parser.add_argument("--spot_cone", type=float, default=300.0) #radius of the directional light size
    parser.add_argument("--spot_exponent", type=float, default=3.0) #make it small to actually view diference 
    parser.add_argument("--spot_center_x", type=float, default=0.5)
    parser.add_argument("--spot_center_y", type=float, default=0.5)
    parser.add_argument("--screen_spot_radius", type=float, default=0.25)
    args = parser.parse_args()

    relight_pipeline(
        image_path=args.input,
        out_dir=args.out,
        device=args.device,
        downsize_long_side=args.downsize,
        light_az=args.az,
        light_el=args.el,
        intensity=args.intensity,
        ambient=args.ambient,
        guided_radius=args.guided_radius,
        guided_eps=args.guided_eps,
        brush_mask_path=args.brush_mask,
        force_cpu=args.force_cpu,
        spot_mode=args.spot_mode,
        spot_cone_deg=args.spot_cone,
        spot_exponent=args.spot_exponent,
        spot_center=(args.spot_center_x, args.spot_center_y),
        screen_spot_radius=args.screen_spot_radius,
    )
