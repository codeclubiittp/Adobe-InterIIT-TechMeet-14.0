import numpy as np
import cv2
import math

# sRGB conversion
def srgb_to_linear(img):
    a = 0.055
    out = np.where(img <= 0.04045, img / 12.92, ((img + a) / (1 + a)) ** 2.4)
    return out

def linear_to_srgb(img):
    a = 0.055
    out = np.where(img <= 0.0031308, img * 12.92, (1 + a) * img ** (1 / 2.4) - a)
    return np.clip(out, 0, 1)

# spotlight + directional shading (same as your code)
def compute_shading(normals, az, el, spot_mode, cone, exponent, center, screen_radius):
    H, W = normals.shape[:2]

    az_r = math.radians(az)
    el_r = math.radians(el)

    L = np.array([
        math.cos(el_r) * math.cos(az_r),
        math.cos(el_r) * math.sin(az_r),
        math.sin(el_r)
    ], dtype=np.float32).reshape(1, 1, 3)

    cos_theta = np.sum(normals * L, axis=-1)
    cos_clamped = np.clip(cos_theta, 0.0, 1.0)

    if spot_mode == "directional":
        cone_r = math.radians(cone)
        angular_diff = np.arccos(np.clip(cos_theta, -1, 1))
        mask = np.where(
            angular_diff <= cone_r,
            ((cone_r - angular_diff) / cone_r) ** exponent,
            0
        )
        return cos_clamped * mask

    if spot_mode == "screen":
        ys = (np.arange(H) + 0.5) / H
        xs = (np.arange(W) + 0.5) / W
        xv, yv = np.meshgrid(xs, ys)

        cx, cy = center
        dist = np.sqrt((xv - cx) ** 2 + (yv - cy) ** 2)

        mask = np.clip(1 - dist / screen_radius, 0, 1) ** (exponent / 10)
        return cos_clamped * mask

    return cos_clamped

# apply lighting
def apply_relighting(img_srgb, shading, intensity, ambient):
    img_lin = srgb_to_linear(img_srgb)
    lighting = ambient + intensity * shading[..., None]
    relit_lin = img_lin * lighting
    return np.clip(linear_to_srgb(relit_lin), 0, 1)

