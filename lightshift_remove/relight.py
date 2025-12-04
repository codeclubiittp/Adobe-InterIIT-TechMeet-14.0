import numpy as np
from image_io import load_image_from_bytes, to_uint8_image, encode_base64
from normal_utils import decode_normals, visualize_normals
from shading import compute_shading, apply_relighting
import torch

def run_relighting_pipeline(
    img_bytes,
    pipe,
    az,
    el,
    intensity,
    ambient,
    spot_mode,
    spot_cone,
    spot_exponent,
    spot_center,
    screen_spot_radius,
    profile: bool = False,
):
    img_rgb, pil = load_image_from_bytes(img_bytes)
    result = None
    shading = None
    normals = None

    # closures definitions
    # shading
    def shade():
        normals = decode_normals(pred)
        shading = compute_shading(
            normals,
            az, el,
            spot_mode,
            spot_cone,
            spot_exponent,
            spot_center,
            screen_spot_radius
        )
        return shading, normals

    # loading pip
    def inference():
        result = pipe(image=pil, num_inference_steps=4)
        return result

    # inference
    if profile:
        with torch.profiler.record_function("Relight_Inference"):
            result = inference()
    else:
        result = inference()

    raw_pred = getattr(result, "prediction", None)

    if raw_pred is not None:
        pred = raw_pred
    else:
        pred = result.images[0]

    # normal computation & shading
    if profile:
        with torch.profiler.record_function("Relight_Shading"):
            shading, normals = shade()
    else:
        shading, normals = shade()

    # postprocess
    if profile:
        with torch.profiler.record_function("Relight_Postprocess"):
            relit = apply_relighting(img_rgb, shading, intensity, ambient)
    else:
        relit = apply_relighting(img_rgb, shading, intensity, ambient)

    return {
        "relit_base64": encode_base64(to_uint8_image(relit)),
        "normals_base64": encode_base64(to_uint8_image(visualize_normals(normals))),
        "shading_base64": encode_base64(to_uint8_image(shading)),
        "normals_raw": normals,
    }
