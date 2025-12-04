import base64
from pathlib import Path

from models import load_models, MODELS
from relight import run_relighting_pipeline
from image_io import load_image_from_bytes, to_uint8_image

import cv2
import numpy as np


# Helper to decode base64 → numpy array
def decode_base64_to_image(b64_str):
    img_bytes = base64.b64decode(b64_str)
    img_array = np.frombuffer(img_bytes, dtype=np.uint8)
    return cv2.imdecode(img_array, cv2.IMREAD_COLOR)


def save_base64_image(b64, path):
    img = decode_base64_to_image(b64)
    cv2.imwrite(path, img)


if __name__ == "__main__":

    # ----------------------------------------------------
    # INPUT FILE PATH (CHANGE THIS)
    # ----------------------------------------------------
    input_path = "../pipelines/relighting-pipeline-v1/inputs/input.jpg"

    # Load bytes
    img_bytes = Path(input_path).read_bytes()

    # ----------------------------------------------------
    # LOAD MODEL
    # ----------------------------------------------------
    print("Loading model...")
    load_models()
    pipe = MODELS["marigold"]

    # ----------------------------------------------------
    # HARDCODED PARAMETERS
    # ----------------------------------------------------
    az = 30.0
    el = 60.0
    intensity = 3
    ambient = 0.2
    spot_mode = "directional"
    spot_cone = 3.0
    spot_exponent = 300.0
    screen_spot_radius = 0.25
    spot_center = (0.5, 0.5)

    # ----------------------------------------------------
    # RUN PIPELINE
    # ----------------------------------------------------
    print("Running relighting...")
    output = run_relighting_pipeline(
        img_bytes=img_bytes,
        pipe=pipe,
        az=az,
        el=el,
        intensity=intensity,
        ambient=ambient,
        spot_mode=spot_mode,
        spot_cone=spot_cone,
        spot_exponent=spot_exponent,
        spot_center=spot_center,
        screen_spot_radius=screen_spot_radius,
        profile=False
    )

    # ----------------------------------------------------
    # SAVE OUTPUT IMAGES
    # ----------------------------------------------------
    print("Saving results...")
    # import os
    # os.makedirs("out/")
    save_base64_image(output["relit_base64"], "relit.png")
    save_base64_image(output["normals_base64"], "normals.png")
    save_base64_image(output["shading_base64"], "shading.png")

    print("Done! Saved:")
    print(" - relit.png")
    print(" - normals.png")
    print(" - shading.png")
