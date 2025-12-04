from fastapi import FastAPI, UploadFile, File
from fastapi.staticfiles import StaticFiles
from typing import Optional
import hashlib
import numpy as np
import torch
from fastapi.middleware.cors import CORSMiddleware
# --------------------------
# Your imports
# --------------------------
from models import load_models, MODELS
from relight import run_relighting_pipeline
from image_io import load_image_from_bytes, to_uint8_image, encode_base64
from normal_utils import decode_normals, visualize_normals
from shading import compute_shading, apply_relighting


# --------------------------
# FastAPI App
# --------------------------
app = FastAPI(title="Marigold Relighting API")
# app.mount("/ui", StaticFiles(directory="ui", html=True), name="ui")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --------------------------
# GLOBAL NORMAL CACHE
# --------------------------
#   image_sha256 -> { "normals": np.ndarray }
# --------------------------
NORMAL_CACHE = {}


# --------------------------
# SHA256 helper
# --------------------------
def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


# --------------------------
# Startup event
# --------------------------
@app.on_event("startup")
async def startup_event():
    load_models()


# --------------------------
# API Endpoint
# --------------------------
@app.post("/infer")
async def infer_relighting(
    input_image: UploadFile = File(...),
    az: float = 45.0,
    el: float = 30.0,
    intensity: float = 1.2,
    ambient: float = 0.25,
    spot_mode: str = "directional",
    spot_cone: float = 10.0,
    spot_exponent: float = 60.0,
    screen_spot_radius: float = 0.25,
    spot_center_x: float = 0.5,
    spot_center_y: float = 0.5
):
    # read bytes
    img_bytes = await input_image.read()

    # compute content hash
    img_hash = sha256_bytes(img_bytes)
    spot_center = (spot_center_x, spot_center_y)

    # ------------------------------------------
    # CACHE HIT — SKIP INFERENCE
    # ------------------------------------------
    if img_hash in NORMAL_CACHE:
        normals = NORMAL_CACHE[img_hash]["normals"]

        # load original image
        img_rgb, _ = load_image_from_bytes(img_bytes)

        # re-compute shading using the cached normals
        shading = compute_shading(
            normals,
            az, el,
            spot_mode,
            spot_cone,
            spot_exponent,
            spot_center,
            screen_spot_radius
        )

        # relight
        relit = apply_relighting(img_rgb, shading, intensity, ambient)

        return {
            "relit_image_base64": encode_base64(to_uint8_image(relit)),
            "normals_base64": encode_base64(to_uint8_image(visualize_normals(normals))),
            "shading_base64": encode_base64(to_uint8_image(shading)),
            "cache_hit": True
        }

    # ------------------------------------------
    # CACHE MISS — RUN FULL INFERENCE
    # ------------------------------------------
    output = run_relighting_pipeline(
        img_bytes=img_bytes,
        pipe=MODELS["marigold"],
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

    # extract normals from output
    normals = output["normals_raw"]

    # save normals to cache
    NORMAL_CACHE[img_hash] = {"normals": normals}

    # remove raw normals from response
    output.pop("normals_raw", None)

    output["cache_hit"] = False

    return output
