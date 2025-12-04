import os
import uuid
import shutil
import subprocess
from pathlib import Path
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse, JSONResponse

app = FastAPI()

ROOT = Path(__file__).parent.resolve()
PREDICT_SCRIPT = ROOT / "bin" / "predict.py"
MODEL_PATH = ROOT / "fourier"     # your model path (contains config.yaml + models/*)
TMP_ROOT = ROOT / "api_tmp"

TMP_ROOT.mkdir(exist_ok=True)


def run_lama_inference(image_bytes: bytes, mask_bytes: bytes):
    """Runs LaMa predict.py exactly like your CLI does."""

    # unique per-request folder
    req_id = uuid.uuid4().hex
    indir = TMP_ROOT / f"in_{req_id}"
    outdir = TMP_ROOT / f"out_{req_id}"

    indir.mkdir()
    outdir.mkdir()

    # file paths
    image_path = indir / "image.png"
    mask_path = indir / "mask.png"

    # save inputs
    image_path.write_bytes(image_bytes)
    mask_path.write_bytes(mask_bytes)

    # run predict.py
    cmd = [
        "python3", str(PREDICT_SCRIPT),
        f"model.path={MODEL_PATH}",
        f"indir={indir}",
        f"outdir={outdir}"
    ]

    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        shutil.rmtree(indir, ignore_errors=True)
        shutil.rmtree(outdir, ignore_errors=True)
        return None, f"Predict.py failed: {e}"

    # locate output
    out_files = list(outdir.glob("*.png"))
    if not out_files:
        shutil.rmtree(indir, ignore_errors=True)
        shutil.rmtree(outdir, ignore_errors=True)
        return None, "No output produced"

    return out_files[0], None


@app.post("/inpaint")
async def inpaint(
    image: UploadFile = File(...),
    mask: UploadFile = File(...)
):
    image_bytes = await image.read()
    mask_bytes = await mask.read()

    output_path, err = run_lama_inference(image_bytes, mask_bytes)

    if err:
        return JSONResponse(status_code=500, content={"error": err})

    # Return the output image
    return FileResponse(
        path=output_path,
        media_type="image/png",
        filename="inpainted.png"
    )


@app.get("/")
def home():
    return {"status": "LaMa FastAPI running"}
