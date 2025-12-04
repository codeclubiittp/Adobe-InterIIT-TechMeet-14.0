import onnxruntime as ort
from config import settings
import torch
from diffusers import DiffusionPipeline

MODELS = {}

def load_models():
    device = settings.DEVICE
    dtype = torch.float16 if device == "cuda" else torch.float32
    
    pipe_normals = DiffusionPipeline.from_pretrained(
        settings.NORMAL_MODEL,
        torch_dtype=dtype,
        safety_checker=None
    ).to(device)
    MODELS["marigold"] = pipe_normals
    print("[STARTUP] Loaded Marigold.")

    # providers = ["GPUExecutionProvider", "CPUExecutionProvider"]
    # ort_session = ort.InferenceSession(
    #     settings.LAMA_ONNX_PATH,
    #     providers=providers
    # )
    # MODELS["lama_onnx"] = ort_session
    # print("[STARTUP] Loaded LaMa Dilated ONNX.")
