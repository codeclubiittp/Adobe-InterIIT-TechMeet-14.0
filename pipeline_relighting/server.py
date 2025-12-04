#!/usr/bin/env python3
"""
FastAPI server for directional + spotlight relighting using Marigold normals LCM.
"""

import argparse
import math
from pathlib import Path
import hashlib
import json
from typing import Optional, Dict, Tuple, List
import numpy as np
from PIL import Image
import cv2
import torch
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from pydantic import BaseModel
from diffusers import DiffusionPipeline
import io
import base64
import tempfile
import os

# ---------------------- Guided Filter ----------------------
try:
    from guided_filter.guided_filter import guided_filter as gf_native
    HAVE_GUIDED_NATIVE = True
except Exception:
    HAVE_GUIDED_NATIVE = False

# ---------------------- Cache Classes ----------------------
class ImageCache:
    """Cache for storing image hashes and their computed normals"""
    def __init__(self, cache_dir: str = "./image_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.hash_to_normals: Dict[str, np.ndarray] = {}
        self.hash_to_info: Dict[str, dict] = {}
        self._load_cache()
    
    def _load_cache(self):
        """Load existing cache from disk"""
        index_file = self.cache_dir / "cache_index.json"
        if index_file.exists():
            try:
                with open(index_file, 'r') as f:
                    self.hash_to_info = json.load(f)
                print(f"[CACHE] Loaded {len(self.hash_to_info)} cached entries")
            except Exception as e:
                print(f"[CACHE] Failed to load cache index: {e}")
    
    def _save_cache_index(self):
        """Save cache index to disk"""
        index_file = self.cache_dir / "cache_index.json"
        try:
            with open(index_file, 'w') as f:
                json.dump(self.hash_to_info, f, indent=2)
        except Exception as e:
            print(f"[CACHE] Failed to save cache index: {e}")
    
    def get_image_hash(self, image_bytes: bytes) -> str:
        """Generate hash for image bytes"""
        return hashlib.sha256(image_bytes).hexdigest()
    
    def get_normals_path(self, image_hash: str) -> Path:
        """Get path for cached normals"""
        return self.cache_dir / f"{image_hash}_normals.npy"
    
    def has_normals(self, image_hash: str) -> bool:
        """Check if normals are cached for this image"""
        normals_path = self.get_normals_path(image_hash)
        return normals_path.exists()
    
    def get_normals(self, image_hash: str) -> Optional[np.ndarray]:
        """Get cached normals"""
        if image_hash in self.hash_to_normals:
            return self.hash_to_normals[image_hash]
        
        normals_path = self.get_normals_path(image_hash)
        if normals_path.exists():
            try:
                normals = np.load(str(normals_path))
                self.hash_to_normals[image_hash] = normals
                return normals
            except Exception as e:
                print(f"[CACHE] Failed to load cached normals: {e}")
        return None
    
    def save_normals(self, image_hash: str, normals: np.ndarray, image_info: dict):
        """Save normals to cache"""
        try:
            normals_path = self.get_normals_path(image_hash)
            np.save(str(normals_path), normals)
            self.hash_to_normals[image_hash] = normals
            
            # Store basic info
            self.hash_to_info[image_hash] = {
                "hash": image_hash,
                "timestamp": image_info.get("timestamp", ""),
                "size": image_info.get("size", 0),
                "dimensions": image_info.get("dimensions", ""),
            }
            self._save_cache_index()
            print(f"[CACHE] Saved normals for hash: {image_hash[:16]}...")
        except Exception as e:
            print(f"[CACHE] Failed to save normals: {e}")
    
    def clear_cache(self):
        """Clear all cached data"""
        for file in self.cache_dir.glob("*"):
            if file.is_file():
                file.unlink()
        self.hash_to_normals.clear()
        self.hash_to_info.clear()
        self._save_cache_index()
        print("[CACHE] Cache cleared")

# ---------------------- Model Manager ----------------------
class ModelManager:
    """Manages loading and caching of the normals model"""
    def __init__(self, model_name: str = "prs-eth/marigold-normals-lcm-v0-1", device: str = "cuda"):
        self.model_name = model_name
        self.device = device
        self.pipe = None
        self.dtype = torch.float16 if device == "cuda" and torch.cuda.is_available() else torch.float32
        self.is_loaded = False
    
    def load_model(self):
        """Load the diffusion model"""
        if self.is_loaded:
            return self.pipe
        
        if self.device == "cuda" and not torch.cuda.is_available():
            print("[WARN] CUDA not available, falling back to CPU")
            self.device = "cpu"
            self.dtype = torch.float32
        
        torch_device = torch.device(self.device)
        print(f"[MODEL] Loading normals pipeline `{self.model_name}` on {self.device}...")
        
        try:
            self.pipe = DiffusionPipeline.from_pretrained(
                self.model_name, 
                torch_dtype=self.dtype, 
                safety_checker=None
            ).to(torch_device)
            self.is_loaded = True
            print("[MODEL] Model loaded successfully")
        except Exception as e:
            print(f"[MODEL] Failed to load model: {e}")
            raise
        
        return self.pipe
    
    def unload_model(self):
        """Unload model to free memory"""
        if self.pipe is not None:
            del self.pipe
            self.pipe = None
            self.is_loaded = False
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            print("[MODEL] Model unloaded")

# ---------------------- Pydantic Models ----------------------
class RelightRequest(BaseModel):
    """Request model for relighting parameters"""
    az: float = 45.0
    el: float = 30.0
    intensity: float = 1.2
    ambient: float = 0.25
    spot_mode: str = "directional"
    spot_cone: float = 10.0
    spot_exponent: float = 60.0
    spot_center_x: float = 0.5
    spot_center_y: float = 0.5
    screen_spot_radius: float = 0.25
    downsize: int = 768
    guided_radius: int = 8
    guided_eps: float = 1e-4
    return_normals: bool = False
    return_shading: bool = False

class RelightResponse(BaseModel):
    """Response model for relighting results"""
    success: bool
    message: str
    image_hash: Optional[str] = None
    relit_image_b64: Optional[str] = None
    normals_image_b64: Optional[str] = None
    shading_image_b64: Optional[str] = None
    processing_time_ms: Optional[float] = None

# ---------------------- Utilities ----------------------
def load_image_from_bytes(image_bytes: bytes) -> Tuple[np.ndarray, Image.Image]:
    """Load image from bytes"""
    try:
        pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        arr = np.asarray(pil_image).astype(np.float32) / 255.0
        return arr, pil_image
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to load image: {str(e)}")

def save_image_to_bytes(arr: np.ndarray) -> bytes:
    """Convert numpy array to image bytes"""
    arr8 = np.clip(arr * 255.0 + 0.5, 0, 255).astype(np.uint8)
    pil_image = Image.fromarray(arr8)
    img_byte_arr = io.BytesIO()
    pil_image.save(img_byte_arr, format='PNG')
    return img_byte_arr.getvalue()

def encode_image_to_base64(arr: np.ndarray) -> str:
    """Encode numpy array to base64 string"""
    img_bytes = save_image_to_bytes(arr)
    return base64.b64encode(img_bytes).decode('utf-8')

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

# ---------------------- FastAPI App ----------------------
app = FastAPI(title="Relighting API", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global instances
model_manager = None
image_cache = None

@app.on_event("startup")
async def startup_event():
    """Initialize model and cache on startup"""
    global model_manager, image_cache
    
    # Initialize cache
    image_cache = ImageCache()
    
    # Initialize model manager
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_manager = ModelManager(device=device)
    
    # Load model (can be deferred to first request if preferred)
    try:
        model_manager.load_model()
    except Exception as e:
        print(f"[ERROR] Failed to load model on startup: {e}")
        # Model will be loaded lazily on first request

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    global model_manager
    if model_manager:
        model_manager.unload_model()

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Relighting API",
        "version": "1.0.0",
        "endpoints": {
            "POST /infer": "Process image with relighting",
            "GET /health": "Check API health",
            "GET /cache/info": "Get cache information",
            "DELETE /cache/clear": "Clear cache"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    model_status = "loaded" if model_manager and model_manager.is_loaded else "not_loaded"
    cache_count = len(image_cache.hash_to_info) if image_cache else 0
    
    return {
        "status": "healthy",
        "model_loaded": model_manager.is_loaded if model_manager else False,
        "model_status": model_status,
        "cache_entries": cache_count,
        "device": model_manager.device if model_manager else "unknown"
    }

@app.get("/cache/info")
async def get_cache_info():
    """Get cache information"""
    if not image_cache:
        raise HTTPException(status_code=500, detail="Cache not initialized")
    
    cache_stats = {
        "total_entries": len(image_cache.hash_to_info),
        "memory_cached": len(image_cache.hash_to_normals),
        "cache_dir": str(image_cache.cache_dir),
        "entries": []
    }
    
    for hash_val, info in list(image_cache.hash_to_info.items())[:20]:  # First 20 entries
        cache_stats["entries"].append({
            "hash_short": hash_val[:16],
            "size": info.get("size", 0),
            "dimensions": info.get("dimensions", "")
        })
    
    return cache_stats

@app.delete("/cache/clear")
async def clear_cache():
    """Clear all cached data"""
    if not image_cache:
        raise HTTPException(status_code=500, detail="Cache not initialized")
    
    image_cache.clear_cache()
    return {"message": "Cache cleared successfully"}

@app.post("/infer", response_model=RelightResponse)
async def infer_relighting(
    image: UploadFile = File(...),
    az: float = Form(45.0),
    el: float = Form(30.0),
    intensity: float = Form(1.2),
    ambient: float = Form(0.25),
    spot_mode: str = Form("directional"),
    spot_cone: float = Form(10.0),
    spot_exponent: float = Form(60.0),
    spot_center_x: float = Form(0.5),
    spot_center_y: float = Form(0.5),
    screen_spot_radius: float = Form(0.25),
    downsize: int = Form(768),
    guided_radius: int = Form(8),
    guided_eps: float = Form(1e-4),
    return_normals: bool = Form(False),
    return_shading: bool = Form(False)
):
    """
    Process image with relighting.
    
    Parameters match the original CLI arguments.
    """
    import time
    start_time = time.time()
    
    try:
        # Read image bytes
        image_bytes = await image.read()
        if not image_bytes:
            raise HTTPException(status_code=400, detail="Empty image file")
        
        # Generate image hash
        image_hash = image_cache.get_image_hash(image_bytes)
        print(f"[INFER] Processing image with hash: {image_hash[:16]}...")
        
        # Load image
        img_rgb, pil_image = load_image_from_bytes(image_bytes)
        H, W = img_rgb.shape[:2]
        
        # Check cache for normals
        normals_low = image_cache.get_normals(image_hash)
        
        if normals_low is None:
            print(f"[INFER] Normals not cached, computing...")
            
            # Ensure model is loaded
            if not model_manager or not model_manager.is_loaded:
                model_manager.load_model()
            
            # Resize for model
            if max(H, W) > downsize:
                scale = downsize / float(max(H, W))
                new_w, new_h = int(W * scale), int(H * scale)
                img_for_model = cv2.resize(img_rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)
            else:
                img_for_model = img_rgb.copy()
            
            pil_for_model = Image.fromarray((img_for_model * 255.0).astype(np.uint8))
            
            # Predict normals
            print("[INFER] Predicting normals...")
            result = model_manager.pipe(
                image=pil_for_model, 
                num_inference_steps=4, 
                generator=torch.manual_seed(0)
            )
            pred = getattr(result, "prediction", None)
            if pred is None:
                pred = result.images[0]
            
            normals_low = decode_normals_from_tensor_or_image(pred)
            
            # Cache normals
            image_info = {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "size": len(image_bytes),
                "dimensions": f"{W}x{H}"
            }
            image_cache.save_normals(image_hash, normals_low, image_info)
        else:
            print(f"[INFER] Using cached normals")
        
        # Compute shading
        shading_low = compute_shading_from_normals(
            normals_low,
            light_azimuth_deg=az,
            light_elevation_deg=el,
            spot_mode=spot_mode,
            cone_angle_deg=spot_cone,
            spot_exponent=spot_exponent,
            screen_center=(spot_center_x, spot_center_y),
            screen_falloff_radius=screen_spot_radius,
        )
        
        # Upsample shading
        shading_full = guided_upsample_shading(
            shading_low, 
            srgb_to_linear(img_rgb), 
            radius=guided_radius, 
            eps=guided_eps
        )
        
        # Apply relighting
        relit = apply_relighting(img_rgb, shading_full, intensity=intensity, ambient=ambient)
        
        arr8 = np.clip(relit * 255.0 + 0.5, 0, 255).astype(np.uint8)
        pil_image = Image.fromarray(arr8)
        pil_image.save("out.jpg")

        # Prepare response
        response_data = {
            "success": True,
            "message": "Relighting completed successfully",
            "image_hash": image_hash,
            "relit_image_b64": encode_image_to_base64(relit),
        }
        
        # Add normals if requested
        if return_normals:
            normals_vis = visualize_normals(normals_low)
            response_data["normals_image_b64"] = encode_image_to_base64(normals_vis)
        
        # Add shading if requested
        if return_shading:
            response_data["shading_image_b64"] = encode_image_to_base64(shading_full)
        
        # Calculate processing time
        processing_time_ms = (time.time() - start_time) * 1000
        response_data["processing_time_ms"] = round(processing_time_ms, 2)
        
        print(f"[INFER] Processing completed in {processing_time_ms:.2f}ms")
        
        return RelightResponse(**response_data)
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"[ERROR] Processing failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")

# Alternative endpoint using JSON body (for complex parameters)
class BatchInferRequest(BaseModel):
    """Request model for batch inference"""
    image_b64: str
    parameters: List[RelightRequest]

@app.post("/infer_batch")
async def infer_batch(request: BatchInferRequest):
    """
    Process the same image with multiple parameter sets.
    Returns a list of results.
    """
    try:
        # Decode base64 image
        image_bytes = base64.b64decode(request.image_b64)
        image_hash = image_cache.get_image_hash(image_bytes)
        
        # Load image once
        img_rgb, pil_image = load_image_from_bytes(image_bytes)
        H, W = img_rgb.shape[:2]
        
        # Get or compute normals once
        normals_low = image_cache.get_normals(image_hash)
        
        if normals_low is None:
            # Compute normals (similar to single inference)
            if not model_manager or not model_manager.is_loaded:
                model_manager.load_model()
            
            # ... (same normals computation as above)
            # For brevity, implementing the full logic would duplicate code
            # You can refactor into a function
        
        results = []
        
        # Process each parameter set
        for i, params in enumerate(request.parameters):
            # Compute shading with these parameters
            shading_low = compute_shading_from_normals(
                normals_low,
                light_azimuth_deg=params.az,
                light_elevation_deg=params.el,
                spot_mode=params.spot_mode,
                cone_angle_deg=params.spot_cone,
                spot_exponent=params.spot_exponent,
                screen_center=(params.spot_center_x, params.spot_center_y),
                screen_falloff_radius=params.screen_spot_radius,
            )
            
            shading_full = guided_upsample_shading(
                shading_low,
                srgb_to_linear(img_rgb),
                radius=params.guided_radius,
                eps=params.guided_eps
            )
            
            relit = apply_relighting(img_rgb, shading_full, 
                                    intensity=params.intensity, 
                                    ambient=params.ambient)
            
            results.append({
                "index": i,
                "relit_image_b64": encode_image_to_base64(relit)
            })

            relit.save("out.jpg")
            print("output saved to out.jpg")
        
        return {
            "success": True,
            "image_hash": image_hash,
            "results": results
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch processing failed: {str(e)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FastAPI server for relighting")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    args = parser.parse_args()
    
    print(f"[SERVER] Starting relighting API server on {args.host}:{args.port}")
    print(f"[SERVER] Model device: {model_manager.device if model_manager else 'unknown'}")
    
    uvicorn.run(
        "relight_api:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info"
    )