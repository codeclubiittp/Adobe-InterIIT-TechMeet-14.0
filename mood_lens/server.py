
import uvicorn
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import shutil
import os
import asyncio
import torch
import gc
import logging
import pandas as pd
import base64

import mood_lens.config as config
from mood_lens.vsearch import VSearchEngine
from mood_lens.lut_generator import NeuralLUTGenerator
from mood_lens.engine_core import colour_correct, colour_correct_multivariant

models = {}
gpu_lock = asyncio.Lock()

@asynccontextmanager
async def lifespan(app: FastAPI):
    config.setup_logging()
    logging.info("Startup | Initializing Models")
    try:
        models['matcher'] = VSearchEngine()
        models['generator'] = NeuralLUTGenerator()
        if os.path.exists(config.CSV_PATH):
            df = pd.read_csv(config.CSV_PATH)
            models['emotions'] = sorted(df['emotion'].unique().tolist())
        else:
            models['emotions'] = []
        logging.info("Models Loaded Successfully")
        yield
    except Exception as e:
        logging.critical(f"fastapi startup failed: {e}")
        raise e
    finally:
        logging.info("shutdown | cleaning up resources")
        models.clear()
        torch.cuda.empty_cache()
        gc.collect()

app = FastAPI(title="Neural Grade API", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def save_upload_file(upload_file: UploadFile, destination: str):
    try:
        with open(destination, "wb") as buffer:
            shutil.copyfileobj(upload_file.file, buffer)
    finally:
        upload_file.file.close()

@app.get("/")
async def root():
    return {"status": "online", "device": config.DEVICE}

@app.get("/emotions")
async def get_emotions():
    return {"count": len(models['emotions']), "emotions": models['emotions']}

@app.post("/grade")
async def grade(
    emotion: str = Form(...),
    file: UploadFile = File(...)
):
    target_emotion = emotion
    if models['emotions'] and target_emotion not in models['emotions']:
        raise HTTPException(status_code=400, detail=f"Emotion not found.")

    temp_dir = os.path.join(config.MISC_ROOT_DIR, "api_temp")
    os.makedirs(temp_dir, exist_ok=True)
    
    input_filename = f"temp_{file.filename}"
    input_path = os.path.join(temp_dir, input_filename)
    save_upload_file(file, input_path)

    async with gpu_lock:
        logging.info(f"Request | Processing {file.filename} & {target_emotion}")
        try:
            colour_correct_multivariant(
                input_path=input_path, 
                target_emotion=target_emotion, 
                matcher=models['matcher'], 
                generator=models['generator']
            )
            
            results_b64 = []
            base_out_name = f"result_{target_emotion}_{input_filename}"
            
            for i in range(1, 4): 
                variant_path = os.path.join(config.OUTPUT_ROOT_DIR, f"{base_out_name}_v{i}.png")
                
                if os.path.exists(variant_path):
                    with open(variant_path, "rb") as img_file:
                        b64_str = base64.b64encode(img_file.read()).decode('utf-8')
                        results_b64.append(f"data:image/png;base64,{b64_str}")
                    os.remove(variant_path)
                else:
                    logging.warning(f"Expected output {variant_path} not found.")

            if os.path.exists(input_path): os.remove(input_path)
            
            if not results_b64:
                raise HTTPException(status_code=500, detail="Generation failed (No outputs found)")

            return JSONResponse(content={"results": results_b64})

        except Exception as e:
            logging.error(f"API Error: {e}")
            raise HTTPException(status_code=500, detail=str(e))
        finally:
            torch.cuda.empty_cache()
            gc.collect()

if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=8001, reload=False)