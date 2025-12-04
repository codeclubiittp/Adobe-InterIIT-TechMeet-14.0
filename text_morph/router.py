from fastapi import APIRouter, File, UploadFile, HTTPException, Body
from fastapi.responses import JSONResponse
import cv2
import numpy as np
from util.sam_utils import AddPointRequest, SessionState
from util.session_manager import session_manager
from util.text_edit import get_crops, edit_text
from fastapi.responses import StreamingResponse
import io
import os
import sys

router = APIRouter(prefix="/sam", tags=["SAM Segmentation"])

# Add TextCtrl path to sys.path
# Assuming router.py is in text_ctrl/ and main.py is in text_ctrl/TextCtrl/
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))  # This is text_ctrl/
TEXTCTRL_PATH = os.path.join(SCRIPT_DIR, 'TextCtrl')  # This is text_ctrl/TextCtrl/

print(f"[DEBUG] SCRIPT_DIR: {SCRIPT_DIR}")
print(f"[DEBUG] TEXTCTRL_PATH: {TEXTCTRL_PATH}")
print(f"[DEBUG] TEXTCTRL_PATH exists: {os.path.exists(TEXTCTRL_PATH)}")

if TEXTCTRL_PATH not in sys.path:
    sys.path.insert(0, TEXTCTRL_PATH)

# Now import model_manager after path is set
from main import model_manager

# Initialize the model immediately when this module is imported
print("[ROUTER] Initializing GaMuSA model...")
try:
    # Change to TextCtrl directory for relative paths (configs, weights, etc.)
    original_cwd = os.getcwd()
    print(f"[DEBUG] Original CWD: {original_cwd}")
    
    os.chdir(TEXTCTRL_PATH)
    print(f"[DEBUG] Changed to: {os.getcwd()}")
    
    model_manager.get_pipeline()
    print("[ROUTER] GaMuSA model loaded successfully!")
    
    # Restore original working directory
    os.chdir(original_cwd)
    print(f"[DEBUG] Restored CWD: {os.getcwd()}")
except Exception as e:
    print(f"[ROUTER ERROR] Failed to load GaMuSA model: {str(e)}")
    import traceback
    print(f"[ROUTER ERROR] Traceback: {traceback.format_exc()}")
    # Restore directory even on error
    try:
        os.chdir(original_cwd)
    except:
        pass
    # You can choose to raise the error or continue
    # raise e


@router.post("/create_session")
async def create_session(file: UploadFile = File(...)):
    """Upload an image and create a new session."""
    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            raise HTTPException(status_code=400, detail="Invalid image file")
        
        session_id = session_manager.create_session(img)
        
        return JSONResponse(content={
            "session_id": session_id,
            "width": img.shape[1],
            "height": img.shape[0]
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/add_point", response_model=SessionState)
async def add_point(request: AddPointRequest):
    """Add a point to the session and run SAM segmentation."""
    if not session_manager.session_exists(request.session_id):
        raise HTTPException(status_code=404, detail="Session not found")
    
    return session_manager.add_point(request.session_id, request.point)


@router.post("/remove_last_point", response_model=SessionState)
async def remove_last_point(session_id: str):
    """Remove the last added point."""
    if not session_manager.session_exists(session_id):
        raise HTTPException(status_code=404, detail="Session not found")
    
    return session_manager.remove_last_point(session_id)


@router.post("/clear_session", response_model=SessionState)
async def clear_session(session_id: str):
    """Clear all points, masks, and bboxes from session."""
    if not session_manager.session_exists(session_id):
        raise HTTPException(status_code=404, detail="Session not found")
    
    return session_manager.clear_session(session_id)


@router.get("/get_state", response_model=SessionState)
async def get_state(session_id: str):
    """Get current state of the session."""
    if not session_manager.session_exists(session_id):
        raise HTTPException(status_code=404, detail="Session not found")
    
    return session_manager.get_session_state(session_id)

@router.get("/get_ocr")
async def get_ocr(session_id: str):
    """Get OCR results for a session."""
    if not session_manager.session_exists(session_id):
        raise HTTPException(status_code=404, detail="Session not found")
    
    try:
        crops = get_crops(session_id)
        # got the crops
        ocr_results = session_manager.get_ocr_results(session_id)
        print(ocr_results)
        return JSONResponse(content={
            "session_id": session_id,
            "ocr_results": ocr_results,
            "count": len(ocr_results)
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@router.post("/update_ocr")
async def update_ocr(data: dict = Body(...)):
    """Update OCR results with target text and apply edits to image."""
    session_id = data.get("session_id")
    updated_results = data.get("ocr_results")
    
    if not session_manager.session_exists(session_id):
        raise HTTPException(status_code=404, detail="Session not found")
    
    try:
        session_manager.set_ocr_results(session_id, updated_results)
        print(updated_results)
        
        # Call edit_text to process and affix edited text onto original image
        edited_results = edit_text(session_id)
        
        return JSONResponse(content={
            "message": "OCR results updated and image edited successfully",
            "count": len(updated_results)
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/delete_session")
async def delete_session(session_id: str):
    """Delete a session and free up memory."""
    if not session_manager.session_exists(session_id):
        raise HTTPException(status_code=404, detail="Session not found")
    
    session_manager.delete_session(session_id)
    return JSONResponse(content={"message": "Session deleted successfully"})

@router.get("/get_edited_image")
async def get_edited_image(session_id: str):
    """Get the current (edited) image from session."""
    if not session_manager.session_exists(session_id):
        raise HTTPException(status_code=404, detail="Session not found")
    
    try:
        img_rgb = session_manager.get_image_rgb(session_id)
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        
        # Encode image to bytes
        _, buffer = cv2.imencode('.png', img_bgr)
        io_buf = io.BytesIO(buffer)
        
        return StreamingResponse(io_buf, media_type="image/png")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))