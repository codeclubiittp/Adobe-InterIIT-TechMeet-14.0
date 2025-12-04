from typing import Dict
import cv2
import numpy as np
from util.sam_utils import run_point_segmentation, merge_overlapping_boxes, IOU_THRESHOLD, Point, BoundingBox, SessionState
import uuid


class SessionManager:
    def __init__(self):
        self.sessions: Dict[str, dict] = {}
    
    def create_session(self, img_bgr: np.ndarray) -> str:
        """Create a new session with an image."""
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        session_id = str(uuid.uuid4())
        
        self.sessions[session_id] = {
            "img_rgb": img_rgb,
            "img_bgr": img_bgr,
            "points": [],
            "masks_list": [],
            "bboxes_list": [],
            "ocr_results":[]
        }
        
        return session_id
    
    def session_exists(self, session_id: str) -> bool:
        """Check if a session exists."""
        return session_id in self.sessions
    
    def add_point(self, session_id: str, point: Point) -> SessionState:
        """Add a point and run SAM segmentation."""
        session = self.sessions[session_id]
        img_rgb = session["img_rgb"]
        
        session["points"].append(point)
        
        mask, bbox = run_point_segmentation(img_rgb, point.x, point.y)
        
        if mask is not None:
            session["masks_list"].append(mask)
        if bbox is not None:
            session["bboxes_list"].append(bbox)
        
        return self._get_session_state(session_id)
    
    def remove_last_point(self, session_id: str) -> SessionState:
        """Remove the last added point."""
        session = self.sessions[session_id]
        
        if session["points"]:
            session["points"].pop()
        if session["masks_list"]:
            session["masks_list"].pop()
        if session["bboxes_list"]:
            session["bboxes_list"].pop()
        
        return self._get_session_state(session_id)
    
    def clear_session(self, session_id: str) -> SessionState:
        """Clear all points, masks, and bboxes."""
        session = self.sessions[session_id]
        session["points"].clear()
        session["masks_list"].clear()
        session["bboxes_list"].clear()
        
        return self._get_session_state(session_id)
    
    def get_session_state(self, session_id: str) -> SessionState:
        """Get current session state."""
        return self._get_session_state(session_id)
    
    def delete_session(self, session_id: str):
        """Delete a session."""
        if session_id in self.sessions:
            del self.sessions[session_id]
    
    def get_active_sessions_count(self) -> int:
        """Get number of active sessions."""
        return len(self.sessions)
    
    def set_ocr_results(self, session_id: str, ocr_results: list):
        """Store OCR results for a session."""
        if session_id not in self.sessions:
            raise ValueError(f"Session '{session_id}' does not exist")
        self.sessions[session_id]["ocr_results"] = ocr_results

    def get_ocr_results(self, session_id: str) -> list:
        """Retrieve OCR results for a session."""
        if session_id not in self.sessions:
            raise ValueError(f"Session '{session_id}' does not exist")
        return self.sessions[session_id].get("ocr_results", [])
    
    def update_image(self, session_id: str, img_rgb: np.ndarray):
        """Update the RGB image for a session."""
        if session_id not in self.sessions:
            raise ValueError(f"Session '{session_id}' does not exist")
        self.sessions[session_id]["img_rgb"] = img_rgb
        # Optionally update BGR version too
        self.sessions[session_id]["img_bgr"] = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    
    def _get_session_state(self, session_id: str) -> SessionState:
        """Helper to convert session data to SessionState response."""
        session = self.sessions[session_id]
        
        bboxes = [BoundingBox(x1=b[0], y1=b[1], x2=b[2], y2=b[3]) for b in session["bboxes_list"]]
        
        merged = merge_overlapping_boxes(session["bboxes_list"], IOU_THRESHOLD)
        merged_bboxes = [BoundingBox(x1=b[0], y1=b[1], x2=b[2], y2=b[3]) for b in merged]
        
        return SessionState(
            session_id=session_id,
            points=session["points"],
            bboxes=bboxes,
            merged_bboxes=merged_bboxes
        )
    
    def get_image_rgb(self, session_id: str):
        try:
            return self.sessions[session_id]["img_rgb"]
        except KeyError:
            raise ValueError(f"Session '{session_id}' does not exist")