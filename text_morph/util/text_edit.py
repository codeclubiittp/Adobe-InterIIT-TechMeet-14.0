from util.state_manager import SessionManager
from util.text_detection import get_boxes
from util.ocr import text_ocr
import cv2 as cv
import numpy as np
import os, sys
from util.session_manager import session_manager
from PIL import Image
import tempfile
import traceback

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TEXTCTRL_PATH = os.path.join(os.path.dirname(SCRIPT_DIR), 'TextCtrl')
if TEXTCTRL_PATH not in sys.path:
    sys.path.insert(0, TEXTCTRL_PATH)
from main import process_single_image,process_multiple_images


def get_crops(session_id):
    print("IN CROPS")
    state = session_manager.get_session_state(session_id)
    merged = state.merged_bboxes
    img = session_manager.get_image_rgb(session_id)
    print("SECOND IN CROPS")
    final_results = []
    print("MERGED: ", merged)
    
    for box in merged:
        x1, y1, x2, y2 = box.x1, box.y1, box.x2, box.y2
        crop = img[y1:y2, x1:x2]
        gb = get_boxes(crop)
        o = text_ocr(crop, gb)
        print("GB", gb)
        print("OCR", o)
        
        if len(gb) != len(o):
            raise Exception("OCR Failed")
        
        for i in range(len(o)):
            temp = {
                "ocr": o[i],
                "gb_coord": gb[i][0],
                "crop_coord": [x1, y1, x2, y2],
                "session_id": session_id
            }
            final_results.append(temp)
    
    session_manager.set_ocr_results(session_id, final_results)
    return final_results


def edit_text(session_id):
    print(f"[DEBUG] Starting edit_text for session: {session_id}")
    
    try:
        ocr = session_manager.get_ocr_results(session_id)
        print(f"[DEBUG] Retrieved OCR results, count: {len(ocr)}")
        
        img = session_manager.get_image_rgb(session_id).copy()
        print(f"[DEBUG] Retrieved image, shape: {img.shape}")
        
        print("INSIDE EDIT RESULTS")
        final_results = []
        
        # Save current working directory
        original_cwd = os.getcwd()
        
        for idx, i in enumerate(ocr):
            print(f"\n[DEBUG] Processing OCR result {idx + 1}/{len(ocr)}")
            print(f"[DEBUG] OCR text: '{i.get('ocr', 'N/A')}'")
            print(f"[DEBUG] Target text: '{i.get('target_text', 'N/A')}'")
            
            x1, y1, x2, y2 = i["crop_coord"]
            print(f"[DEBUG] Crop coordinates: ({x1}, {y1}, {x2}, {y2})")
            
            crop = img[y1:y2, x1:x2]
            print(f"[DEBUG] Crop shape: {crop.shape}")
            
            a1, b1, a2, b2 = i["gb_coord"]
            print(f"[DEBUG] GB coordinates: ({a1}, {b1}, {a2}, {b2})")
            
            bounded_image = crop[b1:b2, a1:a2]
            print(f"[DEBUG] Bounded image shape: {bounded_image.shape}")
            
            # Save bounded_image temporarily
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
                tmp_path = tmp_file.name
                print(f"[DEBUG] Saving to temp file: {tmp_path}")
                Image.fromarray(bounded_image).save(tmp_path)
            
            try:
                print(f"[DEBUG] Calling process_single_image...")
                
                # Change to TextCtrl directory for relative paths
                os.chdir(TEXTCTRL_PATH)
                
                try:
                    _, edited_text_pil = process_single_image(tmp_path, i["ocr"], i["target_text"])
                    print(f"[DEBUG] process_single_image completed, PIL size: {edited_text_pil.size}")
                finally:
                    # Restore original working directory
                    os.chdir(original_cwd)
                
                edited_text_image = np.array(edited_text_pil)
                print(f"[DEBUG] Edited text image shape: {edited_text_image.shape}")
                
            except Exception as e:
                print(f"[ERROR] process_single_image failed: {str(e)}")
                print(f"[ERROR] Traceback: {traceback.format_exc()}")
                raise
            finally:
                os.unlink(tmp_path)
                print(f"[DEBUG] Cleaned up temp file")
            
            # Calculate absolute coordinates
            abs_x1 = x1 + a1
            abs_y1 = y1 + b1
            abs_x2 = x1 + a2
            abs_y2 = y1 + b2
            print(f"[DEBUG] Absolute coordinates: ({abs_x1}, {abs_y1}, {abs_x2}, {abs_y2})")
            
            # Resize edited_text_image
            h_target = abs_y2 - abs_y1
            w_target = abs_x2 - abs_x1
            print(f"[DEBUG] Target dimensions: {w_target}x{h_target}")
            
            edited_text_image = cv.resize(edited_text_image, (w_target, h_target))
            print(f"[DEBUG] Resized edited image shape: {edited_text_image.shape}")
            
            # Affix edited_text_image onto original image
            img[abs_y1:abs_y2, abs_x1:abs_x2] = edited_text_image
            print(f"[DEBUG] Affixed edited text to image")
            
            # Don't include the large image array in results - this is likely causing the 500 error
            temp = {
                "ocr": i["ocr"],
                "target_text": i["target_text"],
                "crop_coord": i["crop_coord"],
                "gb_coord": i["gb_coord"],
                "absolute_coordinates": [abs_x1, abs_y1, abs_x2, abs_y2],
                "edited_shape": list(edited_text_image.shape),  # Just store shape info
                "session_id": session_id
            }
            final_results.append(temp)
            print(f"[DEBUG] Added result {idx + 1} to final_results")
        
        print(f"\n[DEBUG] All edits processed. Updating session...")
        
        # Update the session with the edited image
        session_manager.update_image(session_id, img)
        print(f"[DEBUG] Session image updated")
        #cv.imwrite("output.png",img)
        # from PIL import Image
        # import cv2 as cv

        # final_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        # Image.fromarray(final_rgb).save("output.png")
        Image.fromarray(img).save("output.png")
        session_manager.set_ocr_results(session_id, final_results)
        print(f"[DEBUG] Session OCR results updated")
        
        print(f"[DEBUG] edit_text completed successfully. Returning {len(final_results)} results")
        return final_results
        
    except Exception as e:
        print(f"[ERROR] Exception in edit_text: {str(e)}")
        print(f"[ERROR] Exception type: {type(e).__name__}")
        print(f"[ERROR] Full traceback:\n{traceback.format_exc()}")
        raise  # Re-raise to let FastAPI handle it

# def edit_text(session_id):
#     ocr = session_manager.get_ocr_results(session_id)
#     img = session_manager.get_image_rgb(session_id).copy()
    
#     # Prepare batch data
#     image_data_list = []
#     temp_files = []
    
#     for i in ocr:
#         x1, y1, x2, y2 = i["crop_coord"]
#         crop = img[y1:y2, x1:x2]
#         a1, b1, a2, b2 = i["gb_coord"]
#         bounded_image = crop[b1:b2, a1:a2]
        
#         tmp_file = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
#         tmp_path = tmp_file.name
#         tmp_file.close()
#         Image.fromarray(bounded_image).save(tmp_path)
#         temp_files.append(tmp_path)
        
#         image_data_list.append((tmp_path, i["ocr"], i["target_text"]))
    
#     try:
#         # Process all images with single model load
#         edited_results = process_multiple_images(image_data_list)
        
#         final_results = []
#         for idx, i in enumerate(ocr):
#             x1, y1, x2, y2 = i["crop_coord"]
#             crop = img[y1:y2, x1:x2]
#             a1, b1, a2, b2 = i["gb_coord"]
            
#             _, edited_text_pil = edited_results[idx]
#             edited_text_image = np.array(edited_text_pil)
            
#             abs_x1 = x1 + a1
#             abs_y1 = y1 + b1
#             abs_x2 = x1 + a2
#             abs_y2 = y1 + b2
            
#             h_target = abs_y2 - abs_y1
#             w_target = abs_x2 - abs_x1
#             edited_text_image = cv.resize(edited_text_image, (w_target, h_target))
            
#             img[abs_y1:abs_y2, abs_x1:abs_x2] = edited_text_image
            
#             temp = {
#                 **i,
#                 "edited_text_image": edited_text_image.tolist(),
#                 "absolute_coordinates": [abs_x1, abs_y1, abs_x2, abs_y2]
#             }
#             final_results.append(temp)
#     finally:
#         # Clean up all temporary files
#         for tmp_path in temp_files:
#             os.unlink(tmp_path)
    
#     session_manager.update_image(session_id, img)
#     session_manager.set_ocr_results(session_id, final_results)
    
#     return final_results