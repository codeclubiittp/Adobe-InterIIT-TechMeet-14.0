import torch
from PIL import Image
import numpy as np
import onnxruntime as ort
import cv2
from pipelines.utils.onnx_helpers import log_session_details

def modnet_matting(
    model_path: str = "models/modnet_photographic_portrait_matting.onnx",
    image_path: str = "assets/i1.jpg"
):
    onnx_session = ort.InferenceSession(
        model_path,
        providers=["CPUExecutionProvider"]
    )
    log_session_details(onnx_session)

    input_name = onnx_session.get_inputs()[0].name
    output_name = onnx_session.get_outputs()[0].name

    img = cv2.imread(image_path)
    original_h, original_w, _ = img.shape
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (512,512), interpolation=cv2.INTER_LINEAR)
    img_normalized = img_resized.astype(np.float32) / 255.0
    img_chw = img_normalized.transpose(2, 0, 1)
    input_tensor = np.expand_dims(img_chw, axis=0)
    
    ort_inputs = {input_name: input_tensor}
    ort_outputs = onnx_session.run([output_name], ort_inputs)

    alpha_matte = ort_outputs[0].squeeze() 
    alpha_matte = (alpha_matte * 255).astype(np.uint8)

    alpha_matte_original_size = cv2.resize(
        alpha_matte, 
        (original_w, original_h), 
        interpolation=cv2.INTER_LINEAR
    )
    cv2.imwrite("outputs/alpha_matte.png", alpha_matte_original_size)

if __name__ == "__main__":
    modnet_matting()