import onnxruntime as ort
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pipelines.utils.onnx_helpers import log_session_details

def run_lama_inpainting(
    model_path: str,
    image_path: str = "images/i1.jpg",
    savefig: bool = True,
    save_path_prefix: str = "outputs/lama"
):
    onnx_session = ort.InferenceSession(
        model_path,
        providers=["CPUExecutionProvider"]
    )

    log_session_details(onnx_session)

    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    orig_h, orig_w = image.shape[:2]

    mask = np.zeros((orig_h, orig_w), dtype=np.uint8)
    mask_h, mask_w = orig_h // 4, orig_w // 4
    start_h = int(orig_h * 0.05)  
    end_h = int(orig_h * 0.75)    
    
    start_w = int(orig_w * 0.05)  
    end_w = int(orig_w * 0.35)
    mask[start_h:end_h, start_w:end_w] = 1  

    image_resized = cv2.resize(image, (512, 512))
    mask_resized = cv2.resize(mask, (512, 512))

    
    image_norm = image_resized.astype(np.float32) / 255.0
    mask_broadcastable = np.expand_dims(mask_resized, axis=-1)
    image_masked = image_norm * (1 - mask_broadcastable)
    image_onnx = np.transpose(image_masked, (2, 0, 1))
    image_onnx = np.expand_dims(image_onnx, axis=0).astype(np.float32)

    mask_onnx = np.expand_dims(mask_resized, axis=0)
    mask_onnx = np.expand_dims(mask_onnx, axis=0).astype(np.float32)

    input_names = [input.name for input in onnx_session.get_inputs()]
    output_names = [output.name for output in onnx_session.get_outputs()]

    inputs = {
        input_names[0]: image_onnx,
        input_names[1]: mask_onnx
    }
    
    painted_image_onnx = onnx_session.run(output_names, inputs)[0]

    painted_image_norm = np.squeeze(painted_image_onnx, axis=0)
    painted_image_norm = np.transpose(painted_image_norm, (1, 2, 0))
    painted_image_norm = np.clip(painted_image_norm, 0, 1)
    painted_image_denorm = (painted_image_norm * 255.0).astype(np.uint8) 
    final_image_resized = cv2.resize(painted_image_denorm, (orig_w, orig_h), interpolation=cv2.INTER_CUBIC)
    
    if savefig:
        final_image_bgr = cv2.cvtColor(final_image_resized, cv2.COLOR_RGB2BGR)
        output_filename = f"{save_path_prefix}_inpainted.jpg"
        cv2.imwrite(output_filename, final_image_bgr)

        masked_input_rgb = (image_masked * 255.0).astype(np.uint8)
        masked_input_bgr = cv2.cvtColor(masked_input_rgb, cv2.COLOR_RGB2BGR)
        masked_filename = f"{save_path_prefix}_masked.jpg"
        cv2.imwrite(masked_filename, masked_input_bgr)

    plt.figure(figsize=(18, 6))
    plt.subplot(1, 3, 1)
    plt.imshow(image) 
    plt.title("original image")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow((image_masked * 255.0).astype(np.uint8)) 
    plt.title("masked input (resized 512x512)")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(final_image_resized) 
    plt.title("inpainted result")
    plt.axis("off")

    plt.tight_layout()
    plt.savefig(f"{save_path_prefix}_comparison.png")
    plt.close()