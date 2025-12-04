import onnxruntime as ort
import cv2
import time
import numpy as np
from pipelines.utils.onnx_helpers import log_session_details
from pipelines.utils.visualization import visualize_pose_predictions

def run_mediapipe_pose(
    model_path: str,
    image_path: str = "images/i1.jpg",
    savefig: bool = True,
    save_path: str = "outputs/estimated_pose.png",
    score_thresh: float = 0.5
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    
    onnx_session = ort.InferenceSession(
        model_path,
        providers=["CPUExecutionProvider"]
    )

    log_session_details(onnx_session)

    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_resized = cv2.resize(image_rgb, (128, 128))

    image_np = np.array(image_resized, dtype=np.float32)
    image_np = (image_np / 127.5) - 1.0
    image_np = np.transpose(image_np, (2, 0, 1))
    image_np = np.expand_dims(image_np, axis=0)

    print(f"image (shape) : {image_np.shape}")
    print(f"image (dtype) : {image_np.dtype}")

    input_names = [input.name for input in onnx_session.get_inputs()]
    output_names = [output.name for output in onnx_session.get_outputs()]

    input_feed = {
        input_names[0] : image_np
    } 

    begin = time.perf_counter()
    box_coords_1, box_coords_2, box_scores_1, box_scores_2 = onnx_session.run(
        output_names=output_names,
        input_feed=input_feed
    )
    end = time.perf_counter()

    print(f"inference time : {(end - begin):.4f}s")

    print(box_coords_1.shape)
    print(box_coords_2.shape)
    print(box_scores_1.shape)
    print(box_scores_2.shape)

    if savefig:
        visualize_pose_predictions(
            image, 
            box_coords_1, 
            box_coords_2, 
            box_scores_1, 
            box_scores_2,
            savefile=save_path,
            score_thresh=score_thresh
        )

    return box_coords_1, box_coords_2, box_scores_1, box_scores_2