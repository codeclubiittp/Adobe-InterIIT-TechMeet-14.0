import onnxruntime as ort
import cv2
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from utils import log_session_details

def _get_colors(num_colors):
    cmap = plt.cm.get_cmap('hsv', num_colors)
    return [mcolors.to_hex(cmap(i)) for i in range(num_colors)]

def _visualize_pose_predictions(
    image: np.ndarray, 
    coords1: np.ndarray, 
    coords2: np.ndarray, 
    scores1: np.ndarray, 
    scores2: np.ndarray, 
    savefile: str = "images/estimated_pose.png",
    score_thresh: float = 0.5
):
    # concatenate separate predictions into single ndarray
    coords = np.concatenate([coords1[0], coords2[0]], axis=0)  
    scores = np.concatenate([scores1[0], scores2[0]], axis=0)  
    # rescale it to actual height and width
    h, w, _ = image.shape
    coords_scaled = coords.copy()
    coords_scaled[..., 0::2] *= w   
    coords_scaled[..., 1::2] *= h   
    # logits->probability
    scores = 1/(1 + np.exp(-scores))
    # apply threshold mask
    mask = scores[:, 0] > score_thresh
    coords_scaled = coords_scaled[mask]
    scores = scores[mask]
    
    num_poses = len(coords_scaled)
    print(f"Visualizing {num_poses} poses above threshold {score_thresh}")

    if num_poses == 0:
        return

    colors = _get_colors(num_poses)
    plt.figure(figsize=(10, 10)) 
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    
    for i, c in enumerate(coords_scaled):
        color = colors[i]
        for j in range(0, len(c), 2):
            x, y = c[j], c[j+1]
            plt.scatter(x, y, s=20, c=color, alpha=0.9)
        plt.text(c[0], c[1] - 5, f"{scores[i][0]:.2f}", color='white', fontsize=9, 
                 bbox=dict(facecolor=color, alpha=0.8, pad=1)) 
    plt.axis("off")
    plt.savefig(savefile) 

def pose_estimation_qualcomm_mediapipe(
    image_path: str = "images/i1.jpg",
    savefig: bool = True
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns
    -------
    [{'name': 'box_coords_1', 'shape': [1, 512, 12], 'type': 'tensor(float)'},
    {'name': 'box_coords_2', 'shape': [1, 384, 12], 'type': 'tensor(float)'},
    {'name': 'box_scores_1', 'shape': [1, 512, 1], 'type': 'tensor(float)'},
    {'name': 'box_scores_2', 'shape': [1, 384, 1], 'type': 'tensor(float)'}]
    """
    onnx_session = ort.InferenceSession(
        "models/qualcomm-mediapipe-post-estimation/model.onnx",
        providers=["CPUExecutionProvider"]
    )

    log_session_details(onnx_session)

    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_resized = cv2.resize(image, (128, 128))

    image_np = np.array(image_resized, dtype=np.float32)
    # [-1, 1] normalization
    image_np = (image_np / 127.5) - 1.0
    # HWC->CHW
    image_np = np.transpose(image_np, (2, 0, 1))
    # add batch dimension
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

    print(f"Inference Time : {(end - begin):.4f}s")

    print(box_coords_1.shape)
    print(box_coords_2.shape)
    print(box_scores_1.shape)
    print(box_scores_2.shape)

    if savefig:
        _visualize_pose_predictions(image, box_coords_1, box_coords_2, box_scores_1, box_scores_2)

    return box_coords_1, box_coords_2, box_scores_1, box_scores_2

if __name__ == "__main__":
    pose_estimation_qualcomm_mediapipe()
