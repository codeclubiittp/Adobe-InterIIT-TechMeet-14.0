import onnxruntime as ort
import numpy as np
from PIL import Image

from pipelines.utils.profiling import profiler
from pipelines.utils.onnx_helpers import log_session_details
from pipelines.utils.visualization import visualize_depth_map

def run_depth_estimation(
    model_path: str,
    image_path: str,
    save_path: str = "outputs/depth_map.png",
    show: bool = True
) -> tuple[np.ndarray, float]:

    onnx_session = ort.InferenceSession(
        path_or_bytes=model_path
    )

    log_session_details(onnx_session, other={"sessiontype" : "depth-estimation"})

    pf = profiler()

    image = Image.open(image_path).resize(size=(518, 518))
    image_np = np.array(image, dtype=np.float32).transpose((2, 0, 1))/255.0
    image_np = np.expand_dims(image_np, axis=0)

    output_names = [o.name for o in onnx_session.get_outputs()]
    input_names = [i.name for i in onnx_session.get_inputs()]

    onnx_input_feed = {
        input_names[0] : image_np
    }

    pf.begin()
    depth_estimate = onnx_session.run(
        output_names=output_names,
        input_feed=onnx_input_feed
    )
    inference_time = pf.end("depth estimation time")

    depth_map = depth_estimate[0].squeeze() 
    depth_min = depth_map.min()
    depth_max = depth_map.max()
    normalized_map = (depth_map - depth_min) / (depth_max - depth_min)

    if show:
        visualize_depth_map(normalized_map, save_path)

    return normalized_map, inference_time