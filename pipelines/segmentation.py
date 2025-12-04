import cv2
import numpy as np
from pprint import pprint
import tflite_runtime.interpreter as interpreter
from pipelines.utils.profiling import profiler
from pipelines.utils.image_processing import letterbox

def run_mobilesam_segmentation(
    encoder_path: str,
    decoder_path: str,
    image_path: str,
    point_coords: list,
    point_labels: list,
    save_path_prefix: str = "outputs/mobilesam"
):
    tfi_sam_encoder = interpreter.Interpreter(model_path=encoder_path)
    tfi_sam_decoder = interpreter.Interpreter(model_path=decoder_path)

    pprint(tfi_sam_encoder.get_input_details())
    pprint(tfi_sam_decoder.get_input_details())
    pprint(tfi_sam_encoder.get_output_details())
    pprint(tfi_sam_decoder.get_output_details())

    tfi_sam_encoder.allocate_tensors()
    tfi_sam_decoder.allocate_tensors()

    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_resized = cv2.resize(image_rgb, (1024, 1024))
    image_np = np.array(image_resized, dtype=np.float32) / 255.0
    image_np = np.expand_dims(image_np, axis=0) 

    coords = np.array([point_coords], dtype=np.float32) 
    labels = np.array([point_labels], dtype=np.float32) 

    encoder_inputs = tfi_sam_encoder.get_input_details()
    tfi_sam_encoder.set_tensor(encoder_inputs[0]['index'], image_np)

    tfi_sam_encoder.invoke()

    encoder_outputs = tfi_sam_encoder.get_output_details()
    image_embeddings = tfi_sam_encoder.get_tensor(encoder_outputs[0]['index'])

    decoder_inputs = tfi_sam_decoder.get_input_details()
    tfi_sam_decoder.set_tensor(decoder_inputs[0]['index'], image_embeddings)
    tfi_sam_decoder.set_tensor(decoder_inputs[1]['index'], coords)
    tfi_sam_decoder.set_tensor(decoder_inputs[2]['index'], labels)

    tfi_sam_decoder.invoke()

    decoder_outputs = tfi_sam_decoder.get_output_details()
    final_mask = tfi_sam_decoder.get_tensor(decoder_outputs[0]['index'])
    final_score = tfi_sam_decoder.get_tensor(decoder_outputs[1]['index'])

    print(f"low-res mask shape: {final_mask.shape}")  
    print(f"score: {final_score[0][0]}")

    mask_256 = final_mask[0, :, :, 0]
    mask_1024 = cv2.resize(
        mask_256,
        (1024, 1024),  
        interpolation=cv2.INTER_LINEAR
    )

    binary_mask = (mask_1024 > 0.50)
    binary_mask_uint8 = binary_mask.astype(np.uint8) * 255

    cv2.imwrite(f"{save_path_prefix}_mask.png", binary_mask_uint8)

    original_image = cv2.imread(image_path)
    original_image_resized = cv2.resize(original_image, (1024, 1024))

    contours, _ = cv2.findContours(binary_mask_uint8, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(original_image_resized, contours, -1, (0, 255, 0), 3) 
    cv2.imwrite(f"{save_path_prefix}_overlay.png", original_image_resized)
    print("mask and overlay image saved!")

def run_fastsam_segmentation(
    model_path: str,
    image_path: str,
    conf_threshold: float = 0.4,
    nms_threshold: float = 0.45,
    save_path_prefix: str = "outputs/fastsam"
):
    pf = profiler()

    print("loading fastsam model...")
    tfi_fastsam = interpreter.Interpreter(model_path=model_path)
    tfi_fastsam.allocate_tensors()

    input_details = tfi_fastsam.get_input_details()
    output_details = tfi_fastsam.get_output_details()

    pprint(input_details)
    pprint(output_details)

    input_height = input_details[0]['shape'][1] 
    input_width = input_details[0]['shape'][2]  
    print(f"model requires input: (1, {input_height}, {input_width}, 3)")

    image_orig = cv2.imread(image_path)
    orig_h, orig_w = image_orig.shape[:2] 

    image_rgb = cv2.cvtColor(image_orig, cv2.COLOR_BGR2RGB)

    image_padded, scale, (pad_w, pad_h) = letterbox(image_rgb, (input_width, input_height))
    image_np = np.array(image_padded, dtype=np.float32) / 255.0
    image_np = np.expand_dims(image_np, axis=0)

    tfi_fastsam.set_tensor(input_details[0]['index'], image_np)

    pf.begin()
    tfi_fastsam.invoke()
    inference_time = pf.end("fastsam inference time")

    def get_tensor_by_name(name):
        for detail in output_details:
            if name in detail['name']:
                return tfi_fastsam.get_tensor(detail['index'])
        raise ValueError(f"could not find output tensor with name containing {name}")

    boxes = get_tensor_by_name('boxes')[0]
    scores = get_tensor_by_name('scores')[0]
    mask_coeffs = get_tensor_by_name('mask_coeffs')[0]
    mask_protos = get_tensor_by_name('mask_protos')[0]

    print("filtering boxes...")
    good_indices = np.where(scores > conf_threshold)[0]

    boxes_norm = boxes[good_indices]
    scores_filtered = scores[good_indices]
    coeffs_filtered = mask_coeffs[good_indices]

    boxes_pixel = boxes_norm.copy()
    boxes_pixel[:, [0, 2]] *= input_width  
    boxes_pixel[:, [1, 3]] *= input_height 

    boxes_x1y1wh = boxes_pixel.copy()
    boxes_x1y1wh[:, 0] = boxes_pixel[:, 0] - boxes_pixel[:, 2] / 2 
    boxes_x1y1wh[:, 1] = boxes_pixel[:, 1] - boxes_pixel[:, 3] / 2 

    final_indices = cv2.dnn.NMSBoxes(
        boxes_x1y1wh.tolist(), 
        scores_filtered.tolist(), 
        conf_threshold, 
        nms_threshold
    )

    if len(final_indices) == 0:
        print("no objects found")
        return
    
    if isinstance(final_indices, tuple):
        print("no objects found after nms")
        return

    final_indices = final_indices.flatten()
    final_coeffs = coeffs_filtered[final_indices]

    print(f"found {len(final_indices)} objects. reconstructing masks...")

    proto_h, proto_w, proto_c = mask_protos.shape 
    protos_flat = mask_protos.reshape(-1, proto_c) 
    low_res_masks = (final_coeffs @ protos_flat.T).reshape(-1, proto_h, proto_w)

    low_res_masks_sigmoid = 1 / (1 + np.exp(-low_res_masks))

    final_combined_mask = np.max(low_res_masks_sigmoid, axis=0) 

    pad_w_proto = int((pad_w / input_width) * proto_w)
    pad_h_proto = int((pad_h / input_height) * proto_h)

    mask_cropped = final_combined_mask[pad_h_proto : proto_h - pad_h_proto, pad_w_proto : proto_w - pad_w_proto]

    binary_mask = (mask_cropped > 0.5)
    binary_mask_uint8 = binary_mask.astype(np.uint8) * 255

    final_mask_full_res = cv2.resize(
        binary_mask_uint8,
        (orig_w, orig_h), 
        interpolation=cv2.INTER_LINEAR
    )

    cv2.imwrite(f"{save_path_prefix}_mask.png", final_mask_full_res)

    contours, _ = cv2.findContours(final_mask_full_res, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(image_orig, contours, -1, (0, 255, 0), 3) 
    cv2.imwrite(f"{save_path_prefix}_overlay.png", image_orig)

    print("mask and overlay image saved!")