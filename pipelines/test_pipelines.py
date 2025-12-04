import os
import torch

from pipelines.inpainting import run_lama_inpainting
from pipelines.detection import run_yolox_detection, run_gdino_detection
from pipelines.segmentation import run_mobilesam_segmentation, run_fastsam_segmentation
from pipelines.depth import run_depth_estimation
from pipelines.pose import run_mediapipe_pose
from pipelines.generation import run_controlnet_generation

def main():
    print("--- initializing pipeline tests ---")
    
    base_output_dir = "outputs"
    os.makedirs(base_output_dir, exist_ok=True)
    
    test_image = "assets/i1.jpg"
    
    if not os.path.exists(test_image):
        print(f"test image not found at {test_image}. skipping tests.")
        print("please download a test image and save it as assets/image_1.jpg")
        return
        
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"using device: {device}")

    # --- test inpainting ---
    try:
        print("\n--- testing lama inpainting ---")
        model_path = "models/qualcomm-lama-dilated/model.onnx"
        if os.path.exists(model_path):
            run_lama_inpainting(
                model_path=model_path,
                image_path=test_image,
                save_path_prefix=os.path.join(base_output_dir, "lama")
            )
        else:
            print(f"model not found: {model_path}, skipping.")
    except Exception as e:
        print(f"inpainting test failed: {e}")

    # --- test yolox detection ---
    try:
        print("\n--- testing yolox detection ---")
        model_path = "models/qualcomm-yolo-x/model.onnx"
        if os.path.exists(model_path):
            run_yolox_detection(
                model_path=model_path,
                image_path=test_image,
                save_path=os.path.join(base_output_dir, "yolox_estimated.png")
            )
        else:
            print(f"model not found: {model_path}, skipping.")
    except Exception as e:
        print(f"yolox detection test failed: {e}")

    # --- test gdino detection ---
    try:
        print("\n--- testing gdino detection ---")
        run_gdino_detection(
            image_path=test_image,
            text_prompt="a person. a building.",
            save_path=os.path.join(base_output_dir, "gdino_detected.png"),
            device=device
        )
    except Exception as e:
        print(f"gdino detection test failed: {e}")

    # --- test mobilesam segmentation ---
    try:
        print("\n--- testing mobilesam segmentation ---")
        encoder_path = "models/MobileSam_MobileSAMEncoder_float.tflite"
        decoder_path = "models/MobileSam_MobileSAMDecoder_float.tflite"
        if os.path.exists(encoder_path) and os.path.exists(decoder_path):
            run_mobilesam_segmentation(
                encoder_path=encoder_path,
                decoder_path=decoder_path,
                image_path=test_image,
                point_coords=[[600, 20], [600, 450]],
                point_labels=[1.0, 1.0],
                save_path_prefix=os.path.join(base_output_dir, "mobilesam")
            )
        else:
            print(f"mobilesam models not found, skipping.")
    except Exception as e:
        print(f"mobilesam segmentation test failed: {e}")

    # --- test fastsam segmentation ---
    try:
        print("\n--- testing fastsam segmentation ---")
        model_path = "models/FastSam-X_float.tflite"
        if os.path.exists(model_path):
            run_fastsam_segmentation(
                model_path=model_path,
                image_path=test_image,
                save_path_prefix=os.path.join(base_output_dir, "fastsam")
            )
        else:
            print(f"model not found: {model_path}, skipping.")
    except Exception as e:
        print(f"fastsam segmentation test failed: {e}")

    # --- test depth estimation ---
    try:
        print("\n--- testing depth estimation ---")
        model_path = "models/qualcomm-depth-anything-v2/model.onnx"
        if os.path.exists(model_path):
            run_depth_estimation(
                model_path=model_path,
                image_path=test_image,
                save_path=os.path.join(base_output_dir, "depth_map.png")
            )
        else:
            print(f"model not found: {model_path}, skipping.")
    except Exception as e:
        print(f"depth estimation test failed: {e}")

    # --- test pose estimation ---
    try:
        print("\n--- testing mediapipe pose ---")
        model_path = "models/qualcomm-mediapipe-post-estimation/model.onnx"
        if os.path.exists(model_path):
            run_mediapipe_pose(
                model_path=model_path,
                image_path=test_image,
                save_path=os.path.join(base_output_dir, "estimated_pose.png")
            )
        else:
            print(f"model not found: {model_path}, skipping.")
    except Exception as e:
        print(f"pose estimation test failed: {e}")

    # --- test controlnet generation ---
    try:
        print("\n--- testing controlnet generation ---")
        depth_model_path = "models/qualcomm-depth-anything-v2/model.onnx"
        test_image_2 = "assets/test_2.jpg" 
        
        if not os.path.exists(test_image_2):
            print(f"controlnet test image {test_image_2} not found, using {test_image}")
            test_image_2 = test_image
            
        if os.path.exists(depth_model_path) and device == "cuda":
            run_controlnet_generation(
                source_image_path=test_image_2,
                depth_model_path=depth_model_path,
                prompt="(van gogh style) (black truck) hyperdetailed realism. remove any aliens. 4k high resolution",
                save_path_prefix=os.path.join(base_output_dir, "controlnet_gen"),
                device=device
            )
        elif device != "cuda":
            print("controlnet test requires a cuda device, skipping.")
        else:
            print(f"depth model not found: {depth_model_path}, skipping controlnet.")
    except Exception as e:
        print(f"controlnet generation test failed: {e}")
        
    print("\n--- all tests complete ---")

if __name__ == "__main__":
    main()