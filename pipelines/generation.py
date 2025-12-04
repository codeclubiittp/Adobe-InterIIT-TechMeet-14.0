import cv2 
import onnxruntime as ort
import numpy as np
import torch
from PIL import Image
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
from diffusers.utils import load_image

from pipelines.utils.profiling import profiler
from pipelines.utils.image_processing import upscale_simple_lanczos, upscale_opencv_super_res
from pipelines.depth import run_depth_estimation


def run_controlnet_generation(
    source_image_path: str,
    depth_model_path: str,
    prompt: str,
    base_model_id: str = "runwayml/stable-diffusion-v1-5",
    depth_control_id: str = "lllyasviel/sd-controlnet-depth",
    canny_control_id: str = "lllyasviel/sd-controlnet-canny",
    upscale_method: str = "lanczos",
    upscale_factor: int = 4,
    save_path_prefix: str = "outputs/controlnet",
    device: str = "cuda"
):
    
    pf = profiler()
    cache_dir = ".hf/"

    pf.begin()
    depth_map_array, _ = run_depth_estimation(
        model_path=depth_model_path,
        image_path=source_image_path,
        show=False
    )
    pf.end("depth estimation time")
    
    image_scaled = (depth_map_array * 255).astype(np.uint8)
    image_3_channel = image_scaled[:, :, None]
    image_3_channel = np.concatenate([image_3_channel, image_3_channel, image_3_channel], axis=2)
    control_image_depth = Image.fromarray(image_3_channel)
    control_image_depth.save(f"{save_path_prefix}_debug_depth.jpg") 
    print("generated depth map")

    image_cv = cv2.imread(source_image_path)
    image_cv = cv2.resize(image_cv, (518, 518)) 

    pf.begin()
    canny_edges = cv2.Canny(image_cv, 50, 100)
    canny_image_3_channel = cv2.cvtColor(canny_edges, cv2.COLOR_GRAY2BGR)
    control_image_canny = Image.fromarray(canny_image_3_channel)
    control_image_canny.save(f"{save_path_prefix}_debug_canny.jpg")
    pf.end("canny estimation time")
    print("generated canny map")

    pf.begin()
    controlnet_depth = ControlNetModel.from_pretrained(
        depth_control_id, 
        torch_dtype=torch.float16, 
        load_in_8bit=True,
        cache_dir=cache_dir
    )
    controlnet_canny = ControlNetModel.from_pretrained(
        canny_control_id, 
        torch_dtype=torch.float16,
        load_in_8bit=True,
        cache_dir=cache_dir
    )

    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        base_model_id, 
        controlnet=[controlnet_depth, controlnet_canny], 
        safety_checker=None, 
        torch_dtype=torch.float16,
        load_in_8bit=True, 
        cache_dir=cache_dir
    )
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)

    pipe.to(device)
    pf.end("pipeline load time")

    controlnet_scales = [1.0, 0.9] 

    pf.begin()
    with torch.inference_mode():
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            image = pipe(
                prompt=prompt, 
                image=[control_image_depth, control_image_canny], 
                num_inference_steps=30,
                guidance_scale=10.0,
                controlnet_conditioning_scale=controlnet_scales,
            ).images[0]

    pf.end("base generation time")
    print(f"base image generated: {image.width}x{image.height}")

    image.save(f'{save_path_prefix}_base.jpg')
    print(f"saved base image: {save_path_prefix}_base.jpg")

    print(f"starting upscaling with method: {upscale_method}")
    if upscale_method == "lanczos":
        pf.begin()
        upscaled_image = upscale_simple_lanczos(image, scale_factor=upscale_factor)
        pf.end("lanczos upscaling time")
        
    elif upscale_method == "opencv":
        pf.begin()
        upscaled_image = upscale_opencv_super_res(image, scale_factor=upscale_factor)
        pf.end("opencv upscaling time")
    else:
        print(f"unknown upscale method: {upscale_method}. not upscaling.")
        upscaled_image = image
        
    print(f" upscaled image: {upscaled_image.width}x{upscaled_image.height}")

    upscaled_image.save(f'{save_path_prefix}_upscaled.jpg')