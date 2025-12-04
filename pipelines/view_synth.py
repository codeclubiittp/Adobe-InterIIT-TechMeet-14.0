import torch
from diffusers import DiffusionPipeline, EulerAncestralDiscreteScheduler
from PIL import Image
import requests
from io import BytesIO

def generate_views(image_path, output_path="output_views.png"):
    print("Loading model to VRAM...")
    pipeline = DiffusionPipeline.from_pretrained(
        "sudo-ai/zero123plus-v1.2",
        custom_pipeline="sudo-ai/zero123plus-pipeline",
        torch_dtype=torch.float16
    )
    pipeline.scheduler = EulerAncestralDiscreteScheduler.from_config(
        pipeline.scheduler.config, timestep_spacing="trailing"
    )
    pipeline.to("cuda:0")
    pipeline.enable_attention_slicing() 
    pipeline.enable_vae_tiling() 

    try:
        pipeline.enable_xformers_memory_efficient_attention()
        print("Xformers enabled (Maximum efficiency).")
    except Exception as e:
        print("Xformers not found. Relying on standard slicing.")

    if image_path.startswith("http"):
        response = requests.get(image_path)
        input_image = Image.open(BytesIO(response.content))
    else:
        input_image = Image.open(image_path)

    input_image = input_image.convert("RGB")
    input_image = input_image.resize((320, 320), Image.LANCZOS)

    print("Generating views...")
    with torch.no_grad():
        result = pipeline(
            input_image, 
            num_inference_steps=30, 
            guidance_scale=4.0,
        ).images[0]

    result.save(output_path)
    print(f"Views saved to {output_path}")

if __name__ == "__main__":
    generate_views("./assets/i4.jpg")

