from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import torch
import torch.profiler as tp
import time

def init_profiler():
    activities = [
        tp.ProfilerActivity.CUDA,
        tp.ProfilerActivity.CPU
    ],

    return tp.profile(
        activities=activities,
        record_shapes=True,
        profile_memory=True,
        with_flops=True,
        with_stack=True,
        with_modules=True,
    )

with tp.profile(
activities=[
    tp.ProfilerActivity.CUDA,
    tp.ProfilerActivity.CPU
],
record_shapes=True,
profile_memory=True,
with_stack=True,
with_flops=True,
with_modules=True,
) as profiler:
    with tp.record_function("blip-image-captioning-base-fp16-TOTAL"):
        processor = BlipProcessor.from_pretrained("gospacedev/blip-image-captioning-base-bf16")
        model = BlipForConditionalGeneration.from_pretrained(
            "gospacedev/blip-image-captioning-base-bf16",
            torch_dtype=torch.bfloat16,
        )

        # model_quant = torch.quantization.quantize_dynamic(
        #     model,
        #     {torch.nn.Linear},
        #     dtype=torch.qint8
        # )

        with tp.record_function("blip-image-captioning-base-fp16-LOADING"):
            model = model.to("cuda")
            model.eval()

        with tp.record_function("blip-image-captioning-base-fp16-INFERENCE"):
            image = Image.open("assets/i4.jpg").convert("RGB")
            inputs = processor(images=image, return_tensors="pt").to("cuda")

            out = model.generate(**inputs)
            caption = processor.decode(out[0], skip_special_tokens=True)

        # print(inputs["pixel_values"].device)
        # print(model.device)
        # print("Caption:", caption)

print(
    profiler.key_averages(group_by_input_shape=True).table(
        sort_by="cpu_time_total", row_limit=10
    )
)
profiler.export_chrome_trace("trace.json")
