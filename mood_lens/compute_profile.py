import torch
import torch.profiler
from torch.profiler import profile, ProfilerActivity
import matplotlib.pyplot as plt
import numpy as np
import logging
import config
from fvcore.nn import FlopCountAnalysis
from torchvision import transforms
from PIL import Image
from vsearch import VSearchEngine
from lut_generator import NeuralLUTGenerator
from lut_model import TrilinearLUT
from engine_core import colour_correct

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def set_publication_style():
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman'],
        'font.size': 12,           
        'axes.labelsize': 13,
        'axes.titlesize': 15,
        'xtick.labelsize': 11,
        'ytick.labelsize': 11,
        'legend.fontsize': 11,
        'figure.titlesize': 18,
        'axes.grid': True,
        'grid.alpha': 0.3,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'figure.dpi': 300,
        'savefig.bbox': 'tight'
    })

def format_count(n):
    if n > 1e12: return f"{n/1e12:.2f}T"
    if n > 1e9: return f"{n/1e9:.3f}G"
    if n > 1e6: return f"{n/1e6:.2f}M"
    if n > 1e3: return f"{n/1e3:.2f}K"
    return str(n)

def compute_flops_lut_model(lut_dim, device):
    model = TrilinearLUT(dim=lut_dim).to(device)
    dummy_input = torch.randn(1, 3, 512, 512).to(device)
    
    flops = FlopCountAnalysis(model, dummy_input)
    flops.unsupported_ops_warnings(False)
    return flops, model

def print_stat_card(model, latency_sec, vram_fwd_gb, vram_bwd_gb, flops_count):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    param_size = sum(p.nelement() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.nelement() * b.element_size() for b in model.buffers())
    size_all_mb = (param_size + buffer_size) / 1024**2
    
    throughput = 1.0 / latency_sec if latency_sec > 0 else 0
    total_vram_mb = (vram_fwd_gb + vram_bwd_gb) * 1024 

    print("\n" + "=" * 60)
    print(f"{'COMPUTE STAT CARD: NEURAL LUT PIPELINE':^60}")
    print("=" * 60)
    print(" [Model Architecture]")
    print(f"  • Parameters (Total)  : {total_params / 1e6:.2f} M")
    print(f"  • Parameters (Train)  : {trainable_params / 1e6:.2f} M")
    print(f"  • Weights Size (Disk) : {size_all_mb:.2f} MB")
    print(f"  • Complexity (FLOPs)  : {flops_count / 1e6:.2f} MFLOPs")
    print("-" * 60)
    print(" [Runtime Metrics]")
    print(f"  • VRAM (Fwd+Bwd)      : {total_vram_mb:.2f} MB")
    print(f"  • Total Latency       : {latency_sec * 1000:.2f} ms")
    print(f"  • System Throughput   : {throughput:.2f} img/sec")
    print("=" * 60 + "\n")

def print_detailed_report(model, flops_analysis, max_depth=3):
    flop_counts = flops_analysis.by_module()
    
    print("\n" + "=" * 100)
    print(f"{'DETAILED LAYER-WISE BREAKDOWN':^100}")
    print("=" * 100)
    print(f"{'Module':<60} | {'#Params / Shape':>20} | {'#FLOPs':>15}")
    print("-" * 100)

    def get_params_or_shape(m):
        if hasattr(m, 'weight') and hasattr(m.weight, 'shape'):
             return str(tuple(m.weight.shape))
        count = sum(p.numel() for p in m.parameters())
        if count == 0 and hasattr(m, 'bias') and hasattr(m.bias, 'shape'):
            return str(tuple(m.bias.shape))
        return format_count(count) if count > 0 else ""

    total_flops = flops_analysis.total()
    model_params = sum(p.numel() for p in model.parameters())
    print(f"{'TrilinearLUT':<60} | {format_count(model_params):>20} | {format_count(total_flops):>15}")
    print("-" * 100)

    for name, module in model.named_modules():
        if name == "": continue
        
        depth = name.count('.') + 1
        if depth > max_depth: continue
        
        indent = "  " * (depth - 1)
        display_name = indent + name
        
        f_count = flop_counts.get(name, 0)
        p_str = get_params_or_shape(module)
        
        if f_count > 0 or p_str != "":
            f_str = format_count(f_count) if f_count > 0 else ""
            print(f"{display_name:<60} | {p_str:>20} | {f_str:>15}")

    print("=" * 100 + "\n")

def draw_chart(prof):
    logging.info("generating charts")
    
    events = prof.key_averages()
    time_keys = ["LUT_Preprocess", "LUT_ForwardPass", "LUT_BackwardPass", "LUT_Training", "LUT_Postprocess"]
    metrics = {k: {'cpu_time': 0, 'gpu_time': 0} for k in time_keys}

    for event in events:
        if event.key in metrics:
            metrics[event.key]['cpu_time'] = event.cpu_time_total / 1000.0
            metrics[event.key]['gpu_time'] = event.device_time_total / 1000.0

    train_cpu = metrics["LUT_Training"]['cpu_time']
    train_gpu = metrics["LUT_Training"]['gpu_time']
    
    fwd_cpu = metrics["LUT_ForwardPass"]['cpu_time']
    bwd_cpu = metrics["LUT_BackwardPass"]['cpu_time']
    
    fwd_gpu = metrics["LUT_ForwardPass"]['gpu_time']
    bwd_gpu = metrics["LUT_BackwardPass"]['gpu_time']

    overhead_cpu = max(0, train_cpu - (fwd_cpu + bwd_cpu))
    overhead_gpu = max(0, train_gpu - (fwd_gpu + bwd_gpu))

    labels = ['Preprocess', 'Forward Pass', 'Backward Pass', 'Training Overhead', 'Postprocess']
    
    cpu_times = [
        metrics["LUT_Preprocess"]['cpu_time'],
        fwd_cpu,
        bwd_cpu,
        overhead_cpu,
        metrics["LUT_Postprocess"]['cpu_time']
    ]

    gpu_times = [
        metrics["LUT_Preprocess"]['gpu_time'],
        fwd_gpu,
        bwd_gpu,
        overhead_gpu,
        metrics["LUT_Postprocess"]['gpu_time']
    ]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    c_cpu = '#5D6D7E'  
    c_gpu = '#27AE60'  

    y_pos = np.arange(len(labels))
    ax1.barh(y_pos, cpu_times, color=c_cpu, height=0.65, alpha=0.95)
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(labels)
    ax1.invert_yaxis()  
    ax1.set_xlabel("Execution Time (ms)")
    ax1.set_title("CPU Latency Profile", fontweight='bold', pad=12)

    for i, v in enumerate(cpu_times):
        ax1.text(v + (max(cpu_times) * 0.02), i, f"{v:.1f} ms", va='center', fontsize=10, color='black')

    ax2.barh(y_pos, gpu_times, color=c_gpu, height=0.65, alpha=0.95)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(labels)
    ax2.invert_yaxis()
    ax2.set_xlabel("Execution Time (ms)")
    ax2.set_title("GPU Latency Profile", fontweight='bold', pad=12)

    for i, v in enumerate(gpu_times):
        if v > 0.1:
            ax2.text(v + (max(gpu_times) * 0.02), i, f"{v:.1f} ms", va='center', fontsize=10, color='black')
        else:
            ax2.text(max(gpu_times) * 0.02, i, "Idle (0 ms)", va='center', fontsize=9, color='#7f8c8d', style='italic')

    plt.suptitle("Computational Latency", fontsize=20, fontweight='bold', y=0.98)
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=c_cpu, label='CPU Execution'),
        Patch(facecolor=c_gpu, label='GPU Execution')
    ]
    fig.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 0.04), ncol=2, frameon=False, fontsize=12)

    plt.tight_layout(rect=[0, 0.05, 1, 0.95]) 
    plt.savefig('profile_latency_detailed.png')
    logging.info("Saved profile_latency_detailed.png")

if __name__ == "__main__":
    config.setup_logging()
    set_publication_style()
    
    matcher = VSearchEngine()
    generator = NeuralLUTGenerator()
    
    input_img = "adobe-colour-correction/v1/inputs/1.jpg"
    emotion = "Anxiety"

    vram_forward_pass = 0.0
    vram_backward_pass = 0.0
    
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    total_latency_sec = 0.0

    logging.info("Starting Profiler")
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        schedule=torch.profiler.schedule(wait=1, warmup=1, active=1, repeat=1),
        record_shapes=True,
        with_flops=True,
        on_trace_ready=draw_chart 
    ) as prof:
        for i in range(4):
            if i == 2:
                start_event.record()
                vram_forward_pass, vram_backward_pass = colour_correct(
                    input_path=input_img,
                    target_emotion=emotion,
                    matcher=matcher,
                    generator=generator,
                    profile=True
                )
                end_event.record()
                torch.cuda.synchronize()
                total_latency_sec = start_event.elapsed_time(end_event) / 1000.0
            else:
                colour_correct(
                    input_path=input_img,
                    target_emotion=emotion,
                    matcher=matcher,
                    generator=generator,
                    profile=True 
                )
            prof.step()

    flops_analysis, lut_model_instance = compute_flops_lut_model(
        lut_dim=config.LUT_DIM,
        device=config.DEVICE
    )
    
    print_stat_card(
        model=lut_model_instance,
        latency_sec=total_latency_sec,
        vram_fwd_gb=vram_forward_pass / 1024,
        vram_bwd_gb=vram_backward_pass / 1024,
        flops_count=flops_analysis.total()
    )

    print_detailed_report(lut_model_instance, flops_analysis, max_depth=3)