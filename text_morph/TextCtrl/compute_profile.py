import torch
import torch.profiler
from torch.profiler import profile, ProfilerActivity
import matplotlib.pyplot as plt
import numpy as np
import logging
import os
from PIL import Image
from fvcore.nn import FlopCountAnalysis
from inference import main, create_parser

from utils import create_model, load_state_dict
from main import process_single_image

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def set_publication_style():
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['DejaVu Sans', 'Arial', 'sans-serif'],
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

def compute_flops_gamusa(cfg_path, device):
    model = create_model(cfg_path).to(device)
    model.eval()

    dummy_input = torch.randn(1, 3, 256, 256).to(device)
    
    flops = FlopCountAnalysis(model, dummy_input)
    flops.unsupported_ops_warnings(False)
    return flops, model

def print_stat_card(model, latency_sec, peak_vram_mb, flops_count):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    param_size = sum(p.nelement() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.nelement() * b.element_size() for b in model.buffers())
    size_all_mb = (param_size + buffer_size) / 1024**2
    
    throughput = 1.0 / latency_sec if latency_sec > 0 else 0

    print("\n" + "=" * 60)
    print(f"{'COMPUTE STAT CARD: GaMuSA PIPELINE':^60}")
    print("=" * 60)
    print(" [Model Architecture]")
    print(f"  • Parameters (Total)  : {total_params / 1e6:.2f} M")
    print(f"  • Parameters (Train)  : {trainable_params / 1e6:.2f} M")
    print(f"  • Weights Size (Disk) : {size_all_mb:.2f} MB")
    print(f"  • Complexity (FLOPs)  : {flops_count / 1e9:.2f} GFLOPs")
    print("-" * 60)
    print(" [Runtime Metrics]")
    print(f"  • Peak VRAM Usage     : {peak_vram_mb:.2f} MB")
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
    print(f"{'GaMuSA Model':<60} | {format_count(model_params):>20} | {format_count(total_flops):>15}")
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
    logging.info("Generating performance charts")
    
    events = prof.key_averages()
    
    target_keys = ["Textctrl_Loading", "Textctrl_Inference"]
    
    metrics = {k: {'cpu_time': 0, 'gpu_time': 0} for k in target_keys}
    found_data = False

    for event in events:
        if event.key in metrics:
            found_data = True
            metrics[event.key]['cpu_time'] = event.cpu_time_total / 1000.0 
            metrics[event.key]['gpu_time'] = event.cuda_time_total / 1000.0 

    if not found_data:
        logging.error("No matching keys found. Check record_function names.")
        return

    labels = ['Model Loading', 'Inference (Text Edit)']
    cpu_times = [metrics[k]['cpu_time'] for k in target_keys]
    gpu_times = [metrics[k]['gpu_time'] for k in target_keys]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    c_cpu = '#5D6D7E'  
    c_gpu = '#8E44AD' 

    y_pos = np.arange(len(labels))

    ax1.barh(y_pos, cpu_times, color=c_cpu, height=0.6, alpha=0.95)
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(labels)
    ax1.invert_yaxis()
    ax1.set_xlabel("Execution Time (ms)")
    ax1.set_title("CPU Latency Profile", fontweight='bold', pad=12)
    for i, v in enumerate(cpu_times):
        ax1.text(v + (max(cpu_times or [1]) * 0.02), i, f"{v:.1f} ms", va='center', fontsize=10)

    ax2.barh(y_pos, gpu_times, color=c_gpu, height=0.6, alpha=0.95)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(labels)
    ax2.invert_yaxis()
    ax2.set_xlabel("Execution Time (ms)")
    ax2.set_title("GPU Latency Profile", fontweight='bold', pad=12)
    for i, v in enumerate(gpu_times):
        val_text = f"{v:.1f} ms" if v > 0.1 else "Idle"
        ax2.text(max(v, 0.1) + (max(gpu_times or [1]) * 0.02), i, val_text, va='center', fontsize=10)

    plt.suptitle("GaMuSA Text-Editing Latency", fontsize=20, fontweight='bold', y=0.98)
    
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=c_cpu, label='CPU'), Patch(facecolor=c_gpu, label='GPU')]
    fig.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 0.04), ncol=2, frameon=False, fontsize=12)

    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    plt.savefig('gamusa_latency_profile.png')
    logging.info("Saved gamusa_latency_profile.png")

def create_temp_image(path):
    img = Image.new('RGB', (256, 256), color='blue')
    os.makedirs(os.path.dirname(path), exist_ok=True)
    img.save(path)

if __name__ == "__main__":
    set_publication_style()
    
    #img_path = "example/i_s/temp_profile.png"
    #create_temp_image(img_path)
    
    # params = {
    #     "image_path": img_path,
    #     "style_text": "FOUR",
    #     "target_text": "FIVE",
    #     "ckpt_path": "weights/model.pth",
    #     "profile": True 
    # }

    logging.info("Starting Profiler Loop")
    
    final_peak_vram = 0.0
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    total_latency_sec = 0.0

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        #schedule=torch.profiler.schedule(wait=1, warmup=1, active=1, repeat=1),
        record_shapes=True,
        with_flops=True,
        on_trace_ready=draw_chart
    ) as prof:
        torch.cuda.reset_peak_memory_stats()
        start_mem = torch.cuda.memory_allocated()
        start_event.record()

       # process_single_image(**params)

        parser = create_parser()
        opt = parser.parse_args()
        main(opt)
        
        end_event.record()
        torch.cuda.synchronize()
        
        max_mem = torch.cuda.max_memory_allocated()
        final_peak_vram = (max_mem - start_mem) / 1024**2
        total_latency_sec = start_event.elapsed_time(end_event) / 1000.0

        prof.step()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    flops_analysis, model_instance = compute_flops_gamusa('configs/inference.yaml', device)
    
    print_stat_card(
        model=model_instance,
        latency_sec=total_latency_sec,
        peak_vram_mb=final_peak_vram,
        flops_count=flops_analysis.total()
    )

    print_detailed_report(model_instance, flops_analysis, max_depth=3)
    
    if os.path.exists(img_path):
        os.remove(img_path)