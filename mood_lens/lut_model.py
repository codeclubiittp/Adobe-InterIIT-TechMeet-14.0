import torch
import torch.nn as nn
import numpy as np
import mood_lens.config as config

class TrilinearLUT(nn.Module):
    def __init__(self, dim=33):
        config.setup_logging()
        super(TrilinearLUT, self).__init__()
        
        lut = np.linspace(0, 1, dim).astype(np.float32)
        r, g, b = np.meshgrid(lut, lut, lut, indexing='ij')
        initial_lut = np.stack([r, g, b], axis=-1) 
        self.lut = nn.Parameter(torch.from_numpy(initial_lut).permute(3, 0, 1, 2).unsqueeze(0)) 

    def forward(
        self, 
        x: torch.Tensor
    ):
        grid = x.permute(0, 2, 3, 1) * 2 - 1 
        grid = grid.unsqueeze(1)
        out = torch.nn.functional.grid_sample(
            self.lut, grid, 
            mode='bilinear', 
            padding_mode='border', 
            align_corners=True
        )
        return out.squeeze(2)