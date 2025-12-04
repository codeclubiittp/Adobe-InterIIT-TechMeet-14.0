import torch
import torch.nn as nn
import torch.nn.functional as F

class Embedder(nn.Module):
    """
    SynthID-like embedder:
    - transforms image to frequency domain
    - injects learned low-magnitude watermark
    - uses adaptive masking based on local texture
    """
    def __init__(self, key_dim=128, alpha=0.0005):
        super().__init__()
        self.alpha = alpha

        self.key_fc = nn.Sequential(
            nn.Linear(key_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64*64)  
        )

        # texture extractor to build perceptual mask
        self.texture = nn.Conv2d(3, 1, kernel_size=3, stride=1, padding=1, bias=False)
        self.texture.weight.data.fill_(1/9)

    def forward(self, img, key_vec):
        B, C, H, W = img.shape

        # FFT to frequency domain
        freq = torch.fft.fft2(img)
        freq_shifted = torch.fft.fftshift(freq)

        # Generate per-key watermark pattern
        wm = self.key_fc(key_vec).view(B, 1, 64, 64)
        wm = F.interpolate(wm, size=(H, W), mode='bilinear', align_corners=False)

        # Compute perceptual mask
        texture_map = torch.abs(self.texture(img))
        texture_map = texture_map / (texture_map.max() + 1e-6)

        # Mid frequencies (SynthID-like)
        row_mid = slice(H//4, 3*H//4)
        col_mid = slice(W//4, 3*W//4)

        # Inject watermark only into midband region
        freq_mod = freq_shifted.clone()

        freq_mod[:, :, row_mid, col_mid] += (
            self.alpha * 
            wm[:, :, row_mid, col_mid] * 
            texture_map[:, :, row_mid, col_mid]
        )

        # Reconstruct image
        freq_unshifted = torch.fft.ifftshift(freq_mod)
        img_out = torch.fft.ifft2(freq_unshifted).real

        return torch.clamp(img_out, 0, 1)


class Detector(nn.Module):
    """
    Given possibly edited image (B,3,H,W)
    Returns P(watermarked).
    """
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 3, 1, 1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, 1, 1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, 1, 1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )
        self.fc = nn.Linear(128, 1)

    def forward(self, x):
        f = self.net(x).view(x.size(0), -1)
        return torch.sigmoid(self.fc(f))
