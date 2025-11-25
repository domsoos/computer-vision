# curriculum.py
import torch

def _radial_mask(H, W, cutoff_frac, device):
    yy, xx = torch.meshgrid(
        torch.linspace(-1, 1, H, device=device),
        torch.linspace(-1, 1, W, device=device),
        indexing='ij'
    )
    r = torch.sqrt(xx*xx + yy*yy)
    k = cutoff_frac
    band = 0.08  # smooth rolloff
    t = torch.clamp((k + band - r) / band, 0, 1)
    return 0.5 - 0.5*torch.cos(torch.pi * t)  # Hann-ish taper

@torch.no_grad()
def lowpass_inputs(x, cutoff_frac: float):
    """Low-pass filter inputs in spatial domain via rFFT masking (per proposal §Bandwidth curriculum)."""
    B, C, H, W = x.shape
    W_r = W // 2 + 1
    M_spatial = _radial_mask(H, W, cutoff_frac, x.device)           # (H,W)
    # Implement LP in Fourier domain using rFFT on each channel, broadcast mask
    X = torch.fft.rfft2(x, norm='ortho')                             # (B,C,H,W_r)
    # Construct frequency response from spatial mask (same for all channels)
    # Note: using rFFT of mask’s impulse response ≈ convolutional LP; fast and stable here
    M = torch.fft.rfft2(M_spatial, norm='ortho')                     # (H,W_r), complex
    M = M.abs().unsqueeze(0).unsqueeze(0)                            # (1,1,H,W_r)
    Y = X * M
    y = torch.fft.irfft2(Y, s=(H,W), norm='ortho')
    return y

