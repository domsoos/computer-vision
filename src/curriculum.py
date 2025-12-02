# curriculum.py
import torch

def _freq_radial_mask(H, W, cutoff_frac, device):
    """
    Proper radial low-pass mask in the frequency domain.

    cutoff_frac in [0, 1]:
        0.0 -> keep almost nothing (extreme blur)
        1.0 -> keep everything (identity filter)
    """
    # Raw frequency coords (in cycles per pixel)
    fy = torch.fft.fftfreq(H, d=1.0).to(device)      # length H, symmetric around 0
    fx = torch.fft.rfftfreq(W, d=1.0).to(device)     # length W_r, non-negative

    yy, xx = torch.meshgrid(fy, fx, indexing="ij")
    r = torch.sqrt(xx * xx + yy * yy)               # radius in frequency plane

    # 🔧 KEY FIX: normalize by **max radius**, not per-axis max
    r = r / (r.max() + 1e-8)                        # now r ∈ [0, 1]

    # Clamp cutoff_frac and build smooth Hann-ish rolloff
    k = float(torch.clamp(torch.tensor(cutoff_frac, device=device), 0.0, 1.0))
    band = 0.08                                     # smooth transition width (in normalized radius)
    t = torch.clamp((k + band - r) / band, 0.0, 1.0)

    # Hann( t ) in [0,1]
    mask = 0.5 - 0.5 * torch.cos(torch.pi * t)
    return mask                                     # (H, W_r)


@torch.no_grad()
def lowpass_inputs(x, cutoff_frac: float):
    """
    cutoff_frac in [0, 1]:

        0.0 -> extreme low-pass (super blurry)
        0.3 -> strong blur, recognisable classes
        1.0 -> ~identity (full bandwidth)

    Implemented as X * M(fx, fy) in frequency domain.
    """
    B, C, H, W = x.shape
    device = x.device

    X = torch.fft.rfft2(x, norm="ortho")                # (B, C, H, W_r)
    M = _freq_radial_mask(H, W, cutoff_frac, device)    # (H, W_r)
    M = M.unsqueeze(0).unsqueeze(0)                     # (1, 1, H, W_r)

    Y = X * M
    y = torch.fft.irfft2(Y, s=(H, W), norm="ortho")
    return y

def bandwidth_schedule(epoch, epochs, min_cutoff=0.35, max_cutoff=1.0, gamma=1.5):
    """
    Smooth schedule from strong blur to full bandwidth.

    epoch: 0..epochs-1
    min_cutoff: minimum bandwidth at start (0.3–0.4 is safe)
    max_cutoff: final bandwidth (1.0 = full)
    gamma: >1 makes it stay blurrier longer, <1 speeds up sharpening
    """
    if epochs <= 1:
        return max_cutoff
    progress = epoch / (epochs - 1)
    return min_cutoff + (max_cutoff - min_cutoff) * (progress ** gamma)
