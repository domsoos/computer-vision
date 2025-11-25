# regularizers.py
import torch

def total_variation_phase(phase_list):
    """Sum TV over all phase masks (operating in rFFT half-plane)."""
    tv = 0.0
    for phi in phase_list:  # (C,H,W_r)
        dx = phi[..., 1:, :] - phi[..., :-1, :]
        dy = phi[..., :, 1:] - phi[..., :, :-1]
        tv = tv + dx.abs().mean() + dy.abs().mean()
    return tv

def amplitude_l2(amp_logits_list):
    """L2 on amplitudes a=σ(α) or α itself; we use a for simplicity."""
    if amp_logits_list is None: return torch.tensor(0.0, device='cpu')
    reg = 0.0
    for a_logit in amp_logits_list:
        a = torch.sigmoid(a_logit)
        reg = reg + (a*a).mean()
    return reg

