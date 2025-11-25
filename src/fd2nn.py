import torch
import torch.nn as nn
import torch.nn.functional as F

class FD2NN(nn.Module):
    """
    Fourier-space Diffractive DNN with phase+amplitude masks, pre-norm,
    channel mixing, dropout, and residual layer-scale.
    """
    def __init__(self, in_ch=1, img_size=64, n_layers=4, hidden_channels=16, classes=10,
                 dropout=0.10, layer_scale_init=1e-3):
        super().__init__()
        self.in_ch = in_ch
        self.img_size = img_size
        self.n_layers = n_layers
        self.hidden_channels = hidden_channels

        # Start near identity (phase=0, amp≈1); learnable per-layer masks
        self.phase_masks = nn.ParameterList([
            nn.Parameter(torch.zeros(hidden_channels, img_size, img_size))  # phase ~ 0
            for _ in range(n_layers)
        ])
        # amplitude logits -> amp = 1 + 0.25*tanh(logits)  (bounded ~ [0.75, 1.25])
        self.amp_logits = nn.ParameterList([
            nn.Parameter(torch.zeros(hidden_channels, img_size, img_size))
            for _ in range(n_layers)
        ])

        # Stem / head
        self.input_proj  = nn.Conv2d(in_ch, hidden_channels, kernel_size=1, bias=True)
        self.output_proj = nn.Conv2d(hidden_channels, classes,       kernel_size=1, bias=True)
        self.pool = nn.AdaptiveAvgPool2d(1)

        # Stabilizers inside each block
        self.norms   = nn.ModuleList([
            nn.LayerNorm([hidden_channels, img_size, img_size]) for _ in range(n_layers)
        ])
        self.mixers  = nn.ModuleList([
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=1, bias=True)
            for _ in range(n_layers)
        ])
        self.dropout = nn.Dropout2d(p=dropout)
        # Layer-scale (learned per layer, per channel) keeps residual small at start
        self.layer_scales = nn.ParameterList([
            nn.Parameter(torch.ones(1, hidden_channels, 1, 1) * layer_scale_init)
            for _ in range(n_layers)
        ])
        self.act = nn.SiLU()

    def forward(self, x):
        # x: (B, in_ch, H, W)
        x = self.input_proj(x)  # (B, C, H, W)

        for l in range(self.n_layers):
            # Fourier domain scattering 
            x_c   = x.to(torch.complex64)
            x_fft = torch.fft.fft2(x_c, norm='ortho')

            phase = self.phase_masks[l].unsqueeze(0)               # (1,C,H,W)
            amp   = 1.0 + 0.25 * torch.tanh(self.amp_logits[l])    # (C,H,W)
            amp   = amp.unsqueeze(0)                                # (1,C,H,W)

            mask_c = amp * torch.exp(1j * phase)                    # complex mask
            x_fft  = x_fft * mask_c
            x_real = torch.fft.ifft2(x_fft, norm='ortho').real      # (B,C,H,W)

            # Residual channel mixing (pre-norm)
            y = self.norms[l](x_real)
            y = self.act(y)
            y = self.mixers[l](y)
            y = self.dropout(y)

            x = x_real + self.layer_scales[l] * y                   # residual

        out = self.output_proj(x)                 # (B, classes, H, W)
        out = self.pool(out).view(out.size(0), -1)
        return out

