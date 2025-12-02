import torch
import torch.nn as nn
import torch.nn.functional as F

class Frankenstein(nn.Module):
    """
    Take F2DNN and frankenstein it together with D2NN by replacing the mask layers with convolutional layers.
    So, we have unlearned Fourier layer, then learned convolutional layer, then unlearned Fourier layer, and so on
    up until the end.
    """
    def __init__(self, in_ch=1, img_size=32, n_layers=4, hidden_channels=16, classes=10,
                 dropout=0.10, layer_scale_init=1e-3):
        super().__init__()
        self.in_ch = in_ch
        self.img_size = img_size
        self.n_layers = n_layers
        self.hidden_channels = hidden_channels

        # True complex-valued 3x3 convolutions
        self.complex_weights = nn.ParameterList([
            nn.Parameter(torch.randn(hidden_channels, hidden_channels, 3, 3, dtype=torch.complex64) * 0.02)
            for _ in range(n_layers)
        ])

        # stem / head (remain real-valued)
        self.input_proj = nn.Conv2d(in_ch, hidden_channels, kernel_size=1)
        self.output_proj = nn.Conv2d(hidden_channels, classes, kernel_size=1)
        self.pool = nn.AdaptiveAvgPool2d(1)

        self.norms = nn.ModuleList([
            nn.LayerNorm([hidden_channels, img_size, img_size]) for _ in range(n_layers)
        ])
        self.mixers = nn.ModuleList([
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=1)
            for _ in range(n_layers)
        ])
        self.dropout = nn.Dropout2d(p=dropout)
        self.layer_scales = nn.ParameterList([
            nn.Parameter(torch.ones(1, hidden_channels, 1, 1) * layer_scale_init)
            for _ in range(n_layers)
        ])
        self.act = nn.SiLU()


    def forward(self, x, return_features=False):
        # x: (B, in_ch, H, W)
        x = self.input_proj(x)  # (B, C, H, W)

        for l in range(self.n_layers):
            # Fourier domain scattering
            x_c = x.to(torch.complex64)
            x_fft = torch.fft.fft2(x_c, norm='ortho')

            # instead of doing a simple complex scalar multiplication mask, we do a 3x3 complex convolutional mask
            # this is where the "frankensteining" comes into play
            x_fft = F.conv2d(x_fft, self.complex_weights[l], padding=1)

            # return to real space
            x_real = torch.fft.ifft2(x_fft, norm='ortho').real  # (B,C,H,W)

            y = self.norms[l](x_real)
            y = self.act(y)
            y = self.mixers[l](y)
            y = self.dropout(y)
            x = x_real + self.layer_scales[l] * y  # residual

        if return_features:
            return x  # (B, C, H, W)

        out = self.output_proj(x)
        out = self.pool(out)
        return out.flatten(1)



