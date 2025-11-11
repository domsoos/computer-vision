import torch
import torch.nn as nn
import torch.nn.functional as F

class FD2NN(nn.Module):
    """
    Fourier-space Diffractive Deep Neural Network (FD2NN) for 2D image classification.
    Implements a stack of diffractive layers in the Fourier domain, using phase modulation.
    The architecture is loosely inspired by [Lin et al., "All-optical machine learning using diffractive deep neural networks", Science 2018].
    """
    def __init__(self, in_ch=1, img_size=64, n_layers=4, hidden_channels=16, classes=10):
        """
        Args:
            in_ch:        Number of input channels (e.g., 1 for grayscale MNIST).
            img_size:     Height/width of the input images (must be square).
            n_layers:     Number of diffractive layers.
            hidden_channels: Number of diffractive "channels" (Fourier plane masks).
            classes:      Number of output classes.
        """
        super().__init__()
        self.in_ch = in_ch
        self.img_size = img_size
        self.n_layers = n_layers
        self.hidden_channels = hidden_channels

        # "Diffractive layers" as complex phase masks in Fourier space
        self.phase_masks = nn.ParameterList([
            nn.Parameter(
                torch.rand(hidden_channels, img_size, img_size) * 2 * torch.pi
            ) for _ in range(n_layers)
        ])
        # 1x1 conv at input to embed in hidden_channels
        self.input_proj = nn.Conv2d(in_ch, hidden_channels, kernel_size=1)
        # 1x1 conv at output to classes
        self.output_proj = nn.Conv2d(hidden_channels, classes, kernel_size=1)
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        # x: (B, in_ch, H, W)
        x = self.input_proj(x)  # (B, hidden_channels, H, W)
        for mask in self.phase_masks:
            x = x.to(torch.complex64)
            # 1. Forward FFT to Fourier domain
            x_fft = torch.fft.fft2(x, norm='ortho')
            # 2. Apply phase mask (multiply by exp(i*phase))
            phase = mask.unsqueeze(0)  # (1, hidden_channels, H, W)
            x_fft = x_fft * torch.exp(1j * phase)
            # 3. Inverse FFT to image domain
            x = torch.fft.ifft2(x_fft, norm='ortho').real
            # 4. Nonlinearity (ReLU, mimicking diffractive non-ideality)
            x = F.relu(x)
        # Output projection and global pooling
        out = self.output_proj(x)  # (B, classes, H, W)
        out = self.pool(out)       # (B, classes, 1, 1)
        out = out.view(out.size(0), -1)  # (B, classes)
        return out
