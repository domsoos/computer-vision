import torch
import torch.nn as nn

def fresnel_transfer_fft(H, W, prop_k, device):
    fy = torch.fft.fftfreq(H, d=1.0).to(device).view(H, 1)
    fx = torch.fft.fftfreq(W, d=1.0).to(device).view(1, W)
    phase = -torch.pi * prop_k * (fx*fx + fy*fy)
    return torch.exp(1j * phase)

class D2NN(nn.Module):
    def __init__(self, in_ch=1, img_size=32, n_layers=4, hidden_channels=64, classes=10,
                 prop_k=0.15, sim_nonlinearity=True):
        super().__init__()
        self.H, self.W = img_size, img_size
        self.C = hidden_channels
        self.n = n_layers
        self.sim_nonlinearity = sim_nonlinearity

        self.phase = nn.ParameterList([
            nn.Parameter(torch.randn(self.C, self.H, self.W) * 0.02) # Smaller init
            for _ in range(self.n)
        ])
        
        # Add normalization if nonlinear to keep signal in check
        self.norms = nn.ModuleList([
            nn.GroupNorm(8, self.C) if sim_nonlinearity else nn.Identity()
            for _ in range(n_layers)
        ])

        self.stem = nn.Conv2d(in_ch, self.C, kernel_size=1)
        self.head = nn.Conv2d(self.C, classes, kernel_size=1)
        self.pool = nn.AdaptiveAvgPool2d(1)

        Hc = fresnel_transfer_fft(self.H, self.W, prop_k, device='cpu')
        self.register_buffer('Hc', Hc)

    def forward(self, x):
        x = self.stem(x)
        Hc = self.Hc.to(x.device)

        for k in range(self.n):
            x_complex = x.to(torch.complex64) * torch.exp(1j * self.phase[k].unsqueeze(0))
            
            X = torch.fft.fft2(x_complex, norm='ortho')
            X = X * Hc.view(1, 1, self.H, self.W)
            x_complex = torch.fft.ifft2(X, norm='ortho')

            if self.sim_nonlinearity:
                x = x_complex.abs() 
                x = self.norms[k](x) # Re-center stats
            else:
                x = x_complex 

        x = x.abs() if torch.is_complex(x) else x
        return self.pool(self.head(x)).flatten(1)
