import torch
import torch.nn as nn
from fd2nn import FD2NN
from cnn import CNN

class HybridFD2NN_CNN(nn.Module):
    """
    Fourier-plane front-end feeding a CNN head. Optionally freeze the front-end masks.
    """
    def __init__(self, img_size=32, classes=10,
                 fd_layers=2, fd_channels=32,
                 freeze_frontend=False,init_noise_scale=0.0,
                 cnn_channels=(16,32), linear_head=False):
        super().__init__()
        # Front end: FD2NN 
        # We don't care about 'classes' here, we want features.
        self.front = FD2NN(in_ch=1, img_size=img_size, n_layers=fd_layers,
                           hidden_channels=fd_channels, classes=fd_channels) 
        if freeze_frontend:
            init_noise_scale = 2.0
        if init_noise_scale > 0:
            with torch.no_grad():
                for p in self.front.phase_masks:
                    p.data.uniform_(-init_noise_scale, init_noise_scale)
        
        # Back end: CNN
        # Input channels to CNN = hidden_channels of FD2NN
        
        self.back = CNN(in_ch=fd_channels, classes=classes,
                        channels=cnn_channels, linear=linear_head)

        if freeze_frontend:
            for p in self.front.parameters():
                p.requires_grad = False

    def forward(self, x):
        # Request features directly (B, fd_channels, H, W)
        feat = self.front(x, return_features=True) 
        return self.back(feat)

