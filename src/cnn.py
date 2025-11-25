import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    """
    Small baseline CNN: around 10k param, used for:
      - nonlinear baseline  -> linear=False
      - linearized baseline -> linear=True (no activations)
    """
    def __init__(self, in_ch=1, classes=10, channels=(12, 24), linear=False):
        super().__init__()
        c1, c2 = channels
        self.linear = linear
        self.conv1 = nn.Conv2d(in_ch, c1, kernel_size=3, padding=1, bias=True)
        self.conv2 = nn.Conv2d(c1,   c1, kernel_size=3, padding=1, bias=True)
        self.pool2 = nn.MaxPool2d(2)
        self.conv3 = nn.Conv2d(c1,   c2, kernel_size=3, padding=1, bias=True)
        self.conv4 = nn.Conv2d(c2,   c2, kernel_size=3, padding=1, bias=True)
        self.pool4 = nn.MaxPool2d(2)
        self.gap   = nn.AdaptiveAvgPool2d(1)
        self.fc    = nn.Linear(c2, classes)

    def forward(self, x):
        x = self.conv1(x);  x = x if self.linear else F.relu(x, inplace=True)
        x = self.conv2(x);  x = x if self.linear else F.relu(x, inplace=True)
        x = self.pool2(x)
        x = self.conv3(x);  x = x if self.linear else F.relu(x, inplace=True)
        x = self.conv4(x);  x = x if self.linear else F.relu(x, inplace=True)
        x = self.pool4(x)
        x = self.gap(x).flatten(1)
        return self.fc(x)

def count_params(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)

