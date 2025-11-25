# src/visualize_features.py
import torch
import matplotlib.pyplot as plt
import numpy as np
from torchvision import datasets, transforms
from hybrid import HybridFD2NN_CNN

def visualize_hybrid_features():
    # Load Hybrid Model
    model = HybridFD2NN_CNN(img_size=32, classes=10, fd_channels=32, cnn_channels=(16,32))
    try:
        model.load_state_dict(torch.load("checkpoints/Hybrid.pth", weights_only=False))
    except:
        print("Hybrid checkpoint not found.")
        return

    # Get a sample image
    tfms = transforms.Compose([transforms.Pad(2), transforms.ToTensor(), transforms.Normalize((0.2860,), (0.3530,))])
    ds = datasets.FashionMNIST('./data', train=False, transform=tfms)
    img, label = ds[0] # Ankle Boot usually
    
    # Pass through Optical Front-end
    model.eval()
    with torch.no_grad():
        # model.front returns (1, 32, 32, 32) feature map
        feats = model.front(img.unsqueeze(0), return_features=True)
    
    # Plot Input
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 5, 1)
    plt.imshow(img.squeeze(), cmap='gray')
    plt.title("Input")
    plt.axis('off')

    # Plot first 4 Optical Features
    for i in range(4):
        plt.subplot(1, 5, i+2)
        f = feats[0, i, :, :].numpy()
        plt.imshow(f, cmap='magma')
        plt.title(f"Opt Feat {i}")
        plt.axis('off')
        
    plt.suptitle("Hybrid Model: Optical Feature Extraction")
    plt.tight_layout()
    plt.savefig("hybrid_features.png")
    print("Saved hybrid_features.png")

if __name__ == "__main__":
    visualize_hybrid_features()
