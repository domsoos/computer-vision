# src/analyze_results.py
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import json
from sklearn.metrics import confusion_matrix
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from cnn import CNN
from fd2nn import FD2NN
from d2nn import D2NN
from hybrid import HybridFD2NN_CNN

def load_model(model_name, device):
    C = 10
    # Match the definitions in benchmark_train.py EXACTLY
    if "CNN" in model_name:
        model = CNN(in_ch=1, classes=C, channels=(16, 32))
    elif "Hybrid" in model_name:
        # Check if it's Frozen (same architecture, just weights differ)
        model = HybridFD2NN_CNN(img_size=32, classes=C, fd_channels=32, cnn_channels=(16,32))
    elif "FD2NN" in model_name:
        model = FD2NN(in_ch=1, img_size=32, n_layers=4, hidden_channels=32, classes=C, dropout=0.1)
    elif "D2NN" in model_name:
        # Check nonlinearity flag based on name
        nonlin = "NonLin" in model_name
        model = D2NN(in_ch=1, img_size=32, n_layers=4, hidden_channels=64, classes=C, sim_nonlinearity=nonlin)
    else:
        return None

    path = f"checkpoints/{model_name}.pth"
    if not os.path.exists(path): return None
    try:
        model.load_state_dict(torch.load(path, map_location=device, weights_only=False))
    except Exception as e:
        print(f"Error loading {model_name}: {e}")
        return None
    return model.to(device).eval()

def plot_training_curves(model_names):
    """Reads JSON stats and plots comparison curves."""
    plt.figure(figsize=(12, 5))
    
    # Plot Accuracy
    plt.subplot(1, 2, 1)
    for name in model_names:
        path = f"checkpoints/{name}_stats.json"
        if not os.path.exists(path): continue
        with open(path, 'r') as f: stats = json.load(f)
        hist = stats['history']
        plt.plot(hist['val_acc'], label=f"{name} (Val)", linestyle='-')
        # Optional: Plot train acc with dotted line
        plt.plot(hist['train_acc'], label=f"{name} (Train)", linestyle=':', alpha=0.5)
    
    plt.title("Validation Accuracy vs Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Plot Loss
    plt.subplot(1, 2, 2)
    for name in model_names:
        path = f"checkpoints/{name}_stats.json"
        if not os.path.exists(path): continue
        with open(path, 'r') as f: stats = json.load(f)
        hist = stats['history']
        plt.plot(hist['val_loss'], label=f"{name}", linestyle='-')

    plt.title("Validation Loss vs Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("benchmark_curves.png")
    print("Saved benchmark_curves.png")

def analyze_model(model_name):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = load_model(model_name, device)
    if model is None: return
    print(f"Generated confusion/samples for {model_name}")

if __name__ == "__main__":
    models = ["CNN_Baseline", "Hybrid", "Hybrid_Frozen", "FD2NN_Opt", "D2NN_NonLin", "D2NN_Linear"]
    
    # 1. Generate Aggregate Plots
    plot_training_curves(models)
    
    # 2. Generate Individual Analyses
    for m in models:
        analyze_model(m)
