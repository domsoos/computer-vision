# src/benchmark_train.py
import torch
import os
import json
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from training import train_model

from cnn import CNN
from fd2nn import FD2NN
from d2nn import D2NN
from hybrid import HybridFD2NN_CNN

def get_loaders(dataset='fashion', bs=128):
    MEAN, STD = (0.2860,), (0.3530,)
    root = './data'
    train_tfms = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomAffine(degrees=10, translate=(0.1, 0.1)),
        transforms.Pad(2), transforms.ToTensor(), transforms.Normalize(MEAN, STD)
    ])
    test_tfms = transforms.Compose([
        transforms.Pad(2), transforms.ToTensor(), transforms.Normalize(MEAN, STD)
    ])
    trainset = datasets.FashionMNIST(root, train=True, download=True, transform=train_tfms)
    testset = datasets.FashionMNIST(root, train=False, download=True, transform=test_tfms)
    return (DataLoader(trainset, batch_size=bs, shuffle=True, num_workers=2),
            DataLoader(testset, batch_size=bs, shuffle=False, num_workers=2),
            10)

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Benchmarking on {device}...")
    train_loader, val_loader, C = get_loaders()

    cnn     = CNN(in_ch=1, classes=C, channels=(16, 32)) # Reverted to match checkpoints
    fd2nn   = FD2NN(in_ch=1, img_size=32, n_layers=4, hidden_channels=32, classes=C, dropout=0.1)
    
    # D2NN Variations
    d2nn_nl = D2NN(in_ch=1, img_size=32, n_layers=4, hidden_channels=64, classes=C, sim_nonlinearity=True)
    d2nn_lin= D2NN(in_ch=1, img_size=32, n_layers=4, hidden_channels=64, classes=C, sim_nonlinearity=False)

    # Hybrid Variations
    hybrid  = HybridFD2NN_CNN(img_size=32, classes=C, fd_channels=32, cnn_channels=(16,32))
    hybrid_fz = HybridFD2NN_CNN(img_size=32, classes=C, fd_channels=32, cnn_channels=(16,32), freeze_frontend=True)

    runs = {
        "CNN_Baseline": (cnn, dict(epochs=25, base_lr=1e-3, weight_decay=1e-4, curriculum=False)),
        "FD2NN_Opt":    (fd2nn, dict(epochs=25, base_lr=1e-3, weight_decay=1e-3, curriculum=False, tv_max=1e-4)),
        
        # Experiment D: Nonlinear vs Linear D2NN
        "D2NN_NonLin":  (d2nn_nl, dict(epochs=25, base_lr=1e-3, weight_decay=1e-4, curriculum=False)),
        "D2NN_Linear":  (d2nn_lin, dict(epochs=25, base_lr=1e-3, weight_decay=1e-4, curriculum=False)),

        # Experiment A: Learned vs Frozen Hybrid
        "Hybrid":       (hybrid, dict(epochs=25, base_lr=1e-3, weight_decay=1e-4, curriculum=False)),
        "Hybrid_Frozen":(hybrid_fz, dict(epochs=25, base_lr=1e-3, weight_decay=1e-4, curriculum=False)),
    }

    results = {}
    os.makedirs("checkpoints", exist_ok=True)
    
    for name, (model, args) in runs.items():
        print(f"\nTraining {name}")
        res = train_model(name, model, train_loader, val_loader, device, **args)
        results[name] = res
        torch.save(model.state_dict(), f"checkpoints/{name}.pth")
        with open(f"checkpoints/{name}_stats.json", "w") as f:
            json.dump(res, f)

    print("\n==== Final Results ====")
    for k, v in results.items():
        print(f"{k:>14} | Acc: {v['best_val_acc']:.3f} | Speed: {v['inference_imgs_per_s']:.0f} img/s")

if __name__ == "__main__":
    main()
