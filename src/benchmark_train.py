# benchmark_train.py
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from training import train_model
from fd2nn import FD2NN
from d2nn import D2NN
from hybrid import HybridFD2NN_CNN
from cnn import CNN  # if you renamed it to CNN, import that instead

def get_loaders(H=32, W=32, dataset='mnist', bs=128, aug=True):
    MEAN_STD = {'mnist': ((0.1307,), (0.3081,)), 'fashion': ((0.2860,), (0.3530,))}
    tfms = []
    if (H, W) == (32, 32):  # pad 28->32 (no resize artifacts)
        tfms += [transforms.Pad(2)]
    if aug:
        tfms += [
            transforms.RandomCrop((H, W), padding=2),
            transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        ]
    else:
        tfms += [transforms.CenterCrop((H, W))]
    tfms += [transforms.ToTensor(), transforms.Normalize(*MEAN_STD[dataset])]
    tfms = transforms.Compose(tfms)

    if dataset == 'mnist':
        trainset = datasets.MNIST('./data', train=True, download=True, transform=tfms)
        testset  = datasets.MNIST('./data', train=False, download=True, transform=tfms)
        C = 10
    elif dataset == 'fashion':
        trainset = datasets.FashionMNIST('./data', train=True, download=True, transform=tfms)
        testset  = datasets.FashionMNIST('./data', train=False, download=True, transform=tfms)
        C = 10
    else:
        raise ValueError("dataset must be 'mnist' or 'fashion'")

    train_loader = DataLoader(trainset, batch_size=bs, shuffle=True,  num_workers=2, pin_memory=True)
    val_loader   = DataLoader(testset,  batch_size=bs, shuffle=False, num_workers=2, pin_memory=True)
    return train_loader, val_loader, C

def count_params(m): return sum(p.numel() for p in m.parameters() if p.requires_grad)

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    H = W = 32
    train_loader, val_loader, C = get_loaders(H, W, dataset='mnist', bs=128, aug=True)

    # instantiate models 
    tiny_cnn  = CNN(in_ch=1, classes=C, channels=(8,16), linear=False)   # nonlinear CNN baseline (<=10k)
    lin_cnn   = CNN(in_ch=1, classes=C, channels=(8,16), linear=True)    # linearized CNN baseline
    fd2nn     = FD2NN(in_ch=1, img_size=H, n_layers=6, hidden_channels=96, classes=C)
    d2nn      = D2NN(in_ch=1, img_size=H, n_layers=3, hidden_channels=64, classes=C, prop_k=1.0)
    hybrid_tr = HybridFD2NN_CNN(img_size=H, classes=C, fd_layers=2, fd_channels=64,phi_kind=None, freeze_frontend=False,cnn_channels=(16,32), linear_head=False)

    # training configs per model 
    runs = {
        "CNN":         (tiny_cnn, dict(epochs=12, base_lr=3e-3, weight_decay=1e-4, use_amp=True,
                                       curriculum=False, tv_max=0.0, amp_l2_max=0.0)),
        "CNN_linear":  (lin_cnn,  dict(epochs=12, base_lr=3e-3, weight_decay=1e-4, use_amp=True,
                                       curriculum=False, tv_max=0.0, amp_l2_max=0.0)),
        "FD2NN":       (fd2nn,    dict(epochs=30, base_lr=3e-3, weight_decay=5e-4, use_amp=False,
                                       curriculum=True,  tv_max=1e-4, amp_l2_max=1e-4)),
        "D2NN":        (d2nn,     dict(epochs=30, base_lr=3e-3, weight_decay=5e-4, use_amp=False,
                                       curriculum=True,  tv_max=1e-4, amp_l2_max=0.0)),
        # For Hybrid, keep TV off (regularizer code looks for top-level .phase):
        "Hybrid_train":(hybrid_tr,dict(epochs=30, base_lr=3e-3, weight_decay=5e-4, use_amp=False,
                                       curriculum=True,  tv_max=0.0,  amp_l2_max=0.0)),
    }

    results = {}
    for name, (model, args) in runs.items():
        print(f"{name} params: {count_params(model)}")
        results[name] = train_model(name, model, train_loader, val_loader, device, **args)

    print("\n==== Summary ====")
    for k, v in results.items():
        print(f"{k:>14} | acc {v['best_val_acc']:.3f} | infer {v['inference_imgs_per_s']:.0f} img/s")

if __name__ == "__main__":
    print("Starting benchmark…")
    main()

