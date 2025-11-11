# benchmark_train.py
import time
import torch, torch.nn as nn, torch.optim as optim
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from cnn import CNN
from fd2nn import FD2NN

def get_loaders(H=64, W=64, dataset='fashion', bs=256, aug=True):
    tfms = [transforms.Resize((H,W))]
    if aug:
        tfms += [transforms.RandomAffine(degrees=5, translate=(0.05,0.05), scale=(0.95,1.05))]
    tfms += [transforms.ToTensor()]
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
        raise ValueError
    train_loader = DataLoader(trainset, batch_size=bs, shuffle=True, num_workers=1, pin_memory=True)
    test_loader  = DataLoader(testset,  batch_size=bs, shuffle=False, num_workers=1, pin_memory=True)
    return train_loader, test_loader, C

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    crit = nn.CrossEntropyLoss()
    total, correct, loss_sum = 0, 0, 0.0
    for x,y in loader:
        x,y = x.to(device), y.to(device)
        logits = model(x if hasattr(model, 'net') else x)  # both paths fine
        loss = crit(logits, y)
        loss_sum += loss.item() * x.size(0)
        correct += (logits.argmax(1) == y).sum().item()
        total += x.size(0)
    return loss_sum/total, correct/total

def train_model(name, model, train_loader, val_loader, device, epochs=10, lr=3e-3):
    model.to(device)
    opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scaler = GradScaler('cuda', enabled=(device=='cuda'))
    crit = nn.CrossEntropyLoss()

    epoch_times = []
    best = 0.0
    for ep in range(epochs):
        model.train()
        start = time.perf_counter()
        total, correct, loss_sum = 0, 0, 0.0
        for x,y in train_loader:
            x,y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with autocast(device_type='cuda', dtype=torch.float16, enabled=(device=='cuda')):
                logits = model(x)
                loss = crit(logits, y)
            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update()

            loss_sum += loss.item() * x.size(0)
            correct += (logits.argmax(1) == y).sum().item()
            total += x.size(0)
        epoch_time = time.perf_counter() - start
        epoch_times.append(epoch_time)
        val_loss, val_acc = evaluate(model, val_loader, device)
        print(f"[{name}] epoch {ep:02d} | train_acc {correct/total:.3f} | val_acc {val_acc:.3f} | time {epoch_time:.1f}s")
        best = max(best, val_acc)
    # inference throughput
    model.eval()
    n_imgs, inf_start = 0, time.perf_counter()
    with torch.no_grad():
        for x,_ in val_loader:
            x = x.to(device)
            _ = model(x)
            n_imgs += x.size(0)
    inf_time = time.perf_counter() - inf_start
    imgs_per_s = n_imgs / inf_time
    return {
        "best_val_acc": best,
        "mean_epoch_time_s": sum(epoch_times)/len(epoch_times),
        "inference_imgs_per_s": imgs_per_s
    }

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    H = W = 64
    train_loader, val_loader, C = get_loaders(H, W, dataset='mnist', bs=256, aug=True)

    # CNN baseline
    cnn = CNN(in_ch=1, classes=C)
    fd2nn = FD2NN(in_ch=1, img_size=64, n_layers=4, hidden_channels=16, classes=10)

    results = {}
    for name, model, ep in [("CNN",cnn,12),("FD2NN",fd2nn,12)]:
        results[name] = train_model(name, model, train_loader, val_loader, device, epochs=ep, lr=3e-3)

    print("\n==== Summary ====")
    for k,v in results.items():
        print(f"{k:>14} | acc {v['best_val_acc']:.3f} | epoch {v['mean_epoch_time_s']:.1f}s | infer {v['inference_imgs_per_s']:.0f} img/s")

if __name__ == "__main__":
    main()

