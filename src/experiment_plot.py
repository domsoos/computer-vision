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


def get_test_loader():
    MEAN, STD = (0.2860,), (0.3530,)
    tfms = transforms.Compose([
        transforms.Pad(2),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD)
    ])
    dset = datasets.FashionMNIST('./data', train=False, download=True, transform=tfms)
    return DataLoader(dset, batch_size=256, shuffle=False), dset.classes


def load_model(model_name, device):
    C = 10
    # Match the definitions in benchmark_train.py 
    if "CNN" in model_name:
        model = CNN(in_ch=1, classes=C, channels=(16, 32))
    elif "Hybrid" in model_name:
        # Same architecture for regular and frozen hybrids; weights differ by checkpoint
        model = HybridFD2NN_CNN(img_size=32, classes=C, fd_channels=32, cnn_channels=(16, 32))
    elif "FD2NN" in model_name:
        model = FD2NN(in_ch=1, img_size=32, n_layers=4, hidden_channels=32, classes=C, dropout=0.1)
    elif "D2NN" in model_name:
        # Check nonlinearity flag based on name
        nonlin = "NonLin" in model_name
        model = D2NN(in_ch=1, img_size=32, n_layers=4, hidden_channels=64,
                     classes=C, sim_nonlinearity=nonlin)
    else:
        return None

    path = f"checkpoints/no_curriculum/fashion/{model_name}.pth"
    if not os.path.exists(path):
        print(f"[WARN] Checkpoint not found for {model_name}: {path}")
        return None

    try:
        model.load_state_dict(torch.load(path, map_location=device, weights_only=False))
    except Exception as e:
        print(f"Error loading {model_name}: {e}")
        return None
    return model.to(device).eval()


def load_stats(model_name):
    """Helper: load stats JSON for a model."""
    path = f"checkpoints/no_curriculum/fashion/{model_name}_stats.json"
    if not os.path.exists(path):
        print(f"[WARN] Stats file not found for {model_name}: {path}")
        return None
    with open(path, "r") as f:
        return json.load(f)


def plot_benchmark_tradeoff(model_names, filename="benchmark_tradeoff.png"):
    """
    Crossplot: best validation accuracy vs inference throughput.
    Uses only aggregate benchmark stats (no training curves).
    """
    xs, ys, labels = [], [], []

    for name in model_names:
        stats = load_stats(name)
        if stats is None:
            continue
        xs.append(stats["inference_imgs_per_s"])
        ys.append(stats["best_val_acc"])
        labels.append(name)

    if not xs:
        print("No stats found for any model, not plotting benchmark tradeoff.")
        return

    plt.figure(figsize=(6, 5))
    plt.scatter(xs, ys)

    for x, y, label in zip(xs, ys, labels):
        plt.annotate(label, (x, y),
                     textcoords="offset points",
                     xytext=(5, 5),
                     ha="left",
                     fontsize=8)

    plt.xlabel("Inference throughput [images / s]")
    plt.ylabel("Best validation accuracy")
    plt.title("Accuracy vs Throughput (benchmark crossplot)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    print(f"Saved {filename}")


def plot_training_curves(model_names, filename="benchmark_curves.png"):
    """
    Reads JSON stats and plots ONLY validation curves (no training metrics).
    Keeps this around in case you still want full benchmark curves.
    """
    plt.figure(figsize=(12, 5))

    # Validation Accuracy
    plt.subplot(1, 2, 1)
    for name in model_names:
        stats = load_stats(name)
        if stats is None:
            continue
        hist = stats["history"]
        plt.plot(hist["val_acc"], label=name)
    plt.title("Validation Accuracy vs Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Validation Loss
    plt.subplot(1, 2, 2)
    for name in model_names:
        stats = load_stats(name)
        if stats is None:
            continue
        hist = stats["history"]
        plt.plot(hist["val_loss"], label=name)
    plt.title("Validation Loss vs Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    print(f"Saved {filename}")


def plot_experiment_curves(exp_name, model_names, out_prefix):
    """
    Plot validation accuracy & loss curves for a group of models (an experiment).
    Only validation metrics (no training curves).
    """
    plt.figure(figsize=(12, 5))

    # Validation Accuracy 
    plt.subplot(1, 2, 1)
    for name in model_names:
        stats = load_stats(name)
        if stats is None:
            continue
        hist = stats["history"]
        plt.plot(hist["val_acc"], label=name)
    plt.title(f"{exp_name} – Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Validation Loss
    plt.subplot(1, 2, 2)
    for name in model_names:
        stats = load_stats(name)
        if stats is None:
            continue
        hist = stats["history"]
        plt.plot(hist["val_loss"], label=name)
    plt.title(f"{exp_name} – Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    fname = f"{out_prefix}.png"
    plt.savefig(fname)
    plt.close()
    print(f"Saved {fname}")


def analyze_model(model_name):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Analyzing {model_name}...")

    model = load_model(model_name, device)
    if model is None:
        return

    loader, class_names = get_test_loader()
    all_preds, all_targs, images = [], [], []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            logits = model(x)
            preds = logits.argmax(1).cpu()
            all_preds.extend(preds.numpy())
            all_targs.extend(y.numpy())
            if len(images) == 0:
                images = x.cpu()  # Save first batch only

    # 1. Confusion Matrix
    cm = confusion_matrix(all_targs, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title(f"Confusion Matrix: {model_name}")
    plt.ylabel('True')
    plt.xlabel('Pred')
    plt.tight_layout()
    plt.savefig(f"{model_name}_confusion.png")
    plt.close()

    # 2. Sample Predictions
    all_preds = np.array(all_preds)
    all_targs = np.array(all_targs)

    # images is from the first batch
    num_imgs = len(images)
    if num_imgs == 0:
        print(f"No images recorded for {model_name}, skipping samples.")
        return

    n_show = min(10, num_imgs)
    indices = np.random.choice(num_imgs, n_show, replace=False)

    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    axes = axes.flatten()

    for i, idx in enumerate(indices):
        ax = axes[i]
        img = images[idx].squeeze()
        # Un-normalize using FashionMNIST stats
        img = img * 0.3530 + 0.2860
        ax.imshow(img, cmap='gray')

        pred = class_names[all_preds[idx]]
        true = class_names[all_targs[idx]]
        col = 'green' if pred == true else 'red'

        ax.set_title(f"P:{pred}\nT:{true}", color=col)
        ax.axis('off')

    # Hide any unused subplots (if n_show < 10)
    for j in range(n_show, len(axes)):
        axes[j].axis('off')

    plt.suptitle(f"{model_name} Random Samples")
    plt.tight_layout()
    plt.savefig(f"{model_name}_samples.png")
    plt.close()
    print(f"Saved plots for {model_name}")


if __name__ == "__main__":
    # Models corresponding to your training script
    models = [
        "CNN_Baseline",
        "Hybrid",
        "Hybrid_Frozen",
        "FD2NN_Opt",
        "D2NN_NonLin",
        "D2NN_Linear",
    ]

    # Global benchmark crossplot
    plot_benchmark_tradeoff(models)

    # Experiment 1: CNN vs FD2NN vs D2NN (Nonlinear)
    exp1_models = ["CNN_Baseline", "FD2NN_Opt", "D2NN_NonLin"]
    plot_experiment_curves(
        exp_name="Exp1: CNN vs FD2NN vs D2NN (Nonlin)",
        model_names=exp1_models,
        out_prefix="exp1_baseline_curves"
    )

    # Experiment 2: Hybrid regular vs Hybrid frozen
    exp2_models = ["Hybrid", "Hybrid_Frozen"]
    plot_experiment_curves(
        exp_name="Exp2: Hybrid vs Hybrid_Frozen",
        model_names=exp2_models,
        out_prefix="exp2_hybrid_curves"
    )

    # Experiment 3: D2NN Linear vs Nonlinear (Optical ReLU)
    exp3_models = ["D2NN_Linear", "D2NN_NonLin"]
    plot_experiment_curves(
        exp_name="Exp3: D2NN Linear vs Nonlinear",
        model_names=exp3_models,
        out_prefix="exp3_d2nn_lin_vs_nonlin"
    )

    plot_training_curves(models)

    for m in models:
        analyze_model(m)

