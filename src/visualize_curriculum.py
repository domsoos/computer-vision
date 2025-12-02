# visualize_curriculum_examples.py

import os
import torch
from torchvision.utils import make_grid
import matplotlib.pyplot as plt

from benchmark_train import get_loaders
from curriculum import lowpass_inputs  # adjust if file/module name differs


# Stats copied from get_loaders so we can de-normalize for viewing
DATA_STATS = {
    "fashion": ((0.2860,), (0.3530,)),
    "mnist":   ((0.1307,), (0.3081,)),
}


def denormalize(x, mean, std):
    """
    x: (B, C, H, W), normalized
    mean, std: sequences of length C
    Returns: de-normalized tensor in [roughly 0,1]
    """
    mean = torch.as_tensor(mean, device=x.device)[None, :, None, None]
    std  = torch.as_tensor(std,  device=x.device)[None, :, None, None]
    return x * std + mean


@torch.no_grad()
def save_orig_vs_blur_grids(
    dataset: str = "fashion",
    bs: int = 32,
    cutoff_values=(0.25, 0.5, 0.75, 1.0),
    device: str = None,
):
    """
    Grabs one batch from get_loaders(), applies lowpass_inputs for several cutoff_frac
    values, and saves side-by-side grids: original vs blurred.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    os.makedirs("plots/curriculum_examples", exist_ok=True)

    # Load data (with your augmentations + normalization)
    train_loader, _, _ = get_loaders(dataset=dataset, bs=bs)
    x, y = next(iter(train_loader))   # one real batch
    x = x.to(device)

    mean, std = DATA_STATS[dataset]

    # For reference: a grid of originals
    x_vis = denormalize(x, mean, std).clamp(0, 1)
    grid_orig = make_grid(x_vis[:16], nrow=8)  # first 16
    plt.figure(figsize=(6, 4))
    plt.imshow(grid_orig.permute(1, 2, 0).cpu().numpy())
    plt.axis("off")
    plt.title(f"{dataset} originals (first batch)")
    plt.tight_layout()
    plt.savefig("plots/curriculum_examples/original_batch.png", dpi=150)
    plt.close()
    print("[viz] saved plots/curriculum_examples/original_batch.png")

    # For each cutoff value, blur and save grid
    for cutoff in cutoff_values:
        x_blur = lowpass_inputs(x, cutoff)  # still normalized

        x_blur_vis = denormalize(x_blur, mean, std).clamp(0, 1)

        # Build pairwise [orig0, blur0, orig1, blur1, ...]
        B = x.size(0)
        n = min(B, 8)  # 8 pairs
        pairs = []
        for i in range(n):
            pairs.append(x_vis[i])
            pairs.append(x_blur_vis[i])
        grid_pairs = make_grid(pairs, nrow=2)

        plt.figure(figsize=(6, 6))
        plt.imshow(grid_pairs.permute(1, 2, 0).cpu().numpy())
        plt.axis("off")
        plt.title(f"{dataset} — original vs blurred\ncutoff_frac={cutoff:.2f}")
        plt.tight_layout()
        out_path = f"plots/curriculum_examples/{dataset}_cutoff_{cutoff:.2f}.png"
        plt.savefig(out_path, dpi=150)
        plt.close()
        print(f"[viz] saved {out_path}")


if __name__ == "__main__":
    save_orig_vs_blur_grids(
        dataset="fashion",
        bs=64,
        cutoff_values=(0.25, 0.5, 0.75, 1.0),
    )

