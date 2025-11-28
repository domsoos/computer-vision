import torch
import matplotlib.pyplot as plt
import numpy as np
import os

# Import all optical architectures
from fd2nn import FD2NN
from d2nn import D2NN
from hybrid import HybridFD2NN_CNN

def get_model_and_masks(model_name, device='cpu'):
    """Instantiates the correct model and returns the list of phase mask tensors."""
    C = 10
    path = f"./checkpoints/{model_name}.pth"
    
    if not os.path.exists(path):
        print(f"Skipping {model_name}: {path} not found.")
        return None, None

    # 1. Instantiate the correct architecture
    if "Hybrid" in model_name:
        model = HybridFD2NN_CNN(img_size=32, classes=C, fd_channels=32, cnn_channels=(16,32))
        # Masks are in the front-end
        masks = model.front.phase_masks
    
    elif "FD2NN" in model_name:
        model = FD2NN(in_ch=1, img_size=32, n_layers=4, hidden_channels=32, classes=C, dropout=0.1)
        # Masks are in .phase_masks
        masks = model.phase_masks
        
    elif "D2NN" in model_name:
        # D2NN params must match training
        nonlin = "NonLin" in model_name
        model = D2NN(in_ch=1, img_size=32, n_layers=4, hidden_channels=64, classes=C, sim_nonlinearity=nonlin)
        # Masks are in .phase
        masks = model.phase
    else:
        return None, None

    # 2. Load Weights
    try:
        model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
        print(f"Loaded {model_name}")
    except Exception as e:
        print(f"Error loading {model_name}: {e}")
        return None, None

    return model, masks

def visualize_all_masks():
    """Generates heatmaps for all models with phase masks."""
    # The list of models we want to visualize
    target_models = ["FD2NN_Opt", "D2NN_NonLin","D2NN_Linear","Hybrid", "Hybrid_Frozen"]
    
    # Ensure plots directory exists
    os.makedirs("./plots", exist_ok=True)

    for name in target_models:
        model, masks = get_model_and_masks(name)
        
        if masks is None or len(masks) == 0:
            continue

        # Create Plot
        n_layers = len(masks)
        plt.figure(figsize=(3 * n_layers, 3.5))
        
        for i, mask in enumerate(masks):
            # Shape is (Channels, H, W) -> Average to (H, W) for visualization
            phase_avg = mask.detach().cpu().numpy().mean(axis=0)
            
            plt.subplot(1, n_layers, i+1)
            # 'twilight' is best for phase (-pi to pi)
            im = plt.imshow(phase_avg, cmap='twilight', vmin=-np.pi, vmax=np.pi)
            plt.title(f"Layer {i+1}")
            plt.axis('off')
            plt.colorbar(im, fraction=0.046, pad=0.04)
        
        plt.suptitle(f"Learned Phase: {name}", fontsize=14)
        plt.tight_layout()
        
        # Save
        filename = f"{name}_phase_masks.png"
        save_path = f"../plots/{filename}"
        plt.savefig(save_path)
        print(f"Saved {save_path}")
        plt.close()

if __name__ == "__main__":
    visualize_all_masks()
