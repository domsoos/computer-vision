import json
import matplotlib.pyplot as plt
import os
import numpy as np

def load_stats(path):
    if not os.path.exists(path):
        print(f"Warning: {path} not found")
        return None
    with open(path, 'r') as f:
        return json.load(f)

def plot_curriculum_impact():
    # Define paths based on your directory structure
    base_no = "checkpoints/no_curriculum/fashion"
    base_yes = "checkpoints/with_curriculum/fashion"

    # Models to compare (now includes Hybrid)
    models = ["FD2NN_Opt", "D2NN_NonLin", "CNN_Baseline", "Hybrid"]
    colors = {
        "FD2NN_Opt": "orange",
        "D2NN_NonLin": "green",
        "CNN_Baseline": "blue",
        "Hybrid": "purple",       
    }

    max_epochs_to_plot = 20

    plt.figure(figsize=(10, 6))

    for name in models:
        # 1. Load No Curriculum (Sharp)
        path_no = f"{base_no}/{name}_stats.json"
        stats_no = load_stats(path_no)

        if stats_no is not None:
            # Take epochs 0–19 from the history
            acc_no = stats_no['history']['val_acc'][:max_epochs_to_plot]

            # Shift to 1–20 on the x-axis
            epochs_x_no = np.arange(1, len(acc_no) + 1)

            plt.plot(
                epochs_x_no, acc_no,
                label=f"{name} (No Curr.)",
                color=colors[name],
                linestyle="-",
                linewidth=2.5
            )

        # 2. Load With Curriculum (Blurry)
        path_yes = f"{base_yes}/{name}_stats.json"
        stats_yes = load_stats(path_yes)

        if stats_yes is not None:
            # Take epochs 0–19 from the history
            acc_yes = stats_yes['history']['val_acc'][:max_epochs_to_plot]

            # Shift to 1–20 on the x-axis
            epochs_x_yes = np.arange(1, len(acc_yes) + 1)

            plt.plot(
                epochs_x_yes, acc_yes,
                label=f"{name} (With Curr.)",
                color=colors[name],
                linestyle="--",
                linewidth=1.5,
                alpha=0.7
            )

    plt.title("Impact of Curriculum Learning: Physics Mismatch", fontsize=14)
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Validation Accuracy", fontsize=12)

    # Force X-axis to display 1 to 20
    plt.xlim(1, max_epochs_to_plot)
    plt.xticks(np.arange(1, max_epochs_to_plot + 1))  # 1,2,3,...,20

    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    # Save output
    os.makedirs("plots", exist_ok=True)
    save_path = "plots/curriculum_failure_with_hybrid.png"
    plt.savefig(save_path)
    print(f"Saved comparison plot to {save_path}")

if __name__ == "__main__":
    plot_curriculum_impact()

