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
    base_no = "checkpoints/no_curriculum/fashion"
    base_yes = "checkpoints/with_curriculum/fashion"

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
        # No Curriculum 
        path_no = f"{base_no}/{name}_stats.json"
        stats_no = load_stats(path_no)

        if stats_no is not None:
            # full validation accuracy history (0..N-1, e.g. 0..25)
            acc_no = np.array(stats_no["history"]["val_acc"])

            # cut to first 20 epochs
            acc_no = acc_no[:max_epochs_to_plot]

            # shift indices so x = 1..20
            epochs_x_no = np.arange(1, len(acc_no) + 1)

            plt.plot(
                epochs_x_no,
                acc_no,
                label=f"{name} (No Curr.)",
                color=colors[name],
                linestyle="-",
                linewidth=2.5,
            )

        # With Curriculum 
        path_yes = f"{base_yes}/{name}_stats.json"
        stats_yes = load_stats(path_yes)

        if stats_yes is not None:
            acc_yes = np.array(stats_yes["history"]["val_acc"])
            acc_yes = acc_yes[:max_epochs_to_plot]
            epochs_x_yes = np.arange(1, len(acc_yes) + 1)

            plt.plot(
                epochs_x_yes,
                acc_yes,
                label=f"{name} (With Curr.)",
                color=colors[name],
                linestyle="--",
                linewidth=1.5,
                alpha=0.7,
            )

    plt.title("Impact of Curriculum Learning: Physics Mismatch", fontsize=14)
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Validation Accuracy", fontsize=12)

    # Ticks at every integer epoch 1–20
    plt.xticks(np.arange(1, max_epochs_to_plot + 1))

    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    os.makedirs("plots", exist_ok=True)
    save_path = "plots/curriculum_failure_with_hybrid.png"
    plt.savefig(save_path)
    print(f"Saved comparison plot to {save_path}")

if __name__ == "__main__":
    plot_curriculum_impact()

