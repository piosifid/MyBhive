import os
import matplotlib.pyplot as plt
import numpy as np

DATA_PATH = os.environ["DATA_PATH"].rstrip("/")
OUT_DIR = os.path.join(DATA_PATH, "comparison_plots")
os.makedirs(OUT_DIR, exist_ok=True)

models = ["baseline", "flatpt"]
metrics = ["train acc", "val acc", "test acc"]
values = {
    "baseline": [62.02, 59.29, 55.8],
    "flatpt":   [89.62, 89.99, 2.6],
}
colors = ["#1f4e8c", "#5b9bd5", "#d9534f"]  # train, val, test — test in a distinct warning color

fig, ax = plt.subplots(figsize=(8, 5.5))
x = np.arange(len(models))
width = 0.25

for i, metric in enumerate(metrics):
    vals = [values[m][i] for m in models]
    bars = ax.bar(x + (i - 1) * width, vals, width, label=metric, color=colors[i])
    for b, v in zip(bars, vals):
        ax.annotate(f"{v:.1f}%", xy=(b.get_x() + b.get_width() / 2, v),
                    xytext=(0, 3), textcoords="offset points",
                    ha="center", fontsize=9)

ax.set_xticks(x)
ax.set_xticklabels(models)
ax.set_ylabel("accuracy [%]")
ax.set_ylim(0, 100)
ax.set_title("Train/val accuracy vs. real test accuracy (ParticleNet, epoch 40)")
ax.legend()
ax.grid(True, axis="y", alpha=0.3)
fig.savefig(os.path.join(OUT_DIR, "plot4_accuracy_comparison.png"), dpi=170, bbox_inches="tight")
print("Saved plot4_accuracy_comparison.png")
