# Developed as part of a BSc thesis at the Faculty of Computer Science, Bialystok Univesity of Technology

import json

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import MultipleLocator

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.size"] = 14

split1_file = open(
    "thesis/out/2025-11-19_16-41/model_split1_validation_metrics.json", "r"
)
split1_data = json.load(split1_file)
split2_file = open(
    "thesis/out/2025-11-19_16-41/model_split2_validation_metrics.json", "r"
)
split2_data = json.load(split2_file)
split3_file = open(
    "thesis/out/2025-11-19_16-41/model_split3_validation_metrics.json", "r"
)
split3_data = json.load(split3_file)


x = np.arange(1, 11)
acc = [split1_data["mean"], split2_data["mean"], split3_data["mean"]]

acc = np.array(acc)
acc = 100 * acc

train_losses = [
    [2.3, 1.7, 1.52, 1.41, 1.37, 1.31, 1.27, 1.25, 1.24, 1.23],
    [2.29, 1.7, 1.53, 1.41, 1.35, 1.31, 1.28, 1.26, 1.25, 1.25],  # change...
    [2.3, 1.71, 1.52, 1.41, 1.37, 1.32, 1.27, 1.26, 1.25, 1.24],
]

val_losses = [
    [1.92, 1.63, 1.51, 1.465, 1.43, 1.4, 1.39, 1.38, 1.38, 1.38],
    [1.87, 1.595, 1.49, 1.43, 1.39, 1.375, 1.366, 1.36, 1.36, 1.36],
    [1.86, 1.59, 1.49, 1.43, 1.39, 1.37, 1.36, 1.35, 1.35, 1.36],
]

fig, axes = plt.subplots(
    2,
    3,
    figsize=(12, 5),
    sharex=True,
    sharey="row",
    gridspec_kw={"height_ratios": [1.3, 1]},
)

for i, ax in list(enumerate(axes.flat))[:3]:
    # ax.xaxis.set_major_locator(MultipleLocator(2))
    # ax.xaxis.set_major_locator(MultipleLocator(1))
    ax.yaxis.set_major_locator(MultipleLocator(3))
    (l1,) = ax.plot(x, acc[i], label=f"Split {i + 1}", color=f"C{i}")

    ax.grid(which="major", color="gray", linewidth=0.3, alpha=0.3)

for i, ax in list(enumerate(axes.flat[3:])):
    ax.xaxis.set_major_locator(MultipleLocator(2))
    ax.xaxis.set_minor_locator(MultipleLocator(1))
    ax.yaxis.set_major_locator(MultipleLocator(0.45))
    (l1,) = ax.plot(x, train_losses[i], label=f"Split {i + 1}", color=f"C{i}", ls="--")
    (l1,) = ax.plot(x, val_losses[i], label=f"Split {i + 1}", color=f"C{i}")
    ax.grid(which="major", color="gray", linewidth=0.3, alpha=0.3)

axes.flat[0].set_ylabel("Mean of Cue-F1 & SO")
axes.flat[3].set_ylabel("Loss")
axes.flat[4].set_xlabel("Epoch")

colors = ["C0", "C1", "C2"]
labels = ["Split 1", "Split 2", "Split 3"]

color_handles = [
    Patch(color=color, label=label) for color, label in zip(colors, labels)
]
line_handles = [
    Line2D([0], [0], color="k", lw=2, linestyle="-", label="Validation"),
    Line2D([0], [0], color="k", lw=2, linestyle="--", label="Training"),
]

fig.legend(
    handles=color_handles + line_handles,
    loc="upper left",
    bbox_to_anchor=(0.075, 0.98),
    borderaxespad=0.0,
    ncol=5,
    columnspacing=0.6,
    frameon=False,
)

plt.tight_layout(rect=[0, 0, 1, 0.93])
plt.show()

split1_file.close()
split2_file.close()
split3_file.close()
