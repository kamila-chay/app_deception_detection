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
    "thesis/out/2025-11-15_20-01/model_split1_validation_metrics.json", "r"
)
split1_data = json.load(split1_file)
split2_file = open(
    "thesis/out/2025-11-15_20-01/model_split2_validation_metrics.json", "r"
)
split2_data = json.load(split2_file)
split3_file = open(
    "thesis/out/2025-11-15_20-01/model_split3_validation_metrics.json", "r"
)
split3_data = json.load(split3_file)


offsets = (9, 2, 4)
x = [np.arange(1, offset + 4) for offset in offsets]
x_labels = [
    [str(i) if i <= offsets[j] else str(i - offsets[j]) for i in x[j]] for j in range(3)
]
acc = [
    split1_data["label_acc"][:9],
    split2_data["label_acc"][:2],
    split3_data["label_acc"][:4],
]

for i in range(3):
    for j in range(len(acc[i])):
        acc[i][j] = 100 * acc[i][j]

dpo_acc = [[0.42, 0.54, 0.49], [0.7, 0.55, 0.55], [0.42, 0.52, 0.46]]

for i in range(3):
    for j in range(len(dpo_acc[i])):
        dpo_acc[i][j] = 100 * dpo_acc[i][j]

train_losses = [[0.4, 0.3, 0.26], [0.42, 0.38, 0.35], [0.41, 0.34, 0.3]]

val_losses = [[0.35, 0.28, 0.27], [0.40, 0.36, 0.35], [0.38, 0.33, 0.32]]

fig, axes = plt.subplots(
    2,
    3,
    figsize=(12, 5),
    sharex="none",
    sharey="row",
    gridspec_kw={"height_ratios": [1.3, 1]},
)

for i, ax in list(enumerate(axes.flat))[:3]:
    # ax.xaxis.set_major_locator(MultipleLocator(2))
    # ax.xaxis.set_major_locator(MultipleLocator(1))
    ax.yaxis.set_major_locator(MultipleLocator(10))
    (l1,) = ax.plot(x[i][: offsets[i]], acc[i], label="A", color=f"C{i}", ls=":")
    (l1,) = ax.plot(
        x[i][offsets[i] - 1 :],
        [acc[i][-1]] + dpo_acc[i],
        label=f"Split {i + 1}",
        color=f"C{i}",
    )
    ax.set_xticks(x[i])
    ax.set_xticklabels(x_labels[i])
    for j, label in enumerate(ax.get_xticklabels()):
        if j < offsets[i]:
            label.set_color("darkgray")
    ax.axhline(y=acc[i][-1], ls=":", color=f"C{i}")

    ax.grid(which="major", color="gray", linewidth=0.3, alpha=0.3)

for i, ax in list(enumerate(axes.flat[3:])):
    # ax.xaxis.set_major_locator(MultipleLocator(2))
    ax.xaxis.set_major_locator(MultipleLocator(1))
    # ax.yaxis.set_major_locator(MultipleLocator(0.3))
    (l1,) = ax.plot(
        np.arange(1, 4), train_losses[i], label=f"Split {i + 1}", color=f"C{i}", ls="--"
    )
    (l1,) = ax.plot(
        np.arange(1, 4), val_losses[i], label=f"Split {i + 1}", color=f"C{i}"
    )
    ax.grid(which="major", color="gray", linewidth=0.3, alpha=0.3)

axes.flat[0].set_ylabel("Accuracy")
axes.flat[3].set_ylabel("Loss")
axes.flat[4].set_xlabel("Epoch")

colors = ["C0", "C1", "C2"]
labels = ["Split 1", "Split 2", "Split 3"]

color_handles = [
    Patch(color=color, label=label) for color, label in zip(colors, labels)
]
line_handles = [
    Line2D([0], [0], color="k", lw=2, linestyle=":", label="From Setup A"),
    Line2D([0], [0], color="k", lw=2, linestyle="-", label="Validation"),
    Line2D([0], [0], color="k", lw=2, linestyle="--", label="Training"),
]

fig.legend(
    handles=color_handles + line_handles,
    loc="upper left",
    bbox_to_anchor=(0.075, 0.98),
    borderaxespad=0.0,
    ncol=6,
    columnspacing=0.6,
    frameon=False,
)

plt.tight_layout(rect=[0, 0, 1, 0.93])
plt.show()

split1_file.close()
split2_file.close()
split3_file.close()
