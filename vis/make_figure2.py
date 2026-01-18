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

x = np.arange(1, 13)
x_labels = [str(i) if i <= 9 else str(i - 9) for i in x]
acc = [split1_data["label_acc"][:-1] for _ in range(3)]
acc = np.array(acc)
acc = acc * 100

mrt_acc = [[0.58, 0.56, 0.49], [0.65, 0.50, 0.59], [0.59, 0.54, 0.58]]

mrt_acc = np.array(mrt_acc)
mrt_acc = mrt_acc * 100

train_losses = [[0.51, 0.43, 0.394], [0.54, 0.44, 0.37], [0.56, 0.49, 0.39]]

val_losses = [[0.45, 0.40, 0.399], [0.50, 0.43, 0.421], [0.52, 0.43, 0.425]]

fig, axes = plt.subplots(
    2,
    3,
    figsize=(12, 5),
    sharex="row",
    sharey="row",
    gridspec_kw={"height_ratios": [1.3, 1]},
)

lines = []
labels = []

for i, ax in list(enumerate(axes.flat))[:3]:
    ax.yaxis.set_major_locator(MultipleLocator(10))
    (l1,) = ax.plot(
        x[:9], acc[i], label="A", color=f"C{i}", ls=":"
    )
    (l1,) = ax.plot(
        x[8:],
        np.concatenate((acc[i, -1:], mrt_acc[i]), axis=0),
        label=f"B {i + 1}",
        color=f"C{i}",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels)
    for j, label in enumerate(ax.get_xticklabels()):
        if j < 9:
            label.set_color("darkgray")
    ax.axhline(y=acc[i][-1], ls=":", color=f"C{i}")

    ax.grid(which="major", color="gray", linewidth=0.3, alpha=0.3)

for i, ax in list(enumerate(axes.flat[3:])):
    ax.xaxis.set_major_locator(MultipleLocator(1))
    (l1,) = ax.plot(
        np.arange(1, 4), train_losses[i], label=f"B {i + 1}", color=f"C{i}", ls="--"
    )
    (l1,) = ax.plot(np.arange(1, 4), val_losses[i], label=f"B {i + 1}", color=f"C{i}")
    ax.grid(which="major", color="gray", linewidth=0.3, alpha=0.3)

axes.flat[0].set_ylabel("Accuracy")
axes.flat[3].set_ylabel("Loss")
axes.flat[4].set_xlabel("Epoch")

colors = ["C0", "C1", "C2"]
labels = ["B1", "B2", "B3"]

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
