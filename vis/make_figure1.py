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

x = np.arange(1, 11)
acc = [split1_data["label_acc"], split2_data["label_acc"], split3_data["label_acc"]]

mean = [
    [
        0.4090129567402294,
        0.4166797885714998,
        0.45251974456519917,
        0.48132391969290356,
        0.457285939464732,
        0.4753504912595822,
        0.46926249008468257,
        0.4596523883622814,
        0.4574628604708818,
        0.4435895023813684,
    ],
    [
        0.40285165171528803,
        0.41326786701118245,
        0.4445220183856547,
        0.4220529544727405,
        0.46431973241183766,
        0.48679645023227913,
        0.4814595856708156,
        0.46395853725399183,
        0.45041958613412314,
        0.4492211257251364,
    ],
    [
        0.42299298959746723,
        0.450062582525269,
        0.44066232031552477,
        0.44852535524177306,
        0.4265378485527739,
        0.42751601632198644,
        0.42771560943202735,
        0.41769681396547076,
        0.4177061744225923,
        0.4031350490305714,
    ],
]

acc = np.array(acc)
mean = np.array(mean)

acc = 100 * acc
mean = 100 * mean

train_losses = [
    [2.3, 1.81, 1.66, 1.58, 1.51, 1.46, 1.42, 1.41, 1.4, 1.39],
    [2.28, 1.82, 1.68, 1.58, 1.51, 1.47, 1.43, 1.41, 1.402, 1.4],
    [2.26, 1.8, 1.61, 1.52, 1.47, 1.43, 1.4, 1.39, 1.38, 1.378],
]

val_losses = [
    [1.95, 1.71, 1.64, 1.591, 1.58, 1.57, 1.565, 1.563, 1.563, 1.563],
    [1.96, 1.76, 1.67, 1.608, 1.575, 1.56, 1.54, 1.54, 1.54, 1.54],
    [1.94, 1.73, 1.64, 1.59, 1.57, 1.54, 1.535, 1.53, 1.529, 1.529],
]

fig, axes = plt.subplots(
    3,
    3,
    figsize=(12, 7.83),
    sharex=True,
    sharey="row",
    gridspec_kw={"height_ratios": [1.3, 1.3, 1]},
)

lines = []
labels = []

for i, ax in list(enumerate(axes.flat))[:3]:
    ax.xaxis.set_major_locator(MultipleLocator(2))
    ax.xaxis.set_minor_locator(MultipleLocator(1))
    ax.yaxis.set_major_locator(MultipleLocator(10))
    (l1,) = ax.plot(x, acc[i], label=f"Split {i + 1}", color=f"C{i}")
    ax.grid(which="major", color="gray", linewidth=0.3, alpha=0.3)


for i, ax in list(enumerate(axes.flat[3:6])):
    ax.xaxis.set_major_locator(MultipleLocator(2))
    ax.xaxis.set_minor_locator(MultipleLocator(1))
    ax.yaxis.set_major_locator(MultipleLocator(3))
    (l1,) = ax.plot(x, mean[i], label=f"Split {i + 1}", color=f"C{i}")
    ax.grid(which="major", color="gray", linewidth=0.3, alpha=0.3)

for i, ax in list(enumerate(axes.flat[6:])):
    ax.xaxis.set_major_locator(MultipleLocator(2))
    ax.xaxis.set_minor_locator(MultipleLocator(1))
    ax.yaxis.set_major_locator(MultipleLocator(0.3))
    (l1,) = ax.plot(x, train_losses[i], label=f"Split {i + 1}", color=f"C{i}", ls="--")
    (l1,) = ax.plot(x, val_losses[i], label=f"Split {i + 1}", color=f"C{i}")
    ax.grid(which="major", color="gray", linewidth=0.3, alpha=0.3)

axes.flat[3].set_ylabel("Mean of Cue-F1 & SO")
axes.flat[0].set_ylabel("Accuracy")
axes.flat[6].set_ylabel("Loss")
axes.flat[7].set_xlabel("Epoch")

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
