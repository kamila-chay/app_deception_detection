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


x = np.arange(1, 21)
acc = [
    [
        49.23076923076923,
        60.0,
        60.0,
        53.84615384615385,
        64.61538461538461,
        63.07692307692307,
        69.23076923076923,
        67.6923076923077,
        70.76923076923077,
        67.6923076923077,
        69.23076923076923,
        69.23076923076923,
        69.23076923076923,
        69.23076923076923,
        69.23076923076923,
        69.23076923076923,
        72.3076923076923,
        72.3076923076923,
        72.3076923076923,
        72.3076923076923,
    ],
    [
        49.23076923076923,
        58.46153846153847,
        63.07692307692307,
        58.46153846153847,
        60.0,
        58.46153846153847,
        64.61538461538461,
        58.46153846153847,
        61.53846153846154,
        63.07692307692307,
        61.53846153846154,
        66.15384615384615,
        67.6923076923077,
        61.53846153846154,
        63.07692307692307,
        64.61538461538461,
        63.07692307692307,
        64.61538461538461,
        64.61538461538461,
        64.61538461538461,
    ],
    [
        62.121212121212125,
        57.57575757575758,
        57.57575757575758,
        59.09090909090909,
        63.63636363636363,
        66.66666666666666,
        69.6969696969697,
        74.24242424242425,
        71.21212121212122,
        72.72727272727273,
        71.21212121212122,
        71.21212121212122,
        80.3030303030303,
        72.72727272727273,
        78.78787878787878,
        77.27272727272727,
        77.27272727272727,
        77.27272727272727,
        77.27272727272727,
        77.27272727272727,
    ],
]

train_losses = [
    [
        0.59375,
        0.58203125,
        0.5703125,
        0.58203125,
        0.56640625,
        0.515625,
        0.45703125,
        0.38671875,
        0.365234375,
        0.326171875,
        0.30078125,
        0.27734375,
        0.255859375,
        0.2421875,
        0.21484375,
        0.197265625,
        0.1865234375,
        0.185546875,
        0.17578125,
        0.173828125,
    ],
    [
        0.61328125,
        0.578125,
        0.5703125,
        0.56640625,
        0.55859375,
        0.52734375,
        0.455078125,
        0.404296875,
        0.369140625,
        0.314453125,
        0.279296875,
        0.25390625,
        0.2109375,
        0.1787109375,
        0.1640625,
        0.1494140625,
        0.1396484375,
        0.1357421875,
        0.1298828125,
        0.125,
    ],
    [
        0.59765625,
        0.5625,
        0.546875,
        0.55859375,
        0.54296875,
        0.5234375,
        0.470703125,
        0.400390625,
        0.345703125,
        0.310546875,
        0.2734375,
        0.251953125,
        0.203125,
        0.181640625,
        0.15234375,
        0.1376953125,
        0.1259765625,
        0.115234375,
        0.11376953125,
        0.111328125,
    ],
]

val_losses = [
    [
        0.71484375,
        0.7109375,
        0.6953125,
        0.7265625,
        0.66796875,
        0.65625,
        0.640625,
        0.6328125,
        0.62109375,
        0.609375,
        0.59375,
        0.59765625,
        0.5859375,
        0.5859375,
        0.57421875,
        0.578125,
        0.58203125,
        0.578125,
        0.58203125,
        0.58984375,
    ],
    [
        0.74609375,
        0.71875,
        0.69140625,
        0.6796875,
        0.66015625,
        0.65234375,
        0.6640625,
        0.68359375,
        0.66796875,
        0.671875,
        0.6796875,
        0.68359375,
        0.70703125,
        0.7265625,
        0.72265625,
        0.73828125,
        0.75,
        0.75390625,
        0.7578125,
        0.75390625,
    ],
    [
        0.77734375,
        0.6875,
        0.6796875,
        0.63671875,
        0.6328125,
        0.59375,
        0.5703125,
        0.55078125,
        0.546875,
        0.5546875,
        0.58203125,
        0.57421875,
        0.5625,
        0.671875,
        0.6328125,
        0.63671875,
        0.65625,
        0.65234375,
        0.66796875,
        0.66796875,
    ],
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
    ax.xaxis.set_major_locator(MultipleLocator(4))
    ax.xaxis.set_major_locator(MultipleLocator(2))
    (l1,) = ax.plot(x, acc[i], label=f"Split {i + 1}", color=f"C{i}")

    ax.grid(which="major", color="gray", linewidth=0.3, alpha=0.3)

for i, ax in list(enumerate(axes.flat[3:])):
    ax.xaxis.set_major_locator(MultipleLocator(2))
    ax.xaxis.set_minor_locator(MultipleLocator(1))
    ax.yaxis.set_major_locator(MultipleLocator(0.3))
    (l1,) = ax.plot(x, train_losses[i], label=f"Split {i + 1}", color=f"C{i}", ls="--")
    (l1,) = ax.plot(x, val_losses[i], label=f"Split {i + 1}", color=f"C{i}")
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
