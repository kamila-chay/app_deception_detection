from pathlib import Path

import matplotlib.pyplot as plt
from PIL import Image

root = Path("data/attn_rollout_reasoning_labels")

for item in root.iterdir():
    if item.is_file():
        image_file = root / "heatmaps" / f"{item.stem}.png"
        text_file = item

        img = Image.open(image_file)

    with open(text_file, "r") as f:
        text = f.read().strip()

        fig, axes = plt.subplots(1, 2, figsize=(10, 5))

        axes[0].imshow(img)
        axes[0].axis("off")
        axes[0].set_title("Heatmap")

        axes[1].axis("off")
        axes[1].text(
            0,
            0.5,
            text,
            wrap=True,
            fontsize=10,
            verticalalignment="center",
            horizontalalignment="left",
        )
        axes[1].set_title("Explanation")

        plt.tight_layout()
        plt.show()
