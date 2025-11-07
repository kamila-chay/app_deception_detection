from pathlib import Path

import matplotlib.pyplot as plt
import torch
from peft import PeftModel
from torch.utils.data import DataLoader
from transformers import (
    AutoImageProcessor,
    TimesformerConfig,
    TimesformerForVideoClassification,
    pipeline,
)

from utils.dataset_dolos import DolosClassificationDataset
from utils.utils import overlay_attention, roll_out_attn_map, set_seed

set_seed(42)

BATCH_SIZE = 1
EPOCHS = 20

Path("data/attn_rollout_reasoning_labels/heatmaps").mkdir(parents=True, exist_ok=True)

timestamp = "2025-11-03_00-45"

processor = AutoImageProcessor.from_pretrained(
    "facebook/timesformer-base-finetuned-k600"
)
config = TimesformerConfig()
config.num_labels = 2

pipe = pipeline("image-text-to-text", model="llava-hf/llava-1.5-7b-hf")

for split_id, epoch in ((2, 12), (1, 16), (3, 12)):
    test_dataset = DolosClassificationDataset(
        f"data/test_fold{split_id}.csv", "data/video", processor
    )
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    save_path_lora = f"out/{timestamp}/lora_timesformer_split{split_id}_epoch{epoch}"
    save_path = f"out/{timestamp}/timesformer_split{split_id}_epoch{epoch}.pt"

    model = TimesformerForVideoClassification.from_pretrained(
        "facebook/timesformer-base-finetuned-k600",
        config=config,
        ignore_mismatched_sizes=True,
    )
    lora_model = PeftModel.from_pretrained(model, save_path_lora)
    classifier_state_dict = torch.load(save_path, map_location="cpu")
    lora_model.base_model.model.classifier.load_state_dict(classifier_state_dict)
    lora_model = lora_model.to("cuda").to(torch.bfloat16)
    lora_model.eval()

    for (pixel_values, labels), name in test_dataset:
        pixel_values = pixel_values.to(lora_model.device).to(torch.bfloat16)
        if pixel_values.ndim == 4:
            pixel_values = pixel_values.unsqueeze(0)
        labels = labels.to(lora_model.device)

        with torch.inference_mode():
            output = lora_model(pixel_values=pixel_values, output_attentions=True)

        attn_map = roll_out_attn_map(
            [attention[1::2] for attention in output.attentions], 4, 14, 14
        )

        fig, axes = plt.subplots(2, 2, figsize=(8, 8))
        for i, ax in enumerate(axes.flat):
            ax.imshow(overlay_attention(attn_map, pixel_values[:, 1::2], processor, i))
            ax.axis("off")
        plt.savefig(f"data/attn_rollout_reasoning_labels/heatmaps/{name}.png")

        pred = torch.argmax(output.logits, dim=-1)
        del output, attn_map
        if torch.sum(pred == labels).item():
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "url": f"data/attn_rollout_reasoning_labels/heatmaps/{name}.png",
                        },
                        {
                            "type": "text",
                            "text": """You are given consecutive frames from a video.  
Certain regions are highlighted in **yellow**, **red**, and **black**, representing:
- **Yellow** – very important for the model’s decision  
- **Red** – somewhat important  
- **Black** – unimportant  

Focus **only** on patches located on the **face and body** of the person. Avoid focusing on ears or hair/clothing.

Write a **short, objective summary** describing **where and when** the model focused its attention across the frames.
- Compare different timestamps (e.g., “At the beginning… later… toward the end…”. Don't number frames. ).  
- Use **specific cues** rather than generic terms like “facial area” or “body region.”  
- Output **only** the summary, no extra text.  

**Example summary:**  
“At the beginning, attention is concentrated around the right eyebrow. In the middle and later frames, the model focuses more on the mouth and nose area.”""",
                        },
                    ],
                },
            ]

            out = pipe(text=messages, max_new_tokens=100)[0]["generated_text"][1][
                "content"
            ]
            print(out)
            print("==================")
            with open(f"data/attn_rollout_reasoning_labels/{name}.txt", "w") as f:
                f.write(out)
        else:
            print("NONE===========")
