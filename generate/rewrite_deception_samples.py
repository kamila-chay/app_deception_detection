"""
Rewrite some reasoning labels generated Qwen3-30B-A3B-Thinking-2507 - they contained many
wrong conclusions in the deception category.
"""

from pathlib import Path

import pandas as pd
import torch
from openai import OpenAI

client = OpenAI()

LABELS_DIR = Path("./data/mumin_reasoning_labels")
df = pd.read_excel("./data/traits.xlsx")

with torch.inference_mode():
    for _, row in df.iterrows():
        if row["Label"].lower().strip() == "truth":
            continue
        file = LABELS_DIR / f"{row['Filename']}.txt"
        with open(file, "r") as f:
            prev_text = f.read()

        prompt = f"Read the following text. Rewrite it so that it reaches the conclusion that the person is probably lying - it shouldn't be too confident but also it should be stated that you lean towards deception. Use the exising cues only and make it reasonable. Only output the rewritten text. No bullet points etc. Sometimes the input text might already lean towards deception, in that case don't change anything. TEXT: \n {prev_text}"

        new_text = client.responses.create(
            model="gpt-4.1-mini", input=prompt
        ).output_text

        with open(file, "w") as f:
            f.write(new_text)
