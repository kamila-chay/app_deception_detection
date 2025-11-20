import logging
from pathlib import Path
from typing import List

import pandas as pd
from openai import OpenAI

from thesis.utils.constants import ALL_RELEVANT_TRAITS

# ============================ CONFIGURATION ============================

DATA_PATH = Path("thesis/data/traits.xlsx")
OUTPUT_DIR = Path("thesis/data/mumin_reasoning_labels_concise")
MODEL_NAME = "gpt-4.1-mini"

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)

client = OpenAI()

# ============================ DATA LOADING ============================

logging.info(f"Loading data from {DATA_PATH}")
df = pd.read_excel(DATA_PATH)

# ============================ HELPER FUNCTIONS ============================



def make_prompt(old_text: str) -> str:
    """Make a chat prompt for the LLM."""
    message = f"Rewrite the following textual assessment so that the opposite conclusion is reached (e.g someone is truthful instead of lying). Try to keep the generated text as close to the original one, also layout-wise, but at the same time make sure that the new text is coherent. Both the old and new texts should contain the same behavioral cues, don't change them semantically, change only the final conclusion. Ouput the new text only. Old text: {old_text}"
    return message

# ============================ MAIN LOOP ============================

for i, row in df.iterrows():
    logging.info(f"Row {i}...")
    filename = row["Filename"]
    with open(OUTPUT_DIR / f"{filename}.txt", "r") as f:
        old_text = f.read()

    prompt = make_prompt(old_text)
    response = client.responses.create(model=MODEL_NAME, input=prompt).output_text

    with open(OUTPUT_DIR / f"{filename}_opposing.txt", "w") as f:
        f.write(response)

logging.info("✅ Generation complete.")
