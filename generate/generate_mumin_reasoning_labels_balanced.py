import logging
from pathlib import Path
from typing import List

import pandas as pd
from openai import OpenAI

from thesis.utils.constants import ALL_RELEVANT_TRAITS

# ============================ CONFIGURATION ============================

DATA_PATH = Path("thesis/data/traits.xlsx")
OUTPUT_DIR = Path("thesis/data/mumin_reasoning_labels_balanced")
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


def fill_traits(row: pd.Series) -> List[str]:
    """Fill the traits dictionary from a DataFrame row, converting to bools where needed."""
    traits = list()
    for trait in ALL_RELEVANT_TRAITS:
        try:
            if bool(int(row[trait])):
                traits.append(trait)
        except (KeyError, AttributeError, ValueError, TypeError) as e:
            logging.warning(f"Row {row.name}: failed to parse '{trait}' ({e})")
    return traits


def make_prompt(traits: List[str]) -> str:
    message = f"Imagine you see a video of a person and your task is to reason about their behaviors in the context of spotting possible deception. Write a text that includes both arguments for and against them being truthful/deceptive - emphasize how each behavior could be interpreted in both ways depending on the bigger picture. Make the text neutral - both possibilities should be emphasized equally. The subject's visible behaviors are the following: {repr(traits)}. Avoid lists or bullet points. The output should sound professional and polished. Don't output anything else, just the text."

    return message

# ============================ MAIN LOOP ============================


OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

for i, row in df.iterrows():
    logging.info(f"Row {i}...")
    traits = fill_traits(row)
    message = make_prompt(traits)
    filename = row["Filename"]
    response = client.responses.create(model=MODEL_NAME, input=message).output_text

    with open(OUTPUT_DIR / f"{filename}.txt", "w") as f:
        f.write(response)

logging.info("✅ Generation complete.")
