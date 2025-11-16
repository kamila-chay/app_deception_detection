import logging
from pathlib import Path
from typing import List

import pandas as pd
import json

from thesis.utils.constants import ALL_RELEVANT_TRAITS

# ============================ CONFIGURATION ============================

DATA_PATH = Path("thesis/data/traits.xlsx")
OUTPUT_DIR = Path("thesis/data/mumin_reasoning_labels_concise")

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)

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


# ============================ MAIN LOOP ============================

for i, row in df.iterrows():
    logging.info(f"Row {i}...")
    traits = fill_traits(row)
    filename = row["Filename"]

    with open(OUTPUT_DIR / f"{filename}_raw_cues.json", "w") as f:
        json.dump(traits, f)

logging.info("✅ Generation complete.")
