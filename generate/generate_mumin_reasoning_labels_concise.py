import copy
import logging
from pathlib import Path
from typing import List

import pandas as pd
from openai import OpenAI
from time import sleep

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

ALL_RELEVANT_TRAITS : List[str] = [
    "Smile",
    "Laughter",
    "Scowl",
    "Frowning eyebrows",
    "Raising eyebrows",
    "Frequent blinking",
    "Exaggerated eye opening",
    "Exaggerated eye closing",
    "Exaggerated mouth opening",
    "Mouth closed (awkward pauses)",
    "Lip corners up",
    "Lip corners down",
    "Lips protruded",
    "Lips retracted",
    "Chin moving upwards",
    "Chin moving downwards",
    "Gaze towards interlocutor",
    "Gaze up",
    "Gaze down",
    "Gaze sideways",
    "Nod",
    "Head moving forward",
    "Head moving backward",
    "Head tilting",
    "Head side-turn",
    "Head shaking",
    "Head up",
    "Hands rubbing",
    "Hands scratching",
    "Hands touching head/body/etc",
    "Hands covering face/mouth/etc",
    "Crossing fingers",
    "Palms up",
    "Arms open",
    "Arms folded",
    "Shrugs",
]


# ============================ HELPER FUNCTIONS ============================


def fill_traits(row: pd.Series) -> List[str]:
    """Fill the traits dictionary from a DataFrame row, converting to bools where needed."""
    traits = list()
    traits.append("telling the truth" if str(row["Label"]).strip().lower() == "truth" else "lying")
    for trait in ALL_RELEVANT_TRAITS:
        try:
            if bool(int(row[trait])):
                traits.append(trait)
        except (KeyError, AttributeError, ValueError, TypeError) as e:
            logging.warning(f"Row {row.name}: failed to parse '{trait}' ({e})")
    return traits


def make_prompt(traits: List[str]) -> str:
    """Make a chat prompt for the LLM."""
    message = ( f"Imagine you see a video of a person and you know that the person is {traits[0]}. "
        "On top of that you are given a list of behaviors that were exhibited by that person. "
        "Imagine you don't see the list or the label, just the video file instead. "
        "Write a text assessing the person's behavior and estimating whether or not "
        "they are lying. Reason about both possibilities and make it seem like you arrived at the correct "
        f"conclusion (that they are {traits[0]}). Don't sound overly confident—instead make it seem like " 
        "you're making an educated guess with counterarguments. Avoid lists or bullet points. "
        f"Output the text only. Use only clues given here, nothing else. The output should sound professional "
        "but not too formal. "
        f"List: {repr(traits[1:])}" )

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
