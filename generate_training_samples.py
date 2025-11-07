"""
Generate behavioral deception assessments from structured MUMIN trait data
using Qwen3-30B-A3B-Thinking-2507.
"""

import copy
import logging
from pathlib import Path
from typing import Dict, List

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# ============================ CONFIGURATION ============================

MODEL_NAME = "Qwen/Qwen3-30B-A3B-Thinking-2507"
DATA_PATH = Path("./data/traits.xlsx")
OUTPUT_DIR = Path("./data/mumin_reasoning_labels")
BATCH_SIZE = 32
MAX_NEW_TOKENS = 32768
THINKING_TOKEN_ID = 151668  # Marks end of "thinking" section

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)

# ============================ MODEL & TOKENIZER ============================

logging.info("Loading model and tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, device_map="cuda", dtype="auto"
)
model.eval()

# ============================ DATA LOADING ============================

logging.info(f"Loading data from {DATA_PATH}")
df = pd.read_excel(DATA_PATH)

TRAIT_TEMPLATE: Dict[str, bool | None] = {
    "Label": None,
    "Smile": None,
    "Laughter": None,
    "Scowl": None,
    "Frowning eyebrows": None,
    "Raising eyebrows": None,
    "Frequent blinking": None,
    "Exaggerated eye opening": None,
    "Exaggerated eye closing": None,
    "Exaggerated mouth opening": None,
    "Mouth closed (awkward pauses)": None,
    "Lip corners up": None,
    "Lip corners down": None,
    "Lips protruded": None,
    "Lips retracted": None,
    "Chin moving upwards": None,
    "Chin moving downwards": None,
    "Gaze towards interlocutor": None,
    "Gaze up": None,
    "Gaze down": None,
    "Gaze sideways": None,
    "Nod": None,
    "Head moving forward": None,
    "Head moving backward": None,
    "Head tilting": None,
    "Head side-turn": None,
    "Head shaking": None,
    "Head up": None,
    "Hands rubbing": None,
    "Hands scratching": None,
    "Hands touching head/body/etc": None,
    "Hands covering face/mouth/etc": None,
    "Crossing fingers": None,
    "Palms up": None,
    "Arms open": None,
    "Arms folded": None,
    "Shrugs": None,
}


# ============================ HELPER FUNCTIONS ============================


def fill_traits(row: pd.Series) -> Dict[str, bool | str | None]:
    """Fill the traits dictionary from a DataFrame row, converting to bools where needed."""
    traits = copy.deepcopy(TRAIT_TEMPLATE)
    for key in traits:
        try:
            if key == "Label":
                traits[key] = str(row[key]).strip()
            else:
                traits[key] = bool(int(row[key]))
        except (KeyError, AttributeError, ValueError, TypeError) as e:
            logging.warning(f"Row {row.name}: failed to parse '{key}' ({e})")
    return traits


def make_prompt(traits: Dict[str, bool | str | None]) -> str:
    """Format the chat prompt for the LLM."""
    messages = [
        {
            "role": "user",
            "content": (
                "You are given a structured, accurate, human-written description "
                "of someone's behavior in a video. Based on external reliable information, "
                f"the clip is labeled as {traits['Label']}. "
                "Imagine you don't see the description or the label, just the video file instead. "
                "Write a text where you assess the person's behavior and estimate whether or not "
                "they are lying. Reason about both possibilities and make it seem like you deduced "
                "the correct label on your own using the video. "
                "Don't sound overly confident—make it an educated guess with counterarguments. "
                "Avoid lists or bullet points. "
                f"Description (True = behavior occurred, False = it didn’t): {repr(traits)}"
            ),
        }
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def generate_batch(texts: List[str]) -> List[torch.Tensor]:
    """Generate model outputs for a batch of prompts."""
    inputs = tokenizer(
        texts, return_tensors="pt", padding=True, padding_side="left"
    ).to(model.device)
    with torch.inference_mode():
        return model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS)


def save_outputs(filenames: List[str], generated_ids: torch.Tensor, input_len: int):
    """Decode and save outputs to disk."""
    for name, output in zip(filenames, generated_ids):
        output_ids = output[input_len:].tolist()

        try:
            # Split thinking/content parts
            index = len(output_ids) - output_ids[::-1].index(THINKING_TOKEN_ID)
        except ValueError:
            index = 0

        thinking = tokenizer.decode(
            output_ids[:index], skip_special_tokens=True
        ).strip()
        content = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip()

        (OUTPUT_DIR / f"{name}.txt").write_text(content)
        (OUTPUT_DIR / f"{name}_thinking.txt").write_text(thinking)


# ============================ MAIN LOOP ============================


OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
messages, filenames = [], []

for i, row in df.iterrows():
    traits = fill_traits(row)
    messages.append(make_prompt(traits))
    filenames.append(row["Filename"])

    if len(messages) == BATCH_SIZE:
        logging.info(f"Generating batch ending at row {i}...")
        outputs = generate_batch(messages)
        save_outputs(
            filenames, outputs, input_len=len(tokenizer(messages[0])["input_ids"])
        )
        messages.clear()
        filenames.clear()

# Process any remaining items
if messages:
    logging.info("Generating final incomplete batch...")
    outputs = generate_batch(messages)
    save_outputs(filenames, outputs, input_len=len(tokenizer(messages[0])["input_ids"]))

logging.info("✅ Generation complete.")
