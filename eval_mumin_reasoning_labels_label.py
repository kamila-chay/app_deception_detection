import random
from pathlib import Path

import pandas as pd
from openai import OpenAI

client = OpenAI()

LABELS_DIR = Path("./data/mumin_reasoning_labels")
EVAL_DIR = Path("./out/mumin_reasoning_labels_eval/label")
EVAL_DIR.mkdir(parents=True, exist_ok=True)
df = pd.read_excel("./data/traits.xlsx")

instruction_1 = "Read the following text and choose what side it leans: deception or truthfulness. You have to choose one, even if there are multiple arguments for both sides, infer what the text leans towards from the perspecitive of a human reader. Text:\n"

instruction_2 = "(End of text)\n\nOutput your conclusion as one word and check if the conclusion is **"

instruction_3 = "**. If yes, score it as 1, else 0 points."

for i, row in df.iterrows():
    if random.randint(1, 10) == 6:
        curr = LABELS_DIR / f"{row['Filename']}.txt"
        label = row["Label"]

        with open(curr, "r") as gen:
            generated = gen.read()

        total_string = instruction_1 + generated + instruction_2 + label + instruction_3

        response = client.responses.create(model="gpt-4.1-mini", input=total_string)

        with open(EVAL_DIR / f"{row['Filename']}.txt", "w") as f:
            f.write(response.output_text)

# make it switchable too
total = 0
n = 0

for file in EVAL_DIR.iterdir():
    with open(file, "r") as f:
        n += 1
        text = f.read()
        try:
            total += float(text.split("Score:")[1])

        except (ValueError, IndexError):
            try:
                total += float(text.split("\n")[-1])
            except (ValueError, IndexError):
                print(file.stem)

print(total / n)
