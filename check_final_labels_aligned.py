from pathlib import Path
import pandas as pd
import random
import copy
import time
from openai import OpenAI

client = OpenAI()

out = Path("./data/gen_labels")
eval_out = Path("./data/results_eval/final_label")
eval_out.mkdir(parents=True, exist_ok=True)
df = pd.read_excel("./data/traits.xlsx")

instruction_1 = "Read the following text and choose what side it leans: deception or truthfulness. You have to choose one, even if there are multiple arguments for both sides, infer what the text leans towards from the perspecitive of a human reader. Text:\n"

instruction_2 = "(End of text)\n\nOutput your conclusion as one word and check if the conclusion is **"

instruction_3 = "**. If yes, score it as 1, else 0 points."

for i, row in df.iterrows():
    if random.randint(1, 10) == 6:
        curr = out / f"{row["Filename"]}.txt"
        label = row["Label"]

        with open(curr, "r") as gen:
            generated = gen.read()

        total_string = instruction_1 + generated + instruction_2 + label + instruction_3

        response = client.responses.create(
            model="gpt-4.1-mini",
            input=total_string
        )

        with open(eval_out / f"{row['Filename']}.txt", "w") as f:
            f.write(response.output_text)
        