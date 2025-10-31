from pathlib import Path
import pandas as pd
import random
import copy
import time

from openai import OpenAI

client = OpenAI()

out = Path("./data/gen_labels")
eval_out = Path("./data/results_eval/cues")
eval_out.mkdir(parents=True, exist_ok=True)

df = pd.read_excel("./data/traits.xlsx")

traits_dict_example = {
    # 'Label': None,
    'Smile': None,
    'Laughter': None,
    'Scowl': None,
    'Frowning eyebrows': None,
    'Raising eyebrows': None,
    'Frequent blinking': None,
    'Exaggerated eye opening': None,
    'Exaggerated eye closing': None,
    'Exaggerated mouth opening': None,
    'Mouth closed (awkward pauses)': None,
    'Lip corners up': None,
    'Lip corners down': None,
    'Lips protruded': None,
    'Lips retracted': None,
    'Chin moving upwards': None,
    'Chin moving downwards': None,
    'Gaze towards interlocutor': None,
    'Gaze up': None,
    'Gaze down': None,
    'Gaze sideways': None,
    'Nod': None,
    'Head moving forward': None,
    'Head moving backward': None,
    'Head tilting': None,
    'Head side-turn': None,
    'Head shaking': None,
    'Head up': None,
    'Hands rubbing': None,
    'Hands scratching': None,
    'Hands touching head/body/etc': None,
    'Hands covering face/mouth/etc': None,
    'Crossing fingers': None,
    'Palms up': None,
    'Arms open': None,
    'Arms folded': None,
    'Shrugs': None,
}

query = "Look at this text describing someone's behavior and the list of behaviours marked as either occuring or not. Rate how well those two are aligned. The scale is from 0.00 (not aligned at all) through 0.1, 0.2, etc. to 1.00 (perfectly aligned). Please be strict. The text doesn't have to mention every word from the list to get the maximum number of points but it should never claim that a behavior occured when the list says it didn't and shouldn't claim that a behviour didn't occur when the list says it did. Output the score only. Text:\n\n"

for i, row in df.iterrows():
    if random.randint(1, 10) == 6:
        curr = out / f"{row["Filename"]}.txt"
        traits_dict = copy.deepcopy(traits_dict_example)

        for key in traits_dict:
            traits_dict[key] = bool(int(row[key]))

        with open(curr, "r") as gen:
            generated = gen.read()

        total_string = query + generated + "\nList: " + repr(traits_dict)

        response = client.responses.create(
            model="gpt-4.1-mini",
            input=total_string
        )

        with open(eval_out / f"{row['Filename']}.txt", "w") as f:
            f.write(response.output_text)
