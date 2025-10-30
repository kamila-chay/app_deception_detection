import torch
import pandas as pd
import copy
from transformers import AutoModelForCausalLM, AutoTokenizer
from pathlib import Path
import json

out = Path("./data/gen_labels")

model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-30B-A3B-Thinking-2507", device_map="cuda", dtype="auto", attn_implementation="flash_attention_2"
) # write how another one was used, i think VL 8B and it was a lot worse!
model.eval()
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-30B-A3B-Thinking-2507")

df = pd.read_excel("./data/traits.xlsx")

traits_dict_example = {
    'Label': None,
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

valid_indices = list(range(200, 500)) + list(range(700, 1000)) + list(range(1200, 1529))

list_for_tokenizer = []
list_for_filenames = []

for i, row in df.iterrows(): # 32 per batch
    if i in valid_indices:
        print(f"Row{i}")
        traits_dict = copy.deepcopy(traits_dict_example)
        for key in traits_dict:
            try:
                if key == "Label":
                    traits_dict[key] = row[key].strip()
                else:
                    traits_dict[key] = bool(int(row[key]))
            except:
                print("Exception while filling one field")
        
        messages = [
            {
                "role": "user",
                "content": f"You are given a structured, accurate, human-written description of someone's behavior in a video. Based on external, reliable information, the clip is labeled as {traits_dict['Label']}. Imagine you don't see the description or the label, just the video file instead. Write a text where you assess the person's behavior and estimate whether or not they are lying. Reason about both possibilities and make it seem like you deducted the correct label on your own using the video. You shouldn't sound too confident too, try to make it seem like you're making an educated guess instead of claiming you know if they're lying or not. Include counterarguments as well. Make the response sound natural as a reply to the question 'Is this person lying or telling the truth and why?'. Make sure that your reasoning is aligned with how humans would approach a question like that, avoid lists or bullet points. Description (True - the behavior occured, False - it didn't occur): {repr(traits_dict)}"
            }
        ]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        list_for_tokenizer.append(text)
        list_for_filenames.append(row["Filename"])

        if len(list_for_tokenizer) == 32:
            model_inputs = tokenizer(list_for_tokenizer, return_tensors="pt", padding=True, padding_side="left").to(model.device)
            with torch.inference_mode():
                generated_ids = model.generate(**model_inputs, max_new_tokens=32768)
                for j in range(32):
                    output_ids = generated_ids[j][len(model_inputs.input_ids[0]):].tolist() 

                    try:
                        index = len(output_ids) - output_ids[::-1].index(151668)
                    except ValueError:
                        index = 0

                    thinking_content = tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
                    content = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")

                    with open(out / f'{list_for_filenames[j]}.txt', "w") as f:
                        f.write(content)
                    with open(out / f'{list_for_filenames[j]}_thinking.txt', "w") as f:
                        f.write(thinking_content)
                        
            list_for_tokenizer = []
            list_for_filenames = []

print("Done")
