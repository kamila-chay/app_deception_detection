import torch
import pandas as pd
import copy
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from pathlib import Path
import json

out = Path("./data/gen_labels")

device = "cuda" if torch.cuda.is_available() else "cpu"

model = Qwen3VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen3-VL-8B-Instruct", dtype=torch.bfloat16, device_map=device, attn_implementation="flash_attention_2"
)
model.eval()
processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-8B-Instruct")

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

for _, row in df.iterrows():
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
            "content": [
                {"type": "text", "text": f"You are given a structured, accurate, human-written description of someone's behavior in a video. Based on external, reliable information, the clip is labeled as {traits_dict['Label']}. Imagine you don't see the description or the label, just the video file instead. Write a text where you assess the person's behavior and estimate whether or not they are lying. Reason about both possibilities and make it seem like you deducted the correct label on your own using the video. You shouldn't sound too confident too, try to make it seem like you're making an educated guess instead of claiming you know if they're lying or not. Include counterarguments as well. Make the response sound natural as a reply to the question 'Is this person lying or telling the truth and why?'. Description (True - the behavior occured, False - it didn't occur): {repr(traits_dict)}"},
            ],
        }
    ]
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt"
    )
    inputs = inputs.to(model.device)
    with torch.inference_mode():
        generated_ids = model.generate(inputs.input_ids, max_new_tokens=1000)
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text_trimmed = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
    with open(out / f'{row["Filename"]}.txt', "w") as f:
        f.write(output_text_trimmed)

    with open(out / "json_labels" / f'{row["Filename"]}.json', "w") as f:
        json.dump(traits_dict, f)
