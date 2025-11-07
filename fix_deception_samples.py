from pathlib import Path

import pandas as pd
import torch
from openai import OpenAI

client = OpenAI()

out = Path("./data/gen_labels")
out_new = Path("./data/gen_new_labels")  # should be fixed

out_new.mkdir(parents=True, exist_ok=True)

# MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"

# model = AutoModelForCausalLM.from_pretrained(MODEL_ID, device_map="cuda", dtype="bfloat16", attn_implementation="flash_attention_2")
# model.eval()

# tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

df = pd.read_excel("./data/traits.xlsx")

# count = 0
# row_batch = []
BATCH_SIZE = 2

counter = 0

with torch.inference_mode():
    for _, row in df.iterrows():
        if row["Label"] in {"truth ", "truth", "Truth"}:
            continue
        counter += 1
        if counter <= 146:
            continue
        prev_text_file = out / f"{row['Filename']}.txt"
        with open(prev_text_file, "r") as f:
            prev_text = f.read()

        prompt = f'Read the following text. Rewrite it so that it reaches the conclusion that the person is probably lying - it shouldn\'t be too confident but also it should be stated that you lean towards deception. Use the exising cues only and make it reasonable. Only output the rewritten text. No bullet points etc. Sometimes the input text might already lean towards deception. In this case, simply output "SAME". TEXT: \n {prev_text}'

        response = client.responses.create(model="gpt-4.1-mini", input=prompt)

        print(response.output_text)

        new_file = out_new / f"{row['Filename']}.txt"
        with open(new_file, "w") as f:
            f.write(response.output_text)

        # count += 1
        # row_batch.append(row["Filename"])
        # if count == BATCH_SIZE:
        #     texts = []
        #     for j in range(BATCH_SIZE):
        #         prev_text_file = out / f"{row_batch[j]}.txt"
        #         with open(prev_text_file, "r") as f:
        #             prev_text = f.read()

        #         prompt = f"Read the following text. Rewrite it so that it reaches the conclusion that the person is probably lying - it shouldn't be too confident but also it should be stated that you lean towards deception. Use the exising cues only and make it reasonable. Only output the rewritten text. No bullet points etc. TEXT: \n {prev_text}"

        #         messages = [
        #             {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
        #             {"role": "user", "content": prompt}
        #         ]

        #         text = tokenizer.apply_chat_template(
        #             messages,
        #             tokenize=False,
        #             add_generation_prompt=True
        #         )
        #         texts.append(text)
        #     model_inputs = tokenizer(texts, return_tensors="pt", padding=True, padding_side="left").to(model.device)

        #     generated_ids = model.generate(
        #         **model_inputs,
        #         max_new_tokens=4000
        #     )

        #     generated_ids = [x[model_inputs.input_ids.shape[1]:] for x in generated_ids]

        #     responses = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        #     for i, response in enumerate(responses):
        #         new_file = out_new / f"{row_batch[i]}.txt"
        #         with open(new_file, "w") as f:
        #             f.write(response)
        #     count = 0
        #     row_batch = []
