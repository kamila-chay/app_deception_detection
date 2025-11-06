from transformers import AutoProcessor, AutoModelForImageTextToText
import torch
from dataset_dolos import DolosDataset
from pathlib import Path
from torch.utils.data import DataLoader

MODEL_PATH = "llava-hf/llava-1.5-7b-hf"
processor = AutoProcessor.from_pretrained(MODEL_PATH, use_fast=True)
model = AutoModelForImageTextToText.from_pretrained(MODEL_PATH, dtype=torch.bfloat16).to("cuda")

train_dataset = DolosDataset(f"data/train_fold1.csv", Path("./data"))
train_dataloader = DataLoader(train_dataset, 1, shuffle=False, 
                              collate_fn=lambda batch: ([sample[0] for sample in batch], [sample[1] for sample in batch]))


for x, y in train_dataloader:
    inputs = processor.apply_chat_template(
        x,
        num_frames=16,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        padding=True
    )

    inputs = {k: v.to(model.device, dtype=torch.bfloat16) if torch.is_floating_point(v) else v.to(model.device) for k, v in inputs.items()}
    generated_ids = model.generate(**inputs, max_new_tokens=1000, do_sample=False)
    generated_ids_trimmed = generated_ids[:, inputs["input_ids"].shape[1]:]
    generated_text_trimmed = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )

    for text in generated_text_trimmed:
        print(text)