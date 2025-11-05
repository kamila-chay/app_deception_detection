from transformers import AutoProcessor, AutoModelForImageTextToText
import torch
from dataset_dolos import DolosDataset
from pathlib import Path
from torch.utils.data import DataLoader
import json
from transformers import logging
logging.set_verbosity_error()
from peft import PeftModel

DEFAULT_BATCH_SIZE = 8

timestamp = "2025-11-05_13-36"

MODEL_PATH = "facebook/Perception-LM-1B"
NUM_EPOCHS = 10

processor = AutoProcessor.from_pretrained(MODEL_PATH, use_fast=True)
base = AutoModelForImageTextToText.from_pretrained(MODEL_PATH, torch_dtype=torch.bfloat16)

for split_id in range(1, 4):
    print(f"Split id: {split_id}")
    
    val_dataset = DolosDataset(f"data/val_fold{split_id}.csv", Path("./data"))
    val_dataloader = DataLoader(val_dataset, DEFAULT_BATCH_SIZE, shuffle=False, 
                                collate_fn=lambda batch: ([sample[0] for sample in batch], [sample[1] for sample in batch]))
    
    for epoch in range(NUM_EPOCHS):
        print(f"Epoch: {epoch}")
        model = PeftModel.from_pretrained(base, f"out/{timestamp}/model_split{split_id}_epoch{epoch}")
        model = model.to("cuda:0").eval()
        all_scores = []
        for X, Y in val_dataloader:
            X = processor.apply_chat_template(
                X,
                num_frames=16,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
                padding=True
            )
            Y = processor.apply_chat_template(
                Y,
                num_frames=16,
                add_generation_prompt=False,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
                padding=True
            )
            inputs = {k: v.to("cuda:0", dtype=torch.bfloat16) if torch.is_floating_point(v) else v.to("cuda:0") for k, v in X.items()}
            with torch.inference_mode():
                generated_ids = model.generate(**inputs, max_new_tokens=1000)
            generated_ids_trimmed = generated_ids[:, inputs["input_ids"].shape[1]:]
            expected_ids = Y["input_ids"]
            expected_ids_trimmed = expected_ids[:, inputs["input_ids"].shape[1]:]
            generated_text_trimmed = processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            expected_text_trimmed = processor.batch_decode(
                expected_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )

            print(generated_text_trimmed)
            print("------------------------------Now expected: ")
            print(expected_text_trimmed)
            print("------------------------")
        
            for pred, ref in zip(generated_text_trimmed, expected_text_trimmed):
                
                all_scores.append(1)
        with open(f"out/{timestamp}/model_split{split_id}_info.json", "w") as f:
            json.dump({"best_epoch": epoch, 
                    "val_score": all_scores}, # to change 
                    f)
