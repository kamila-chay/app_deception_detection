import json
from pathlib import Path

import torch
from peft import PeftModel
from torch.utils.data import DataLoader
from transformers import AutoModelForImageTextToText, AutoProcessor, logging

from thesis.utils.dataset_dolos import DolosDataset

logging.set_verbosity_error()

DEFAULT_BATCH_SIZE = 8

timestamp = "2025-11-08_00-10"

MODEL_PATH = "facebook/Perception-LM-1B"

processor = AutoProcessor.from_pretrained(MODEL_PATH, use_fast=True)
base = AutoModelForImageTextToText.from_pretrained(
    MODEL_PATH, torch_dtype=torch.bfloat16
)

for split_id in range(1, 2):
    print(f"Split id: {split_id}")

    train_dataset = DolosDataset(
        f"thesis/data/train_fold{split_id}.csv", Path("thesis/data")
    )
    train_dataloader = DataLoader(
        train_dataset,
        DEFAULT_BATCH_SIZE,
        shuffle=True,
        collate_fn=lambda batch: (
            [sample[0] for sample in batch],
            [sample[1] for sample in batch],
        ),
    )

    val_dataset = DolosDataset(
        f"thesis/data/val_fold{split_id}.csv", Path("thesis/data")
    )
    val_dataloader = DataLoader(
        val_dataset,
        DEFAULT_BATCH_SIZE,
        shuffle=True,
        collate_fn=lambda batch: (
            [sample[0] for sample in batch],
            [sample[1] for sample in batch],
        ),
    )

    for epoch in range(2):
        print(f"Epoch: {epoch}")
        model = PeftModel.from_pretrained(
            base, f"thesis/out/{timestamp}/model_split{split_id}_epoch{epoch}"
        )
        model = model.to("cuda:0").eval()
        print("*********Train set**********")
        for i, (X, Y) in enumerate(train_dataloader):
            if i == 1: # only 8 random samples
                break
            X = processor.apply_chat_template(
                X,
                num_frames=16,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
                padding=True,
            )
            Y = processor.apply_chat_template(
                Y,
                num_frames=16,
                add_generation_prompt=False,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
                padding=True,
            )
            inputs = {
                k: (
                    v.to("cuda:0", dtype=torch.bfloat16)
                    if torch.is_floating_point(v)
                    else v.to("cuda:0")
                )
                for k, v in X.items()
            }
            with torch.inference_mode():
                generated_ids = model.generate(**inputs, 
                                               max_new_tokens=1000, 
                                               do_sample=True,
                                               top_k=10,
                                               repetition_penalty=1.2,
                                               no_repeat_ngram_size=3)
            generated_ids_trimmed = generated_ids[:, inputs["input_ids"].shape[1] :]
            expected_ids = Y["input_ids"]
            expected_ids_trimmed = expected_ids[:, inputs["input_ids"].shape[1] :]
            generated_text_trimmed = processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            expected_text_trimmed = processor.batch_decode(
                expected_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )

            for pred, ref in zip(generated_text_trimmed, expected_text_trimmed):
                print("   Generated:")
                print(pred)
                print("   Gold:")
                print(ref)

        print("*********Val set**********")

        for i, (X, Y) in enumerate(val_dataloader):
            if i == 1: # only 8 random samples
                break
            X = processor.apply_chat_template(
                X,
                num_frames=16,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
                padding=True,
            )
            Y = processor.apply_chat_template(
                Y,
                num_frames=16,
                add_generation_prompt=False,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
                padding=True,
            )
            inputs = {
                k: (
                    v.to("cuda:0", dtype=torch.bfloat16)
                    if torch.is_floating_point(v)
                    else v.to("cuda:0")
                )
                for k, v in X.items()
            }
            with torch.inference_mode():
                generated_ids = model.generate(**inputs, 
                                               max_new_tokens=1000,
                                               do_sample=True,
                                               top_k=10, 
                                               repetition_penalty=1.2,
                                               no_repeat_ngram_size=3)
            generated_ids_trimmed = generated_ids[:, inputs["input_ids"].shape[1] :]
            expected_ids = Y["input_ids"]
            expected_ids_trimmed = expected_ids[:, inputs["input_ids"].shape[1] :]
            generated_text_trimmed = processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            expected_text_trimmed = processor.batch_decode(
                expected_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )

            for pred, ref in zip(generated_text_trimmed, expected_text_trimmed):
                print("   Generated:")
                print(pred)
                print("   Gold:")
                print(ref)
