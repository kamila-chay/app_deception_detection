import json
from pathlib import Path

import numpy as np
import torch
from peft import PeftModel
from rouge_score import rouge_scorer
from torch.utils.data import DataLoader
from transformers import AutoModelForImageTextToText, AutoProcessor, logging

from utils.dataset_dolos import DolosDataset

logging.set_verbosity_error()

DEFAULT_BATCH_SIZE = 8

timestamp = "2025-11-01_01-36"
dir_path = Path(f"out/{timestamp}")

MODEL_PATH = "facebook/Perception-LM-1B"
NUM_EPOCHS = 10

best_test_scores_per_split = []
scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
processor = AutoProcessor.from_pretrained(MODEL_PATH, use_fast=True)
base = AutoModelForImageTextToText.from_pretrained(
    MODEL_PATH, torch_dtype=torch.bfloat16
)

for split_id in range(1, 2):
    print(f"Split id: {split_id}")

    val_dataset = DolosDataset(f"thesis/data/val_fold{split_id}.csv", Path("./data"))
    val_dataloader = DataLoader(
        val_dataset,
        DEFAULT_BATCH_SIZE,
        shuffle=False,
        collate_fn=lambda batch: (
            [sample[0] for sample in batch],
            [sample[1] for sample in batch],
        ),
    )

    test_dataset = DolosDataset(f"thesis/data/test_fold{split_id}.csv", Path("./data"))
    test_dataloader = DataLoader(
        test_dataset,
        DEFAULT_BATCH_SIZE,
        shuffle=False,
        collate_fn=lambda batch: (
            [sample[0] for sample in batch],
            [sample[1] for sample in batch],
        ),
    )

    best_rouge_val_score = -float("inf")
    best_rouge_test_score = 0.0

    for epoch in range(NUM_EPOCHS):
        print(f"Epoch: {epoch}")
        model = PeftModel.from_pretrained(
            base, f"out/{timestamp}/model_split{split_id}_epoch{epoch}"
        )
        model = model.to("cuda:0").eval()
        all_scores = []
        with torch.inference_mode():
            for X, Y in val_dataloader:
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
                generated_ids = model.generate(**inputs, max_new_tokens=1000)
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
                    score = scorer.score(ref, pred)
                    all_scores.append(
                        np.mean(
                            [
                                score["rouge1"].fmeasure,
                                score["rouge2"].fmeasure,
                                score["rougeL"].fmeasure,
                            ]
                        )
                    )
            rouge_val_score = np.mean(all_scores)
            print(f"Rouge validation score: {rouge_val_score}")
            if rouge_val_score > best_rouge_val_score:
                best_rouge_val_score = rouge_val_score
                all_test_scores = []
                for X, Y in test_dataloader:
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
                    generated_ids = model.generate(**inputs, max_new_tokens=1000)
                    generated_ids_trimmed = generated_ids[
                        :, inputs["input_ids"].shape[1] :
                    ]
                    expected_ids = Y["input_ids"]
                    expected_ids_trimmed = expected_ids[
                        :, inputs["input_ids"].shape[1] :
                    ]
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
                        score = scorer.score(ref, pred)
                        all_test_scores.append(
                            np.mean(
                                [
                                    score["rouge1"].fmeasure,
                                    score["rouge2"].fmeasure,
                                    score["rougeL"].fmeasure,
                                ]
                            )
                        )
                best_rouge_test_score = np.mean(all_test_scores)
                print(
                    f"  Top val score for this split, the corresponding test score is {best_rouge_test_score}"
                )
                with open(f"out/{timestamp}/model_split{split_id}_info.json", "w") as f:
                    json.dump(
                        {
                            "best_epoch": epoch,
                            "val_rouge_score": best_rouge_val_score,
                            "test_rouge_score": best_rouge_test_score,
                        },
                        f,
                    )
