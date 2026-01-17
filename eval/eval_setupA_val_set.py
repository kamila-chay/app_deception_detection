# Developed as part of a BSc thesis at the Faculty of Computer Science, Bialystok Univesity of Technology

import json
from pathlib import Path

import numpy as np
import torch
from openai import OpenAI
from peft import PeftModel
from torch.utils.data import DataLoader
from transformers import AutoModelForImageTextToText, AutoProcessor, logging

from thesis.utils.constants import ALL_RELEVANT_TRAITS
from thesis.utils.dataset_dolos import DolosDataset
from thesis.utils.utils import set_seed

set_seed(42)
logging.set_verbosity_error()

DEFAULT_BATCH_SIZE = 8

timestamp = "2025-11-15_20-01"

MODEL_PATH = "facebook/Perception-LM-1B"

processor = AutoProcessor.from_pretrained(MODEL_PATH, use_fast=True)

client = OpenAI()

prompt_cue_f1 = f"Please read the text below. Look for behaviors that are mentioned in the text from the following list: {repr(ALL_RELEVANT_TRAITS)}. Output those using the same exact wording as in the list, one per line. Don't ouput anything else. \n\nText:\n"

prompt_reasoning_overlap_p1 = "Read those 2 texts describing the behavior of the same person and how it can be interpreted as a cue to deception/truthfulness. Score the logical overlap between those texts, you should pay attention to both the cues themselves and how they are interpreted and reasoned about. The score should be lower if e.g one of the texts focuses just on one interpretation of a specific cue etc. The score should be anywhere between 0.0 and 1.0 (both inclusive). Output the score only, nothing else. \n\nTEXT 1:\n"

prompt_reasoning_overlap_p2 = "\n\nTEXT 2:\n"

for split_id in range(1, 4):
    print(f"Split id: {split_id}")

    val_dataset = DolosDataset(
        f"thesis/data/val_fold{split_id}.csv",
        Path("thesis/data"),
        "joint_configuration_reasoning_labels",
    )

    val_dataset.include_raw_clues_(True)
    val_dataloader = DataLoader(
        val_dataset,
        DEFAULT_BATCH_SIZE,
        shuffle=False,
        collate_fn=lambda batch: (
            [sample[0] for sample in batch],
            [sample[1] for sample in batch],
            [sample[2] for sample in batch],
        ),
    )

    all_SoftOverlap_scores = []
    all_Cue_F1_scores = []
    all_mean_reasoning_scores = []

    for epoch in range(10):
        print(f"Epoch: {epoch}")
        base = AutoModelForImageTextToText.from_pretrained(
            MODEL_PATH, torch_dtype=torch.bfloat16
        )
        checkpoint = f"thesis/out/{timestamp}/model_split{split_id}_epoch{epoch}"
        model = PeftModel.from_pretrained(base, checkpoint)
        model = model.to("cuda:0").eval()

        all_SoftOverlap_scores_per_epoch = []
        all_Cue_F1_scores_per_epoch = []
        for i, (X, Y, raw_cues) in enumerate(val_dataloader):
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
                generated_ids = model.generate(
                    **inputs, max_new_tokens=1000, do_sample=False
                )
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

            for pred, ref, raw_clues_per_sample in zip(
                generated_text_trimmed, expected_text_trimmed, raw_cues
            ):
                full_prompt = prompt_cue_f1 + pred
                try:
                    response = None
                    response = client.responses.create(
                        model="gpt-4.1-mini", input=full_prompt, top_p=1, temperature=0
                    ).output_text

                    pred_cues = list(
                        map(
                            lambda z: z.strip(),
                            filter(lambda x: len(x) > 0, response.split("\n")),
                        )
                    )
                    init_len = len(pred_cues)
                    pred_cues = [cue for cue in pred_cues if cue in ALL_RELEVANT_TRAITS]
                    if diff := init_len - len(pred_cues):
                        print(
                            f"WARNING: {diff} cues were output that don't match ALL_RELEVANT_TRAITS"
                        )

                    pred_cues = set(pred_cues)
                    raw_clues_per_sample = set(raw_clues_per_sample)
                    intersection = pred_cues & raw_clues_per_sample
                    precision = (
                        len(intersection) / len(pred_cues)
                        if len(pred_cues) > 0
                        else 0.0
                    )
                    recall = (
                        len(intersection) / len(raw_clues_per_sample)
                        if len(raw_clues_per_sample) > 0
                        else 0.0
                    )

                    f1_for_cues = (
                        2 * precision * recall / (precision + recall)
                        if (precision + recall) > 0.0
                        else 0.0
                    )
                    all_Cue_F1_scores_per_epoch.append(f1_for_cues)
                except Exception:
                    print(f"ERROR: Incorrect response formatting: {response}")

                full_prompt = (
                    prompt_reasoning_overlap_p1
                    + pred
                    + prompt_reasoning_overlap_p2
                    + ref
                )
                try:
                    response = None
                    response = client.responses.create(
                        model="gpt-4.1-mini", input=full_prompt, top_p=1, temperature=0
                    ).output_text

                    score = float(response)
                    all_SoftOverlap_scores_per_epoch.append(score)
                except Exception:
                    print(f"ERROR: Incorrect response formatting: {response}")

        all_Cue_F1_scores.append(np.mean(all_Cue_F1_scores_per_epoch))
        all_SoftOverlap_scores.append(np.mean(all_SoftOverlap_scores_per_epoch))
        all_mean_reasoning_scores.append(
            (all_Cue_F1_scores[-1] + all_SoftOverlap_scores[-1]) / 2
        )

        print(all_mean_reasoning_scores)

    with open(
        f"thesis/out/{timestamp}/model_split{split_id}_validation_only_info_reasoning_only.json",
        "w",
    ) as f:
        json.dump(
            {
                "cue-f1": all_Cue_F1_scores,
                "SO": all_SoftOverlap_scores,
                "mean_reasoning": all_mean_reasoning_scores,
            },
            f,
        )
