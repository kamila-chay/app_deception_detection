# Developed as part of a BSc thesis at the Faculty of Computer Science, Bialystok Univesity of Technology

import json
from pathlib import Path

import numpy as np
import torch
from openai import OpenAI
from peft import PeftModel
from rouge_score import rouge_scorer
from torch.utils.data import DataLoader
from transformers import AutoModelForImageTextToText, AutoProcessor, logging

from thesis.utils.constants import (
    ALL_RELEVANT_TRAITS,
    cue_f1_template,
    so_template_part1,
    so_template_part2,
)
from thesis.utils.dataset_dolos import DolosDataset
from thesis.utils.utils import make_conversation_for_separate_configuration, set_seed

set_seed(42)
logging.set_verbosity_error()

DEFAULT_BATCH_SIZE = 8

timestamp = "2025-11-19_16-41"

MODEL_PATH = "facebook/Perception-LM-1B"

processor = AutoProcessor.from_pretrained(MODEL_PATH, use_fast=True)

scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
client = OpenAI()

for split_id in range(1, 4):
    print(f"Split id: {split_id}")

    val_dataset = DolosDataset(
        f"thesis/data/val_fold{split_id}.csv",
        Path("thesis/data"),
        label_folder="separate_configuration_reasoning_labels",
        conversation_making_func=make_conversation_for_separate_configuration,
    )

    val_dataset.include_raw_cues_(True)

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

    all_rouge_scores = []
    all_cue_f1_scores = []
    all_so_scores = []
    mean_so_cue_f1 = []

    for epoch in range(10):
        print(f"Epoch: {epoch}")
        base = AutoModelForImageTextToText.from_pretrained(
            MODEL_PATH, torch_dtype=torch.bfloat16
        )
        checkpoint = f"thesis/out/{timestamp}/model_split{split_id}_epoch{epoch}"
        model = PeftModel.from_pretrained(base, checkpoint)
        model = model.to("cuda:0").eval()

        rouge_scores_this_epoch = []
        cue_f1_scores_this_epoch = []
        so_scores_this_epoch = []
        for X, Y, raw_cues in val_dataloader:
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

            with torch.no_grad():
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

            for pred, ref, ref_cues in zip(
                generated_text_trimmed, expected_text_trimmed, raw_cues
            ):
                cue_f1_prompt = cue_f1_template + pred
                try:
                    response = client.responses.create(
                        model="gpt-4.1-mini",
                        input=cue_f1_prompt,
                        top_p=1,
                        temperature=0,
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
                    ref_cues = set(ref_cues)
                    intersection = pred_cues & ref_cues
                    precision = (
                        len(intersection) / len(pred_cues)
                        if len(pred_cues) > 0
                        else 0.0
                    )
                    recall = (
                        len(intersection) / len(ref_cues) if len(ref_cues) > 0 else 0.0
                    )

                    cue_f1 = (
                        2 * precision * recall / (precision + recall)
                        if (precision + recall) > 0.0
                        else 0.0
                    )
                    cue_f1_scores_this_epoch.append(cue_f1)
                except Exception as e:
                    print(
                        f"ERROR: Didn't process OpenAI API cue F1 ouput properly: {e}"
                    )

                so_prompt = so_template_part1 + pred + so_template_part2 + ref

                try:
                    response = client.responses.create(
                        model="gpt-4.1-mini", input=so_prompt, top_p=1, temperature=0
                    ).output_text

                    score = float(response)
                    so_scores_this_epoch.append(score)

                except Exception as e:
                    print(
                        f"ERROR: Didn't process OpenAI API cue overlap ouput properly: {e}"
                    )

                rouge_score = scorer.score(ref, pred)
                rouge_scores_this_epoch.append(
                    np.mean(
                        [
                            rouge_score["rouge1"].fmeasure,
                            rouge_score["rouge2"].fmeasure,
                            rouge_score["rougeL"].fmeasure,
                        ]
                    )
                )

        all_rouge_scores.append(np.mean(rouge_scores_this_epoch))
        all_cue_f1_scores.append(np.mean(cue_f1_scores_this_epoch))
        all_so_scores.append(np.mean(so_scores_this_epoch))
        mean_so_cue_f1.append(
            (np.mean(cue_f1_scores_this_epoch) + np.mean(so_scores_this_epoch)) / 2
        )

    with open(
        f"thesis/out/{timestamp}/model_split{split_id}_validation_metrics.json", "w"
    ) as f:
        json.dump(
            {
                "ROUGE": all_rouge_scores,
                "Cue-F1": all_cue_f1_scores,
                "SO": all_so_scores,
                "Mean of SO and Cue-F1": mean_so_cue_f1,
            },
            f,
        )
