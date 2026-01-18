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
    classification_template_part1,
    classification_template_part2,
)
from thesis.utils.dataset_dolos import DolosDataset
from thesis.utils.utils import set_seed

set_seed(42)
logging.set_verbosity_error()

DEFAULT_BATCH_SIZE = 8

timestamp = "2025-11-17_01-23"

MODEL_PATH = "facebook/Perception-LM-1B"

processor = AutoProcessor.from_pretrained(MODEL_PATH, use_fast=True)

scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
client = OpenAI()

for split_id in range(1, 2):
    print(f"Split id: {split_id}")

    val_dataset = DolosDataset(
        f"thesis/data/val_fold{split_id}.csv",
        Path("thesis/data"),
        "joint_configuration_reasoning_labels",
    )
    val_dataloader = DataLoader(
        val_dataset,
        DEFAULT_BATCH_SIZE,
        shuffle=False,
        collate_fn=lambda batch: (
            [sample[0] for sample in batch],
            [sample[1] for sample in batch],
        ),
    )

    all_rouge_scores = []
    all_acc = []
    all_precision = []
    all_recall = []
    all_f1 = []

    for epoch in range(3):
        print(f"Epoch: {epoch}")
        base = AutoModelForImageTextToText.from_pretrained(
            MODEL_PATH, torch_dtype=torch.bfloat16
        )
        model = PeftModel.from_pretrained(
            base, f"thesis/out/{timestamp}/model_split{split_id}_epoch{epoch}"
        )
        model = model.to("cuda:0").eval()
        all_rouge_scores_this_epoch = []
        all_gt_this_epoch = []
        all_pred_this_epoch = []
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

            for pred, ref in zip(generated_text_trimmed, expected_text_trimmed):
                classification_prompt = (
                    classification_template_part1
                    + pred
                    + classification_template_part2
                    + ref
                )
                response = client.responses.create(
                    model="gpt-4.1-mini",
                    input=classification_prompt,
                    top_p=1,
                    temperature=0,
                ).output_text
                try:
                    predicted, gt = response.split(",")
                    if predicted.replace("Text 1: ", "").lower().strip() == "deception":
                        predicted = 1
                    elif predicted.replace("Text 1: ", "").lower().strip() == "truth":
                        predicted = 0
                    else:
                        raise ValueError()
                    if gt.replace("Text 2: ", "").lower().strip() == "deception":
                        gt = 1
                    elif gt.replace("Text 2: ", "").lower().strip() == "truth":
                        gt = 0
                    else:
                        raise ValueError()
                    all_pred_this_epoch.append(predicted)
                    all_gt_this_epoch.append(gt)
                except ValueError:
                    print(f"WARNING: incorrect response formatting: {response}")

                rouge_score = scorer.score(ref, pred)
                all_rouge_scores_this_epoch.append(
                    np.mean(
                        [
                            rouge_score["rouge1"].fmeasure,
                            rouge_score["rouge2"].fmeasure,
                            rouge_score["rougeL"].fmeasure,
                        ]
                    )
                )

        all_rouge_scores.append(np.mean(all_rouge_scores_this_epoch))

        all_gt_this_epoch = np.array(all_gt_this_epoch)
        all_pred_this_epoch = np.array(all_pred_this_epoch)
        acc = (all_gt_this_epoch == all_pred_this_epoch).sum() / all_gt_this_epoch.size
        tp = ((all_gt_this_epoch == 1) & (all_pred_this_epoch == 1)).sum()
        fp = ((all_gt_this_epoch == 0) & (all_pred_this_epoch == 1)).sum()
        fn = ((all_gt_this_epoch == 1) & (all_pred_this_epoch == 0)).sum()

        precision = tp / (tp + fp) if (tp + fp) > 0.0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0.0 else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0.0
            else 0.0
        )

        all_acc.append(acc)
        all_precision.append(precision)
        all_recall.append(recall)
        all_f1.append(f1)

    with open(
        f"thesis/out/{timestamp}/model_split{split_id}_validation_metrics.json",
        "w",
    ) as f:
        json.dump(
            {
                "ROUGE": all_rouge_scores,
                "Precision": all_precision,
                "Recall": all_recall,
                "F1": all_f1,
                "Accuracy": all_acc,
            },
            f,
        )
### add loss calculations???
