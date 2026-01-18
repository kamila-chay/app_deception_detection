# Developed as part of a BSc thesis at the Faculty of Computer Science, Bialystok Univesity of Technology

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from openai import OpenAI
from rouge_score import rouge_scorer
from torch.utils.data import DataLoader
from transformers import AutoModelForImageTextToText, AutoProcessor, logging

from thesis.utils.constants import (
    ALL_RELEVANT_TRAITS,
    classification_template_part1,
    classification_template_part2,
    cue_f1_template,
    so_template_part1,
    so_template_part2,
)
from thesis.utils.dataset_dolos import DolosDataset
from thesis.utils.utils import set_seed

set_seed(42)
logging.set_verbosity_error()

DEFAULT_BATCH_SIZE = 8

timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
dir_path = Path(f"thesis/out/{timestamp}")
dir_path.mkdir(parents=True, exist_ok=True)

processor = AutoProcessor.from_pretrained("llava-hf/LLaVA-NeXT-Video-7B-hf")
model = AutoModelForImageTextToText.from_pretrained(
    "llava-hf/LLaVA-NeXT-Video-7B-hf", dtype=torch.bfloat16, device_map="cuda"
)

scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
client = OpenAI()

for split_id in range(1, 4):
    print(f"Split id: {split_id}")

    test_dataset = DolosDataset(
        f"thesis/data/test_fold{split_id}.csv",
        Path("thesis/data"),
        "joint_configuration_reasoning_labels",
    )

    test_dataset.include_raw_cues_(True)

    test_dataloader = DataLoader(
        test_dataset,
        DEFAULT_BATCH_SIZE,
        shuffle=False,
        collate_fn=lambda batch: (
            [sample[0] for sample in batch],
            [sample[1] for sample in batch],
            [sample[2] for sample in batch],
        ),
    )

    model = model.eval()

    soft_overlap_scores = []
    cue_f1_scores = []
    rouge_scores = []
    gts = []
    preds = []

    for X, Y, raw_cues in test_dataloader:
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
        Y = {
            k: (
                v.to("cuda:0", dtype=torch.bfloat16)
                if torch.is_floating_point(v)
                else v.to("cuda:0")
            )
            for k, v in Y.items()
        }
        with torch.no_grad():
            generated_ids = model.generate(
                **inputs, max_new_tokens=1000, do_sample=False
            )
        expected_ids = Y["input_ids"]
        generated_text = processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        expected_text = processor.batch_decode(
            expected_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

        for pred, ref, ref_cues in zip(generated_text, expected_text, raw_cues):
            pred = pred.split("ASSISTANT:")[1]
            ref = ref.split("ASSISTANT:")[1]
            cue_f1_prompt = cue_f1_template + pred
            try:
                response = None
                response = client.responses.create(
                    model="gpt-4.1-mini", input=cue_f1_prompt, top_p=1, temperature=0
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
                cue_precision = (
                    len(intersection) / len(pred_cues) if len(pred_cues) > 0 else 0.0
                )
                cue_recall = (
                    len(intersection) / len(ref_cues) if len(ref_cues) > 0 else 0.0
                )

                cue_f1 = (
                    2 * cue_precision * cue_recall / (cue_precision + cue_recall)
                    if (cue_precision + cue_recall) > 0.0
                    else 0.0
                )
                cue_f1_scores.append(cue_f1)
            except Exception:
                print(f"ERROR: Incorrect response formatting: {response}")

            so_prompt = so_template_part1 + pred + so_template_part2 + ref
            try:
                response = None
                response = client.responses.create(
                    model="gpt-4.1-mini", input=so_prompt, top_p=1, temperature=0
                ).output_text

                so_score = float(response)
                soft_overlap_scores.append(so_score)

            except Exception:
                print(f"ERROR: Incorrect response formatting: {response}")

            prompt_classification = (
                classification_template_part1
                + pred
                + classification_template_part2
                + ref
            )
            try:
                response = None
                response = client.responses.create(
                    model="gpt-4.1-mini",
                    input=prompt_classification,
                    temperature=0,
                    top_p=1,
                ).output_text
                pred, gt = response.split(",")
                if pred.replace("Text 1: ", "").lower().strip() == "deception":
                    pred = 1
                elif pred.replace("Text 1: ", "").lower().strip() == "truth":
                    pred = 0
                else:
                    raise ValueError()
                if gt.replace("Text 2: ", "").lower().strip() == "deception":
                    gt = 1
                elif gt.replace("Text 2: ", "").lower().strip() == "truth":
                    gt = 0
                else:
                    raise ValueError()
                preds.append(pred)
                gts.append(gt)
            except ValueError:
                print(f"ERROR: Incorrect response formatting: {response}")

            rouge_score = scorer.score(ref, pred)
            rouge_scores.append(
                np.mean(
                    [
                        rouge_score["rouge1"].fmeasure,
                        rouge_score["rouge2"].fmeasure,
                        rouge_score["rougeL"].fmeasure,
                    ]
                )
            )

    gts = np.array(gts)
    preds = np.array(preds)
    acc = (gts == preds).sum() / gts.size
    tp = ((gts == 1) & (preds == 1)).sum()
    fp = ((gts == 0) & (preds == 1)).sum()
    fn = ((gts == 1) & (preds == 0)).sum()

    precision = tp / (tp + fp) if (tp + fp) > 0.0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0.0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0.0
        else 0.0
    )

    with open(
        f"thesis/out/{timestamp}/model_split{split_id}_test_metrics.json", "w"
    ) as f:
        json.dump(
            {
                "ROUGE": np.mean(rouge_scores),
                "Accuracy": acc,
                "Precision": precision,
                "Recall": recall,
                "F1": f1,
                "Cue-F1": np.mean(cue_f1_scores),
                "SO": np.mean(soft_overlap_scores),
            },
            f,
        )
