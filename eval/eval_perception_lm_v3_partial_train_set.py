import json
from pathlib import Path

import numpy as np
import torch
from openai import OpenAI
from peft import PeftModel
from rouge_score import rouge_scorer
from torch.utils.data import DataLoader
from transformers import AutoModelForImageTextToText, AutoProcessor, logging

from thesis.utils.dataset_dolos import DolosDataset

logging.set_verbosity_error()

DEFAULT_BATCH_SIZE = 8

timestamp = "2025-11-08_00-10"

MODEL_PATH = "facebook/Perception-LM-1B"
NUM_EPOCHS = 10

processor = AutoProcessor.from_pretrained(MODEL_PATH, use_fast=True)
base = AutoModelForImageTextToText.from_pretrained(
    MODEL_PATH, torch_dtype=torch.bfloat16
)

scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
client = OpenAI()

prompt_1 = "Please read 2 texts describing human behavior. Output two scores based on them. The first one should be 1 if the texts agree on whether the person is lying or telling the truth. Please consider the main conclusion of each text, they will include various counterarguments but focus only on the final/dominating direction. Otherwise, if the texts disagree, output 0. Then rate how different behavioral cues are described and reasoned about. Check how much semantic overlap there is between those 2 texts in the context of those cues. The result should be in range 0.0(no overlap) through 0.1, 0.2 etc. to 1.0 (perfect overlap). Be strict. Only output those two numbers, starting from 0 or 1 and then a number in the range 0.0-1.0.\n\nText 1:\n"

prompt_2 = "\n\nText 2:\n"

for split_id in range(1, 2):
    print(f"Split id: {split_id}")

    train_dataset = DolosDataset(
        f"thesis/data/train_fold{split_id}.csv", Path("thesis/data")
    )
    train_dataloader = DataLoader(
        train_dataset,
        DEFAULT_BATCH_SIZE,
        shuffle=False,
        collate_fn=lambda batch: (
            [sample[0] for sample in batch],
            [sample[1] for sample in batch],
        ),
    )

    all_rouge_scores = []
    all_label_scores = []
    all_cue_scores = []

    for epoch in range(2):
        print(f"Epoch: {epoch}")
        model = PeftModel.from_pretrained(
            base, f"thesis/out/{timestamp}/model_split{split_id}_epoch{epoch}"
        )
        model = model.to("cuda:0").eval()
        all_rouge_scores_per_epoch = []
        all_label_scores_per_epoch = []
        all_cue_scores_per_epoch = []
        for i, (X, Y) in enumerate(train_dataloader):
            if i % 10 != 0:
                continue
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
                                               top_k=6, # a bit stricter than in training 
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
                full_prompt = prompt_1 + pred + prompt_2 + ref
                response = client.responses.create(
                    model="gpt-4.1-mini", input=full_prompt
                ).output_text
                try:
                    label_score, cue_score = map(float, response.split())
                except ValueError:
                    label_score = 0
                    cue_score = 0
                    print(f"WARNING: incorrect response formatting: {response}")
                all_label_scores_per_epoch.append(label_score)
                all_cue_scores_per_epoch.append(cue_score)

                rouge_score = scorer.score(ref, pred)
                all_rouge_scores_per_epoch.append(
                    np.mean(
                        [
                            rouge_score["rouge1"].fmeasure,
                            rouge_score["rouge2"].fmeasure,
                            rouge_score["rougeL"].fmeasure,
                        ]
                    )
                )

        all_rouge_scores.append(np.mean(all_rouge_scores_per_epoch))
        all_label_scores.append(np.mean(all_label_scores_per_epoch))
        all_cue_scores.append(np.mean(all_cue_scores_per_epoch))
        print(all_rouge_scores)
        print(all_label_scores)
        print(all_cue_scores)

    with open(
        f"thesis/out/{timestamp}/model_split{split_id}_train_only_info.json", "w"
    ) as f:
        json.dump(
            {
                "rouge_scores": all_rouge_scores,
                "label_scores": all_label_scores,
                "cue_scores": all_cue_scores,
            },
            f,
        )
