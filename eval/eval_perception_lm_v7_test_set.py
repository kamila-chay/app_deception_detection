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
from thesis.utils.utils import set_seed, make_conv_for_classification_cond
from thesis.utils.constants import ALL_RELEVANT_TRAITS

# if we include -oracle classification scores -real classification scores does anything change in the metrcis?

set_seed(42)
logging.set_verbosity_error()

DEFAULT_BATCH_SIZE = 8

timestamp = "2025-11-18_22-39"

MODEL_PATH = "facebook/Perception-LM-1B"

processor = AutoProcessor.from_pretrained(MODEL_PATH, use_fast=True)

scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
client = OpenAI()

prompt_cue_f1 = f"Please read the text below. Look for behaviors that are mentioned in the text from the following list: {repr(ALL_RELEVANT_TRAITS)}. Output those using the same exact wording as in the list, one per line. Don't ouput anything else. \n\nText:\n" 

prompt_reasoning_overlap_p1 = f"Read those 2 texts describing the behavior of the same person and how it can be interpreted as a clue to deception/truthfulness. Score the logical overlap between those texts, you should pay attention to both the clues themselves and how they are interpreted and reasoned about. Thse score should be lower if e.g one of the texts focuses just on one interpretation of a specific cue etc. The score should be anywhere between 0.0 and 1.0 (both inclusive). Output the score only, nothing else. \n\nTEXT 1:\n"

prompt_reasoning_overlap_p2 = "\n\nTEXT 2:\n"

for split_id in range(1, 4):
    print(f"Split id: {split_id}")

    test_dataset = DolosDataset(
        f"thesis/data/test_fold{split_id}.csv", Path("thesis/data"), label_folder="mumin_reasoning_labels_balanced",
        conv_making_func=make_conv_for_classification_cond
    )

    test_dataset.include_raw_clues_(True)

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

    all_rouge_scores = []
    all_f1_cue_scores = []
    all_cue_overlap_scores = []

    for epoch in [8]:
        print(f"Epoch: {epoch}")
        base = AutoModelForImageTextToText.from_pretrained(
            MODEL_PATH, torch_dtype=torch.bfloat16
        )
        checkpoint = f"thesis/out/{timestamp}/model_split{split_id}_epoch{epoch}"
        model = PeftModel.from_pretrained(
            base, checkpoint
        )
        model = model.to("cuda:0").eval()

        all_rouge_scores_per_epoch = []
        all_f1_cue_scores_per_epoch = []
        all_cue_overlap_scores_per_epoch = []
        for X, Y, raw_cues in test_dataloader:
            X = processor.apply_chat_template(
                X,
                num_frames=16,
                add_generation_prompt=False,
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

            print("Before trimming:")
            print(inputs["input_ids"])

            inputs["input_ids"] = inputs["input_ids"][:, :-1]
            inputs["attention_mask"] = inputs["attention_mask"][:, :-1] # what about PAD???

            print("After trimming:")
            print(inputs["input_ids"])
            with torch.inference_mode():
                generated_ids = model.generate(**inputs, 
                                               max_new_tokens=1000,
                                               do_sample=True,
                                               top_k=3, 
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

            print(f"!!!The length of raw_cues is {len(raw_cues)}")

            for pred, ref, raw_clues_per_sample in zip(generated_text_trimmed, expected_text_trimmed, raw_cues):
                print(pred)
                print("*********")
                print(ref)

                print("^^^^^^^^")
                full_prompt = prompt_cue_f1 + pred
                try:
                    response = client.responses.create(
                        model="gpt-4.1-mini", input=full_prompt, top_p=1, temperature=0
                    ).output_text

                    pred_cues = list(map(lambda z: z.strip(), filter(lambda x: len(x) > 0, response.split("\n"))))
                    init_len = len(pred_cues)
                    pred_cues = [cue for cue in pred_cues if cue in ALL_RELEVANT_TRAITS]
                    if diff:= init_len - len(pred_cues):
                        print(f"WARNING: {diff} cues were output that don't match ALL_RELEVANT_TRAITS")
                    
                    pred_cues = set(pred_cues)
                    raw_clues_per_sample = set(raw_clues_per_sample)
                    intersection = pred_cues & raw_clues_per_sample

                    precision = len(intersection) / len(pred_cues) if len(pred_cues) > 0 else 0.0
                    recall = len(intersection) / len(raw_clues_per_sample) if len(raw_clues_per_sample) > 0 else 0.0

                    f1_for_cues = 2 * precision * recall / (precision + recall) if (precision + recall) > 0.0 else 0.0
                    all_f1_cue_scores_per_epoch.append(f1_for_cues)
                except Exception as e:
                    print(f"ERROR: Didn't process OpenAI API cue F1 ouput properly: {e}")

                full_prompt = prompt_reasoning_overlap_p1 + pred + prompt_reasoning_overlap_p2 + ref

                try:
                    response = client.responses.create(
                        model="gpt-4.1-mini", input=full_prompt, top_p=1, temperature=0
                    ).output_text

                    score = float(response)
                    all_cue_overlap_scores_per_epoch.append(score)

                except Exception as e:
                    print(f"ERROR: Didn't process OpenAI API clue overlap ouput properly: {e}")

                
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

            print(all_cue_overlap_scores_per_epoch) # should be the same between runs
            print(all_f1_cue_scores_per_epoch)

        all_rouge_scores.append(np.mean(all_rouge_scores_per_epoch))
        all_f1_cue_scores.append(np.mean(all_f1_cue_scores_per_epoch))
        all_cue_overlap_scores.append(np.mean(all_cue_overlap_scores_per_epoch))

    with open(
        f"thesis/out/{timestamp}/model_split{split_id}_test_only_info.json", "w"
    ) as f:
        json.dump(
            {
                "rouge_scores": all_rouge_scores,
                "f1_cue_scores": all_f1_cue_scores,
                "cue_overlap_scores": all_cue_overlap_scores 
            },
            f,
        )
