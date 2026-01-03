import json
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
from openai import OpenAI
from rouge_score import rouge_scorer
from torch.utils.data import DataLoader
from transformers import logging
from transformers import LlavaNextVideoProcessor, LlavaNextVideoForConditionalGeneration


from thesis.utils.dataset_dolos import DolosDataset
from thesis.utils.utils import set_seed
from thesis.utils.constants import ALL_RELEVANT_TRAITS

set_seed(42)
logging.set_verbosity_error()

DEFAULT_BATCH_SIZE = 8

timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
dir_path = Path(f"thesis/out/{timestamp}")
dir_path.mkdir(parents=True, exist_ok=True)

processor = LlavaNextVideoProcessor.from_pretrained("llava-hf/LLaVA-NeXT-Video-7B-hf")
model = LlavaNextVideoForConditionalGeneration.from_pretrained("llava-hf/LLaVA-NeXT-Video-7B-hf", dtype=torch.bfloat16, device_map="cuda")

scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
client = OpenAI()

prompt_1 = "Please read the 2 texts below. Each of them contains an assesment of whether or not a person is lying. Each one of them contains arguments for and against both deception and truth. At the same time they both lead to a specific, more likely conclusion. Read them and output the final conclusions only. Do it in the following, example format: \"Text 1: truth, Text 2: deception\". The output values should be aligned with the following texts. The results should be limited to \"truth\" and \"deception\". Dont output \"inconclusive\" unless absolutely no hints are made. \n\nText 1:\n"

prompt_2 = "\n\nText 2:\n"

prompt_cue_f1 = f"Please read the text below. Look for behaviors that are mentioned in the text from the following list: {repr(ALL_RELEVANT_TRAITS)}. Output those using the same exact wording as in the list, one per line. Don't ouput anything else. \n\nText:\n" 

prompt_reasoning_overlap_p1 = f"Read those 2 texts describing the behavior of the same person and how it can be interpreted as a clue to deception/truthfulness. Score the logical overlap between those texts, you should pay attention to both the clues themselves and how they are interpreted and reasoned about. The score should be lower if e.g one of the texts focuses just on one interpretation of a specific cue etc. The score should be anywhere between 0.0 and 1.0 (both inclusive). Output the score only, nothing else. \n\nTEXT 1:\n"

prompt_reasoning_overlap_p2 = "\n\nTEXT 2:\n"

for split_id in range(1, 4):
    print(f"Split id: {split_id}")

    test_dataset = DolosDataset(
        f"thesis/data/test_fold{split_id}.csv", Path("thesis/data"), "mumin_reasoning_labels_concise"
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

    model = model.eval()

    all_cue_overlap_scores_per_epoch = []
    all_f1_cue_scores_per_epoch = []
    all_rouge_scores_per_epoch = []
    all_label_gt_per_epoch = []
    all_label_pred_per_epoch = []

    for i, (X, Y, raw_cues) in enumerate(test_dataloader):
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
        with torch.inference_mode():
            generated_ids = model.generate(**inputs, 
                                            max_new_tokens=1000,
                                            do_sample=False)
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

        for pred, ref, raw_clues_per_sample in zip(generated_text_trimmed, expected_text_trimmed, raw_cues):
            print("********start*********")
            print(pred)
            print("==============")
            print(ref)
            print("*********")
            print(raw_clues_per_sample)
            full_prompt = prompt_cue_f1 + pred
            try:
                response = None
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
                print(f"ERROR: Incorrect response formatting: {response}")

            full_prompt = prompt_reasoning_overlap_p1 + pred + prompt_reasoning_overlap_p2 + ref
            try:
                response = None
                response = client.responses.create(
                    model="gpt-4.1-mini", input=full_prompt, top_p=1, temperature=0
                ).output_text

                score = float(response)
                all_cue_overlap_scores_per_epoch.append(score)

            except Exception as e:
                print(f"ERROR: Incorrect response formatting: {response}")

            full_prompt = prompt_1 + pred + prompt_2 + ref
            try:
                response = None
                response = client.responses.create(
                    model="gpt-4.1-mini", input=full_prompt, temperature=0, top_p=1
                ).output_text
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
                all_label_pred_per_epoch.append(predicted)
                all_label_gt_per_epoch.append(gt)
            except ValueError:
                print(f"ERROR: Incorrect response formatting: {response}")

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

    rouge_score = np.mean(all_rouge_scores_per_epoch)
    cue_f1 = np.mean(all_f1_cue_scores_per_epoch)
    cue_soft_overlap = np.mean(all_cue_overlap_scores_per_epoch)

    all_label_gt_per_epoch = np.array(all_label_gt_per_epoch)
    all_label_pred_per_epoch = np.array(all_label_pred_per_epoch)
    label_acc = (all_label_gt_per_epoch == all_label_pred_per_epoch).sum() / all_label_gt_per_epoch.size
    tp = ((all_label_gt_per_epoch == 1) & (all_label_pred_per_epoch == 1)).sum()
    fp = ((all_label_gt_per_epoch == 0) & (all_label_pred_per_epoch == 1)).sum()
    fn = ((all_label_gt_per_epoch == 1) & (all_label_pred_per_epoch == 0)).sum()

    precision = tp / (tp + fp) if (tp + fp) > 0.0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0.0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0.0 else 0.0

    with open(
        f"thesis/out/{timestamp}/model_split{split_id}_test_only_info.json", "w"
    ) as f:
        json.dump(
            {
                "rouge_scores": rouge_score,
                "classification_acc": label_acc,
                "classification_precision": precision,
                "classification_recall": recall,
                "classification_f1": f1,
                "cue-f1": cue_f1,
                "SoftOverlap": cue_soft_overlap,
            },
            f,
        )
