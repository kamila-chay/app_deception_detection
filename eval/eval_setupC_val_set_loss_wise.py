# Developed as part of a BSc thesis at the Faculty of Computer Science, Bialystok Univesity of Technology

from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from peft import PeftModel
from torch.utils.data import DataLoader
from transformers import AutoModelForImageTextToText, AutoProcessor, logging

from thesis.utils.dataset_dolos import DolosDataset
from thesis.utils.utils import set_seed

logging.set_verbosity_error()

set_seed(42)

NUM_DEVICES = 1
GRAD_ACCU_STEPS = 16
MICRO_BATCH = 1
DEFAULT_BATCH_SIZE = NUM_DEVICES * GRAD_ACCU_STEPS * MICRO_BATCH
BETA = 0.01

timestamp = "2025-12-01_20-44"
timestamp_token_level = "2025-11-15_20-01"

MODEL_PATH = "facebook/Perception-LM-1B"
NUM_EPOCHS = 3

processor = AutoProcessor.from_pretrained(MODEL_PATH, use_fast=True)

for split_id, reference_token_level_epoch in ((1, 8), (2, 1), (3, 3)):
    print(f"Split id: {split_id}")
    all_total_losses = []

    model_ref = AutoModelForImageTextToText.from_pretrained(
        MODEL_PATH, dtype=torch.bfloat16
    ).to("cuda")
    model_ref = PeftModel.from_pretrained(
        model_ref,
        f"thesis/out/{timestamp_token_level}/model_split{split_id}_epoch{reference_token_level_epoch}",
    ).to("cuda")
    model_ref.eval()

    val_dataset = DolosDataset(
        f"thesis/data/val_fold{split_id}.csv",
        Path("thesis/data"),
        "joint_configuration_reasoning_labels",
    )

    val_dataset.include_opposing_(True)
    val_dataloader = DataLoader(
        val_dataset,
        MICRO_BATCH,
        shuffle=False,
        collate_fn=lambda batch: (
            [sample[0] for sample in batch],
            [sample[1] for sample in batch],
            [sample[2] for sample in batch],
        ),
    )

    for epoch in range(NUM_EPOCHS):
        print(f"Epoch: {epoch}")
        model = AutoModelForImageTextToText.from_pretrained(
            MODEL_PATH, dtype=torch.bfloat16
        ).to("cuda")
        model = PeftModel.from_pretrained(
            model, f"thesis/out/{timestamp}/model_split{split_id}_epoch{epoch}"
        ).to("cuda")
        model.eval()

        total_loss = 0
        for input, input_completed, input_completed_opposing in val_dataloader:
            input = processor.apply_chat_template(
                input,
                num_frames=16,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
                padding=True,
            )
            input_completed_two_way = processor.apply_chat_template(
                [input_completed[0], input_completed_opposing[0]],
                num_frames=16,
                add_generation_prompt=False,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
                padding=True,
            )

            input_completed_two_way = {
                k: (
                    v.to(model.device, dtype=torch.bfloat16)
                    if torch.is_floating_point(v)
                    else v.to(model.device)
                )
                for k, v in input_completed_two_way.items()
            }

            input_completed_two_way_ids_trimmed = input_completed_two_way["input_ids"][
                :, input["input_ids"].size(1) :
            ]

            with torch.no_grad():
                output = model(**input_completed_two_way)

            logits = output.logits[:, input["input_ids"].size(1) - 1 : -1, :].to(
                torch.float32
            )
            log_probs = F.log_softmax(logits, dim=-1)

            token_log_probs = log_probs.gather(
                -1, input_completed_two_way_ids_trimmed.unsqueeze(-1)
            ).squeeze(-1)
            token_log_probs = (
                token_log_probs
                * input_completed_two_way["attention_mask"][
                    :, -token_log_probs.shape[1] :
                ]
            )

            sequence_log_probs = token_log_probs.sum(dim=-1)

            with torch.no_grad():
                output_ref = model_ref(**input_completed_two_way)

            logits_ref = output_ref.logits[
                :, input["input_ids"].size(1) - 1 : -1, :
            ].to(torch.float32)
            log_probs_ref = F.log_softmax(logits_ref, dim=-1)

            token_log_probs_ref = log_probs_ref.gather(
                -1, input_completed_two_way_ids_trimmed.unsqueeze(-1)
            ).squeeze(-1)

            token_log_probs_ref = (
                token_log_probs_ref
                * input_completed_two_way["attention_mask"][
                    :, -token_log_probs_ref.shape[1] :
                ]
            )

            sequence_log_probs_ref = token_log_probs_ref.sum(dim=-1)

            r_plus = sequence_log_probs[0] - sequence_log_probs_ref[0]
            r_minus = sequence_log_probs[1] - sequence_log_probs_ref[1]

            loss = (
                -torch.log(torch.sigmoid(BETA * (r_plus - r_minus))) / GRAD_ACCU_STEPS
            )
            total_loss += loss

        total_loss /= len(val_dataset)
        all_total_losses.append(total_loss.cpu().item())
        print(all_total_losses)

    plt.plot(all_total_losses, marker="o")
    plt.title("Val Loss Plot")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")

    plt.grid(True)
    plt.tight_layout()

    plt.savefig(f"thesis/out/{timestamp}/model_split{split_id}_val_losses.png")
