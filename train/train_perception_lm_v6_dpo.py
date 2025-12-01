from math import ceil
from pathlib import Path

import torch
import torch.nn.functional as F
from peft import PeftModel
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoModelForImageTextToText, AutoProcessor, logging
import matplotlib.pyplot as plt
from transformers import get_cosine_schedule_with_warmup
from datetime import datetime

from thesis.utils.dataset_dolos import DolosDataset
from thesis.utils.utils import set_seed

logging.set_verbosity_error()

set_seed(42)

NUM_DEVICES = 1
GRAD_ACCU_STEPS = 16
MICRO_BATCH = 1
DEFAULT_BATCH_SIZE = NUM_DEVICES * GRAD_ACCU_STEPS * MICRO_BATCH
BETA = 0.01

timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
dir_path = Path(f"thesis/out/{timestamp}")
dir_path.mkdir(parents=True, exist_ok=True)

token_level_timestamp = "2025-11-15_20-01"

MODEL_PATH = "facebook/Perception-LM-1B"
NUM_EPOCHS = 3

processor = AutoProcessor.from_pretrained(MODEL_PATH, use_fast=True)

for split_id, relevant_epoch in ((1, 8), (2, 1), (3, 3)): 
    print(f"Split id: {split_id}")
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_PATH, dtype=torch.bfloat16
    ).to("cuda")
    model = PeftModel.from_pretrained(
        model, f"thesis/out/{token_level_timestamp}/model_split{split_id}_epoch{relevant_epoch}"
    ).to("cuda")
    model.train()

    model_ref = AutoModelForImageTextToText.from_pretrained(
        MODEL_PATH, dtype=torch.bfloat16
    ).to("cuda")
    model_ref = PeftModel.from_pretrained(
        model_ref, f"thesis/out/{token_level_timestamp}/model_split{split_id}_epoch{relevant_epoch}"
    ).to("cuda")
    model_ref.eval()

    for name, param in model.named_parameters():
        if "lora" not in name:
            param.requires_grad_(False)
        else:
            param.requires_grad_(True)

    for name, param in model_ref.named_parameters():
        param.requires_grad_(False)

    train_dataset = DolosDataset(
        f"thesis/data/train_fold{split_id}.csv", Path("thesis/data"), "mumin_reasoning_labels_concise"
    )

    train_dataset.include_opposing_(True)
    train_dataloader = DataLoader(
        train_dataset,
        MICRO_BATCH,
        shuffle=True,
        collate_fn=lambda batch: (
            [sample[0] for sample in batch],
            [sample[1] for sample in batch],
            [sample[2] for sample in batch],
        ),
    )

    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=7e-6)
    total_steps = (
        ceil(len(train_dataset) / DEFAULT_BATCH_SIZE) * NUM_EPOCHS
    )
    warmup_steps = int(0.1 * total_steps)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    all_total_losses = []

    for epoch in range(NUM_EPOCHS):
        print(f"Epoch: {epoch}")
        total_loss = 0
        for i, (input, input_completed, input_completed_opposing) in enumerate(train_dataloader):
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

            input_completed_two_way_ids_trimmed = input_completed_two_way["input_ids"][:, input["input_ids"].size(1):]

            output = model(
                **input_completed_two_way
            )

            logits = output.logits[:, input["input_ids"].size(1) - 1 : -1, :].to(
                torch.float32
            )
            log_probs = F.log_softmax(logits, dim=-1)

            token_log_probs = log_probs.gather(
                -1, input_completed_two_way_ids_trimmed.unsqueeze(-1)
            ).squeeze(-1)
            token_log_probs = (
                token_log_probs * input_completed_two_way["attention_mask"][:, -token_log_probs.shape[1] :]
            )

            sequence_log_probs = token_log_probs.sum(dim=-1)

            with torch.no_grad():
                output_ref = model_ref(
                    **input_completed_two_way
                )

            logits_ref = output_ref.logits[:, input["input_ids"].size(1) - 1 : -1, :].to(
                torch.float32
            )
            log_probs_ref = F.log_softmax(logits_ref, dim=-1)

            token_log_probs_ref = log_probs_ref.gather(
                -1, input_completed_two_way_ids_trimmed.unsqueeze(-1)
            ).squeeze(-1)

            token_log_probs_ref = (
                token_log_probs_ref * input_completed_two_way["attention_mask"][:, -token_log_probs_ref.shape[1] :]
            )

            sequence_log_probs_ref = token_log_probs_ref.sum(dim=-1)

            r_plus = sequence_log_probs[0] - sequence_log_probs_ref[0]
            r_minus = sequence_log_probs[1] - sequence_log_probs_ref[1]

            loss = -torch.log(torch.sigmoid(BETA * (r_plus - r_minus))) / GRAD_ACCU_STEPS
            total_loss += loss
            print(loss)

            loss.backward() 

            if i % GRAD_ACCU_STEPS == GRAD_ACCU_STEPS - 1:
                per_layer_norm = torch.stack([
                    torch.norm(p.grad.detach(), 2)
                    for p in model.parameters() if p.grad is not None
                ])
                total_norm = torch.norm(
                    per_layer_norm,
                    2
                ).item()

                print(f"Total grad norm: {total_norm}")
                print(f"Per-layer grad norm: {per_layer_norm[0::3]}")
                torch.nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, model.parameters()), max_norm=10.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
            
        if any(p.grad is not None and p.grad.abs().sum().item() > 0 for p in optimizer.param_groups[0]['params']):
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        total_loss /= len(train_dataset)
        all_total_losses.append(total_loss.cpu().item())
        print(all_total_losses)

        save_dir = f"thesis/out/{timestamp}/model_split{split_id}_epoch{epoch}"
        model.save_pretrained(save_dir)

        torch.save({
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
        }, Path(save_dir) / "training_state.pt")

    plt.plot(all_total_losses, marker='o')
    plt.title("Train Loss Plot")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")

    plt.grid(True)
    plt.tight_layout()

    plt.savefig(f"thesis/out/{timestamp}/model_split{split_id}_train_losses.png")

