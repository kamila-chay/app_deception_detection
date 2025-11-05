from transformers import AutoProcessor, AutoModelForImageTextToText
import torch
from dataset_dolos import DolosDataset
from pathlib import Path
from torch.utils.data import DataLoader, DistributedSampler
from peft import LoraConfig, get_peft_model
from rouge_score import rouge_scorer
import numpy as np
import json
from transformers import logging
logging.set_verbosity_error()
import deepspeed
from deepspeed.ops.adam import DeepSpeedCPUAdam
from deepspeed.runtime.lr_schedules import WarmupCosineLR
from torch.distributed import get_rank, barrier
from datetime import datetime
import torch.distributed as dist
from math import ceil
from peft import PeftModel
import os
import matplotlib.pyplot as plt

DEFAULT_BATCH_SIZE = 24
GRAD_ACCU_STEPS = 12 ## change here depending on the devices available

DS_CONFIG = {
    "train_batch_size": DEFAULT_BATCH_SIZE, # 1(samples in microbatch) x 12(acc steps) x 4(devices)
    "gradient_accumulation_steps": GRAD_ACCU_STEPS, 
    "zero_optimization": {
        "stage": 3,
        "offload_optimizer": {
            "device": "cpu",
            "pin_memory": True
        }
    },
    "bf16": {
        "enabled": True
    },
    "activation_checkpointing": {
        "partition_activations": True,
        "contiguous_memory_optimization": True
    },
}

timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
dir_path = Path(f"out/{timestamp}")
dir_path.mkdir(parents=True, exist_ok=True)

deepspeed.init_distributed()

MODEL_PATH = "facebook/Perception-LM-1B"
NUM_EPOCHS = 10

processor = AutoProcessor.from_pretrained(MODEL_PATH, use_fast=True)

for split_id in range(1, 4):
    print(f"Split id: {split_id}")
    model = AutoModelForImageTextToText.from_pretrained(MODEL_PATH, dtype=torch.bfloat16).to("cuda")

    lora_config = LoraConfig(
        r=8,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj", "k_proj"],
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, lora_config)

    for name, param in model.named_parameters():
        if "lora" not in name:
            param.requires_grad_ = False
            param.requires_grad = False
        else:
            print(name) ## check if only the language model is trained

    
    train_dataset = DolosDataset(f"data/train_fold{split_id}.csv", Path("./data"))
    train_sampler = DistributedSampler(train_dataset, num_replicas=dist.get_world_size(), rank=get_rank())
    train_dataloader = DataLoader(train_dataset, (DEFAULT_BATCH_SIZE // GRAD_ACCU_STEPS) // dist.get_world_size(), 
                                  sampler=train_sampler, collate_fn=lambda batch: ([sample[0] for sample in batch], [sample[1] for sample in batch]))

    total_steps = ceil(len(train_dataset) / DEFAULT_BATCH_SIZE) * NUM_EPOCHS
    warmup_steps = int(0.1 * total_steps)
    optimizer = DeepSpeedCPUAdam(filter(lambda p: p.requires_grad, model.parameters()), lr=2e-4)
    scheduler = WarmupCosineLR(optimizer, total_num_steps=total_steps, warmup_num_steps=warmup_steps)

    model_engine, optimizer, _, _ = deepspeed.initialize(
        model=model,
        optimizer=optimizer,
        lr_scheduler=scheduler,
        config=DS_CONFIG
    )

    all_total_losses = []

    with torch.enable_grad():
        for epoch in range(NUM_EPOCHS):
            model_engine.module.train()
            train_sampler.set_epoch(epoch)
            print(f"Epoch: {epoch}")
            total_loss = 0
            for i, (X, Y) in enumerate(train_dataloader):
                X = processor.apply_chat_template(
                    X,
                    num_frames=16,
                    add_generation_prompt=True,
                    tokenize=True,
                    return_dict=True,
                    return_tensors="pt",
                    padding=True
                )
                Y = processor.apply_chat_template(
                    Y,
                    num_frames=16,
                    add_generation_prompt=False,
                    tokenize=True,
                    return_dict=True,
                    return_tensors="pt",
                    padding=True
                )
                inputs = Y
                labels = inputs["input_ids"].clone()
                labels[:, : X["input_ids"].shape[1]] = -100
                labels[labels == processor.tokenizer.pad_token_id] = -100
                inputs["labels"] = labels
                inputs = {k: v.to(model_engine.device, dtype=torch.bfloat16) if torch.is_floating_point(v) else v.to(model_engine.device) for k, v in inputs.items()}
                output = model_engine(**inputs)

                loss = output.loss
                total_loss += loss.item() * labels.size(0)
                model_engine.backward(loss)
                model_engine.step()

            total_loss = torch.tensor(total_loss).to("cuda")
            dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
            total_loss /= len(train_dataset)

            if get_rank() == 0:
                print(f"Train loss: {total_loss}")
                all_total_losses.append(total_loss.cpu().item())
            barrier()
            
            model_engine.save_pretrained(f"out/{timestamp}/model_split{split_id}_epoch{epoch}")  

        if get_rank() == 0:
            plt.plot(all_total_losses, marker='o')
            plt.title("Train Loss Plot")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")

            plt.grid(True)
            plt.tight_layout()

            plt.savefig(f"out/{timestamp}/model_split{split_id}_train_losses.png") ## check if underfitting
        barrier()