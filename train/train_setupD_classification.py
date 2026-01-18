# Developed as part of a BSc thesis at the Faculty of Computer Science, Bialystok Univesity of Technology

import json
from datetime import datetime
from math import ceil
from pathlib import Path

import deepspeed
import torch
from deepspeed.ops.adam import DeepSpeedCPUAdam
from deepspeed.runtime.lr_schedules import WarmupCosineLR
from peft import LoraConfig, TaskType, get_peft_model
from torch.distributed import get_rank, get_world_size
from torch.utils.data import DataLoader, DistributedSampler
from transformers import (
    AutoImageProcessor,
    TimesformerConfig,
    TimesformerForVideoClassification,
)

from thesis.utils.dataset_dolos import DolosClassificationDataset
from thesis.utils.utils import set_seed

set_seed(42)

BATCH_SIZE = 72
GRAD_ACCU_STEPS = 36
EPOCHS = 20

timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
out_dir = Path(f"thesis/out/{timestamp}")
out_dir.mkdir(parents=True, exist_ok=True)

lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["qkv"],
    lora_dropout=0.1,
    bias="none",
    task_type=TaskType.FEATURE_EXTRACTION,
)

ds_config = {
    "train_batch_size": BATCH_SIZE,
    "gradient_accumulation_steps": GRAD_ACCU_STEPS,
    "zero_optimization": {
        "stage": 3,
        "offload_optimizer": {"device": "cpu", "pin_memory": True},
        "offload_param": {"device": "cpu", "pin_memory": True},
    },
    "fp16": {"enabled": False},
    "bf16": {"enabled": True},
    "activation_checkpointing": {
        "partition_activations": True,
        "contiguous_memory_optimization": True,  # could make this script truly compatible with DS
    },
}

for split_id in range(1, 4):
    print(f"Split {split_id}")
    processor = AutoImageProcessor.from_pretrained(
        "facebook/timesformer-base-finetuned-k600"
    )
    config = TimesformerConfig()
    config.num_labels = 2
    model = TimesformerForVideoClassification.from_pretrained(
        "facebook/timesformer-base-finetuned-k600",
        config=config,
        ignore_mismatched_sizes=True,
    )

    train_dataset = DolosClassificationDataset(
        f"thesis/data/train_fold{split_id}.csv", "thesis/data/video", processor
    )

    val_dataset = DolosClassificationDataset(
        f"thesis/data/val_fold{split_id}.csv", "thesis/data/video", processor
    )

    lora_model = get_peft_model(model, lora_config)
    lora_model.base_model.model.classifier.weight.requires_grad = True
    lora_model.base_model.model.classifier.bias.requires_grad = True

    lora_model.print_trainable_parameters()

    total_steps = ceil(len(train_dataset) / BATCH_SIZE) * EPOCHS
    warmup_steps = int(0.1 * total_steps)
    optimizer = DeepSpeedCPUAdam(
        filter(lambda p: p.requires_grad, lora_model.parameters()), lr=2e-4
    )
    scheduler = WarmupCosineLR(
        optimizer, total_num_steps=total_steps, warmup_num_steps=warmup_steps
    )

    model_engine, optimizer, _, _ = deepspeed.initialize(
        model=lora_model, optimizer=optimizer, lr_scheduler=scheduler, config=ds_config
    )

    train_sampler = DistributedSampler(train_dataset, get_world_size(), get_rank())
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE // get_world_size() // GRAD_ACCU_STEPS,
        sampler=train_sampler,
    )

    val_sampler = DistributedSampler(val_dataset, get_world_size(), get_rank())
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE // get_world_size() // GRAD_ACCU_STEPS,
        sampler=val_sampler,
    )

    model_engine.train()

    all_train_losses = []
    all_val_losses = []

    for epoch in range(EPOCHS):
        print(f"Epoch {epoch + 1}")
        train_sampler.set_epoch(epoch)
        train_loss = 0.0
        for batch_x, batch_y in train_dataloader:
            batch_x = batch_x.to(model_engine.device).to(torch.bfloat16)
            batch_y = batch_y.to(model_engine.device)

            output = model_engine(pixel_values=batch_x, labels=batch_y)
            model_engine.backward(output.loss)
            train_loss += output.loss * batch_x.shape[0]
            model_engine.step()

        train_loss /= len(train_dataset)

        all_train_losses.append(train_loss.item())

        save_path_lora = (
            f"thesis/out/{timestamp}/lora_timesformer_split{split_id}_epoch{epoch}"
        )
        save_path = (
            f"thesis/out/{timestamp}/timesformer_split{split_id}_epoch{epoch}.pt"
        )
        lora_model.save_pretrained(save_path_lora)
        torch.save(lora_model.base_model.model.classifier.state_dict(), save_path)

        val_loss = 0.0

        with torch.no_grad():
            for batch_x, batch_y in val_dataloader:
                batch_x = batch_x.to(model_engine.device).to(torch.bfloat16)
                batch_y = batch_y.to(model_engine.device)

                output = model_engine(pixel_values=batch_x, labels=batch_y)
                val_loss += output.loss * batch_x.shape[0]

            val_loss /= len(val_dataset)

            all_val_losses.append(val_loss.item())

    with open(f"thesis/out/{timestamp}/model_split{split_id}_losses.json", "w") as f:
        json.dump(
            {
                "train_losses": all_train_losses,
                "val_losses": all_val_losses,
            },
            f,
        )
