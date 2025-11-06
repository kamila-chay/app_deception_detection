from transformers import AutoProcessor, AutoModelForImageTextToText
import torch
from dataset_dolos import DolosDataset
from pathlib import Path
from torch.utils.data import DataLoader, DistributedSampler
from peft import LoraConfig, PeftModel
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
import torch.functional as F
# finetuning only the language model + checking training loss + using MRT from epoch 5

DEFAULT_BATCH_SIZE = 2 # cus we have 8 outputs per input
GRAD_ACCU_STEPS = 1

DS_CONFIG = {
    "train_batch_size": DEFAULT_BATCH_SIZE, # (samples in microbatch) x (acc steps) x (devices)
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

prev_timestamp = "2025-11-05_13-36_perception_lm_v2_language_only__better_eval__train_loss"

# deepspeed.init_distributed()
# try using generate() with deepspeed

MODEL_PATH = "facebook/Perception-LM-1B" # kept, for fair comparison, more capacity and better video understanding abilities
NUM_EPOCHS = 1 # change!

processor = AutoProcessor.from_pretrained(MODEL_PATH, use_fast=True)

prompt_1 = "Please read 2 texts describing human behavior. Output two scores based on them. The first one should be 1 if the texts agree on whether the person is lying or telling the truth. Please consider the main conclusion of each text, they will include various counterarguments but focus only on the final/dominating direction. Otherwise, if the texts disagree, output 0. Then rate how different behavioral cues are described and reasoned about. Check how much semantic overlap there is between those 2 texts in the context of those cues. The result should be in range 0.0(no overlap) through 0.1, 0.2 etc. to 1.0 (perfect overlap). Be strict. Only output those two numbers, starting from 0 or 1 and then a number in the range 0.0-1.0.\n\nText 1:\n"

prompt_2 = "\n\nText 2:\n"

for split_id in range(1, 2): # change!
    print(f"Split id: {split_id}")
    model = AutoModelForImageTextToText.from_pretrained(MODEL_PATH, dtype=torch.bfloat16).to("cuda")
    model = PeftModel.from_pretrained(model, f"out/{prev_timestamp}/model_split{split_id}_epoch4")

    for name, param in model.named_parameters():
        if "lora" not in name:
            param.requires_grad_(False)
        else:
            param.requires_grad_(True)

    print("Step 1")
    
    train_dataset = DolosDataset(f"data/train_fold{split_id}.csv", Path("./data"))
    train_sampler = DistributedSampler(train_dataset, num_replicas=dist.get_world_size(), rank=get_rank())
    train_dataloader = DataLoader(train_dataset, (DEFAULT_BATCH_SIZE // GRAD_ACCU_STEPS) // dist.get_world_size(), 
                                  sampler=train_sampler, collate_fn=lambda batch: ([sample[0] for sample in batch], [sample[1] for sample in batch]))

    total_steps = ceil(len(train_dataset) / DEFAULT_BATCH_SIZE) * NUM_EPOCHS
    warmup_steps = int(0.1 * total_steps)
    optimizer = DeepSpeedCPUAdam(filter(lambda p: p.requires_grad, model.parameters()), lr=2e-4)
    scheduler = WarmupCosineLR(optimizer, total_num_steps=total_steps, warmup_num_steps=warmup_steps)

    print("Step 2")  

    model_engine, optimizer, _, _ = deepspeed.initialize(
        model=model,
        optimizer=optimizer,
        lr_scheduler=scheduler,
        config=DS_CONFIG
    )

    all_total_losses = []

    print("Step 3")   

    for epoch in range(NUM_EPOCHS):
        model_engine.module.train()
        train_sampler.set_epoch(epoch)
        print("Step 4")  
        print(f"Epoch: {epoch}")
        total_loss = 0
        for i, (X, Y) in enumerate(train_dataloader):
            if i == 1:
                break
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

            print(X.keys())
            generated_ids = model_engine.module.generate(
                                         **X, 
                                         max_new_tokens=1000, 
                                         do_sample=True, 
                                         top_k=10,
                                         num_return_sequences=8)
            
            print(f"[Rank {get_rank()}]: generated ids are of shape {generated_ids.shape}")
            print(f"Size of input_ids: {X['input_ids'].size(1)}")

            generated_ids_trimmed = generated_ids[:, X["input_ids"].size(1):]
            generated_text_trimmed = processor.batch_decode(
                    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
                )
            
            print(f"[Rank {get_rank()}: {repr(generated_ids_trimmed)}")
            print(f"[Rank {get_rank()}: {repr(generated_text_trimmed)}")


            
            logits = model_engine(input_ids=generated_ids, pixel_values=X["pixel_values"]).logits[:, X["input_ids"].size(1)-1:-1, :] 
            print(f"[Rank {get_rank()}]: trimmed logits' shape = {logits.shape}")

            # how should it be shifted? we should probably pass in pixel values, also probably a mask

            # log_probs = F.log_softmax(logits, dim=-1)
            # token_log_probs = log_probs.gather(
            #     -1, generated_ids[:, 1:].unsqueeze(-1)
            # ).squeeze(-1)

            # probs_alpha = token_log_probs.exp() ** 0.1
            # q = probs_alpha / probs_alpha.sum()
            
            # generated_ids_trimmed = generated_ids[:, X["input_ids"].size(1):]
            # generated_text_trimmed = processor.batch_decode(
            #         generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            #     )
            
            # expected_ids_trimmed = Y["input_ids"][:, X["input_ids"].size(1):]
            # expected_text_trimmed = processor.batch_decode(
            #         expected_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            #     )
            
            # print(expected_text_trimmed)
            # print(generated_text_trimmed)

            # loss = ...
            # total_loss += loss.item() * labels.size(0)
            # model_engine.backward(loss)
            # model_engine.step()

    #     total_loss = torch.tensor(total_loss).to("cuda")
    #     dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
    #     total_loss /= len(train_dataset)

    #     if get_rank() == 0:
    #         print(f"Train loss: {total_loss}")
    #         all_total_losses.append(total_loss.cpu().item())
    #     barrier()
        
    #     model_engine.save_pretrained(f"out/{timestamp}/model_split{split_id}_epoch{epoch}")  

    # if get_rank() == 0:
    #     plt.plot(all_total_losses, marker='o')
    #     plt.title("Train Loss Plot")
    #     plt.xlabel("Epoch")
    #     plt.ylabel("Loss")

    #     plt.grid(True)
    #     plt.tight_layout()

    #     plt.savefig(f"out/{timestamp}/model_split{split_id}_train_losses.png") ## check if underfitting
    # barrier()
