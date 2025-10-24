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

DEFAULT_BATCH_SIZE = 36
GRAD_ACCU_STEPS = 12

DS_CONFIG = {
    "train_batch_size": DEFAULT_BATCH_SIZE,
    "gradient_accumulation_steps": GRAD_ACCU_STEPS, # 4(microbatch) x 3(acc steps) x 3 (devices)
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

MODEL_PATH = "facebook/Perception-LM-3B" # check deepspeed
NUM_EPOCHS = 10

best_test_scores_per_subject = []

scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
dataset = DolosDataset(Path("data/traits.xlsx"), Path("data/"))

for subject_id, (train_dataset, val_dataset, test_dataset) in enumerate(dataset.iter_subjects()):
    print(f"Subject id: {subject_id}")
    best_rouge_val_score = 0
    best_rouge_test_score = 0
    processor = AutoProcessor.from_pretrained(MODEL_PATH, use_fast=True)
    model = AutoModelForImageTextToText.from_pretrained(MODEL_PATH, dtype=torch.bfloat16).to("cuda")

    lora_config = LoraConfig(
        r=8,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj", "k_proj", "qkv"],
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, lora_config)

    for name, param in model.named_parameters():
        if "lora" not in name:
            param.requires_grad_ = False

    train_sampler = DistributedSampler(train_dataset, num_replicas=dist.get_world_size())
    train_dataloader = DataLoader(train_dataset, (DEFAULT_BATCH_SIZE // GRAD_ACCU_STEPS) // dist.get_world_size(), 
                                  sampler=train_sampler, collate_fn=lambda batch: ([sample[0] for sample in batch], [sample[1] for sample in batch]))
    val_sampler = DistributedSampler(val_dataset, num_replicas=dist.get_world_size())
    val_dataloader = DataLoader(val_dataset, (DEFAULT_BATCH_SIZE // GRAD_ACCU_STEPS) // dist.get_world_size(), 
                                sampler=val_sampler, collate_fn=lambda batch: ([sample[0] for sample in batch], [sample[1] for sample in batch]))
    test_sampler = DistributedSampler(test_dataset, num_replicas=dist.get_world_size())
    test_dataloader = DataLoader(test_dataset, (DEFAULT_BATCH_SIZE // GRAD_ACCU_STEPS) // dist.get_world_size(), 
                                sampler=test_sampler, collate_fn=lambda batch: ([sample[0] for sample in batch], [sample[1] for sample in batch]))

    total_steps = ceil(len(train_dataset) / DEFAULT_BATCH_SIZE) * NUM_EPOCHS
    warmup_steps = int(0.1 * total_steps)
    optimizer = DeepSpeedCPUAdam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)
    scheduler = WarmupCosineLR(optimizer, total_num_steps=total_steps, warmup_num_steps=warmup_steps)

    model_engine, optimizer, _, _ = deepspeed.initialize(
        model=model,
        optimizer=optimizer,
        lr_scheduler=scheduler,
        config=DS_CONFIG
    )

    with torch.enable_grad():
        for epoch in range(NUM_EPOCHS):
            model_engine.train()
            train_sampler.set_epoch(epoch)
            print(f"Epoch: {epoch}")
            for X, Y in train_dataloader:
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
                model_engine.backward(loss)
                model_engine.step()
            with torch.inference_mode():
                model_engine.eval()
                all_scores = []
                for X, Y in val_dataloader:
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
                    inputs = {k: v.to(model_engine.device, dtype=torch.bfloat16) if torch.is_floating_point(v) else v.to(model_engine.device) for k, v in X.items()}
                    generated_ids = model_engine.generate(**inputs, max_new_tokens=1000)
                    generated_ids_trimmed = generated_ids[:, inputs["input_ids"].shape[1]:]
                    expected_ids = Y
                    expected_ids_trimmed = expected_ids["input_ids"][:, inputs["input_ids"].shape[1]:]
                    generated_text_trimmed = processor.batch_decode(
                        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
                    )
                    expected_text_trimmed = processor.batch_decode(
                        expected_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
                    )
                
                    for pred, ref in zip(generated_text_trimmed, expected_text_trimmed):
                        score = scorer.score(ref, pred)
                        all_scores.append(np.mean([score["rouge1"].fmeasure, score["rouge2"].fmeasure, score["rougeL"].fmeasure]))
                rouge_val_score = np.mean(all_scores)
                print(f"Rouge validation score {rouge_val_score}")
                if rouge_val_score > best_rouge_val_score:
                    best_rouge_val_score = rouge_val_score
                    model_engine.save_pretrained(f"out/{timestamp}/model_subject{subject_id}")
                    all_test_scores = []
                    for X, Y in test_dataloader:
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
                        inputs = {k: v.to(model_engine.device, dtype=torch.bfloat16) if torch.is_floating_point(v) else v.to(model_engine.device) for k, v in X.items()}
                        generated_ids = model_engine.generate(**inputs, max_new_tokens=1000)
                        generated_ids_trimmed = generated_ids[:, inputs["input_ids"].shape[1]:]
                        expected_ids = Y
                        expected_ids_trimmed = expected_ids["input_ids"][:, inputs["input_ids"].shape[1]:]
                        generated_text_trimmed = processor.batch_decode(
                            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
                        )
                        expected_text_trimmed = processor.batch_decode(
                            expected_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
                        )
                    
                        for pred, ref in zip(generated_text_trimmed, expected_text_trimmed):
                            score = scorer.score(ref, pred)
                            all_test_scores.append(np.mean([score["rouge1"].fmeasure, score["rouge2"].fmeasure, score["rougeL"].fmeasure]))
                    print(f"  Top val score for this subject, the test score is {np.mean(all_test_scores)}")
                    best_rouge_test_score = max(np.mean(all_test_scores), best_rouge_test_score)

    best_test_scores_per_subject.append(best_rouge_test_score)
    print(best_test_scores_per_subject) 
# check learning rates 

with open(f"out/{timestamp}/test_scores_per_subject.json", "w") as f:
    json.dump(best_test_scores_per_subject, f)
