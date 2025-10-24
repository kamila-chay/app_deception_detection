from transformers import AutoProcessor, AutoModelForImageTextToText
import torch
from dataset_dolos import DolosDataset
from pathlib import Path
from torch.utils.data import DataLoader
from peft import LoraConfig, get_peft_model
from transformers import get_scheduler
from rouge_score import rouge_scorer
import numpy as np
import json
from transformers import logging
logging.set_verbosity_error()


MODEL_PATH = "facebook/Perception-LM-3B" # add deepspeed
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

    model.train()
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=1e-4,
        weight_decay=0.01
    )

    train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True, collate_fn=lambda batch: ([sample[0] for sample in batch], [sample[1] for sample in batch]))
    val_dataloader = DataLoader(val_dataset, batch_size=4, shuffle=False, collate_fn=lambda batch: ([sample[0] for sample in batch], [sample[1] for sample in batch]))
    test_dataloader = DataLoader(test_dataset, batch_size=4, shuffle=False, collate_fn=lambda batch: ([sample[0] for sample in batch], [sample[1] for sample in batch]))

    num_training_steps = NUM_EPOCHS * len(train_dataloader)
    num_warmup_steps = num_training_steps // 10

    scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
    )

    with torch.enable_grad():
        for epoch in range(NUM_EPOCHS):
            print(f"Epoch: {epoch}")
            for X, Y in train_dataloader:
                X = processor.apply_chat_template(
                    X,
                    num_frames=32,
                    add_generation_prompt=True,
                    tokenize=True,
                    return_dict=True,
                    return_tensors="pt",
                    padding=True
                )
                Y = processor.apply_chat_template(
                    Y,
                    num_frames=32,
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
                inputs = {k: v.to(model.device, dtype=torch.bfloat16) if torch.is_floating_point(v) else v.to(model.device) for k, v in inputs.items()}
                output = model(**inputs)

                loss = output.loss
                torch.nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, model.parameters()), max_norm=1.0)
                loss.backward()
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
            with torch.inference_mode():
                all_scores = []
                for X, Y in val_dataloader:
                    X = processor.apply_chat_template(
                        X,
                        num_frames=32,
                        add_generation_prompt=True,
                        tokenize=True,
                        return_dict=True,
                        return_tensors="pt",
                        padding=True
                    )
                    Y = processor.apply_chat_template(
                        Y,
                        num_frames=32,
                        add_generation_prompt=False,
                        tokenize=True,
                        return_dict=True,
                        return_tensors="pt",
                        padding=True
                    )
                    inputs = {k: v.to(model.device, dtype=torch.bfloat16) if torch.is_floating_point(v) else v.to(model.device) for k, v in inputs.items()}
                    generated_ids = model.generate(**inputs, max_new_tokens=1000)
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
                    model.save_pretrained(f"out/model_subject{subject_id}")
                    torch.save({
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scheduler_state_dict": scheduler.state_dict(),
                        "epoch": epoch,
                    }, f"out/model_subject{subject_id}/training_state.pt")
                    all_test_scores = []
                    for X, Y in test_dataloader:
                        X = processor.apply_chat_template(
                            X,
                            num_frames=32,
                            add_generation_prompt=True,
                            tokenize=True,
                            return_dict=True,
                            return_tensors="pt",
                            padding=True
                        )
                        Y = processor.apply_chat_template(
                            Y,
                            num_frames=32,
                            add_generation_prompt=False,
                            tokenize=True,
                            return_dict=True,
                            return_tensors="pt",
                            padding=True
                        )
                        inputs = {k: v.to(model.device, dtype=torch.bfloat16) if torch.is_floating_point(v) else v.to(model.device) for k, v in inputs.items()}
                        generated_ids = model.generate(**inputs, max_new_tokens=1000)
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

with open("out/test_scores_per_subject.json", "w") as f:
    json.dump(best_test_scores_per_subject, f)
