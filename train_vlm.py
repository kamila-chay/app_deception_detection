from transformers import AutoProcessor, AutoModelForImageTextToText
import torch
from dataset_dolos import DolosDataset
from pathlib import Path
from torch.utils.data import DataLoader
from peft import LoraConfig, get_peft_model
from transformers import get_scheduler

MODEL_PATH = "facebook/Perception-LM-1B" # change before moving to HPC
NUM_EPOCHS = 10

# a training loop


dataset = DolosDataset(Path("data/traits.xlsx"), Path("data/"))

for subject_id, (train_dataset, val_dataset, test_dataset) in enumerate(dataset.iter_subjects()):
    processor = AutoProcessor.from_pretrained(MODEL_PATH, use_fast=True)
    model = AutoModelForImageTextToText.from_pretrained(MODEL_PATH, dtype=torch.bfloat16).to("cuda")

    lora_config = LoraConfig(
        r=8,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj", "k_proj", "qkv"], # visual backbone too
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
    num_training_steps = NUM_EPOCHS * len(train_dataloader) # for now this formula
    num_warmup_steps = num_training_steps // 10

    scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
    )

    train_dataloader = DataLoader(train_dataset, batch_size=2, shuffle=True, collate_fn=lambda batch: ([sample[0] for sample in batch], [sample[1] for sample in batch]))
    val_dataloader = DataLoader(val_dataset, batch_size=2, shuffle=False, collate_fn=lambda batch: ([sample[0] for sample in batch], [sample[1] for sample in batch]))
    test_dataloader = DataLoader(test_dataset, batch_size=2, shuffle=False, collate_fn=lambda batch: ([sample[0] for sample in batch], [sample[1] for sample in batch]))

    with torch.enable_grad():
        for epoch in range(NUM_EPOCHS):
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
                labels = inputs["input_ids"]
                labels[:, : X["input_ids"].shape[1]] = -100
                labels[labels == processor.tokenizer.pad_token_id] = -100
                inputs["labels"] = labels
                inputs = inputs.to(model.device)
                output = model(**inputs)

                loss = output.loss
                loss.backward()
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
            with torch.inference_mode():
                for X, Y in val_dataloader:
                    X = processor.apply_chat_template(
                        X,
                        # num_frames=32,
                        add_generation_prompt=True,
                        padding=True
                    )
                    Y = processor.apply_chat_template(
                        Y,
                        # num_frames=32,
                        add_generation_prompt=False, # are those ok?
                        padding=True
                    )
                    inputs = processor() # how do i do this here?

                    model.generate()



