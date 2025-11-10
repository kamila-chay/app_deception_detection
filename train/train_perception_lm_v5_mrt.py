from datetime import datetime
from math import ceil
from pathlib import Path

import torch
import torch.nn.functional as F
from openai import OpenAI
from peft import PeftModel
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from transformers import AutoModelForImageTextToText, AutoProcessor, logging
import matplotlib.pyplot as plt
from rouge_score import rouge_scorer
import numpy as np
from transformers import get_cosine_schedule_with_warmup

from thesis.utils.dataset_dolos import DolosDataset
from thesis.utils.utils import set_seed

logging.set_verbosity_error()

set_seed(42)

client = OpenAI()

NUM_DEVICES = 1 # fixed here
GRAD_ACCU_STEPS = 8
MICRO_BATCH = 1
DEFAULT_BATCH_SIZE = NUM_DEVICES * GRAD_ACCU_STEPS * MICRO_BATCH
VAL_BATCH = 16
VAL_RUN_FREQ = 20
TEMP = 0.1

RET_SEQUENCES = 4

timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
dir_path = Path(f"thesis/out/{timestamp}")
dir_path.mkdir(parents=True, exist_ok=True)

token_level_timestamp = "2025-11-05_13-36"

MODEL_PATH = "facebook/Perception-LM-1B"
NUM_EPOCHS = 3

processor = AutoProcessor.from_pretrained(MODEL_PATH, use_fast=True)

prompt_1 = "Please read 2 texts describing human behavior. Output two scores based on them. The first one should be 1 if the texts agree on whether the person is lying or telling the truth. Please consider the main conclusion of each text, they will include various counterarguments but focus only on the final/dominating direction. Otherwise, if the texts disagree or one of them doesn't lead to any specific conclusion, output 0. Then rate how different behavioral cues are described and reasoned about. Check how much semantic overlap there is between those 2 texts in the context of those cues. The result should be in range 0.0(no overlap) through 0.1, 0.2 etc. to 1.0 (perfect overlap). Be strict, also give lower scores if one of the texts is very generic/bland, without mentioning specific clues. Only output those two numbers, starting from 0 or 1 and then a number in the range 0.0-1.0.\n\nText 1:\n"

prompt_2 = "\n\nText 2:\n"

scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)

for split_id in range(1, 2):  # change!
    print(f"Split id: {split_id}")
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_PATH, dtype=torch.bfloat16
    ).to("cuda")
    model = PeftModel.from_pretrained(
        model, f"thesis/out/{token_level_timestamp}/model_split{split_id}_epoch4"
    ).to("cuda")
    model.train()

    for name, param in model.named_parameters():
        if "lora" not in name:
            param.requires_grad_(False)
        else:
            param.requires_grad_(True)

    train_dataset = DolosDataset(
        f"thesis/data/train_fold{split_id}.csv", Path("thesis/data")
    )
    train_dataloader = DataLoader(
        train_dataset,
        MICRO_BATCH,
        shuffle=True,
        collate_fn=lambda batch: (
            [sample[0] for sample in batch],
            [sample[1] for sample in batch],
        ),
    )

    val_dataset = DolosDataset(
        f"thesis/data/val_fold{split_id}.csv", Path("thesis/data")
    )
    val_dataloader = DataLoader(
        val_dataset,
        VAL_BATCH,
        shuffle=True,
        collate_fn=lambda batch: (
            [sample[0] for sample in batch],
            [sample[1] for sample in batch],
        ),
    )

    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=2e-5)
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
        for i, (input, input_completed) in enumerate(train_dataloader):
            input = processor.apply_chat_template(
                input,
                num_frames=16,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
                padding=True,
            )
            input_completed = processor.apply_chat_template(
                input_completed,
                num_frames=16,
                add_generation_prompt=False,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
                padding=True,
            )
            input = {
                k: (
                    v.to(model.device, dtype=torch.bfloat16)
                    if torch.is_floating_point(v)
                    else v.to(model.device)
                )
                for k, v in input.items()
            }

            with torch.inference_mode():
                generated_ids = model.generate(
                    **input,
                    max_new_tokens=1000,
                    do_sample=True,
                    top_k=4,
                    num_return_sequences=RET_SEQUENCES,
                    repetition_penalty=1.2,
                    no_repeat_ngram_size=3
                )

            generated_ids_trimmed = generated_ids[
                :, input["input_ids"].size(1) :
            ].clone()
            generated_text_trimmed = processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )

            attention_mask = (
                (generated_ids != processor.tokenizer.pad_token_id)
                .to("cuda")
                .to(torch.long)
            )
            with torch.enable_grad():
                output = model(
                    input_ids=generated_ids,
                    pixel_values_videos=input["pixel_values_videos"].repeat(
                        RET_SEQUENCES, 1, 1, 1, 1
                    ),
                    attention_mask=attention_mask,
                )

            logits = output.logits[:, input["input_ids"].size(1) - 1 : -1, :].to(
                torch.float32
            )
            log_probs = F.log_softmax(logits, dim=-1)

            token_log_probs = log_probs.gather(
                -1, generated_ids_trimmed.unsqueeze(-1)
            ).squeeze(-1)
            token_log_probs = (
                token_log_probs * attention_mask[:, -token_log_probs.shape[1] :]
            )

            sequence_log_probs = token_log_probs.sum(dim=-1)
            q = torch.softmax(sequence_log_probs * TEMP, dim=0)

            expected_ids_trimmed = input_completed["input_ids"][
                :, input["input_ids"].size(1) :
            ]
            expected_text_trimmed = processor.batch_decode(
                expected_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0]

            risk_values = []

            for text in generated_text_trimmed:
                full_prompt = prompt_1 + text + prompt_2 + expected_text_trimmed

                response = client.responses.create(
                    model="gpt-4.1-mini", input=full_prompt
                )

                label_score = 0
                cues_score = 0

                try:
                    label_score, cues_score = map(float, response.output_text.split())
                except ValueError:
                    print(
                        f"WARNING: Incorrect answer from OpenAI: {response.output_text}"
                    )
                  
                rouge_score = scorer.score(expected_text_trimmed, text)
                rouge_score = np.mean(
                    [
                        rouge_score["rouge1"].fmeasure,
                        rouge_score["rouge2"].fmeasure,
                        rouge_score["rougeL"].fmeasure,
                    ]
                )

                total_score = 0.3 * label_score + 0.4 * cues_score + 0.3 * rouge_score
                risk_values.append(1 - total_score)

            risk_values = torch.tensor(risk_values).to(q.device)

            loss = (q * risk_values).sum() / GRAD_ACCU_STEPS
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
                torch.nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, model.parameters()), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
            
            if i % VAL_RUN_FREQ == VAL_RUN_FREQ - 1:
                save_dir = f"thesis/out/{timestamp}/model_split{split_id}_epoch{epoch}_minibatch{i}"
                model.save_pretrained(save_dir)

                torch.save({
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                }, Path(save_dir) / "training_state.pt")
                
                with open("validation_output.txt", "a") as f:
                    print(f"From minibatch {i} ====>", file=f)
                    for j, (input, input_completed) in enumerate(val_dataloader):
                        if j == 2:
                            break
                        input = processor.apply_chat_template(
                            input,
                            num_frames=16,
                            add_generation_prompt=True,
                            tokenize=True,
                            return_dict=True,
                            return_tensors="pt",
                            padding=True,
                        )
                        input_completed = processor.apply_chat_template(
                            input_completed,
                            num_frames=16,
                            add_generation_prompt=False,
                            tokenize=True,
                            return_dict=True,
                            return_tensors="pt",
                            padding=True,
                        )
                        input = {
                            k: (
                                v.to(model.device, dtype=torch.bfloat16)
                                if torch.is_floating_point(v)
                                else v.to(model.device)
                            )
                            for k, v in input.items()
                        }
                        with torch.inference_mode():
                            generated_ids = model.generate(
                                **input,
                                max_new_tokens=1000,
                                do_sample=True,
                                top_k=10,
                                num_return_sequences=RET_SEQUENCES,
                                repetition_penalty=1.2,
                                no_repeat_ngram_size=3
                            )

                        generated_ids_trimmed = generated_ids[
                            :, input["input_ids"].size(1) :
                        ]
                        generated_text_trimmed = processor.batch_decode(
                            generated_ids_trimmed,
                            skip_special_tokens=True,
                            clean_up_tokenization_spaces=False,
                        )
                        expected_ids_trimmed = input_completed["input_ids"][
                            :, input["input_ids"].size(1) :
                        ]
                        expected_text_trimmed = processor.batch_decode(
                            expected_ids_trimmed,
                            skip_special_tokens=True,
                            clean_up_tokenization_spaces=False,
                        )

                        for ref, pred in zip(expected_text_trimmed, generated_text_trimmed):    
                            print("*******Gold********", file=f)
                            print(ref, file=f)
                            print("*******Generated********", file=f)
                            print(pred, file=f)
                            rouge_score = scorer.score(ref, pred)
                            rouge_score = np.mean(
                                [
                                    rouge_score["rouge1"].fmeasure,
                                    rouge_score["rouge2"].fmeasure,
                                    rouge_score["rougeL"].fmeasure,
                                ]
                            )
                            print(f"*******Rouge score********", file=f)
                            print(rouge_score, file=f)
        try:
            if any(p.grad is not None and p.grad.abs().sum()>0 for p in optimizer.param_groups[0]['params']):
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
        except:
            print("Error in leftovers")

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
