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

from thesis.utils.dataset_dolos import DolosDataset

logging.set_verbosity_error()

client = OpenAI()
# finetuning only the language model + checking training loss + using MRT from epoch 5

DEFAULT_BATCH_SIZE = 2  # cus we have 8 outputs per input
GRAD_ACCU_STEPS = 2
RET_SEQUENCES = 4

timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
dir_path = Path(f"out/{timestamp}")
dir_path.mkdir(parents=True, exist_ok=True)

prev_timestamp = (
    "2025-11-05_13-36_perception_lm_v2_language_only__better_eval__train_loss"
)

MODEL_PATH = "facebook/Perception-LM-1B"  # kept, for fair comparison, more capacity and better video understanding abilities
NUM_EPOCHS = 1  # change!

processor = AutoProcessor.from_pretrained(MODEL_PATH, use_fast=True)

prompt_1 = "Please read 2 texts describing human behavior. Output two scores based on them. The first one should be 1 if the texts agree on whether the person is lying or telling the truth. Please consider the main conclusion of each text, they will include various counterarguments but focus only on the final/dominating direction. Otherwise, if the texts disagree, output 0. Then rate how different behavioral cues are described and reasoned about. Check how much semantic overlap there is between those 2 texts in the context of those cues. The result should be in range 0.0(no overlap) through 0.1, 0.2 etc. to 1.0 (perfect overlap). Be strict. Only output those two numbers, starting from 0 or 1 and then a number in the range 0.0-1.0.\n\nText 1:\n"

prompt_2 = "\n\nText 2:\n"

for split_id in range(1, 2):  # change!
    print(f"Split id: {split_id}")
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_PATH, dtype=torch.bfloat16
    ).to("cuda")
    model = PeftModel.from_pretrained(
        model, f"out/{prev_timestamp}/model_split{split_id}_epoch4"
    ).to("cuda")
    model.train()

    for name, param in model.named_parameters():
        if "lora" not in name:
            param.requires_grad_(False)
        else:
            param.requires_grad_(True)

    train_dataset = DolosDataset(
        f"thesis/data/train_fold{split_id}.csv", Path("./data")
    )
    train_dataloader = DataLoader(
        train_dataset,
        DEFAULT_BATCH_SIZE // GRAD_ACCU_STEPS,
        shuffle=True,
        collate_fn=lambda batch: (
            [sample[0] for sample in batch],
            [sample[1] for sample in batch],
        ),
    )

    total_steps = (
        ceil(len(train_dataset) / DEFAULT_BATCH_SIZE) * NUM_EPOCHS
    )  # check if makes sense!
    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=2e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=1e-6)

    all_total_losses = []

    for epoch in range(NUM_EPOCHS):
        print(f"Epoch: {epoch}")
        total_loss = 0
        for i, (input, input_completed) in enumerate(train_dataloader):
            if i == 1:
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
                )

            generated_ids_trimmed = generated_ids[
                :, input["input_ids"].size(1) :
            ].clone()
            generated_text_trimmed = processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )

            attention_mask = (generated_ids != processor.tokenizer.pad_token_id).to(
                torch.long, "cuda"
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
            sequence_log_probs *= 0.05
            sequence_log_probs = sequence_log_probs - torch.max(sequence_log_probs)

            sequence_probs = sequence_log_probs.exp()
            q = sequence_probs / sequence_probs.sum()

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

                total_score = 0.6 * label_score + 0.6 * cues_score
                risk_values.append(1 - total_score)

            risk_values = torch.tensor(risk_values)

            loss = (q * risk_values).mean()  # maybe sum? maybe log q?

            print(loss)

            loss.backward()
            # change this to include grad accumulation
            optimizer.step()
            optimizer.no_grad()

    #     model.save_pretrained(f"out/{timestamp}/model_split{split_id}_epoch{epoch}")

    # if get_rank() == 0:
    #     plt.plot(all_total_losses, marker='o')
    #     plt.title("Train Loss Plot")
    #     plt.xlabel("Epoch")
    #     plt.ylabel("Loss")

    #     plt.grid(True)
    #     plt.tight_layout()

    #     plt.savefig(f"out/{timestamp}/model_split{split_id}_train_losses.png") ## check if underfitting
    # barrier()
