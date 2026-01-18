# Developed as part of a BSc thesis at the Faculty of Computer Science, Bialystok Univesity of Technology

import torch
from peft import PeftModel
from sklearn.metrics import auc, f1_score, roc_curve
from torch.utils.data import DataLoader
from transformers import (
    AutoImageProcessor,
    TimesformerConfig,
    TimesformerForVideoClassification,
)

from thesis.utils.dataset_dolos import DolosClassificationDataset
from thesis.utils.utils import set_seed

set_seed(42)

BATCH_SIZE = 72
EPOCHS = 20

timestamp = "2025-11-03_00-45"

processor = AutoImageProcessor.from_pretrained(
    "facebook/timesformer-base-finetuned-k600"
)
config = TimesformerConfig()
config.num_labels = 2


for split_id in range(1, 4):
    print(f"Split {split_id}")

    val_dataset = DolosClassificationDataset(
        f"thesis/data/val_fold{split_id}.csv", "thesis/data/video", processor
    )
    test_dataset = DolosClassificationDataset(
        f"thesis/data/test_fold{split_id}.csv", "thesis/data/video", processor
    )
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    best_val_accuracy = float("-inf")
    best_test_accuracy = 0
    best_test_auc = 0
    best_test_f1 = 0

    for epoch in range(EPOCHS):
        print(f"Epoch {epoch + 1}")

        save_path_lora = (
            f"thesis/out/{timestamp}/lora_timesformer_split{split_id}_epoch{epoch}"
        )
        save_path = (
            f"thesis/out/{timestamp}/timesformer_split{split_id}_epoch{epoch}.pt"
        )

        model = TimesformerForVideoClassification.from_pretrained(
            "facebook/timesformer-base-finetuned-k600",
            config=config,
            ignore_mismatched_sizes=True,
        )
        lora_model = PeftModel.from_pretrained(model, save_path_lora)
        classifier_state_dict = torch.load(save_path, map_location="cpu")
        lora_model.base_model.model.classifier.load_state_dict(classifier_state_dict)
        lora_model = lora_model.to("cuda").to(torch.bfloat16)
        lora_model.eval()

        correct_val_preds = 0
        all_val_preds = 0

        for pixel_values, labels in val_dataloader:
            pixel_values = pixel_values.to(lora_model.device).to(torch.bfloat16)
            labels = labels.to(lora_model.device)

            with torch.no_grad():
                output = lora_model(pixel_values=pixel_values)
                preds = torch.argmax(output.logits, dim=-1)
            correct_val_preds += torch.sum(preds == labels).item()
            all_val_preds += preds.size(0)

        val_accuracy = correct_val_preds / all_val_preds * 100
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            all_test_preds = []
            all_test_probs = []
            all_test_labels = []
            for pixel_values, labels in test_dataloader:
                pixel_values = pixel_values.to(lora_model.device).to(torch.bfloat16)
                labels = labels.to(lora_model.device)

                with torch.no_grad():
                    output = lora_model(pixel_values=pixel_values)
                    preds = torch.argmax(output.logits, dim=-1)
                    probs = torch.softmax(output.logits, dim=-1)[:, 1]
                all_test_preds.append(preds)
                all_test_probs.append(probs)
                all_test_labels.append(labels)
            all_test_preds = torch.cat(all_test_preds, 0).cpu().numpy()
            all_test_probs = (
                torch.cat(all_test_probs, 0).to(torch.float32).cpu().numpy()
            )
            all_test_labels = torch.cat(all_test_labels, 0).cpu().numpy()
            best_test_accuracy = (
                torch.sum(all_test_preds == all_test_labels).item()
                / all_test_preds.size(0)
                * 100
            )
            best_test_f1 = f1_score(all_test_labels, all_test_preds, pos_label=0)
            fpr, tpr, thresholds = roc_curve(all_test_labels, all_test_probs)
            best_test_auc = auc(fpr, tpr)

    print(f"Test accuracy for split {split_id}: {best_test_accuracy}")
    print(f"Test F1 for split {split_id}: {best_test_f1}")
    print(f"Test AUC for split {split_id}: {best_test_auc}")
